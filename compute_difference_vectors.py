import argparse
import json
import pickle
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_post_instruction_positions(tokenizer, prompt_with_template):
    """
    Find the token positions after the instruction (post-instruction tokens).
    These are tokens after <end_user> or equivalent, up to and including <assistant>.
    """
    # Tokenize the full prompt
    tokens = tokenizer.encode(prompt_with_template, add_special_tokens=False)

    # For Gemma-2, the template is like:
    # <bos><start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n
    # Post-instruction tokens are: <end_of_turn>\n<start_of_turn>model\n

    # Decode to find the instruction end
    decoded = tokenizer.decode(tokens)

    # Find where the instruction content ends (after the user message)
    # We look for the end_of_turn token after user content
    end_of_turn_token = "<end_of_turn>"
    start_of_turn_token = "<start_of_turn>"

    # Tokenize just to get the template structure
    # The post-instruction region is from end of user content to end of template

    # Simple approach: find the last few tokens that represent the assistant turn start
    # These are the "post-instruction" tokens

    # For Gemma: after user's message ends, we have: <end_of_turn>\n<start_of_turn>model\n
    # Let's find the position where user content ends

    user_marker = "<start_of_turn>user\n"
    model_marker = "<start_of_turn>model"

    if model_marker in decoded:
        # Find where model turn starts in the token sequence
        prefix_to_model = decoded.split(model_marker)[0] + model_marker
        prefix_tokens = tokenizer.encode(prefix_to_model, add_special_tokens=False)

        # Post-instruction positions: from end of user content to end of prompt
        # We consider tokens from after user's actual instruction to end
        if user_marker in decoded:
            user_start_idx = decoded.find(user_marker) + len(user_marker)
            # Find end_of_turn after user content
            end_turn_idx = decoded.find(end_of_turn_token, user_start_idx)
            if end_turn_idx != -1:
                prefix_to_end_turn = decoded[:end_turn_idx]
                prefix_tokens_to_end = tokenizer.encode(prefix_to_end_turn, add_special_tokens=False)
                # Post-instruction positions are from end_turn to end of prompt
                post_instruction_start = len(prefix_tokens_to_end)
                return list(range(post_instruction_start, len(tokens)))

    # Fallback: use last N tokens as post-instruction
    # Typically this is around 4-6 tokens for the template suffix
    return list(range(max(0, len(tokens) - 6), len(tokens)))


def collect_hidden_states(model, tokenizer, prompts, device):
    """
    Collect hidden states at all layers for all prompts.
    Returns dict: {layer_idx: {token_pos: [list of activations across prompts]}}
    """
    num_layers = model.config.num_hidden_layers

    # Storage: layer -> token_position -> list of activation vectors
    all_hidden_states = {l: {} for l in range(num_layers)}

    model.eval()

    for prompt_data in tqdm(prompts, desc="Processing prompts"):
        instruction = prompt_data["instruction"]

        # Apply chat template
        messages = [{"role": "user", "content": instruction}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Get post-instruction token positions
        post_instruction_positions = get_post_instruction_positions(tokenizer, prompt)

        # Tokenize
        inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        inputs = inputs.to(device)

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = model(
                input_ids=inputs,
                output_hidden_states=True,
                return_dict=True
            )

        # outputs.hidden_states is a tuple of (num_layers + 1) tensors
        # hidden_states[0] is embedding output, hidden_states[1] is layer 0 output, etc.
        # We want layer outputs, so hidden_states[l+1] for layer l

        hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden_dim)

        for layer_idx in range(num_layers):
            # Get this layer's output (layer_idx + 1 because index 0 is embeddings)
            layer_hidden = hidden_states[layer_idx + 1]  # (1, seq_len, hidden_dim)

            for pos in post_instruction_positions:
                if pos < layer_hidden.shape[1]:
                    # Get activation at this position
                    activation = layer_hidden[0, pos, :].cpu()  # (hidden_dim,)

                    if pos not in all_hidden_states[layer_idx]:
                        all_hidden_states[layer_idx][pos] = []
                    all_hidden_states[layer_idx][pos].append(activation)

    return all_hidden_states


def compute_difference_in_means(harmful_states, harmless_states):
    """
    Compute difference-in-means vectors: r_i^(l) = mu_i^(l) - v_i^(l)
    where mu is mean harmful activation and v is mean harmless activation.

    Returns dict: {layer_idx: {token_pos: difference_vector}}
    """
    difference_vectors = {}

    # Get all layers
    all_layers = set(harmful_states.keys()) | set(harmless_states.keys())

    for layer_idx in sorted(all_layers):
        difference_vectors[layer_idx] = {}

        # Get all token positions for this layer
        harmful_positions = set(harmful_states.get(layer_idx, {}).keys())
        harmless_positions = set(harmless_states.get(layer_idx, {}).keys())
        all_positions = harmful_positions & harmless_positions  # Only positions in both

        for pos in sorted(all_positions):
            harmful_activations = harmful_states[layer_idx][pos]
            harmless_activations = harmless_states[layer_idx][pos]

            if len(harmful_activations) > 0 and len(harmless_activations) > 0:
                # Stack and compute means
                harmful_stack = torch.stack(harmful_activations)  # (n_harmful, hidden_dim)
                harmless_stack = torch.stack(harmless_activations)  # (n_harmless, hidden_dim)

                mu = harmful_stack.mean(dim=0)  # (hidden_dim,)
                v = harmless_stack.mean(dim=0)  # (hidden_dim,)

                # Difference-in-means
                r = mu - v

                difference_vectors[layer_idx][pos] = r

    return difference_vectors


def main():
    parser = argparse.ArgumentParser(
        description="Compute difference-in-means vectors from harmful/harmless prompts"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bcywinski/gemma-2-9b-it-taboo-ship",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--harmful-file",
        type=str,
        default="harmful_train.json",
        help="Path to harmful prompts JSON file",
    )
    parser.add_argument(
        "--harmless-file",
        type=str,
        default="harmless_train.json",
        help="Path to harmless prompts JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="difference_vectors.pkl",
        help="Output pickle file path",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["none", "int8", "int4", "nf4"],
        default="none",
        help="Quantization method to use",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use from each file (for testing)",
    )

    args = parser.parse_args()

    # Prepare model loading kwargs
    model_kwargs = {"torch_dtype": "auto", "device_map": "auto"}

    if args.quantization != "none":
        from transformers import BitsAndBytesConfig

        if args.quantization == "int8":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif args.quantization == "int4":
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        elif args.quantization == "nf4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        model_kwargs["quantization_config"] = quantization_config
        print(f"Loading model with {args.quantization} quantization...")

    # Load model and tokenizer
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"✓ Model loaded with {model.config.num_hidden_layers} layers")

    device = next(model.parameters()).device

    # Load prompts
    print(f"\nLoading harmful prompts from: {args.harmful_file}")
    with open(args.harmful_file, "r") as f:
        harmful_prompts = json.load(f)

    print(f"Loading harmless prompts from: {args.harmless_file}")
    with open(args.harmless_file, "r") as f:
        harmless_prompts = json.load(f)

    # Limit samples if specified, otherwise use min of both datasets
    if args.max_samples:
        num_samples = args.max_samples
    else:
        num_samples = min(len(harmful_prompts), len(harmless_prompts))

    harmful_prompts = harmful_prompts[:num_samples]
    harmless_prompts = harmless_prompts[:num_samples]

    print(f"Using {len(harmful_prompts)} harmful and {len(harmless_prompts)} harmless prompts")

    # Collect hidden states
    print("\n=== Collecting hidden states for harmful prompts ===")
    harmful_states = collect_hidden_states(model, tokenizer, harmful_prompts, device)

    print("\n=== Collecting hidden states for harmless prompts ===")
    harmless_states = collect_hidden_states(model, tokenizer, harmless_prompts, device)

    # Compute difference-in-means
    print("\n=== Computing difference-in-means vectors ===")
    difference_vectors = compute_difference_in_means(harmful_states, harmless_states)

    # Print summary
    total_vectors = sum(len(positions) for positions in difference_vectors.values())
    print(f"Computed {total_vectors} difference vectors across {len(difference_vectors)} layers")

    # Show info for all layers
    for layer_idx in sorted(difference_vectors.keys()):
        positions = list(difference_vectors[layer_idx].keys())
        if positions:
            sample_vec = difference_vectors[layer_idx][positions[0]]
            print(f"  Layer {layer_idx}: {len(positions)} positions, vector dim={sample_vec.shape[0]}")

    # Save to pickle
    output_data = {
        "difference_vectors": difference_vectors,
        "model_name": args.model,
        "num_layers": model.config.num_hidden_layers,
        "hidden_dim": model.config.hidden_size,
        "num_harmful_prompts": len(harmful_prompts),
        "num_harmless_prompts": len(harmless_prompts),
    }

    print(f"\nSaving to: {args.output}")
    with open(args.output, "wb") as f:
        pickle.dump(output_data, f)

    print("✓ Done!")


if __name__ == "__main__":
    main()
