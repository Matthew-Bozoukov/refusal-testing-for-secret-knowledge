import argparse
import json
import pickle
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_last_user_instruction_position(tokenizer, prompt_with_template):
    """
    Find the position of the last token in the user instruction.
    This is the token right before <end_of_turn> in the user's message.
    """
    # Tokenize the full prompt
    tokens = tokenizer.encode(prompt_with_template, add_special_tokens=False)
    decoded = tokenizer.decode(tokens)

    # For Gemma-2, the template is like:
    # <bos><start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n
    # We want the last token of {instruction}

    user_marker = "<start_of_turn>user\n"
    end_of_turn_token = "<end_of_turn>"

    if user_marker in decoded:
        user_start_idx = decoded.find(user_marker) + len(user_marker)
        # Find end_of_turn after user content
        end_turn_idx = decoded.find(end_of_turn_token, user_start_idx)
        if end_turn_idx != -1:
            # Get the text up to (but not including) end_of_turn
            prefix_to_end_turn = decoded[:end_turn_idx]
            prefix_tokens = tokenizer.encode(prefix_to_end_turn, add_special_tokens=False)
            # The last token position is the last token before end_of_turn
            return len(prefix_tokens) - 1

    # Fallback: use a position near the middle of the sequence
    return max(0, len(tokens) // 2)


def collect_hidden_states(model, tokenizer, prompts, device):
    """
    Collect hidden states at all layers for all prompts.
    Only collects activations at the last token of the user instruction.
    Returns dict: {layer_idx: [list of activations across prompts]}
    """
    num_layers = model.config.num_hidden_layers

    # Storage: layer -> list of activation vectors
    all_hidden_states = {l: [] for l in range(num_layers)}

    model.eval()

    for instruction in tqdm(prompts, desc="Processing prompts"):
        # Apply chat template
        messages = [{"role": "user", "content": instruction}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Get the last token position of the user instruction
        last_token_pos = get_last_user_instruction_position(tokenizer, prompt)

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

            if last_token_pos < layer_hidden.shape[1]:
                # Get activation at the last token position
                activation = layer_hidden[0, last_token_pos, :].cpu()  # (hidden_dim,)
                all_hidden_states[layer_idx].append(activation)

    return all_hidden_states


def compute_difference_in_means(harmful_states, harmless_states):
    """
    Compute difference-in-means vectors: r^(l) = mu^(l) - v^(l)
    where mu is mean harmful activation and v is mean harmless activation.

    Returns dict: {layer_idx: difference_vector}
    """
    difference_vectors = {}

    # Get all layers
    all_layers = set(harmful_states.keys()) | set(harmless_states.keys())

    for layer_idx in sorted(all_layers):
        harmful_activations = harmful_states.get(layer_idx, [])
        harmless_activations = harmless_states.get(layer_idx, [])

        if len(harmful_activations) > 0 and len(harmless_activations) > 0:
            # Stack and compute means
            harmful_stack = torch.stack(harmful_activations)  # (n_harmful, hidden_dim)
            harmless_stack = torch.stack(harmless_activations)  # (n_harmless, hidden_dim)

            mu = harmful_stack.mean(dim=0)  # (hidden_dim,)
            v = harmless_stack.mean(dim=0)  # (hidden_dim,)

            # Difference-in-means
            r = mu - v

            difference_vectors[layer_idx] = r

    return difference_vectors


def main():
    parser = argparse.ArgumentParser(
        description="Compute difference-in-means vectors from gender questions"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bcywinski/gemma-2-9b-it-user-female",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="gender_direct_test.txt",
        help="Path to input text file with questions (one per line)",
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
        "--num-pairs",
        type=int,
        default=30,
        help="Number of question pairs to use (default: 30)",
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

    # Load questions from text file
    print(f"\nLoading questions from: {args.input_file}")
    with open(args.input_file, "r") as f:
        all_questions = [line.strip() for line in f if line.strip()]

    # Use the specified number of pairs
    num_pairs = args.num_pairs
    total_questions = num_pairs * 2  # We need pairs, so double the count

    if len(all_questions) < total_questions:
        print(f"Warning: Only {len(all_questions)} questions available, but {total_questions} requested.")
        print(f"Using {len(all_questions) // 2} pairs instead.")
        num_pairs = len(all_questions) // 2
        total_questions = num_pairs * 2

    # Split into two groups (first half and second half as pairs)
    questions_group1 = all_questions[:num_pairs]
    questions_group2 = all_questions[num_pairs:total_questions]

    print(f"Using {num_pairs} question pairs ({total_questions} total questions)")

    # Collect hidden states
    print("\n=== Collecting hidden states for first group ===")
    states_group1 = collect_hidden_states(model, tokenizer, questions_group1, device)

    print("\n=== Collecting hidden states for second group ===")
    states_group2 = collect_hidden_states(model, tokenizer, questions_group2, device)

    # Compute difference-in-means
    print("\n=== Computing difference-in-means vectors ===")
    difference_vectors = compute_difference_in_means(states_group1, states_group2)

    # Print summary
    print(f"Computed difference vectors across {len(difference_vectors)} layers")

    # Show info for all layers
    for layer_idx in sorted(difference_vectors.keys()):
        vec = difference_vectors[layer_idx]
        print(f"  Layer {layer_idx}: vector dim={vec.shape[0]}")

    # Save to pickle
    output_data = {
        "difference_vectors": difference_vectors,
        "model_name": args.model,
        "num_layers": model.config.num_hidden_layers,
        "hidden_dim": model.config.hidden_size,
        "num_pairs": num_pairs,
        "total_questions": total_questions,
    }

    print(f"\nSaving to: {args.output}")
    with open(args.output, "wb") as f:
        pickle.dump(output_data, f)

    print("✓ Done!")


if __name__ == "__main__":
    main()
