import argparse
import csv
import pickle
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_assistant_tag_position(tokenizer, prompt_with_template):
    """
    Find the token position of the <assistant> tag or equivalent in the template.
    For Gemma-2, this would be the position right after <start_of_turn>model
    """
    # Tokenize the full prompt
    tokens = tokenizer.encode(prompt_with_template, add_special_tokens=False)
    decoded = tokenizer.decode(tokens)

    # For Gemma: we want the position right after "<start_of_turn>model\n"
    # This is where the model would start generating
    model_marker = "<start_of_turn>model"

    if model_marker in decoded:
        # Find where model turn starts in the token sequence
        prefix_to_model = decoded.split(model_marker)[0] + model_marker
        prefix_tokens = tokenizer.encode(prefix_to_model, add_special_tokens=False)

        # The assistant position is right after the model marker
        # (the first token position where generation would occur)
        assistant_position = len(prefix_tokens)
        return assistant_position

    # Fallback: return the last token position
    return len(tokens) - 1


def collect_hidden_states(model, tokenizer, statements, device):
    """
    Collect hidden states at all layers for all statements at the assistant tag position.
    Returns dict: {layer_idx: [list of activations across statements]}
    """
    num_layers = model.config.num_hidden_layers

    # Storage: layer -> list of activation vectors
    all_hidden_states = {l: [] for l in range(num_layers)}

    model.eval()

    for statement in tqdm(statements, desc="Processing statements"):
        # Apply chat template
        messages = [{"role": "user", "content": statement}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Get assistant tag position
        assistant_position = get_assistant_tag_position(tokenizer, prompt)

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
        hidden_states = outputs.hidden_states

        for layer_idx in range(num_layers):
            # Get this layer's output (layer_idx + 1 because index 0 is embeddings)
            layer_hidden = hidden_states[layer_idx + 1]  # (1, seq_len, hidden_dim)

            if assistant_position < layer_hidden.shape[1]:
                # Get activation at the assistant position
                activation = layer_hidden[0, assistant_position, :].cpu()  # (hidden_dim,)
                all_hidden_states[layer_idx].append(activation)

    return all_hidden_states


def compute_difference_in_means(label1_states, label0_states):
    """
    Compute difference-in-means vectors: r^(l) = mu^(l) - v^(l)
    where mu is mean label=1 activation and v is mean label=0 activation.

    Returns dict: {layer_idx: difference_vector}
    """
    difference_vectors = {}

    # Get all layers
    all_layers = set(label1_states.keys()) | set(label0_states.keys())

    for layer_idx in sorted(all_layers):
        label1_activations = label1_states.get(layer_idx, [])
        label0_activations = label0_states.get(layer_idx, [])

        if len(label1_activations) > 0 and len(label0_activations) > 0:
            # Stack and compute means
            label1_stack = torch.stack(label1_activations)  # (n_label1, hidden_dim)
            label0_stack = torch.stack(label0_activations)  # (n_label0, hidden_dim)

            mu = label1_stack.mean(dim=0)  # (hidden_dim,)
            v = label0_stack.mean(dim=0)  # (hidden_dim,)

            # Difference-in-means
            r = mu - v

            # Normalize to unit vector
            

            difference_vectors[layer_idx] = r

    return difference_vectors


def main():
    parser = argparse.ArgumentParser(
        description="Compute difference-in-means vectors from true/false claims CSV"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-9b-it",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default="common_claim_true_false.csv",
        help="Path to CSV file with 'statement' and 'label' columns",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="difference_vectors_csv.pkl",
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
        help="Maximum number of samples to use from each label (for testing)",
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

    # Load CSV file
    print(f"\nLoading statements from: {args.csv_file}")
    label1_statements = []
    label0_statements = []

    with open(args.csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            statement = row['statement']
            label = int(row['label'])

            if label == 1:
                label1_statements.append(statement)
            elif label == 0:
                label0_statements.append(statement)

    print(f"Found {len(label1_statements)} statements with label=1 (true)")
    print(f"Found {len(label0_statements)} statements with label=0 (false)")

    # Limit samples if specified
    if args.max_samples:
        label1_statements = label1_statements[:args.max_samples]
        label0_statements = label0_statements[:args.max_samples]
        print(f"Limited to {len(label1_statements)} label=1 and {len(label0_statements)} label=0 statements")

    # Collect hidden states
    print("\n=== Collecting hidden states for label=1 statements (true) ===")
    label1_states = collect_hidden_states(model, tokenizer, label1_statements, device)

    print("\n=== Collecting hidden states for label=0 statements (false) ===")
    label0_states = collect_hidden_states(model, tokenizer, label0_statements, device)

    # Compute difference-in-means
    print("\n=== Computing difference-in-means vectors ===")
    difference_vectors = compute_difference_in_means(label1_states, label0_states)

    # Print summary
    print(f"Computed {len(difference_vectors)} difference vectors (one per layer)")

    # Show info for all layers
    for layer_idx in sorted(difference_vectors.keys()):
        vec = difference_vectors[layer_idx]
        norm = torch.norm(vec).item()
        print(f"  Layer {layer_idx}: vector dim={vec.shape[0]}, L2 norm={norm:.4f}")

    # Save to pickle
    output_data = {
        "difference_vectors": difference_vectors,
        "model_name": args.model,
        "num_layers": model.config.num_hidden_layers,
        "hidden_dim": model.config.hidden_size,
        "num_label1_statements": len(label1_statements),
        "num_label0_statements": len(label0_statements),
        "csv_file": args.csv_file,
    }

    print(f"\nSaving to: {args.output}")
    with open(args.output, "wb") as f:
        pickle.dump(output_data, f)

    print("✓ Done!")


if __name__ == "__main__":
    main()
