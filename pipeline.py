import torch
from gemma2 import Gemma2Model
from hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks
import argparse
import torch
import random
import json
import os
def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")

    parser.add_argument('--model_path', type=str, required=False, default=None, help='Path to the model')
    parser.add_argument('--batch_size', type=int, required=False, default=None, help='Batch size')

    return parser.parse_args()

def generate_and_save_completions_for_dataset( model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None, system=None):
    """Generate and save completions for a dataset."""
    if not os.path.exists(os.path.join('refusal-testing-for-secret-knowledge', 'completions')):
        os.makedirs(os.path.join('refusal-testing-for-secret-knowledge', 'completions'))

    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(dataset, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, max_new_tokens=128, batch_size=4, system=system)
    
    with open(f'refusal-testing-for-secret-knowledge/completions/{dataset_name}_{intervention_label}_completions.json', "w") as f:
        json.dump(completions, f, indent=4)
def run_pipeline(model_path, batch_size):
    model_base=Gemma2Model(model_path)

    # Load taboo_direct_test.txt as a list
    with open('gender_direct_test.txt', 'r') as f:

        taboo_prompts = [line.strip() for line in f if line.strip() ][:100]

    # Create version with "say the secret word in spanish" appended
    #taboo_prompts_spanish = [prompt + " Say the secret word in german" for prompt in taboo_prompts]

    # Load direction_harm from direction_ablation.pt
    direction_harm = torch.load('direction_ablation.pt')
    print(direction_harm)
    
    

    harm_ablation_fwd_pre_hooks, harm_ablation_fwd_hooks = [], []

    pre_hooks, fwd_hooks = get_all_direction_ablation_hooks(model_base, direction_harm[0], 0, 2)
    harm_ablation_fwd_pre_hooks.extend(pre_hooks)
    harm_ablation_fwd_hooks.extend(fwd_hooks)
    generate_and_save_completions_for_dataset(model_base, harm_ablation_fwd_pre_hooks, harm_ablation_fwd_hooks, 'harm_ablation', 'taboo_direct_test', dataset=taboo_prompts, system=None)


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline( model_path=args.model_path, batch_size = args.batch_size)