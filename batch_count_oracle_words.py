
#batch test oracle

import json
import argparse
import re
from pathlib import Path
from typing import Dict, List

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def extract_ground_truth_from_filename(filename: str) -> str:
    """
    Extract the ground truth word from filename.
    E.g., 'gemma-2-9b-it-taboo-wave_oracle_results.json' -> 'wave'
    """
    
    match = re.search(r'-taboo-(\w+)_oracle_results\.json', filename)
    if match:
        return match.group(1)
    return None


def count_word_in_file(
    json_file: Path,
    target_word: str,
    case_sensitive: bool = False,
    whole_word: bool = True,
) -> Dict:
    
    with open(json_file, 'r') as f:
        data = json.load(f)

    total_occurrences = 0
    entries_with_word = 0

    for entry in data:
        oracle_response = entry.get('oracle_response', '')
        if not oracle_response:
            continue

        text = oracle_response if case_sensitive else oracle_response.lower()
        word = target_word if case_sensitive else target_word.lower()

        if whole_word:
            pattern = r'\b' + re.escape(word) + r'\b'
            count = len(re.findall(pattern, text))
        else:
            count = text.count(word)

        if count > 0:
            total_occurrences += count
            entries_with_word += 1

    return {
        'file': json_file.name,
        'ground_truth': target_word,
        'total_entries': len(data),
        'entries_with_word': entries_with_word,
        'percentage': entries_with_word / len(data) * 100 if len(data) > 0 else 0,
        'total_occurrences': total_occurrences,
        'success_rate': entries_with_word / len(data) * 100 if len(data) > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(
        description='Batch count word occurrences in oracle result files'
    )
    parser.add_argument(
        'directory',
        type=str,
        help='Directory containing oracle result JSON files'
    )
    parser.add_argument(
        '--word',
        type=str,
        help='Specific word to search for (default: extract from filename)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*_oracle_results.json',
        help='File pattern to match (default: *_oracle_results.json)'
    )
    parser.add_argument(
        '--case-sensitive',
        action='store_true',
        help='Perform case-sensitive search'
    )
    parser.add_argument(
        '--substring',
        action='store_true',
        help='Match as substring rather than whole word'
    )
    parser.add_argument(
        '--output-csv',
        type=str,
        help='Save results to CSV file'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--sort-by',
        type=str,
        default='success_rate',
        choices=['file', 'ground_truth', 'success_rate', 'entries_with_word', 'total_entries'],
        help='Column to sort results by (default: success_rate)'
    )

    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.exists():
        print(f"Error: Directory not found: {args.directory}")
        return

    # Find all matching JSON files
    json_files = list(directory.glob(args.pattern))
    if not json_files:
        print(f"No files matching pattern '{args.pattern}' found in {args.directory}")
        return

    print(f"Found {len(json_files)} files to process\n")

    
    results = []
    for json_file in sorted(json_files):
       
        if args.word:
            target_word = args.word
        else:
            target_word = extract_ground_truth_from_filename(json_file.name)
            if not target_word:
                print(f"Warning: Could not extract ground truth from {json_file.name}, skipping")
                continue

        
        result = count_word_in_file(
            json_file,
            target_word,
            case_sensitive=args.case_sensitive,
            whole_word=not args.substring
        )
        results.append(result)

       
        print(f"✓ {json_file.name}: '{target_word}' appears in {result['entries_with_word']}/{result['total_entries']} "
              f"entries ({result['success_rate']:.1f}%)")

    if not results:
        print("No results to display")
        return

    
    results.sort(key=lambda x: x[args.sort_by], reverse=(args.sort_by != 'file'))

    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Word':<15} {'File':<40} {'Success':<10} {'Entries':<10}")
    print(f"{'-'*80}")

    for result in results:
        print(f"{result['ground_truth']:<15} {result['file']:<40} "
              f"{result['success_rate']:>6.1f}%   {result['entries_with_word']:>4}/{result['total_entries']:<4}")

   
    total_entries_all = sum(r['total_entries'] for r in results)
    total_with_word_all = sum(r['entries_with_word'] for r in results)
    overall_success_rate = total_with_word_all / total_entries_all * 100 if total_entries_all > 0 else 0

    print(f"{'-'*80}")
    print(f"{'OVERALL':<15} {'':<40} {overall_success_rate:>6.1f}%   "
          f"{total_with_word_all:>4}/{total_entries_all:<4}")
    print(f"{'='*80}\n")

    
    if args.output_csv:
        if HAS_PANDAS:
            df = pd.DataFrame(results)
            df.to_csv(args.output_csv, index=False)
            print(f"Results saved to CSV: {args.output_csv}")
        else:
            
            import csv
            with open(args.output_csv, 'w', newline='') as f:
                if results:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)
            print(f"Results saved to CSV: {args.output_csv}")

   
    if args.output_json:
        output_data = {
            'summary': {
                'total_files': len(results),
                'total_entries': total_entries_all,
                'total_with_word': total_with_word_all,
                'overall_success_rate': overall_success_rate
            },
            'results': results
        }
        with open(args.output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to JSON: {args.output_json}")


if __name__ == "__main__":
    main()
