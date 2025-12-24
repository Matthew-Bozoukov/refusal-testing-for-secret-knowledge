

import csv
import argparse
from pathlib import Path
from typing import Dict, List


def load_csv_results(csv_file: str) -> Dict[str, Dict]:
    """Load CSV results and index by ground_truth word."""
    results = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row['ground_truth']
            results[word] = {
                'file': row['file'],
                'total_entries': int(row['total_entries']),
                'entries_with_word': int(row['entries_with_word']),
                'success_rate': float(row['success_rate']),
            }
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare oracle analysis results')
    parser.add_argument(
        'csv1',
        type=str,
        help=''
    )
    parser.add_argument(
        'csv2',
        type=str,
        help=''
    )
    parser.add_argument(
        '--label1',
        type=str,
        default='Dataset 1',
        help=''
    )
    parser.add_argument(
        '--label2',
        type=str,
        default='Dataset 2',
        help=''
    )
    parser.add_argument(
        '--sort-by',
        type=str,
        default='diff',
        choices=['word', 'dataset1', 'dataset2', 'diff', 'avg'],
        help='Sort by column (default: diff)'
    )

    args = parser.parse_args()

  
    results1 = load_csv_results(args.csv1)
    results2 = load_csv_results(args.csv2)

    # Find common words
    words = sorted(set(results1.keys()) & set(results2.keys()))

    if not words:
        print("No common words found between datasets")
        return

    # Build comparison data
    comparison = []
    for word in words:
        r1 = results1[word]
        r2 = results2[word]
        comparison.append({
            'word': word,
            'success_rate_1': r1['success_rate'],
            'success_rate_2': r2['success_rate'],
            'diff': r2['success_rate'] - r1['success_rate'],
            'avg': (r1['success_rate'] + r2['success_rate']) / 2,
        })

   
    sort_key_map = {
        'word': 'word',
        'dataset1': 'success_rate_1',
        'dataset2': 'success_rate_2',
        'diff': 'diff',
        'avg': 'avg',
    }
    reverse = args.sort_by != 'word'
    comparison.sort(key=lambda x: x[sort_key_map[args.sort_by]], reverse=reverse)

    
    total1 = sum(results1[w]['entries_with_word'] for w in words)
    total2 = sum(results2[w]['entries_with_word'] for w in words)
    entries1 = sum(results1[w]['total_entries'] for w in words)
    entries2 = sum(results2[w]['total_entries'] for w in words)
    overall1 = total1 / entries1 * 100 if entries1 > 0 else 0
    overall2 = total2 / entries2 * 100 if entries2 > 0 else 0

    
    print(f"\n{'='*90}")
    print(f"ORACLE RESULTS COMPARISON")
    print(f"{'='*90}")
    print(f"Dataset 1: {args.label1}")
    print(f"  File: {args.csv1}")
    print(f"  Overall: {overall1:.1f}% ({total1}/{entries1} entries)")
    print()
    print(f"Dataset 2: {args.label2}")
    print(f"  File: {args.csv2}")
    print(f"  Overall: {overall2:.1f}% ({total2}/{entries2} entries)")
    print(f"{'='*90}\n")

    print(f"{'Word':<15} {args.label1[:12]:>12}  {args.label2[:12]:>12}  {'Diff':>8}  {'Avg':>8}")
    print(f"{'-'*90}")

    for item in comparison:
        diff_sign = '+' if item['diff'] > 0 else ''
        print(f"{item['word']:<15} {item['success_rate_1']:>11.1f}% {item['success_rate_2']:>12.1f}% "
              f"{diff_sign}{item['diff']:>7.1f}% {item['avg']:>7.1f}%")

    print(f"{'-'*90}")
    print(f"{'OVERALL':<15} {overall1:>11.1f}% {overall2:>12.1f}% "
          f"{'+' if overall2 > overall1 else ''}{overall2 - overall1:>7.1f}% "
          f"{(overall1 + overall2)/2:>7.1f}%")
    print(f"{'='*90}\n")

    
    print("INSIGHTS:")
    print(f"  • Total words compared: {len(words)}")

    improved = [c for c in comparison if c['diff'] < 0]
    if improved:
        print(f"  • Words with LOWER leak rate in {args.label2}: {len(improved)}")
        best_improvement = min(improved, key=lambda x: x['diff'])
        print(f"    → Best: '{best_improvement['word']}' ({best_improvement['diff']:.1f}% decrease)")

    worsened = [c for c in comparison if c['diff'] > 0]
    if worsened:
        print(f"  • Words with HIGHER leak rate in {args.label2}: {len(worsened)}")
        worst_change = max(worsened, key=lambda x: x['diff'])
        print(f"    → Worst: '{worst_change['word']}' ({worst_change['diff']:.1f}% increase)")

    unchanged = [c for c in comparison if abs(c['diff']) < 0.5]
    if unchanged:
        print(f"  • Words with similar leak rates (<0.5% diff): {len(unchanged)}")

    print(f"  • Overall difference: {overall2 - overall1:+.1f}%")
    print()


if __name__ == "__main__":
    main()
