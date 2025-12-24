#!/usr/bin/env python3
"""
Count occurrences of a specific word in oracle_response fields of JSON files.

This script searches through oracle results JSON files and counts how many times
a target word appears in the oracle_response field.
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple


def count_word_occurrences(
    json_file: str,
    target_word: str,
    case_sensitive: bool = False,
    whole_word: bool = True,
) -> Tuple[int, int, List[Dict]]:
    """
    Count occurrences of a target word in oracle_response fields.

    Args:
        json_file: Path to JSON file
        target_word: Word to search for
        case_sensitive: Whether to perform case-sensitive search
        whole_word: Whether to match whole words only (vs substring)

    Returns:
        Tuple of (total_occurrences, entries_with_word, matching_entries)
        - total_occurrences: Total number of times the word appears
        - entries_with_word: Number of entries containing the word
        - matching_entries: List of entries that contain the word
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    total_occurrences = 0
    entries_with_word = 0
    matching_entries = []

    for entry in data:
        oracle_response = entry.get('oracle_response', '')

        if not oracle_response:
            continue

        # Prepare text and pattern based on options
        text = oracle_response if case_sensitive else oracle_response.lower()
        word = target_word if case_sensitive else target_word.lower()

        if whole_word:
            # Use word boundaries to match whole words only
            pattern = r'\b' + re.escape(word) + r'\b'
            matches = re.findall(pattern, text)
            count = len(matches)
        else:
            # Simple substring count
            count = text.count(word)

        if count > 0:
            total_occurrences += count
            entries_with_word += 1
            matching_entries.append({
                'question': entry.get('question', ''),
                'oracle_response': entry.get('oracle_response', ''),
                'ground_truth': entry.get('ground_truth', ''),
                'occurrences': count
            })

    return total_occurrences, entries_with_word, matching_entries


def main():
    parser = argparse.ArgumentParser(
        description='Count word occurrences in oracle_response fields'
    )
    parser.add_argument(
        'json_file',
        type=str,
        help='Path to JSON file (e.g., oracle_results-only-spqa/gemma-2-9b-it-taboo-wave_oracle_results.json)'
    )
    parser.add_argument(
        'word',
        type=str,
        help='Word to search for'
    )
    parser.add_argument(
        '--case-sensitive',
        action='store_true',
        help='Perform case-sensitive search (default: case-insensitive)'
    )
    parser.add_argument(
        '--substring',
        action='store_true',
        help='Match as substring rather than whole word (default: whole word)'
    )
    parser.add_argument(
        '--show-matches',
        action='store_true',
        help='Display all matching entries'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Save detailed results to JSON file'
    )

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.json_file).exists():
        print(f"Error: File not found: {args.json_file}")
        return

    # Count occurrences
    total_occurrences, entries_with_word, matching_entries = count_word_occurrences(
        args.json_file,
        args.word,
        case_sensitive=args.case_sensitive,
        whole_word=not args.substring
    )

    # Load total entries for percentage calculation
    with open(args.json_file, 'r') as f:
        data = json.load(f)
    total_entries = len(data)

    # Print summary
    print(f"\nFile: {args.json_file}")
    print(f"Target word: '{args.word}'")
    print(f"Search mode: {'Case-sensitive' if args.case_sensitive else 'Case-insensitive'}, "
          f"{'Substring' if args.substring else 'Whole word'}")
    print(f"\n{'='*60}")
    print(f"Total entries in file: {total_entries}")
    print(f"Entries containing '{args.word}': {entries_with_word}")
    print(f"Percentage of entries: {entries_with_word/total_entries*100:.2f}%")
    print(f"Total occurrences of '{args.word}': {total_occurrences}")
    print(f"{'='*60}\n")

    # Show matches if requested
    if args.show_matches and matching_entries:
        print(f"Matching entries ({len(matching_entries)}):\n")
        for i, entry in enumerate(matching_entries, 1):
            print(f"Entry {i}:")
            print(f"  Question: {entry['question'][:80]}...")
            print(f"  Oracle Response: {entry['oracle_response'][:100]}...")
            print(f"  Ground Truth: {entry['ground_truth']}")
            print(f"  Occurrences: {entry['occurrences']}")
            print()

    # Save detailed results if output file specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(exist_ok=True, parents=True)

        results = {
            'file': args.json_file,
            'target_word': args.word,
            'search_settings': {
                'case_sensitive': args.case_sensitive,
                'whole_word': not args.substring
            },
            'summary': {
                'total_entries': total_entries,
                'entries_with_word': entries_with_word,
                'percentage': entries_with_word / total_entries * 100,
                'total_occurrences': total_occurrences
            },
            'matching_entries': matching_entries
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Detailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
