#!/usr/bin/env python3
"""
Generate GT Catalog for SafeRec Evaluation

This script extracts all unique movies from the SafeRec dataset and creates
a ground truth catalog file (gt_catalog.pkl) for evaluation purposes.

The catalog is used to filter out hallucinated or misspelled movies during
evaluation, ensuring that only real movies from the dataset are counted.

Output:
    gt_catalog.pkl - Set of (movie_title, year) tuples
"""

import json
import pickle
import argparse
from pathlib import Path
from collections import Counter


def extract_movies_from_saferec(saferec_json_path):
    """Extract all unique movies from SafeRec dataset."""
    print(f"📂 Loading SafeRec dataset from {saferec_json_path}")

    with open(saferec_json_path, 'r') as f:
        data = json.load(f)

    catalog = set()
    movie_sources = Counter()

    for sample in data['samples']:
        # 1. Ground truth movies
        for movie, year in sample.get('groundtruth_with_release_year', []):
            if movie and year:
                catalog.add((movie, int(year)))
                movie_sources['groundtruth'] += 1

        # 2. Original recommendations
        for rec in sample.get('original_recommendations', []):
            if 'title' in rec and 'year' in rec and rec['title'] and rec['year']:
                catalog.add((rec['title'], int(rec['year'])))
                movie_sources['original_recs'] += 1

        # 3. Safe recommendations (after filtering)
        for rec in sample.get('safe_recommendations', []):
            if 'title' in rec and 'year' in rec and rec['title'] and rec['year']:
                catalog.add((rec['title'], int(rec['year'])))
                movie_sources['safe_recs'] += 1

        # 4. Filtered recommendations
        for rec in sample.get('filtered_recommendations', []):
            if 'title' in rec and 'year' in rec and rec['title'] and rec['year']:
                catalog.add((rec['title'], int(rec['year'])))
                movie_sources['filtered_recs'] += 1

    return catalog, movie_sources


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground truth catalog from SafeRec dataset"
    )
    parser.add_argument(
        "--saferec_json",
        type=str,
        default="data/phase2_3_saferec/saferec_sft_final.json",
        help="Path to SafeRec SFT dataset JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gt_catalog.pkl",
        help="Output path for catalog pickle file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print sample movies from catalog"
    )

    args = parser.parse_args()

    # Extract movies
    catalog, sources = extract_movies_from_saferec(args.saferec_json)

    # Convert to list and sort for consistency
    catalog_list = sorted(list(catalog))

    # Save catalog
    print(f"\n💾 Saving catalog to {args.output}")
    with open(args.output, 'wb') as f:
        pickle.dump(catalog_list, f)

    # Statistics
    print("\n" + "="*60)
    print("📊 Catalog Statistics")
    print("="*60)
    print(f"Total unique movies: {len(catalog_list)}")
    print(f"\nMovie sources:")
    for source, count in sources.items():
        print(f"  - {source}: {count:,} entries")

    # Year distribution
    years = [int(year) for _, year in catalog_list]
    print(f"\nYear range: {min(years)} - {max(years)}")
    print(f"Average year: {sum(years) / len(years):.1f}")

    if args.verbose:
        print("\n📽️  Sample movies (first 20):")
        for i, (title, year) in enumerate(catalog_list[:20], 1):
            print(f"  {i:2d}. {title} ({year})")

    print("\n" + "="*60)
    print(f"✅ Catalog generated successfully!")
    print(f"📁 Saved to: {Path(args.output).resolve()}")
    print("="*60)


if __name__ == "__main__":
    main()
