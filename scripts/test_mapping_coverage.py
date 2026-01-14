#!/usr/bin/env python3
"""
Test title-to-imdbId mapping coverage against SFT dataset.

Usage:
    python scripts/test_mapping_coverage.py \
        --sft_dataset_path downloaded_datasets/processed_datasets/sft_dataset \
        --title_mapping_path data/title_to_imdb.pkl \
        --trait_sensitivity_path downloaded_datasets/movie_trait_sensitivity.json
"""

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path
from datasets import load_from_disk


def normalize_title(title: str) -> str:
    """Normalize title (same as in build_title_mapping.py)."""
    if not title:
        return ""
    title = title.lower()
    title = re.sub(r"[^\w\s']", " ", title)
    title = " ".join(title.split())
    return title


def parse_year(year_str: str) -> int | None:
    """Parse year from string."""
    if not year_str:
        return None
    try:
        if isinstance(year_str, int):
            return year_str
        # Handle "1995" or 1995
        return int(str(year_str).strip())
    except ValueError:
        return None


def lookup_imdb_id(title: str, year: int, mapping: dict, year_tolerance: int = 2) -> str | None:
    """
    Lookup imdbId with year tolerance.

    Args:
        title: Movie title
        year: Release year
        mapping: {(normalized_title, year): imdbId}
        year_tolerance: Allow ±N years difference

    Returns:
        imdbId if found, else None
    """
    norm_title = normalize_title(title)

    # Exact match
    if (norm_title, year) in mapping:
        return mapping[(norm_title, year)]

    # Year tolerance
    for d in range(1, year_tolerance + 1):
        for y in [year + d, year - d]:
            if (norm_title, y) in mapping:
                return mapping[(norm_title, y)]

    return None


def extract_movies_from_dataset(dataset_path: str, max_samples: int = 10000):
    """Extract unique movies from SFT dataset."""
    print(f"Loading dataset from {dataset_path}...")
    train_dataset = load_from_disk(Path(dataset_path) / "train")

    total_samples = len(train_dataset)
    samples_to_check = min(max_samples, total_samples)

    print(f"Extracting movies from {samples_to_check:,} / {total_samples:,} samples...")

    groundtruth_movies = set()
    completion_movies = set()

    for i, sample in enumerate(train_dataset.select(range(samples_to_check))):
        # Extract from groundtruth
        for title, year in sample["groundtruth_with_release_year"]:
            y = parse_year(year)
            if y:
                groundtruth_movies.add((title, y))

        # Extract from completion (parse "Title (Year)" format)
        content = sample["completion"][0]["content"]
        for match in re.finditer(r"^(.+?)\s+\((\d{4})\)\s*$", content, re.MULTILINE):
            title = match.group(1).strip()
            year = parse_year(match.group(2))
            if year:
                completion_movies.add((title, year))

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1:,} samples...")

    all_movies = groundtruth_movies | completion_movies

    print(f"\n📊 Extraction Results:")
    print(f"  Groundtruth movies: {len(groundtruth_movies):,}")
    print(f"  Completion movies: {len(completion_movies):,}")
    print(f"  Total unique movies: {len(all_movies):,}")

    return all_movies, groundtruth_movies, completion_movies


def test_coverage(movies: set, mapping: dict, trait_data: dict, year_tolerance: int = 2):
    """Test mapping coverage."""
    print(f"\n🔍 Testing mapping coverage (year_tolerance={year_tolerance})...")

    stats = {
        "total_movies": len(movies),
        "mapped_to_imdb": 0,
        "found_in_traits": 0,
        "not_mapped": 0,
        "mapped_but_no_traits": 0,
    }

    unmapped_movies = []
    mapped_without_traits = []
    year_distribution = defaultdict(int)

    for title, year in movies:
        year_distribution[year] += 1

        imdb_id = lookup_imdb_id(title, year, mapping, year_tolerance)

        if imdb_id:
            stats["mapped_to_imdb"] += 1

            # Check if in trait data
            if imdb_id in trait_data:
                stats["found_in_traits"] += 1
            else:
                stats["mapped_but_no_traits"] += 1
                mapped_without_traits.append((title, year, imdb_id))
        else:
            stats["not_mapped"] += 1
            unmapped_movies.append((title, year))

    # Calculate percentages
    stats["mapping_rate"] = stats["mapped_to_imdb"] / stats["total_movies"] * 100
    stats["trait_coverage_rate"] = stats["found_in_traits"] / stats["total_movies"] * 100

    print(f"\n✅ Coverage Statistics:")
    print(f"  Total movies: {stats['total_movies']:,}")
    print(f"  Mapped to imdbId: {stats['mapped_to_imdb']:,} ({stats['mapping_rate']:.1f}%)")
    print(f"  Found in trait data: {stats['found_in_traits']:,} ({stats['trait_coverage_rate']:.1f}%)")
    print(f"  Not mapped: {stats['not_mapped']:,}")
    print(f"  Mapped but no traits: {stats['mapped_but_no_traits']:,}")

    # Year distribution
    print(f"\n📅 Year Distribution (top 10):")
    for year, count in sorted(year_distribution.items(), key=lambda x: -x[1])[:10]:
        print(f"  {year}: {count:,} movies")

    # Show unmapped samples
    if unmapped_movies:
        print(f"\n⚠️ Sample Unmapped Movies (first 20):")
        for title, year in unmapped_movies[:20]:
            print(f"  {title} ({year})")

    return stats, unmapped_movies, mapped_without_traits


def main():
    parser = argparse.ArgumentParser(description="Test mapping coverage")
    parser.add_argument("--sft_dataset_path",
                        default="downloaded_datasets/processed_datasets/sft_dataset",
                        help="Path to SFT dataset")
    parser.add_argument("--title_mapping_path",
                        default="data/title_to_imdb.pkl",
                        help="Path to title mapping pickle")
    parser.add_argument("--trait_sensitivity_path",
                        default="downloaded_datasets/movie_trait_sensitivity.json",
                        help="Path to trait sensitivity JSON")
    parser.add_argument("--max_samples", type=int, default=10000,
                        help="Max samples to check from dataset")
    parser.add_argument("--year_tolerance", type=int, default=2,
                        help="Year tolerance for matching")
    args = parser.parse_args()

    # Load mapping
    print(f"Loading title mapping from {args.title_mapping_path}...")
    with open(args.title_mapping_path, "rb") as f:
        mapping = pickle.load(f)
    print(f"Loaded {len(mapping):,} mappings")

    # Load trait data
    print(f"Loading trait sensitivity from {args.trait_sensitivity_path}...")
    with open(args.trait_sensitivity_path) as f:
        trait_list = json.load(f)
    trait_data = {m["imdbId"]: m for m in trait_list}
    print(f"Loaded {len(trait_data):,} movies with trait data")

    # Extract movies from dataset
    all_movies, gt_movies, comp_movies = extract_movies_from_dataset(
        args.sft_dataset_path, args.max_samples
    )

    # Test coverage
    stats, unmapped, mapped_no_traits = test_coverage(
        all_movies, mapping, trait_data, args.year_tolerance
    )

    # Save results
    output = {
        "stats": stats,
        "unmapped_sample": [{"title": t, "year": y} for t, y in unmapped[:100]],
        "mapped_but_no_traits_sample": [
            {"title": t, "year": y, "imdbId": i}
            for t, y, i in mapped_no_traits[:100]
        ],
    }

    output_path = "data/mapping_coverage_report.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Saved detailed report to {output_path}")


if __name__ == "__main__":
    main()
