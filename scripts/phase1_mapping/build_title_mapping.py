#!/usr/bin/env python3
"""
Build title-to-imdbId mapping from IMDb title.basics dataset.

Usage:
    python scripts/build_title_mapping.py \
        --imdb_path data/title.basics.tsv.gz \
        --output_path data/title_to_imdb.pkl \
        --trait_sensitivity_path downloaded_datasets/movie_trait_sensitivity.json
"""

import argparse
import gzip
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path


def normalize_title(title: str) -> str:
    """Normalize movie title for matching."""
    if not title:
        return ""
    # Lowercase
    title = title.lower()
    # Remove punctuation except apostrophes
    title = re.sub(r"[^\w\s']", " ", title)
    # Normalize whitespace
    title = " ".join(title.split())
    return title


def parse_year(year_str: str) -> int | None:
    """Parse year string, return None if invalid."""
    if not year_str or year_str == "\\N":
        return None
    try:
        return int(year_str)
    except ValueError:
        return None


def build_mapping(imdb_path: str, trait_imdb_ids: set) -> dict:
    """
    Build (normalized_title, year) -> imdbId mapping.

    Args:
        imdb_path: Path to title.basics.tsv.gz
        trait_imdb_ids: Set of imdbIds in trait sensitivity data (for filtering)

    Returns:
        dict mapping (title, year) -> imdbId
    """
    mapping = {}
    title_variants = defaultdict(list)  # For debugging duplicates

    print(f"Reading {imdb_path}...")

    with gzip.open(imdb_path, "rt", encoding="utf-8") as f:
        # Skip header
        header = f.readline().strip().split("\t")
        print(f"Columns: {header}")

        # Find column indices
        tconst_idx = header.index("tconst")
        type_idx = header.index("titleType")
        primary_idx = header.index("primaryTitle")
        original_idx = header.index("originalTitle")
        year_idx = header.index("startYear")

        movies_count = 0
        trait_matches = 0

        for line_num, line in enumerate(f, start=2):
            parts = line.strip().split("\t")
            if len(parts) < len(header):
                continue

            title_type = parts[type_idx]

            # Only include movies, tvMovies, and tvSpecials
            if title_type not in ("movie", "tvMovie", "tvSpecial"):
                continue

            movies_count += 1

            tconst = parts[tconst_idx]  # e.g., "tt0114709"
            imdb_id = tconst[2:] if tconst.startswith("tt") else tconst  # Remove "tt" prefix

            primary_title = parts[primary_idx]
            original_title = parts[original_idx]
            year = parse_year(parts[year_idx])

            if not year:
                continue

            # Check if this imdbId is in our trait data
            if imdb_id in trait_imdb_ids:
                trait_matches += 1

            # Add mapping for primary title
            norm_primary = normalize_title(primary_title)
            if norm_primary:
                key = (norm_primary, year)
                if key not in mapping:
                    mapping[key] = imdb_id
                title_variants[key].append(("primary", primary_title, imdb_id))

            # Add mapping for original title if different
            if original_title != primary_title:
                norm_original = normalize_title(original_title)
                if norm_original and norm_original != norm_primary:
                    key = (norm_original, year)
                    if key not in mapping:
                        mapping[key] = imdb_id
                    title_variants[key].append(("original", original_title, imdb_id))

            if line_num % 500000 == 0:
                print(f"  Processed {line_num:,} lines, {movies_count:,} movies, {len(mapping):,} mappings...")

    print(f"\nTotal movies processed: {movies_count:,}")
    print(f"Total mappings created: {len(mapping):,}")
    print(f"Trait data matches: {trait_matches:,} / {len(trait_imdb_ids):,}")

    # Check for duplicates (multiple imdbIds for same title+year)
    duplicates = {k: v for k, v in title_variants.items() if len(set(x[2] for x in v)) > 1}
    if duplicates:
        print(f"\nWarning: {len(duplicates)} title+year combinations have multiple imdbIds")
        # Show first few
        for i, (key, variants) in enumerate(list(duplicates.items())[:5]):
            print(f"  {key}: {variants}")

    return mapping


def main():
    parser = argparse.ArgumentParser(description="Build title-to-imdbId mapping")
    parser.add_argument("--imdb_path", default="data/title.basics.tsv.gz",
                        help="Path to IMDb title.basics.tsv.gz")
    parser.add_argument("--output_path", default="data/title_to_imdb.pkl",
                        help="Output pickle file path")
    parser.add_argument("--trait_sensitivity_path",
                        default="downloaded_datasets/movie_trait_sensitivity.json",
                        help="Path to trait sensitivity JSON for filtering")
    args = parser.parse_args()

    # Load trait sensitivity data to get the set of imdbIds we care about
    print(f"Loading trait sensitivity data from {args.trait_sensitivity_path}...")
    with open(args.trait_sensitivity_path) as f:
        trait_data = json.load(f)
    trait_imdb_ids = {m["imdbId"] for m in trait_data}
    print(f"Trait data contains {len(trait_imdb_ids):,} unique imdbIds")

    # Build mapping
    mapping = build_mapping(args.imdb_path, trait_imdb_ids)

    # Save mapping
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(mapping, f)
    print(f"\nSaved mapping to {output_path}")

    # Save stats
    stats = {
        "total_mappings": len(mapping),
        "trait_imdb_ids": len(trait_imdb_ids),
        "sample_mappings": {f"{k[0]} ({k[1]})": v for k, v in list(mapping.items())[:10]}
    }
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
