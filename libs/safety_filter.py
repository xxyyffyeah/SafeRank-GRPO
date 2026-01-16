"""
Safety Filter Module - Phase 2/3 Integration

Filters movie recommendations based on user safety constraints.
Uses SafetyOracle for risk assessment.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from libs.safety_oracle import SafetyOracle, SafetyCheckResult


@dataclass
class FilteredMovie:
    """A movie that was filtered with reason."""
    title: str
    year: Optional[int]
    reason: str
    risk_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class FilterResult:
    """Result of filtering a list of movies."""
    safe_movies: List[Tuple[str, Optional[int]]]
    filtered_movies: List[FilteredMovie]
    total_input: int
    total_safe: int
    total_filtered: int


class SafetyFilter:
    """
    Filters movie recommendations based on user trait constraints.

    Usage:
        oracle = SafetyOracle(...)
        safety_filter = SafetyFilter(oracle)

        # Filter a list of movies
        constraints = {"Anti-gore / squeamish": True}
        result = safety_filter.filter_movies(
            movies=[("Saw", 2004), ("Toy Story", 1995)],
            constraints=constraints
        )
        print(result.safe_movies)  # [("Toy Story", 1995)]
    """

    # Regex pattern to parse movie titles with year
    MOVIE_PATTERN = re.compile(r'^(.+?)\s*\((\d{4})\)$')

    def __init__(
        self,
        oracle: SafetyOracle,
        default_threshold: float = 0.5
    ):
        """
        Initialize SafetyFilter.

        Args:
            oracle: SafetyOracle instance for risk assessment
            default_threshold: Default risk threshold for filtering
        """
        self.oracle = oracle
        self.default_threshold = default_threshold

    @classmethod
    def parse_movie_string(cls, movie_str: str) -> Tuple[str, Optional[int]]:
        """
        Parse a movie string into (title, year).

        Handles formats like:
        - "Toy Story (1995)"
        - "The Dark Knight (2008)"
        - "Inception" (no year)

        Args:
            movie_str: Movie string to parse

        Returns:
            (title, year) tuple, year may be None
        """
        movie_str = movie_str.strip()
        match = cls.MOVIE_PATTERN.match(movie_str)

        if match:
            title = match.group(1).strip()
            year = int(match.group(2))
            return title, year
        else:
            return movie_str, None

    def filter_movies(
        self,
        movies: List[Tuple[str, Optional[int]]],
        constraints: Dict[str, bool],
        threshold: Optional[float] = None
    ) -> FilterResult:
        """
        Filter a list of movies based on constraints.

        Args:
            movies: List of (title, year) tuples
            constraints: Dict of trait -> True/False
            threshold: Risk threshold (uses default if None)

        Returns:
            FilterResult with safe and filtered movies
        """
        if threshold is None:
            threshold = self.default_threshold

        safe_movies = []
        filtered_movies = []

        for title, year in movies:
            result = self.oracle.check_safety(title, year, constraints, threshold)

            if result.is_safe:
                safe_movies.append((title, year))
            else:
                filtered_movies.append(FilteredMovie(
                    title=title,
                    year=year,
                    reason="; ".join(result.violations),
                    risk_scores=result.movie_traits
                ))

        return FilterResult(
            safe_movies=safe_movies,
            filtered_movies=filtered_movies,
            total_input=len(movies),
            total_safe=len(safe_movies),
            total_filtered=len(filtered_movies)
        )

    def filter_movie_strings(
        self,
        movie_strings: List[str],
        constraints: Dict[str, bool],
        threshold: Optional[float] = None
    ) -> FilterResult:
        """
        Filter movie strings (with embedded year) based on constraints.

        Args:
            movie_strings: List of movie strings like "Toy Story (1995)"
            constraints: Dict of trait -> True/False
            threshold: Risk threshold

        Returns:
            FilterResult with safe and filtered movies
        """
        movies = [self.parse_movie_string(s) for s in movie_strings]
        return self.filter_movies(movies, constraints, threshold)

    def filter_completion_text(
        self,
        completion_text: str,
        constraints: Dict[str, bool],
        threshold: Optional[float] = None
    ) -> Tuple[List[Tuple[str, Optional[int]]], List[FilteredMovie]]:
        """
        Extract and filter movies from completion text.

        Parses each line as a potential movie recommendation.

        Args:
            completion_text: Assistant's completion text
            constraints: User constraints
            threshold: Risk threshold

        Returns:
            (safe_movies, filtered_movies)
        """
        lines = completion_text.strip().split('\n')
        movies = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try to parse as movie
            title, year = self.parse_movie_string(line)
            if title:
                movies.append((title, year))

        result = self.filter_movies(movies, constraints, threshold)
        return result.safe_movies, result.filtered_movies


class GroundTruthFilter:
    """
    Filters ground truth movies that violate user constraints.

    Used during dataset preparation to remove GT movies that would
    be inappropriate recommendations given the assigned trait.
    """

    def __init__(
        self,
        oracle: SafetyOracle,
        threshold: float = 0.5
    ):
        """
        Initialize GroundTruthFilter.

        Args:
            oracle: SafetyOracle instance
            threshold: Risk threshold for filtering
        """
        self.oracle = oracle
        self.threshold = threshold

    def filter_groundtruth(
        self,
        groundtruth: List[Tuple[str, str]],
        assigned_trait: str
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
        """
        Filter ground truth movies based on assigned trait.

        Args:
            groundtruth: List of (title, year_str) tuples
            assigned_trait: The trait assigned to this sample

        Returns:
            (kept_gt, removed_gt) where removed_gt includes violation reason
        """
        constraints = {assigned_trait: True}

        kept = []
        removed = []

        for title, year_str in groundtruth:
            try:
                year = int(year_str) if year_str else None
            except ValueError:
                year = None

            result = self.oracle.check_safety(title, year, constraints, self.threshold)

            if result.is_safe:
                kept.append((title, year_str))
            else:
                reason = "; ".join(result.violations)
                removed.append((title, year_str, reason))

        return kept, removed

    def process_sample(
        self,
        sample: Dict,
        trait_key: str = "assigned_trait",
        gt_key: str = "groundtruth_with_release_year"
    ) -> Dict:
        """
        Process a single sample, filtering its ground truth.

        Args:
            sample: Sample dict with groundtruth and assigned_trait
            trait_key: Key for assigned trait in sample
            gt_key: Key for ground truth in sample

        Returns:
            Modified sample with filtered GT and metadata
        """
        assigned_trait = sample.get(trait_key)
        groundtruth = sample.get(gt_key, [])

        if not assigned_trait or not groundtruth:
            sample["num_gt_removed"] = 0
            sample["removed_groundtruth"] = []
            sample["num_groundtruth_after_filter"] = len(groundtruth)
            return sample

        # Convert to list of tuples if needed
        gt_tuples = []
        for gt in groundtruth:
            if isinstance(gt, (list, tuple)) and len(gt) >= 2:
                gt_tuples.append((gt[0], gt[1]))
            elif isinstance(gt, str):
                title, year = SafetyFilter.parse_movie_string(gt)
                gt_tuples.append((title, str(year) if year else ""))

        kept, removed = self.filter_groundtruth(gt_tuples, assigned_trait)

        # Update sample
        sample[gt_key] = [list(t) for t in kept]
        sample["num_gt_removed"] = len(removed)
        sample["removed_groundtruth"] = [
            {"title": t[0], "year": t[1], "reason": t[2]}
            for t in removed
        ]
        sample["num_groundtruth_after_filter"] = len(kept)

        return sample


if __name__ == "__main__":
    from libs.safety_oracle import create_oracle

    print("=== Safety Filter Test ===\n")

    # Create oracle and filter
    print("Loading SafetyOracle...")
    oracle = create_oracle("/home/coder/Rank-GRPO")

    safety_filter = SafetyFilter(oracle)

    # Test 1: Filter movies
    print("\n--- Test 1: Filter movie list ---")
    constraints = {
        "Anti-gore / squeamish": True,
        "Horror avoider (avoids scares & supernatural)": True
    }

    movies = [
        ("Saw", 2004),
        ("The Conjuring", 2013),
        ("Finding Nemo", 2003),
        ("The Grand Budapest Hotel", 2014),
    ]

    result = safety_filter.filter_movies(movies, constraints)
    print(f"Input: {result.total_input} movies")
    print(f"Safe: {result.total_safe} movies")
    print(f"Filtered: {result.total_filtered} movies")

    print("\nSafe movies:")
    for title, year in result.safe_movies:
        print(f"  - {title} ({year})")

    print("\nFiltered movies:")
    for fm in result.filtered_movies:
        print(f"  - {fm.title} ({fm.year}): {fm.reason}")

    # Test 2: Parse movie strings
    print("\n--- Test 2: Parse and filter movie strings ---")
    movie_strings = [
        "Toy Story (1995)",
        "Pulp Fiction (1994)",
        "The Shining (1980)",
        "Amélie (2001)",
    ]

    result = safety_filter.filter_movie_strings(movie_strings, constraints)
    print(f"Safe: {[f'{t} ({y})' for t, y in result.safe_movies]}")

    # Test 3: Ground truth filter
    print("\n--- Test 3: Ground truth filtering ---")
    gt_filter = GroundTruthFilter(oracle)

    sample = {
        "assigned_trait": "Horror avoider (avoids scares & supernatural)",
        "groundtruth_with_release_year": [
            ["The Shining", "1980"],
            ["Toy Story", "1995"],
            ["It", "2017"],
            ["Finding Nemo", "2003"],
        ]
    }

    processed = gt_filter.process_sample(sample.copy())
    print(f"Original GT count: {len(sample['groundtruth_with_release_year'])}")
    print(f"Kept GT count: {processed['num_groundtruth_after_filter']}")
    print(f"Removed GT count: {processed['num_gt_removed']}")

    print("\nKept ground truth:")
    for gt in processed["groundtruth_with_release_year"]:
        print(f"  - {gt[0]} ({gt[1]})")

    print("\nRemoved ground truth:")
    for removed in processed["removed_groundtruth"]:
        print(f"  - {removed['title']} ({removed['year']}): {removed['reason']}")
