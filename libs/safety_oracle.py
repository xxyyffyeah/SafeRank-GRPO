"""
SafetyOracle Module - Phase 2

Core oracle for checking movie safety against user trait constraints.
Uses movie_trait_sensitivity data and title-to-imdb mapping.
"""

import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class SafetyCheckResult:
    """Result of a safety check for a single movie."""
    is_safe: bool
    violations: List[str]
    movie_traits: Dict[str, float]
    matched_imdb_id: Optional[str] = None


class SafetyOracle:
    """
    Oracle for checking movie safety against user trait constraints.

    Usage:
        oracle = SafetyOracle(
            trait_sensitivity_path="downloaded_datasets/movie_trait_sensitivity.json",
            title_mapping_path="data/phase1_mapping/title_to_imdb.pkl"
        )

        # Check if a movie is safe for a user with specific constraints
        result = oracle.check_safety("Saw", 2004, {"Anti-gore / squeamish": True})
        print(result.is_safe)  # False
        print(result.violations)  # ["Anti-gore / squeamish: risk=0.92"]
    """

    # Default risk threshold above which a movie is considered unsafe
    DEFAULT_RISK_THRESHOLD = 0.66

    def __init__(
        self,
        trait_sensitivity_path: Union[str, Path],
        title_mapping_path: Union[str, Path],
        risk_threshold: float = DEFAULT_RISK_THRESHOLD,
        year_tolerance: int = 1
    ):
        """
        Initialize SafetyOracle.

        Args:
            trait_sensitivity_path: Path to movie_trait_sensitivity.json
            title_mapping_path: Path to title_to_imdb.pkl
            risk_threshold: Risk score above which a movie is considered unsafe (default: 0.5)
            year_tolerance: Year tolerance for fuzzy matching (default: 1)
        """
        self.risk_threshold = risk_threshold
        self.year_tolerance = year_tolerance

        # Load trait sensitivity data (imdbId -> traits)
        self._trait_data: Dict[str, Dict] = {}
        self._load_trait_sensitivity(trait_sensitivity_path)

        # Load title mapping (title, year) -> imdbId
        self._title_mapping: Dict[Tuple[str, int], str] = {}
        self._load_title_mapping(title_mapping_path)

        # Cache for normalized titles
        self._title_cache: Dict[str, str] = {}

    def _load_trait_sensitivity(self, path: Union[str, Path]) -> None:
        """Load movie trait sensitivity data from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Trait sensitivity file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Build imdbId -> traits lookup
        for movie in data:
            imdb_id = movie.get('imdbId')
            if imdb_id:
                self._trait_data[imdb_id] = movie.get('traits', {})

        print(f"[SafetyOracle] Loaded trait sensitivity for {len(self._trait_data)} movies")

    def _load_title_mapping(self, path: Union[str, Path]) -> None:
        """Load title-to-imdb mapping from pickle file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Title mapping file not found: {path}")

        with open(path, 'rb') as f:
            self._title_mapping = pickle.load(f)

        print(f"[SafetyOracle] Loaded title mapping for {len(self._title_mapping)} titles")

    @staticmethod
    def normalize_title(title: str) -> str:
        """
        Normalize movie title for matching.

        - Lowercase
        - Remove articles (the, a, an) from beginning
        - Remove special characters
        - Collapse whitespace
        """
        title = title.lower().strip()

        # Remove leading articles
        title = re.sub(r'^(the|a|an)\s+', '', title)

        # Remove special characters but keep spaces
        title = re.sub(r'[^\w\s]', '', title)

        # Collapse whitespace
        title = re.sub(r'\s+', ' ', title).strip()

        return title

    def _lookup_imdb_id(self, title: str, year: Optional[int] = None) -> Optional[str]:
        """
        Look up IMDb ID for a movie title.

        Args:
            title: Movie title
            year: Release year (optional but recommended)

        Returns:
            IMDb ID if found, None otherwise
        """
        normalized = self.normalize_title(title)

        # Try exact match with year
        if year is not None:
            # Try exact year first
            key = (normalized, year)
            if key in self._title_mapping:
                return self._title_mapping[key]

            # Try year tolerance
            for delta in range(1, self.year_tolerance + 1):
                for y in [year - delta, year + delta]:
                    key = (normalized, y)
                    if key in self._title_mapping:
                        return self._title_mapping[key]

        # If no year provided or not found, try to find any match
        # This is a fallback and may return wrong movie for common titles
        for (t, y), imdb_id in self._title_mapping.items():
            if t == normalized:
                return imdb_id

        return None

    def get_movie_traits(
        self,
        title: str,
        year: Optional[int] = None
    ) -> Optional[Dict[str, Dict]]:
        """
        Get all trait sensitivity scores for a movie.

        Args:
            title: Movie title
            year: Release year (optional but recommended)

        Returns:
            Dict of trait -> {final, unsafe, source, ddd_trigger, pg_risk}
            Returns None if movie not found
        """
        imdb_id = self._lookup_imdb_id(title, year)
        if imdb_id is None:
            return None

        return self._trait_data.get(imdb_id)

    def get_movie_traits_by_imdb(self, imdb_id: str) -> Optional[Dict[str, Dict]]:
        """
        Get all trait sensitivity scores for a movie by IMDb ID.

        Args:
            imdb_id: IMDb ID (without 'tt' prefix)

        Returns:
            Dict of trait -> {final, unsafe, source, ddd_trigger, pg_risk}
            Returns None if movie not found
        """
        # Remove 'tt' prefix if present
        imdb_id = imdb_id.lstrip('t')
        return self._trait_data.get(imdb_id)

    def check_safety(
        self,
        title: str,
        year: Optional[int],
        constraints: Dict[str, bool],
        threshold: Optional[float] = None
    ) -> SafetyCheckResult:
        """
        Check if a movie is safe given user trait constraints.

        Args:
            title: Movie title
            year: Release year (optional but recommended)
            constraints: Dict of trait -> True/False (True means user wants to avoid this)
            threshold: Risk threshold (uses instance default if not specified)

        Returns:
            SafetyCheckResult with is_safe, violations, and movie_traits
        """
        if threshold is None:
            threshold = self.risk_threshold

        # Lookup movie
        imdb_id = self._lookup_imdb_id(title, year)
        if imdb_id is None:
            # Movie not found - consider it safe (no data to contradict)
            return SafetyCheckResult(
                is_safe=True,
                violations=[],
                movie_traits={},
                matched_imdb_id=None
            )

        traits = self._trait_data.get(imdb_id, {})
        violations = []

        # Check each constraint
        for trait, should_avoid in constraints.items():
            if not should_avoid:
                continue

            trait_info = traits.get(trait, {})
            risk_score = trait_info.get('final', 0.0)

            if risk_score >= threshold:
                violations.append(f"{trait}: risk={risk_score:.2f}")

        # Extract final scores for all traits
        movie_traits = {
            trait: info.get('final', 0.0)
            for trait, info in traits.items()
        }

        return SafetyCheckResult(
            is_safe=len(violations) == 0,
            violations=violations,
            movie_traits=movie_traits,
            matched_imdb_id=imdb_id
        )

    def check_safety_by_imdb(
        self,
        imdb_id: str,
        constraints: Dict[str, bool],
        threshold: Optional[float] = None
    ) -> SafetyCheckResult:
        """
        Check if a movie is safe given user trait constraints (by IMDb ID).

        Args:
            imdb_id: IMDb ID (without 'tt' prefix)
            constraints: Dict of trait -> True/False (True means user wants to avoid this)
            threshold: Risk threshold (uses instance default if not specified)

        Returns:
            SafetyCheckResult with is_safe, violations, and movie_traits
        """
        if threshold is None:
            threshold = self.risk_threshold

        # Remove 'tt' prefix if present
        imdb_id = imdb_id.lstrip('t')

        traits = self._trait_data.get(imdb_id, {})
        if not traits:
            return SafetyCheckResult(
                is_safe=True,
                violations=[],
                movie_traits={},
                matched_imdb_id=imdb_id
            )

        violations = []

        # Check each constraint
        for trait, should_avoid in constraints.items():
            if not should_avoid:
                continue

            trait_info = traits.get(trait, {})
            risk_score = trait_info.get('final', 0.0)

            if risk_score >= threshold:
                violations.append(f"{trait}: risk={risk_score:.2f}")

        # Extract final scores for all traits
        movie_traits = {
            trait: info.get('final', 0.0)
            for trait, info in traits.items()
        }

        return SafetyCheckResult(
            is_safe=len(violations) == 0,
            violations=violations,
            movie_traits=movie_traits,
            matched_imdb_id=imdb_id
        )

    def batch_check_safety(
        self,
        movies: List[Tuple[str, Optional[int]]],
        constraints: Dict[str, bool],
        threshold: Optional[float] = None
    ) -> List[SafetyCheckResult]:
        """
        Check safety for multiple movies at once.

        Args:
            movies: List of (title, year) tuples
            constraints: Dict of trait -> True/False
            threshold: Risk threshold

        Returns:
            List of SafetyCheckResult in the same order as input
        """
        return [
            self.check_safety(title, year, constraints, threshold)
            for title, year in movies
        ]

    def get_all_trait_names(self) -> List[str]:
        """Get list of all trait names."""
        # Get from first movie that has traits
        for imdb_id, traits in self._trait_data.items():
            if traits:
                return list(traits.keys())
        return []

    def has_movie(self, title: str, year: Optional[int] = None) -> bool:
        """Check if a movie exists in the database."""
        imdb_id = self._lookup_imdb_id(title, year)
        return imdb_id is not None and imdb_id in self._trait_data


# Convenience function for quick setup
def create_oracle(
    base_path: Union[str, Path] = ".",
    risk_threshold: float = SafetyOracle.DEFAULT_RISK_THRESHOLD
) -> SafetyOracle:
    """
    Create SafetyOracle with default paths relative to base_path.

    Args:
        base_path: Base directory (default: current directory)
        risk_threshold: Risk threshold for safety checks

    Returns:
        Configured SafetyOracle instance
    """
    base = Path(base_path)
    return SafetyOracle(
        trait_sensitivity_path=base / "downloaded_datasets" / "movie_trait_sensitivity.json",
        title_mapping_path=base / "data" / "phase1_mapping" / "title_to_imdb.pkl",
        risk_threshold=risk_threshold
    )


if __name__ == "__main__":
    # Quick test
    import sys

    base_path = sys.argv[1] if len(sys.argv) > 1 else "/home/coder/Rank-GRPO"

    print("Creating SafetyOracle...")
    oracle = create_oracle(base_path)

    print("\n--- Test 1: Get movie traits ---")
    traits = oracle.get_movie_traits("Toy Story", 1995)
    if traits:
        print(f"Toy Story (1995) has {len(traits)} traits")
        for trait, info in list(traits.items())[:3]:
            print(f"  {trait}: {info.get('final', 0):.2f}")

    print("\n--- Test 2: Check safety ---")
    constraints = {
        "Anti-gore / squeamish": True,
        "Horror avoider (avoids scares & supernatural)": True
    }

    # Test with a horror movie
    result = oracle.check_safety("Saw", 2004, constraints)
    print(f"Saw (2004): safe={result.is_safe}, violations={result.violations}")

    # Test with a family movie
    result = oracle.check_safety("Toy Story", 1995, constraints)
    print(f"Toy Story (1995): safe={result.is_safe}, violations={result.violations}")

    print("\n--- Test 3: List all traits ---")
    all_traits = oracle.get_all_trait_names()
    print(f"Total traits: {len(all_traits)}")
    for t in all_traits:
        print(f"  - {t}")
