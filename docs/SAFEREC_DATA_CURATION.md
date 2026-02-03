# SafeRec Dataset Curation

## Overview

This document describes the data pipeline for building the SafeRec dataset — a safety-aware extension of the Reddit-v2 conversational recommendation dataset. The pipeline assigns user sensitivity traits, filters constraint-violating ground truth, injects constraint instructions into prompts, and converts everything into Hugging Face `datasets` format for SFT and GRPO training.

## Existing Assets

| File | Path | Description |
|------|------|-------------|
| Trait Sensitivity | `downloaded_datasets/movie_trait_sensitivity.json` | 24,408 movies x 20 traits |
| Trait Config | `traits_with_imdb_parentguide_weights.json` | Trait weight configuration |
| SFT Dataset | `downloaded_datasets/processed_datasets/sft_dataset/` | 95,753 training samples |

---

## Phase 0: Trait Assignment

### 0.1 Goal

**Objective**: Use GPT API to automatically assign user sensitivity traits to ~24,000 high-quality SFT samples.

**Detailed plan**: See [TRAIT_ASSIGNMENT_PLAN.md](./TRAIT_ASSIGNMENT_PLAN.md)

### 0.2 Pipeline Overview

```
SFT Dataset (95,753)
    | filter GT >= 3
24,000 samples
    | GPT API annotation
Assigned Traits
    | filter violating GT
SafeRec Training Data (~7,200 samples)
```

### 0.3 Key Outputs

| Output | Description |
|--------|-------------|
| `data/saferec_sft_8k_dataset.json` | Training data with assigned traits |
| `data/trait_stats/` | Trait distribution statistics and visualizations |

### 0.4 Why Is This Step Needed?

| Challenge | Trait Assignment Solution |
|-----------|--------------------------|
| **High manual labeling cost** | GPT-4o auto-annotation, cost ~$22 |
| **Uneven trait distribution** | Smart assignment ensures each trait has representative samples |
| **GT noise** | Automatically filters GT movies that violate assigned traits |

---

## Phase 1: Title-to-imdbId Mapping

### 1.1 Data Source

Uses the official IMDb dataset:
- **URL**: https://datasets.imdbws.com/title.basics.tsv.gz
- **Fields**: tconst (imdbId), titleType, primaryTitle, originalTitle, startYear, ...

### 1.2 Mapping Structure

```python
# data/title_to_imdb.pkl
{
    ("toy story", 1995): "0114709",
    ("jumanji", 1995): "0113497",
    ...
}
```

### 1.3 Script

```bash
scripts/phase1_mapping/build_title_mapping.py
```

**Output files**:
- `data/title_to_imdb.pkl` — primary mapping table
- `data/title_mapping_stats.json` — coverage statistics

---

## Phase 2: SafetyOracle Module

### 2.1 File Structure

```
libs/
├── safety_oracle.py      # Oracle core class
├── constraint_injector.py # Constraint injection
└── safety_filter.py       # Safety filter
```

### 2.2 SafetyOracle Interface

```python
class SafetyOracle:
    def __init__(self, trait_sensitivity_path, title_mapping_path):
        ...

    def get_movie_traits(self, title: str, year: int) -> dict:
        """Return all trait sensitivity scores for a movie."""

    def check_safety(self, title: str, year: int, constraints: dict) -> Tuple[bool, List[str]]:
        """Check if a movie satisfies constraints. Returns (is_safe, violation_reasons)."""
```

---

## Phase 3: Constraint Injection

### 3.1 Constraint Templates (20 Traits)

```python
TRAIT_CONSTRAINT_TEMPLATES = {
    "Anti-gore / squeamish": [
        "I don't like gory or violent content.",
        "Please avoid movies with excessive blood and gore.",
        "No overly graphic violence please.",
    ],
    "Horror avoider (avoids scares & supernatural)": [
        "I don't watch horror movies.",
        "Nothing with jump scares or supernatural elements.",
        "I'm easily scared, so no horror please.",
    ],
    "Kid-safety / child harm sensitive": [
        "Something suitable to watch with kids.",
        "Family-friendly movies only please.",
        "Avoid anything involving harm to children.",
    ],
    # ... remaining traits
}
```

### 3.2 Injection Strategy

- **Injection rate**: 30-40% of samples receive constraint injection
- **Constraints per sample**: 1-3 traits
- **Position**: Prepended to the prompt or embedded in the user preference description

---

## Phase 4: Dataset Generation

### 4.1 Data Structure

```python
{
    "prompt": [{"role": "user", "content": "I don't like gory violence... recommend movies"}],
    "completion": [{"role": "assistant", "content": """
1. The Grand Budapest Hotel (2014)
2. ...
"""}],
    "constraints": {"Anti-gore / squeamish": True},
    "groundtruth_with_release_year": [("The Grand Budapest Hotel", 2014), ...],
    "seen_titles": ["Saw (2004)", ...]
}
```

### 4.2 Scripts

```bash
# Generate SafeRec dataset with constraint injection
python scripts/phase2_3_saferec/generate_saferec_dataset.py \
    --input_path downloaded_datasets/processed_datasets/sft_dataset \
    --output_path downloaded_datasets/processed_datasets/saferec_sft_dataset

# Convert to HF datasets format
python scripts/phase2_3_saferec/convert_to_hf_dataset.py \
    --input_path downloaded_datasets/processed_datasets/saferec_sft_dataset \
    --output_path downloaded_datasets/processed_datasets/saferec_grpo_dataset
```

---

## Implementation Status

| Phase | Task | Dependencies | Status |
|-------|------|--------------|--------|
| **0.1** | Filter samples with GT >= 3 (8k) | - | Done |
| **0.2** | Develop GPT annotation script | - | Done |
| **0.3** | Run GPT API annotation | 0.1, 0.2 | Done |
| **0.4** | Filter violating GT movies | 0.3, 1.3 | Done |
| **0.5** | Compute trait distribution statistics | 0.4 | Done |
| **1.1** | Download IMDb title.basics | - | Done |
| **1.2** | Build title-to-imdbId mapping | 1.1 | Done |
| **1.3** | Test SFT dataset coverage | 1.2 | Done |
| **2.1** | Implement SafetyOracle | 1.2 | Done |
| **3.1** | Implement constraint injection module | 2.1, 0.5 | Done |
| **4.1** | Generate SafeRec SFT dataset | 3.1, 0.4 | Done |

---

## Outputs

### Phase 0 Outputs
1. **Trait annotation data**: `data/phase0_trait_assignment/` (train, val, test splits with traits)
2. **Trait distribution statistics**: `data/phase0_trait_assignment/expanded/*_stats/`

### Phase 1-2 Outputs
3. **Title mapping table**: `data/title_to_imdb.pkl`
4. **SafetyOracle module**: `libs/safety_oracle.py`

### Phase 3-4 Outputs
5. **SafeRec SFT dataset**: `downloaded_datasets/processed_datasets/saferec_sft_dataset/`
6. **SafeRec GRPO dataset**: `downloaded_datasets/processed_datasets/saferec_grpo_dataset/` (~38,582 samples)

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Low title match rate | Cannot annotate safety attributes for many movies | Fuzzy matching + year tolerance |
| Too few constrained samples | Model fails to learn safety alignment | Increase injection rate or apply data augmentation |
| Noisy trait assignments | Model learns incorrect safety associations | Manual spot-checks + quality filtering |
