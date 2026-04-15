# Verification of Semantic Dictionaries with Semantic Embedding and LLMs

Source code for the paper:

> Sebastian Heppner, Ablay Abdimalinov, Moritz Sommer, Marko Ristin, Tobias Kleinert, Hans Wernher van de Venn.
> **Verification of Semantic Dictionaries with Semantic Embedding and LLMs.**
> To appear, 2026.

## Overview

This repository implements a modular pipeline for detecting quality issues in semantic dictionaries. The approach works in two stages: semantic embeddings are used to efficiently identify candidate entries that may contain inconsistencies, and an LLM then analyses each candidate at a fine-grained level. Three verification objectives are addressed independently — term–definition misalignment, overly similar terms, and overly similar definitions. The pipeline is evaluated on [ECLASS](https://eclass.eu), a large industrial semantic dictionary used for product classification and data exchange.

## Project Structure

```
├── data/                                        # not tracked — created locally
│   └── original/                                # place raw ECLASS XML files here
├── llm_verification_of_eclass/
│   ├── common/                                  # shared logger
│   ├── preprocessing/
│   │   ├── csv_1_extract.py                     # extract XML to CSV
│   │   ├── csv_2_duplicate_pairs.py             # deduplicate entries by (term, definition)
│   │   └── embeddings.py                        # compute sentence embeddings
│   └── experiments/
│       ├── find_placeholder_definitions.py      # Experiment 1: locate placeholder definitions
│       ├── similarity_based_threshold.py        # Experiment 2: analyse cosine similarity distribution and establish thresholds
│       ├── synthetic_similarity_threshold.py    # Experiment 3: calibrate threshold on synthetic synonym pairs
│       ├── similarity_threshold.py              # Experiment 4: identify candidate pairs within the distance threshold
│       ├── similarity_clustering.py             # Experiment 4 (alt): cluster similar entries
│       ├── llm_definition_comparison.py         # Experiment 5: LLM audit of candidate pairs
│       └── prompts/                             # LLM prompt version history (CHANGELOG.md)
├── pyproject.toml
└── uv.lock
```

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/mristin/llm-verification-of-eclass.git
cd llm-verification-of-eclass
uv sync
```

## Reproducing the Experiments

### Preprocessing

The preprocessing steps must be run in order before any experiment can be executed.

**1. Obtain the raw data**

Add the raw ECLASS Basic XML files to `./data/original/`:

```
data/original/ECLASS15_0_BASIC_EN_SG_01.xml
```

ECLASS data is available at [eclass.eu](https://eclass.eu) and requires a free registration.

**2. Extract XML to CSV**

```bash
uv run python llm_verification_of_eclass/preprocessing/csv_1_extract.py
```

**3. Deduplicate entries**

```bash
uv run python llm_verification_of_eclass/preprocessing/csv_2_duplicate_pairs.py
```

**4. Compute embeddings**

```bash
uv run python llm_verification_of_eclass/preprocessing/embeddings.py
```

Embeddings are computed using `BAAI/bge-large-en-v1.5` and saved as pickle files alongside the CSV outputs.

### Experiments

Each experiment script can be run independently once preprocessing is complete. Refer to the docstring at the top of each file for a description of its purpose and configurable parameters.

**Experiment 5 (LLM audit)** additionally requires [Ollama](https://ollama.com) running locally:

```bash
ollama pull llama3.1
ollama serve
uv run python llm_verification_of_eclass/experiments/llm_definition_comparison.py --mode real
```

Results are saved to `data/5-experiment/llm_analysis.csv`. Use `--limit N` to process only the first N pairs. The model can be changed via the `LLM_MODEL` environment variable:

```bash
LLM_MODEL=gemma2:9b uv run python llm_verification_of_eclass/experiments/llm_definition_comparison.py --mode real
```

## Citation

```bibtex
@inproceedings{heppner2025verification,
  title     = {Verification of Semantic Dictionaries with Semantic Embedding and {LLMs}},
  author    = {Heppner, Sebastian and Abdimalinov, Ablay and Sommer, Moritz and Ristin, Marko and Kleinert, Tobias and van de Venn, Hans Wernher},
  booktitle = {to appear},
  year      = {2025}
}
```
