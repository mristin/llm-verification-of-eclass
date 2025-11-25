"""Experiment 1. Finding placeholder definitions via NLP and embeddings"""

import logging
import pickle
import random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from llm_verification_of_eclass.common.logger import LoggerFactory


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Computes cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def evaluate_placeholder_coverage(
    closest_definitions: List[Tuple[float, str]],
    selected_seeds: List[str],
    all_placeholders: List[str],
    logger: logging.Logger,
) -> None:
    """Find appearances of other placeholder definitions in the top 100 results.

    Args:
        closest_definitions: List of (score, definition) tuples from the experiment
        selected_seeds: The seed definitions used in this experiment
        all_placeholders: Complete list of placeholder definitions
        logger: Logger instance
    """
    # list of placeholders that were not used as seeds
    unused_placeholders = [p for p in all_placeholders if p not in selected_seeds]

    if not unused_placeholders:
        logger.info("No unused placeholders to evaluate (all were used as seeds)")
        return

    logger.info(
        f"\nEvaluating coverage of {len(unused_placeholders)} unused placeholder(s):"
    )

    # mapping the definition to (rank, score) for quick lookup
    definition_ranking = {
        defn: (i + 1, score) for i, (score, defn) in enumerate(closest_definitions)
    }

    found_count = 0
    not_found = []

    for placeholder in unused_placeholders:
        if placeholder in definition_ranking:
            rank, score = definition_ranking[placeholder]
            logger.info(
                f"Found a placeholder:\n---> '{placeholder}' - Rank: {rank}/100, Score: {score:.4f}"
            )
            found_count += 1
        else:
            logger.warning(
                f"Didn't find a placeholder:\n---> '{placeholder}' - NOT in top 100"
            )
            not_found.append(placeholder)

    # Summary
    logger.info(f"\nCoverage Summary:")
    logger.info(f"  Found in top 100: {found_count}/{len(unused_placeholders)}")
    logger.info(f"  Not found: {len(not_found)}/{len(unused_placeholders)}")

    if not_found:
        logger.info(f"  Missing placeholders: {not_found}")


def finding_100_closest_definitions(
    seed_definitions: List[str],
    embedding_map: Dict[str, np.ndarray],
    logger: logging.Logger,
) -> List[Tuple[float, str]]:
    """Finds the 100 closest definitions to a set of seed definitions.

    Args:
        seed_definitions: List of seed definition strings
        embedding_map: definitions to embeddings
        logger: Logger instance

    Returns:
        List[(score, definition)] of the top 100 closest definitions
    """
    # find embeddings for seeds in mapping
    seed_embeddings = []
    for seed in seed_definitions:
        if seed in embedding_map:
            seed_embeddings.append(embedding_map[seed])
        else:
            logger.warning(f"Seed definition not found in embedding_map: '{seed}'")

    if not seed_embeddings:
        logger.error("No valid seed embeddings found")
        return []

    logger.info(f"Computing similarities for {len(embedding_map)} definitions...")

    scored_definitions = []
    for definition, embedding in embedding_map.items():
        # skipping if definition is a seed
        if definition in seed_definitions:
            continue

        # find max similarity to any seed
        max_score = max(
            cosine_similarity(embedding, seed_emb) for seed_emb in seed_embeddings
        )
        scored_definitions.append((max_score, definition))

    # sort by similarity desc (higher means closer)
    scored_definitions.sort(key=lambda entry: entry[0], reverse=True)

    return scored_definitions[:100]


if __name__ == "__main__":

    classes_placeholder_definitions = [
        "-no definition",
        "Definition is still due",
        "tbd",
        "no definition available",
        "No definition available",
        "to be defined",
        "Tbd",
        "Definition is missing",
        "To be defined later",
        "To be defined",
        "[Definition: missing]",
    ]

    properties_placeholder_definitions = [
        "-no definition",
        "tbd",
        "TBD",
        # "Description of product features, product advantages, product benefits",
        # "Search term color",
        # "Used materials and colors are indiscriminately identical with the main description of the product (switch)",
        # "Material options (switch)"
    ]

    mode = "properties"  # should be either "classes" or "properties"

    if mode == "properties":
        in_dir = Path("../../data/extracted-properties/2-deduplicated-pair-properties")
        out_dir = Path("../../data/1-experiment/properties-placeholder-definitions")
        placeholder_definitions = properties_placeholder_definitions
        experiment_numbers = [1, 2, 3]
    elif mode == "classes":
        in_dir = Path("../../data/extracted-classes/2-deduplicated-pair-classes")
        out_dir = Path("../../data/1-experiment/classes-placeholder-definitions")
        placeholder_definitions = classes_placeholder_definitions
        experiment_numbers = [1, 2, 3, 7, 10]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Create output directory if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = LoggerFactory.get_logger(__name__)
    file_handler = logging.FileHandler(
        f"experiment-1-placeholder-{mode}.txt", mode="w", encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # load embeddings
    embedding_path = in_dir / "definition_embedding_map.pickle"
    logger.info(f"Loading embeddings from: {embedding_path}")
    with open(embedding_path, "rb") as f:
        embedding_map = pickle.load(f)
    logger.info(f"Loaded {len(embedding_map)} embeddings")

    # experiment loop
    for num_seeds in experiment_numbers:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running experiment with {num_seeds} seed(s)")
        logger.info(f"{'='*60}")

        # Randomly select seeds
        selected_seeds = random.sample(placeholder_definitions, num_seeds)
        logger.info(f"Selected seeds: {selected_seeds}")

        # Find closest definitions
        closest_definitions = finding_100_closest_definitions(
            selected_seeds, embedding_map, logger
        )

        logger.info(f"Found {len(closest_definitions)} closest definitions")

        # save .csv of 100 matches
        output_path = out_dir / f"{num_seeds}-seed{'s' if num_seeds > 1 else ''}.csv"
        df = pd.DataFrame(
            closest_definitions, columns=["similarity_score", "definition"]
        )
        df.to_csv(output_path, index=False)
        logger.info(f"Saved results to: {output_path}")

        evaluate_placeholder_coverage(
            closest_definitions, selected_seeds, placeholder_definitions, logger
        )

        # Log top 5 for inspection
        logger.info("Top 5 closest definitions:")
        for i, (score, definition) in enumerate(closest_definitions[:5], 1):
            def_preview = (
                definition[:50] + "..." if len(definition) > 50 else definition
            )
            logger.info(f"  {i}. [{score:.3f}] {def_preview}")

    logger.info(f"\n{'='*60}")
    logger.info("All experiments completed!")
