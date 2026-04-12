"""
Experiment 2. Finding thresholds of cosine similarity for too similar and too distant pairs of (preferred_name, definition)
"""

import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from llm_verification_of_eclass.common.logger import LoggerFactory


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Computes cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def comparing_eclass_pairs(
    eclass_df: pd.DataFrame,
    preferred_name_embedding_map: Dict[str, np.ndarray],
    definition_embedding_map: Dict[str, np.ndarray],
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Calculate cosine similarities between preferred names and definitions.

    Args:
        eclass_df: DataFrame with columns ['id', 'preferred-name', 'definition']
        preferred_name_embedding_map: Dictionary mapping preferred names to embeddings
        definition_embedding_map: Dictionary mapping definitions to embeddings
        logger: Logger instance

    Returns:
        DataFrame with columns ['id', 'preferred-name', 'definition', 'cosine_similarity']
        sorted by cosine_similarity in descending order
    """
    results = []
    skipped = 0

    logger.info(f"Processing {len(eclass_df)} eclass entries...")

    for idx, row in eclass_df.iterrows():
        eclass_id = row["id"]
        preferred_name = row["preferred-name"]
        definition = row["definition"]

        # Get embeddings
        if preferred_name not in preferred_name_embedding_map:
            logger.warning(
                f"Preferred name '{preferred_name}' not found in embedding map"
            )
            skipped += 1
            continue

        if definition not in definition_embedding_map:
            logger.warning(f"Definition '{definition}' not found in embedding map")
            skipped += 1
            continue

        pn_embedding = preferred_name_embedding_map[preferred_name]
        def_embedding = definition_embedding_map[definition]

        # Calculate cosine similarity
        similarity: float = cosine_similarity(pn_embedding, def_embedding)

        results.append(
            {
                "id": eclass_id,
                "preferred-name": preferred_name,
                "definition": definition,
                "cosine_similarity": similarity,
            }
        )

        id_int = int(str(idx))
        if (id_int + 1) % 1000 == 0:
            logger.info(f"Processed {id_int + 1} entries...")

    logger.info(
        f"Completed processing. Skipped {skipped} entries due to missing embeddings"
    )

    # Create DataFrame and sort by similarity (descending)
    results_df = pd.DataFrame(results).astype(
        {
            "id": "string",
            "preferred-name": "string",
            "definition": "string",
            "cosine_similarity": np.float64,
        }
    )
    results_df = results_df.sort_values(
        "cosine_similarity", ascending=False
    ).reset_index(drop=True)

    return results_df


def create_distribution_plot(
    similarities: np.ndarray, output_path: Path, mode: str, logger: logging.Logger
) -> None:
    """
    Create a bar chart showing the distribution of cosine similarities.

    Args:
        similarities: Array of cosine similarity values
        output_path: Path to save the plot
        mode: 'properties' or 'classes'
        logger: Logger instance
    """
    # Define bins from 0.0 to 1.0 in intervals of 0.1
    bins = np.arange(0, 1.1, 0.1)
    bin_labels = [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(len(bins) - 1)]

    # Calculate histogram
    counts, _ = np.histogram(similarities, bins=bins)

    # Create bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        range(len(counts)), counts, color="steelblue", edgecolor="black", alpha=0.7
    )

    # Customize plot
    plt.xlabel("Cosine Similarity Range", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(
        f"Distribution of Cosine Similarities: Preferred Name vs Definition ({mode.capitalize()})",
        fontsize=14,
        fontweight="bold",
    )
    plt.xticks(range(len(bin_labels)), bin_labels, rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Add statistics
    stats_text = f"Total: {len(similarities)}\n"
    stats_text += f"Mean: {np.mean(similarities):.3f}\n"
    stats_text += f"Median: {np.median(similarities):.3f}\n"
    stats_text += f"Std: {np.std(similarities):.3f}\n"
    stats_text += f"Min: {np.min(similarities):.3f}\n"
    stats_text += f"Max: {np.max(similarities):.3f}"

    plt.text(
        0.98,
        0.97,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Distribution plot saved to: {output_path}")
    logger.info(
        f"Statistics - Mean: {np.mean(similarities):.3f}, "
        f"Median: {np.median(similarities):.3f}, "
        f"Std: {np.std(similarities):.3f}"
    )


if __name__ == "__main__":
    mode = "properties"  # should be either "classes" or "properties"

    if mode == "properties":
        in_dir = Path("../../data/extracted-properties/2-deduplicated-pair-properties")
        out_dir = Path("../../data/2-experiment/properties-placeholder-definitions")
    elif mode == "classes":
        in_dir = Path("../../data/extracted-classes/2-deduplicated-pair-classes")
        out_dir = Path("../../data/2-experiment/classes-placeholder-definitions")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    combined_eclass_path = in_dir / "eclass-0.csv"

    # Create output directory if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = LoggerFactory.get_logger(__name__)
    file_handler = logging.FileHandler(
        f"experiment-2-similarity-threshold-{mode}.txt", mode="w", encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Load embeddings
    definition_embedding_path = in_dir / "definition_embedding_map.pickle"
    preferred_name_embedding_path = in_dir / "preferred-name_embedding_map.pickle"

    logger.info(f"Loading definition embeddings from: {definition_embedding_path}")
    with open(definition_embedding_path, "rb") as f:
        definition_embedding_map = pickle.load(f)
    logger.info(f"Loaded {len(definition_embedding_map)} definition embeddings")

    logger.info(
        f"Loading preferred name embeddings from: {preferred_name_embedding_path}"
    )
    with open(preferred_name_embedding_path, "rb") as f:
        preferred_name_embedding_map = pickle.load(f)
    logger.info(f"Loaded {len(preferred_name_embedding_map)} preferred name embeddings")

    logger.info(f"Loading eclass dictionary: {combined_eclass_path}")
    eclass_df = pd.read_csv(combined_eclass_path)
    logger.info(f"Loaded {len(eclass_df)} eclass entries")

    # Calculate cosine similarities
    logger.info("Calculating cosine similarities...")
    results_df = comparing_eclass_pairs(
        eclass_df, preferred_name_embedding_map, definition_embedding_map, logger
    )

    # Save results to CSV
    output_csv_path = out_dir / f"cosine_similarities_{mode}.csv"
    results_df.to_csv(output_csv_path, index=False, encoding="utf-8")
    logger.info(f"Results saved to: {output_csv_path}")

    # Create distribution plot
    similarities = results_df["cosine_similarity"].to_numpy(dtype=np.float64)
    plot_path = out_dir / f"similarity_distribution_{mode}.png"
    create_distribution_plot(similarities, plot_path, mode, logger)

    # Log summary statistics
    logger.info("=" * 10)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 10)
    logger.info(f"Total pairs analyzed: {len(results_df)}")
    logger.info(f"Mean similarity: {similarities.mean():.4f}")
    logger.info(f"Median similarity: {np.median(similarities):.4f}")
    logger.info(f"Std deviation: {similarities.std():.4f}")
    logger.info(f"Min similarity: {similarities.min():.4f}")
    logger.info(f"Max similarity: {similarities.max():.4f}")
    logger.info(f"25th percentile: {np.percentile(similarities, 25):.4f}")
    logger.info(f"75th percentile: {np.percentile(similarities, 75):.4f}")
    logger.info("=" * 10)

    # Log top 10 most similar pairs
    logger.info("\nTop 10 most similar pairs:")
    for idx, row in results_df.head(10).iterrows():
        logger.info(
            f"  {row['cosine_similarity']:.4f} - {row['preferred-name']} | {row['definition'][:100]}"
        )

    # Log bottom 10 least similar pairs
    logger.info("\nBottom 10 least similar pairs:")
    for idx, row in results_df.tail(10).iterrows():
        logger.info(
            f"  {row['cosine_similarity']:.4f} - {row['preferred-name']} | {row['definition'][:100]}"
        )

    logger.info(f"\nExperiment completed successfully!")
