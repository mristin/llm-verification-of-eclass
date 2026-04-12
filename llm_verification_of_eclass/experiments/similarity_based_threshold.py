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


CLASSES_PLACEHOLDER_DEFINITIONS = [
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


PROPERTIES_PLACEHOLDER_DEFINITIONS = [
    "-no definition",
    "tbd",
    "TBD",
]


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
    Creates a standard bar chart with the distribution of cosine similarities.
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
    plt.xlabel("Cosine Similarity Range", fontsize=15)
    plt.ylabel("Frequency", fontsize=15)
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
                fontsize=12,
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


def create_stacked_distribution_plot(
    results_df: pd.DataFrame,
    placeholder_list: List[str],
    output_path: Path,
    mode: str,
    logger: logging.Logger,
) -> None:
    """
    Creates a stacked bar chart with placeholders and regular definitions.
    """
    # Separates placeholder values into separate list
    placeholders_df = results_df["definition"].isin(placeholder_list)

    sim_placeholders = results_df[placeholders_df]["cosine_similarity"].to_numpy(
        dtype=np.float64
    )
    sim_regular = results_df[~placeholders_df]["cosine_similarity"].to_numpy(
        dtype=np.float64
    )

    # Calculate statistics for placeholders
    if len(sim_placeholders) > 0:
        ph_mean = np.mean(sim_placeholders)
        ph_std = np.std(sim_placeholders)
        logger.info(f"\nPlaceholder Definitions Statistics ({mode}):")
        logger.info(f"  Count: {len(sim_placeholders)}")
        logger.info(f"  Mean: {ph_mean:.4f}")
        logger.info(f"  Standard Deviation: {ph_std:.4f}")
    else:
        ph_mean = np.float64(0.0)
        ph_std = np.float64(0.0)

        logger.info(f"\nNo placeholder definitions found for mode: {mode}")

    # Bins for graph with range [0.3, 1] and step 0.1
    bins = np.arange(0, 1.1, 0.1)
    all_bin_labels = [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(len(bins) - 1)]

    # Calculate histograms over full range, then slice from 0.3
    counts_ph_full, _ = np.histogram(sim_placeholders, bins=bins)
    counts_reg_full, _ = np.histogram(sim_regular, bins=bins)

    start_idx = 3  # 0.3-0.4 and onwards
    counts_ph = counts_ph_full[start_idx:]
    counts_reg = counts_reg_full[start_idx:]
    bin_labels = all_bin_labels[start_idx:]

    # Colors: dark navy for regular, golden yellow for placeholders
    color_regular =(242 / 255, 208 / 255, 131 / 255)
    color_placeholder =  (16 / 255, 47 / 255, 71 / 255)

    # Create stacked bar chart (scoped font settings)
    with plt.rc_context({"font.family": "Libertinus Serif"}):
        plt.figure(figsize=(12, 6))

        # Placeholders are at the bottom
        plt.bar(
            range(len(bin_labels)),
            counts_ph,
            color=color_placeholder,
            edgecolor="black",
            alpha=0.8,
            label="Placeholder Definitions",
        )

        # Rest of the definitions are on top
        plt.bar(
            range(len(bin_labels)),
            counts_reg,
            bottom=counts_ph,
            color=color_regular,
            edgecolor="black",
            alpha=0.9,
            label="Rest of the Definitions",
        )

        plt.xlabel("Cosine Similarity Range", fontsize=24)
        plt.ylabel("Frequency", fontsize=24)
        plt.xticks(range(len(bin_labels)), bin_labels, rotation=45, ha="right", fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(axis="y", alpha=0.3, linestyle="--")
        plt.legend(loc="upper right", fontsize=24)

        # Add value labels for total height
        total_counts = counts_ph + counts_reg
        for i, count in enumerate(total_counts):
            if count > 0:
                plt.text(
                    i,
                    count,
                    f"{int(count)}",
                    ha="center",
                    va="bottom",
                    fontsize=18,
                )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    logger.info(f"Stacked distribution plot saved to: {output_path}")


if __name__ == "__main__":
    mode = "classes"  # Switch between "classes" or "properties"

    # Define base paths relative to this script file
    script_dir = Path(__file__).resolve().parent

    # Setting parameters based on mode
    if mode == "properties":
        in_dir = (
            script_dir
            / "../../data/extracted-properties/2-deduplicated-pair-properties"
        )
        out_dir = (
            script_dir / "../../data/2-experiment/properties-placeholder-definitions"
        )
        placeholder_list = PROPERTIES_PLACEHOLDER_DEFINITIONS
    elif mode == "classes":
        in_dir = script_dir / "../../data/extracted-classes/2-deduplicated-pair-classes"
        out_dir = script_dir / "../../data/2-experiment/classes-placeholder-definitions"
        placeholder_list = CLASSES_PLACEHOLDER_DEFINITIONS
    else:
        raise ValueError(f"Unknown mode: {mode}")

    in_dir = in_dir.resolve()
    out_dir = out_dir.resolve()

    combined_eclass_path = in_dir / "eclass-0.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = LoggerFactory.get_logger(__name__)
    file_handler = logging.FileHandler(
        out_dir / f"experiment-2-similarity-threshold-{mode}.txt",
        mode="w",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Input directory: {in_dir}")
    logger.info(f"Output directory: {out_dir}")

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

    # Standard distribution plot
    similarities = results_df["cosine_similarity"].to_numpy(dtype=np.float64)
    plot_path = out_dir / f"similarity_distribution_{mode}.png"
    create_distribution_plot(similarities, plot_path, mode, logger)

    # Stacked distribution plot for known placeholder values
    stacked_plot_path = out_dir / f"similarity_distribution_stacked_{mode}.png"
    create_stacked_distribution_plot(
        results_df, placeholder_list, stacked_plot_path, mode, logger
    )

    # Log summary statistics (General)
    logger.info("Statistics summary:\n")
    logger.info(f"Total pairs analyzed: {len(results_df)}")
    logger.info(f"Mean similarity: {similarities.mean():.4f}")
    logger.info(f"Median similarity: {np.median(similarities):.4f}")
    logger.info(f"Std deviation: {similarities.std():.4f}")
    logger.info(f"Min similarity: {similarities.min():.4f}")
    logger.info(f"Max similarity: {similarities.max():.4f}")
    logger.info(f"25th percentile: {np.percentile(similarities, 25):.4f}")
    logger.info(f"75th percentile: {np.percentile(similarities, 75):.4f}")

    logger.info(f"\nExperiment completed successfully!")
