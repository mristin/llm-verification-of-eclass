"""
Experiment 3: HDBSCAN for duplicate identification

Steps taken:
1. Compute O(N^2) Distance Matrix.
2. MASK the matrix: Distance > Threshold -> set to 2.0.
3. Run HDBSCAN.
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, cast

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN  # type: ignore
from sklearn.metrics.pairwise import cosine_distances  # type: ignore

from llm_verification_of_eclass.common.logger import LoggerFactory


def load_embeddings(pickle_path: Path) -> Dict[str, np.ndarray]:
    with open(pickle_path, "rb") as f:
        data = cast(Dict[str, np.ndarray], pickle.load(f))
    return data


def get_masked_matrix(
    embeddings: np.ndarray, threshold: float, cache_file: Path, logger: logging.Logger
) -> np.ndarray:
    """
    Computes O(N^2) matrix and MASKS it.
    """
    # 1. Check if we have the RAW matrix cached (to save re-computing O(N^2))
    raw_matrix_path = cache_file.parent / "full_raw_matrix.npy"

    if raw_matrix_path.exists():
        logger.info("Loading raw distance matrix...")
        matrix = np.load(raw_matrix_path)
    else:
        logger.info("Computing O(N^2) distance matrix...")
        matrix = cosine_distances(embeddings).astype(np.float32)
        np.save(raw_matrix_path, matrix)

    # 2. Apply the Mask (The "Shattering" Step)
    logger.info(f"Applying Threshold Mask (Cutoff > {threshold})...")

    # We copy to avoid destroying the cached raw matrix in memory
    masked_dist = matrix.copy()

    # Set distant points to 2.0 (Max Cosine Distance is 2.0)
    # This tells HDBSCAN: "These points are NOT connected."
    masked_dist[masked_dist > threshold] = 2.0

    return cast(np.ndarray, masked_dist)


def run_masked_hdbscan(
    masked_matrix: np.ndarray,
    threshold: float,
    texts: List[str],
    logger: logging.Logger,
) -> pd.DataFrame:
    logger.info(f"Running HDBSCAN on masked matrix (t={threshold})...")

    # metric='precomputed' is essential here.
    clusterer = HDBSCAN(
        min_cluster_size=2,  # We want pairs
        min_samples=1,  # Connect even isolated pairs
        metric="precomputed",  # Use our masked matrix
        cluster_selection_epsilon=0.0,  # Let hierarchy decide INSIDE the islands
        allow_single_cluster=False,
        n_jobs=-1,
    )

    labels = clusterer.fit_predict(masked_matrix)

    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    logger.info(f"Found {n_clusters} clusters.")

    results = []
    for label in unique_labels:
        if label == -1:
            continue  # Noise

        indices = np.where(labels == label)[0]
        cluster_texts = [texts[i] for i in indices]

        results.append(
            {
                "threshold": threshold,
                "cluster_id": label,
                "size": len(cluster_texts),
                "definitions": " || ".join(cluster_texts),
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":

    mode = "classes"  # can be either "classes" or "properties"

    # trying very small values to find pathological cases
    thresholds = [0.15, 0.1, 0.05, 0.01]

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    base_data_dir = repo_root / "data"

    if mode == "properties":
        in_dir = base_data_dir / "extracted-properties/2-deduplicated-pair-properties"
        out_dir = base_data_dir / "3-experiment/properties-clustering-masked"
    else:
        in_dir = base_data_dir / "extracted-classes/2-deduplicated-pair-classes"
        out_dir = base_data_dir / "3-experiment/classes-clustering-masked"

    out_dir.mkdir(parents=True, exist_ok=True)
    input_pickle = in_dir / "definition_embedding_map.pickle"
    # Cache raw matrix to avoid computing it separately for every threshold
    cache_path = out_dir / "full_raw_matrix.npy"

    logger = LoggerFactory.get_logger(__name__)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())

    if not input_pickle.exists():
        logger.error(f"Input not found: {input_pickle}")
        sys.exit(1)

    # 1: load embeddings
    logger.info("Loading embeddings...")
    data_map = load_embeddings(input_pickle)
    texts = list(data_map.keys())
    embeddings = np.array(list(data_map.values()))

    # 2: try list of thresholds
    for t in thresholds:
        # get matrix with links cut > thresh
        masked_matrix = get_masked_matrix(embeddings, t, cache_path, logger)

        # clustering
        df = run_masked_hdbscan(masked_matrix, t, texts, logger)

        # save result
        out_csv = out_dir / f"clusters_masked_t{t}.csv"
        df.sort_values("size", ascending=False).to_csv(out_csv, index=False)
        logger.info(f"Saved results to {out_csv}\n")
