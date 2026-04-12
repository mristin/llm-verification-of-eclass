"""
Experiment 4b: Duplicate discovery (Sanity Check & Pathological Cases).
"""

import gc
import logging
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors  # type: ignore

from llm_verification_of_eclass.common.logger import LoggerFactory


def load_embedding_map(pickle_path: Path) -> Dict[str, np.ndarray]:
    with open(pickle_path, "rb") as f:
        data = cast(Dict[str, np.ndarray], pickle.load(f))
    return data


def load_and_clean_data(csv_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Loads CSV, normalizes column names, filters out non-alphanumeric definitions.
    Returns DataFrame: [id, preferred_name, definition]
    """
    logger.info(f"Loading data from: {csv_path.name}")
    df = pd.read_csv(csv_path, dtype=str).fillna("")

    df.columns = [
        c.lower().strip().replace("-", "_").replace(" ", "_") for c in df.columns
    ]

    col_map = {}
    for c in df.columns:
        if "definition" in c:
            col_map[c] = "definition"
        elif "preferred" in c and "name" in c:
            col_map[c] = "preferred_name"
        elif "id" == c or "irde" in c:
            col_map[c] = "id"

    if "definition" not in col_map.values():
        raise ValueError(
            f"Column 'definition' not found in {csv_path}. Columns: {df.columns}"
        )

    df = df.rename(columns=col_map)

    # ensure that preferred_name exists
    if "preferred_name" not in df.columns:
        logger.warning("'preferred_name' column missing. Using ID or empty string.")
        df["preferred_name"] = df["id"] if "id" in df.columns else ""

    bad_regex = re.compile(r"^[^a-zA-Z]*$")
    initial_count = len(df)

    # clean whitespace
    df["definition"] = df["definition"].astype(str).str.strip()

    # filter
    mask_valid = ~df["definition"].apply(lambda x: bool(bad_regex.match(x)))
    df_clean = df[mask_valid].copy()

    skipped = initial_count - len(df_clean)
    if skipped > 0:
        logger.info(f"---> Filtered {skipped} entries.")

    return df_clean


def process_exact_duplicates(
    df: pd.DataFrame, out_dir: Path, logger: logging.Logger
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, str]]:
    """
    Finds rows with same definitions.
    1. Saves 'exact_duplicates.csv' with mapping from definition to preferred names that used it
    2. Returns a deduplicated df for the future search
    3. Returns a dict: definition -> "Name1 (ID1) | Name2 (ID2)"
    """
    logger.info("Checking for EXACT duplicates (textual identity)...")

    # group by definition to find multiple uses
    # and aggregate preferred-names into one string
    df["ref_str"] = df.apply(
        lambda row: f"{row.get('preferred_name', '')} ({row.get('id', 'N/A')})", axis=1
    )

    grouped = (
        df.groupby("definition")
        .agg({"ref_str": lambda x: " | ".join(sorted(set(x))), "id": "count"})
        .rename(columns={"id": "count", "ref_str": "all_preferred_names"})
        .reset_index()
    )

    exact_dupes = grouped[grouped["count"] > 1]

    if not exact_dupes.empty:
        f_path = out_dir / "exact_duplicates.csv"
        exact_dupes.sort_values("count", ascending=False).to_csv(f_path, index=False)
        logger.info(
            f"---> Found {len(exact_dupes)} unique definitions with multiple appearances."
        )
        logger.info(f"---> Saved exact duplicate report to: {f_path.name}")
    else:
        logger.info("---> No exact duplicates found.")

    # Create a lookup dictionary: Definition -> String of all names
    def_to_names_map = pd.Series(
        grouped.all_preferred_names.values, index=grouped.definition
    ).to_dict()

    # Return unique definitions, needed for embedding lookup
    df_unique = df.drop_duplicates(subset=["definition"]).copy()

    def_to_single_name_map = (
        df.drop_duplicates(subset=["definition"])
        .set_index("definition")["ref_str"]
        .to_dict()
    )

    return df_unique, def_to_names_map, def_to_single_name_map


def get_embeddings_for_search(
    df_unique: pd.DataFrame,
    embedding_map: Dict[str, np.ndarray],
    logger: logging.Logger,
) -> Tuple[List[str], np.ndarray]:
    """
    Matches unique definitions to the pickle.
    """
    valid_texts = []
    valid_embeddings = []
    skipped = 0

    for text in df_unique["definition"]:
        if text in embedding_map:
            valid_texts.append(text)
            valid_embeddings.append(embedding_map[text])
        else:
            skipped += 1

    logger.info(f"---> {len(valid_texts)} unique definitions have embeddings.")
    if skipped > 0:
        logger.warning(
            f"---> {skipped} definitions missing from pickle (will be skipped)."
        )

    return valid_texts, np.array(valid_embeddings)


def find_neighbors_batched(
    embeddings: np.ndarray,
    texts: List[str],
    max_threshold: float,
    logger: logging.Logger,
    temp_dir: Path,
    batch_size: int = 500,
) -> pd.DataFrame:
    """
    Finds pairs with distance < max_threshold.
    Offloads to disk to handle OOM.
    """
    if len(embeddings) == 0:
        return pd.DataFrame()

    total_items = len(embeddings)
    temp_csv = temp_dir / "temp_nn_buffer.csv"

    # Initialize temp file
    with open(temp_csv, "w", encoding="utf-8") as f:
        f.write("distance,text_a,text_b\n")

    logger.info(f"Running NN (Max Radius < {max_threshold}) on {total_items} items...")

    nn = NearestNeighbors(
        radius=max_threshold, metric="cosine", algorithm="brute", n_jobs=1
    )
    nn.fit(embeddings)

    start_time = time.time()
    match_count = 0

    # Batch Processing
    for start_idx in range(0, total_items, batch_size):
        end_idx = min(start_idx + batch_size, total_items)

        if start_idx % (batch_size * 10) == 0:
            logger.info(f"... scanning batch {start_idx}/{total_items}")

        batch_query = embeddings[start_idx:end_idx]

        try:
            batch_dists, batch_indices = nn.radius_neighbors(batch_query)
        except MemoryError:
            logger.error("OOM in radius_neighbors. Reduce batch_size.")
            sys.exit(1)

        batch_results = []

        # Extract pairs (Upper triangle only: i < j)
        for local_i, (nbr_dists, nbr_indices) in enumerate(
            zip(batch_dists, batch_indices)
        ):
            source_global_idx = start_idx + local_i
            source_text = texts[source_global_idx]

            for dist, target_global_idx in zip(nbr_dists, nbr_indices):
                # Ensure we only keep A < B to avoid self-matches and duplicates
                if source_global_idx < target_global_idx:
                    batch_results.append(
                        {
                            "distance": dist,
                            "text_a": source_text,
                            "text_b": texts[target_global_idx],
                        }
                    )

        if batch_results:
            df_batch = pd.DataFrame(batch_results)
            df_batch.to_csv(temp_csv, mode="a", header=False, index=False)
            match_count += len(batch_results)
            del df_batch
            del batch_results

        del batch_dists, batch_indices
        gc.collect()

    elapsed = time.time() - start_time
    logger.info(f"Search done in {elapsed:.2f}s. Raw pairs found: {match_count}")

    if match_count == 0:
        if temp_csv.exists():
            temp_csv.unlink()
        return pd.DataFrame()

    # Aggregation (Deduplicate pairs if any, though index check prevents most)
    logger.info("Aggregating results from disk buffer...")
    chunk_size = 100_000
    aggregated_counts = {}

    for chunk in pd.read_csv(temp_csv, chunksize=chunk_size):
        # Normalize order text_a < text_b for string comparison consistency
        mask = chunk["text_a"] > chunk["text_b"]
        chunk.loc[mask, ["text_a", "text_b"]] = chunk.loc[
            mask, ["text_b", "text_a"]
        ].values

        # Just to be safe, though index logic usually suffices
        grouped = (
            chunk.groupby(["text_a", "text_b"])
            .agg(dist_mean=("distance", "mean"))
            .reset_index()
        )

        for _, row in grouped.iterrows():
            aggregated_counts[(row["text_a"], row["text_b"])] = row["dist_mean"]

        del chunk
        gc.collect()

    final_data = [
        {"text_a": k[0], "text_b": k[1], "distance": v}
        for k, v in aggregated_counts.items()
    ]

    if temp_csv.exists():
        temp_csv.unlink()

    return pd.DataFrame(final_data)


if __name__ == "__main__":

    # mode is either "classes" or "properties"
    mode = "classes"

    # from experiment 4a: Average Distance: 0.01079, Maximum Distance/Threshold: 0.04735, Threshold (+10%):   0.05208
    thresholds_to_check = [0.04735, 0.03, 0.01, 0.05208]

    # only max threshold is used for function, rest are filtered from it
    max_threshold = max(thresholds_to_check)

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    base_data_dir = repo_root / "data"

    if mode == "properties":
        path_pickle = (
            base_data_dir
            / "extracted-properties/2-deduplicated-pair-properties/definition_embedding_map.pickle"
        )
        path_experiment = (
            base_data_dir
            / "extracted-properties/2-deduplicated-pair-properties/eclass-0.csv"
        )
        out_dir = base_data_dir / "4-experiment/properties-similarity-threshold"
    else:
        path_pickle = (
            base_data_dir
            / "extracted-classes/2-deduplicated-pair-classes/definition_embedding_map.pickle"
        )
        path_experiment = (
            base_data_dir / "extracted-classes/2-deduplicated-pair-classes/eclass-0.csv"
        )
        out_dir = base_data_dir / "4-experiment/classes-similarity-threshold"

    out_dir.mkdir(parents=True, exist_ok=True)

    # Logger setup
    logger = LoggerFactory.get_logger(__name__)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
        logger.addHandler(logging.FileHandler(out_dir / "experiment.log", mode="w"))

    logger.info("Starting Duplicate Discovery Experiment")
    logger.info(f"Thresholds: {thresholds_to_check}")

    # 1. Load Embeddings
    if not path_pickle.exists():
        logger.error(f"Pickle not found: {path_pickle}")
        sys.exit(1)
    emb_map = load_embedding_map(path_pickle)

    # 2. Load and clean data
    if not path_experiment.exists():
        logger.error(f"CSV not found: {path_experiment}")
        sys.exit(1)
    df_data = load_and_clean_data(path_experiment, logger)

    # 3. Exact duplicates, deduplication
    # df_unique contains unique definitions only
    # name_map contains the mapping from a definition to ALL preferred names
    df_unique, name_map, single_name_map = process_exact_duplicates(df_data, out_dir, logger)

    # 4. embeddings
    search_texts, search_embs = get_embeddings_for_search(df_unique, emb_map, logger)

    # 5. Run Similarity Search (using Max Threshold)
    logger.info(f"Starting Nearest Neighbors search (Max Threshold={max_threshold})...")
    df_matches = find_neighbors_batched(
        search_embs, search_texts, max_threshold, logger, out_dir
    )

    if df_matches.empty:
        logger.info("No matches found at maximum threshold.")
        sys.exit(0)

    # 6. Enrich with Preferred Names
    logger.info("Enriching results with preferred names...")
    df_matches["preferred_names_a"] = df_matches["text_a"].map(single_name_map)
    df_matches["preferred_names_b"] = df_matches["text_b"].map(single_name_map)

    cols = ["distance", "preferred_names_a", "text_a", "preferred_names_b", "text_b"]
    df_matches = df_matches[cols]

    # 7. file for each threshold
    logger.info("Saving results for each threshold...")

    for thresh in thresholds_to_check:
        subset = df_matches[df_matches["distance"] <= thresh]

        if not subset.empty:
            filename = out_dir / f"similarity_matches_thresh_{thresh}.csv"
            subset.sort_values("distance", ascending=True).to_csv(filename, index=False)
            logger.info(
                f"---> Saved {len(subset)} pairs for threshold <= {thresh} to: {filename.name}"
            )
        else:
            logger.info(f"---> No pairs found for threshold <= {thresh}")

    logger.info("Done.")
