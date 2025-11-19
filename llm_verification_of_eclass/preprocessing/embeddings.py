"""Module to compute embeddings for unique definitions from ECLASS CSV files."""

import logging
import pickle
import re
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from llm_verification_of_eclass.common.logger import LoggerFactory


def embed(
    texts: List[str], model: SentenceTransformer, logger: logging.Logger
) -> Dict[str, np.ndarray]:
    """Computes embeddings for a list of texts using the specified model.

    Args:
        texts: List of strings to embed
        model: SentenceTransformer model for embeddings
        logger: Logger instance

    Returns:
        Dict: mapping each text to its embedding vector
    """
    logger.info(f"Computing embeddings for {len(texts)} unique definitions...")

    # computing embeddings in batches
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # creating the mapping
    embedding_map = {text: emb for text, emb in zip(texts, embeddings)}

    logger.info(f"Generated {len(embedding_map)} embeddings")
    return embedding_map


def compute_embeddings_for_file(input_path: Path, logger: logging.Logger) -> None:
    """
    Reads a CSV file, extracts unique definitions, filters non-alphabetic definitions out, computes embeddings, and saves them.
    """
    logger.info(f"Processing file: {input_path}")

    # .csv loading
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} entries from {input_path}")
    except Exception as e:
        logger.error(f"Failed to read CSV {input_path}: {e}")
        return

    if "definition" not in df.columns:
        logger.error(f"Missing 'definition' column in {input_path}")
        return

    # extracting definitions without NaNs
    definitions = df["definition"].dropna().tolist()
    logger.info(f"Found {len(definitions)} non-null definitions")

    # Regex matches strings with NO alphabetic characters
    invalid_pattern = re.compile(r"^[^a-zA-Z]*$")
    valid_definitions = [
        d for d in definitions if isinstance(d, str) and not invalid_pattern.match(d)
    ]
    logger.info(
        f"After filtering out invalid definitions: {len(valid_definitions)} remain"
    )

    # keep unique definitions only
    unique_definitions = list(set(valid_definitions))
    logger.info(f"Unique definitions to embed: {len(unique_definitions)}")

    if not unique_definitions:
        logger.warning("No valid definitions to embed. Exiting.")
        return

    logger.info("Loading embedding model: BAAI/bge-large-en-v1.5")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    # Compute embeddings.
    embedding_map = embed(unique_definitions, model, logger)

    # Save embeddings to pickle file in the same directory
    output_path = input_path.parent / "embedding_map.pickle"
    try:
        with open(output_path, "wb") as f:
            pickle.dump(embedding_map, f)
        logger.info(f"Saved embeddings to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save embeddings at {output_path}: {e}")


if __name__ == "__main__":
    # Path to properties:
    input_file = Path(
        "../../data/extracted-properties/2-deduplicated-pair-properties/eclass-0.csv"
    )

    # Path to classes:
    # input_file = Path("../../data/extracted-classes/2-deduplicated-pair-classes/eclass-0.csv")

    # Setup logger
    logger = LoggerFactory.get_logger(__name__)
    file_handler = logging.FileHandler(
        "eclass-embedding-computation.txt", mode="w", encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Compute embeddings
    compute_embeddings_for_file(input_file, logger)
