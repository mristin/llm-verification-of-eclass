"""Module to deduplicate ECLASS entries by (preferred_name, definition) pairs, keeping only the smallest ID."""

import logging
import pandas as pd

from llm_verification_of_eclass.common.logger import LoggerFactory


def deduplicate_by_pairs(
    input_path: str, output_path: str, logger: logging.Logger, mode: str = "classes"
) -> None:
    """Removes duplicate (preferred_name, definition) pairs from CSV file, keeping only the entry with the
    smallest ID. Saves as CSV file in output_path."""

    logger.info(f"Processing CSV: {input_path} (mode={mode})")

    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} entries from {input_path}")
    except Exception as e:
        logger.error(f"Failed to read CSV {input_path}: {e}")
        return

    if df.empty:
        logger.warning(f"Input file {input_path} is empty")
        df.to_csv(output_path, index=False)
        logger.info(f"Saved empty CSV to: {output_path}")
        return

    # check for needed columns
    required_cols = ["id", "preferred-name", "definition"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logger.error(f"Missing needed columns: {missing}")
        return

    initial_count = len(df)

    # creating a tuple for each (preferred_name, definition) pair
    df["content_pair"] = list(zip(df["preferred-name"], df["definition"]))

    # Track duplicates for logging
    duplicates_found = {}
    seen_pairs = {}
    rows_to_keep = []

    for idx, row in df.iterrows():
        content_pair = row["content_pair"]
        current_id = row["id"]

        if content_pair not in seen_pairs:
            # keep the first occurrence
            seen_pairs[content_pair] = current_id
            rows_to_keep.append(idx)
        else:
            # duplicate identified
            kept_id = seen_pairs[content_pair]
            if content_pair not in duplicates_found:
                duplicates_found[content_pair] = {"kept_id": kept_id, "removed_ids": []}
            duplicates_found[content_pair]["removed_ids"].append(current_id)

            # if current ID is smaller, update which one we keep
            if current_id < kept_id:
                # remove previous kept row and keep this one instead
                rows_to_keep.remove(df[df["id"] == kept_id].index[0])
                rows_to_keep.append(idx)
                duplicates_found[content_pair]["removed_ids"].remove(current_id)
                duplicates_found[content_pair]["removed_ids"].append(kept_id)
                duplicates_found[content_pair]["kept_id"] = current_id
                seen_pairs[content_pair] = current_id

    # keep only the selected rows
    df_deduplicated = df.loc[rows_to_keep].copy()
    df_deduplicated = df_deduplicated.drop(columns=["content_pair"])

    # maintain original order

    final_count = len(df_deduplicated)
    removed_count = initial_count - final_count

    # Log summary
    logger.info(f"Initial entries: {initial_count}")
    logger.info(f"Unique entries: {final_count}")
    logger.info(f"Removed entries: {removed_count}")

    if duplicates_found:
        logger.warning(
            f"Found {len(duplicates_found)} unique pairs with duplicates (mode={mode})"
        )
        for content_pair, info in duplicates_found.items():
            pref_name, definition = content_pair
            def_preview = (
                definition[:50] + "..."
                if definition and len(definition) > 50
                else definition
            )
            logger.warning(
                f"Duplicated pair ('{pref_name}', '{def_preview}'): "
                f"kept ID {info['kept_id']}, removed {len(info['removed_ids'])} duplicate(s): {info['removed_ids']} (mode={mode})"
            )

    # save deduplicated CSV
    df_deduplicated.to_csv(output_path, index=False)
    logger.info(f"Saved deduplicated CSV to: {output_path}")


if __name__ == "__main__":

    mode = "classes"  # Categorisation classes - "classes", properties - "properties"

    # Input and output folders depend on mode
    if mode == "properties":
        in_dir = "../../data/extracted-properties/1-original-properties"
        out_dir = "../../data/extracted-properties/2-deduplicated-pair-properties"
    elif mode == "classes":
        in_dir = "../../data/extracted-classes/1-original-classes"
        out_dir = "../../data/extracted-classes/2-deduplicated-pair-classes"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # logger setup
    logger = LoggerFactory.get_logger(__name__)
    file_handler = logging.FileHandler(
        f"eclass-{mode}-deduplication-run.txt", mode="w", encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    input_path = f"{in_dir}/eclass-0.csv"
    output_path = f"{out_dir}/eclass-0.csv"

    deduplicate_by_pairs(input_path, output_path, logger, mode)
