#!/usr/bin/env python3
"""Create a validation set containing one complete benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

MUSIQUE_DATA_SOURCE = "searchR1_musique"
MUSIQUE_EXPECTED_ROWS = 2417


def filter_musique_validation_set(input_path: Path, output_path: Path) -> None:
    table = pq.read_table(input_path)
    filtered = table.filter(pc.equal(table["data_source"], MUSIQUE_DATA_SOURCE))
    if filtered.num_rows != MUSIQUE_EXPECTED_ROWS:
        raise ValueError(
            f"expected {MUSIQUE_EXPECTED_ROWS} rows for {MUSIQUE_DATA_SOURCE}, found {filtered.num_rows}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(filtered, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/test.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/test_musique.parquet"))
    args = parser.parse_args()
    filter_musique_validation_set(args.input, args.output)


if __name__ == "__main__":
    main()
