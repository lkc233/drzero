from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.filter_validation_set import filter_musique_validation_set


def _write_sources(path: Path, sources: list[str]) -> pa.Schema:
    table = pa.table({"data_source": sources, "value": list(range(len(sources)))})
    pq.write_table(table, path)
    return table.schema


def test_filter_musique_validation_set_keeps_every_musique_row(tmp_path: Path) -> None:
    input_path = tmp_path / "test.parquet"
    output_path = tmp_path / "test_musique.parquet"
    sources = ["searchR1_musique"] * 2417 + ["searchR1_hotpotqa"]
    schema = _write_sources(
        input_path,
        sources,
    )

    filter_musique_validation_set(input_path, output_path)

    result = pq.read_table(output_path)
    assert result.schema == schema
    assert result.num_rows == 2417
    assert set(result["data_source"].to_pylist()) == {"searchR1_musique"}
    assert result["value"].to_pylist() == list(range(2417))


def test_filter_musique_validation_set_rejects_incomplete_musique(tmp_path: Path) -> None:
    input_path = tmp_path / "test.parquet"
    _write_sources(input_path, ["searchR1_musique"] * 2416)

    with pytest.raises(ValueError, match="expected 2417.*found 2416"):
        filter_musique_validation_set(input_path, tmp_path / "output.parquet")
