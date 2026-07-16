from __future__ import annotations

import argparse
import json
from pathlib import Path

from verl.iteration.generation import repair_candidate_snapshot


def _default_progress_path(candidates_path: Path) -> Path:
    return candidates_path.with_name(candidates_path.stem + "_progress.jsonl")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Repair a generation candidate snapshot from its persisted trajectories.",
    )
    parser.add_argument("candidates_path", type=Path)
    parser.add_argument("--backup-path", type=Path)
    parser.add_argument("--progress-path", type=Path)
    args = parser.parse_args(argv)

    progress_path = args.progress_path or _default_progress_path(args.candidates_path)
    if progress_path.exists() and progress_path.stat().st_size:
        parser.error(
            f"candidate progress journal is not empty: {progress_path}; "
            "compact or archive it before repairing the snapshot"
        )

    summary = repair_candidate_snapshot(
        args.candidates_path,
        backup_path=args.backup_path,
    )
    progress_path.unlink(missing_ok=True)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
