from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from verl.iteration.core import (
    Candidate,
    IterationState,
    atomic_write_json,
    extract_evidence_bundle,
    stable_group_split,
)
from verl.prompts import DEFAULT_SOLVER_PREFIX


def candidate_id(
    *,
    iteration: int,
    doc_id: str,
    generation_index: int,
    question: str,
    reference_answer: str,
) -> str:
    payload = json.dumps(
        [iteration, doc_id, generation_index, question, reference_answer],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"candidate-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def build_candidate(
    *,
    state: IterationState,
    metadata: dict[str, Any],
    hop_count: int,
    trajectory: str | list[Any],
    response: str,
    question: str,
    reference_answer: str,
    format_score: float,
    generation_index: int,
) -> Candidate:
    required = {"doc_id", "source_document"}
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(
            "legacy proposer data lacks required structured metadata "
            f"{missing}; regenerate it with process_train.py"
        )
    source_document = metadata["source_document"]
    normalized_trajectory, evidence = extract_evidence_bundle(
        source_document,
        trajectory,
        hop_count=hop_count,
    )
    return Candidate(
        candidate_id=candidate_id(
            iteration=state.iteration,
            doc_id=str(metadata["doc_id"]),
            generation_index=generation_index,
            question=question,
            reference_answer=reference_answer,
        ),
        iteration=state.iteration,
        doc_id=str(metadata["doc_id"]),
        hop_count=hop_count,
        source_document=source_document,
        proposer_trajectory=normalized_trajectory,
        evidence_bundle=evidence,
        question=question,
        reference_answer=reference_answer,
        format_score=float(format_score),
        generation_index=generation_index,
    )


def atomic_write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary_path = Path(handle.name)
    os.replace(temporary_path, destination)


def persist_candidates(path: str | Path, candidates: list[Candidate]) -> None:
    atomic_write_jsonl(path, [candidate.model_dump(mode="json") for candidate in candidates])


def atomic_write_parquet(frame: pd.DataFrame, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".parquet.tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        frame.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _solver_row(source_row: pd.Series, candidate: Candidate) -> dict[str, Any]:
    row = source_row.to_dict()
    row["prompt"] = [{"role": "user", "content": DEFAULT_SOLVER_PREFIX.format(question=candidate.question.strip())}]
    reward_model = dict(row.get("reward_model") or {})
    ground_truth = dict(reward_model.get("ground_truth") or {})
    ground_truth["target"] = [candidate.reference_answer]
    reward_model["ground_truth"] = ground_truth
    row["reward_model"] = reward_model
    metadata = dict(row.get("metadata") or {})
    metadata.update(
        {
            "candidate_id": candidate.candidate_id,
            "doc_id": candidate.doc_id,
            "hop_count": candidate.hop_count,
            "source_document": candidate.source_document,
            "candidate_json": candidate.model_dump_json(),
            "evidence_bundle": [item.model_dump(mode="json") for item in candidate.evidence_bundle],
            "verify_result": candidate.verify_result.model_dump(mode="json") if candidate.verify_result else None,
        }
    )
    row["metadata"] = metadata
    return row


def write_generation_datasets(
    source_rows: list[pd.Series],
    selected_candidates: list[Candidate],
    *,
    train_path: str,
    keepout_path: str,
    split_manifest_path: str,
    train_ratio: float,
    split_seed: int,
) -> dict[str, Any]:
    if len(source_rows) != len(selected_candidates):
        raise ValueError("selected candidate count must match source document count")
    train, keepout, manifest = stable_group_split(
        selected_candidates,
        train_ratio=train_ratio,
        seed=split_seed,
    )
    row_by_candidate_id = {
        candidate.candidate_id: _solver_row(row, candidate)
        for row, candidate in zip(source_rows, selected_candidates, strict=True)
    }
    train_ids = {candidate.candidate_id for candidate in train}
    keepout_ids = {candidate.candidate_id for candidate in keepout}
    if train_ids & keepout_ids:
        raise AssertionError("train and keepout candidate ids overlap")

    train_frame = pd.DataFrame([row_by_candidate_id[candidate.candidate_id] for candidate in train])
    keepout_frame = pd.DataFrame([row_by_candidate_id[candidate.candidate_id] for candidate in keepout])
    for frame, path in (
        (train_frame, train_path),
        (keepout_frame, keepout_path),
    ):
        atomic_write_parquet(frame, path)
    atomic_write_json(split_manifest_path, manifest)
    return manifest


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def build_generation_summary(
    candidate_groups: list[list[Candidate]],
    selected_candidates: list[Candidate],
    split_manifest: dict[str, Any],
    *,
    model_call_metrics: dict[str, Any],
) -> dict[str, Any]:
    candidates = [candidate for group in candidate_groups for candidate in group]
    verified_rank_counts: Counter[int] = Counter()
    failure_reasons: Counter[str] = Counter()
    for group in candidate_groups:
        ordered = sorted(group, key=lambda item: (-item.rank_score, item.generation_index))
        for rank, candidate in enumerate(ordered, start=1):
            if candidate.status in {"verify_passed", "verify_failed", "verify_error"}:
                verified_rank_counts[rank] += 1
            result = candidate.verify_result
            if candidate.status == "verify_error":
                failure_reasons["invocation_error"] += 1
            elif result and not result.passed:
                if not result.evidence_support:
                    failure_reasons["evidence_unsupported"] += 1
                if not result.question_is_determinate:
                    failure_reasons["question_indeterminate"] += 1
                if not any(item.semantically_equivalent for item in result.candidate_judgments):
                    failure_reasons["no_equivalent_solver_answer"] += 1
    return {
        "generated_candidate_count": len(candidates),
        "document_count": len(candidate_groups),
        "accepted_candidate_count": len(selected_candidates),
        "accepted_document_rate": len(selected_candidates) / len(candidate_groups) if candidate_groups else 0.0,
        "status_counts": dict(Counter(candidate.status for candidate in candidates)),
        "verified_rank_counts": {str(key): value for key, value in sorted(verified_rank_counts.items())},
        "verify_failure_reasons": dict(failure_reasons),
        "format_score": _distribution([candidate.format_score for candidate in candidates]),
        "rubric_score": _distribution(
            [
                sum((item.score - 1) / 4 for item in candidate.rubric_evaluation)
                / len(candidate.rubric_evaluation)
                for candidate in candidates
                if candidate.rubric_evaluation
            ]
        ),
        "rank_score": _distribution([candidate.rank_score for candidate in candidates]),
        "hop_counts": dict(Counter(candidate.hop_count for candidate in selected_candidates)),
        "split": split_manifest,
        "model_call_metrics": model_call_metrics,
    }
