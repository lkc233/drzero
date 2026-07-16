from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from verl.custom_reward.format_scoring import (
    challenger_answer_reward,
    challenger_format_score,
    challenger_think_reward,
)
from verl.iteration.core import (
    Candidate,
    EvidenceItem,
    IterationState,
    atomic_write_json,
    canonical_hash,
    dynamic_state_hash,
    extract_evidence_bundle,
    normalize_trajectory,
    stable_group_split,
)
from verl.prompts import DEFAULT_SOLVER_PREFIX

GenerationPhase = Literal["all", "generate", "verify"]


def resolve_generation_phase(value: Any) -> GenerationPhase:
    phase = str(value or "all")
    if phase not in {"all", "generate", "verify"}:
        raise ValueError("data.phase must be one of: all, generate, verify")
    return phase


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
    candidate_format_score = float(format_score)
    format_failure = None
    try:
        normalized_trajectory, evidence = extract_evidence_bundle(
            source_document,
            trajectory,
            hop_count=hop_count,
        )
    except ValueError as error:
        # Proposer output is sampled, so malformed tool markup is an expected
        # per-candidate format failure rather than a fatal generation error.
        normalized_trajectory = normalize_trajectory(trajectory)
        evidence = [
            EvidenceItem(
                evidence_id="evidence-0",
                kind="seed_document",
                content=source_document,
                source="seed_document",
                trajectory_index=0,
            )
        ]
        candidate_format_score = 0.0
        format_failure = {
            "error_type": type(error).__name__,
            "reason": str(error),
        }

    status = "generated" if candidate_format_score > 0 else "format_invalid"
    if status == "format_invalid" and format_failure is None:
        format_failure = {
            "error_type": "FormatScoreError",
            "reason": "proposer format score is zero",
        }
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
        format_score=candidate_format_score,
        format_failure=format_failure,
        generation_index=generation_index,
        status=status,
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


def build_candidate_snapshot_contract(
    *,
    state: IterationState,
    metadata_rows: list[dict[str, Any]],
    candidate_count: int,
    model_path: str | Path,
    rollout_config: dict[str, Any],
    verification_config: dict[str, Any],
    tool_config_path: str | Path,
) -> dict[str, Any]:
    metadata_hasher = hashlib.sha256()
    for metadata in metadata_rows:
        item = [metadata["doc_id"], metadata["source_document"], int(metadata["hop_count"])]
        metadata_hasher.update(json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        metadata_hasher.update(b"\n")

    local_model_path = Path(model_path)
    if local_model_path.is_dir():
        model_files = [
            {
                "path": str(path.relative_to(local_model_path)),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in sorted(local_model_path.rglob("*"))
            if path.is_file()
        ]
    else:
        model_files = []
    tool_path = Path(tool_config_path)
    return {
        "schema_version": 1,
        "state_hash": dynamic_state_hash(state),
        "metadata_hash": metadata_hasher.hexdigest(),
        "document_count": len(metadata_rows),
        "candidate_count_per_document": candidate_count,
        "model_path": str(local_model_path.resolve()) if local_model_path.exists() else str(model_path),
        "model_files": model_files,
        "rollout_config": rollout_config,
        "verification_config": verification_config,
        "tool_config_hash": hashlib.sha256(tool_path.read_bytes()).hexdigest(),
    }


def write_candidate_snapshot_manifest(path: str | Path, contract: dict[str, Any]) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "fingerprint": canonical_hash(contract),
            "contract": contract,
        },
    )


def validate_candidate_snapshot_manifest(path: str | Path, contract: dict[str, Any]) -> None:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise ValueError(
            "candidate snapshot has no generation manifest; set data.resume_candidates=false to regenerate it"
        )
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1 or manifest.get("fingerprint") != canonical_hash(contract):
        raise ValueError(
            "candidate snapshot was produced by a different generation run; "
            "set data.resume_candidates=false to regenerate it"
        )


def reset_candidate_progress(path: str | Path) -> None:
    Path(path).unlink(missing_ok=True)


def append_candidate_progress(path: str | Path, candidates: list[Candidate]) -> None:
    """Append one fully processed candidate group to the verify journal."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
    }
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_candidates_with_progress(
    candidates_path: str | Path,
    progress_path: str | Path,
) -> list[Candidate]:
    """Restore the candidate snapshot and apply durable per-group updates."""
    snapshot_path = Path(candidates_path)
    candidates = []
    with snapshot_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                candidates.append(Candidate.model_validate_json(line))
    index_by_id = {candidate.candidate_id: index for index, candidate in enumerate(candidates)}
    if len(index_by_id) != len(candidates):
        raise ValueError(f"candidate snapshot contains duplicate ids: {snapshot_path}")

    journal_path = Path(progress_path)
    if not journal_path.exists():
        return candidates
    with journal_path.open(encoding="utf-8") as handle:
        line_index = 0
        while line := handle.readline():
            line_index += 1
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if not handle.read():
                    break
                raise
            if record.get("schema_version") != 1 or not isinstance(record.get("candidates"), list):
                raise ValueError(f"invalid candidate progress record at line {line_index}: {journal_path}")
            updates = [Candidate.model_validate(item) for item in record["candidates"]]
            update_ids = [candidate.candidate_id for candidate in updates]
            if len(update_ids) != len(set(update_ids)):
                raise ValueError(f"candidate progress record contains duplicate ids at line {line_index}")
            unknown_ids = sorted(set(update_ids) - index_by_id.keys())
            if unknown_ids:
                raise ValueError(f"candidate progress references unknown ids: {unknown_ids}")
            for candidate in updates:
                candidates[index_by_id[candidate.candidate_id]] = candidate
    return candidates


def compact_candidate_progress(
    candidates_path: str | Path,
    progress_path: str | Path,
    candidates: list[Candidate],
) -> None:
    persist_candidates(candidates_path, candidates)
    reset_candidate_progress(progress_path)


def candidate_group_is_complete(candidates: list[Candidate], *, verify_enabled: bool) -> bool:
    if not candidates:
        return False
    if not verify_enabled:
        return all(candidate.status in {"format_invalid", "not_verified"} for candidate in candidates)
    if any(
        candidate.verify_result is not None
        and candidate.verify_result.verification_mode != "two_condition_em"
        for candidate in candidates
    ):
        return False
    passed = [candidate for candidate in candidates if candidate.status == "verify_passed"]
    if len(passed) == 1:
        allowed = {
            "format_invalid",
            "verify_passed",
            "verify_failed",
            "verify_error",
            "not_verified",
        }
        return all(candidate.status in allowed for candidate in candidates)
    return not passed and all(
        candidate.status in {"format_invalid", "verify_failed", "verify_error"}
        for candidate in candidates
    )


def reset_candidate_group(candidates: list[Candidate]) -> None:
    for candidate in candidates:
        candidate.rubric_evaluation = []
        candidate.rubric_raw_output = ""
        candidate.rubric_failure = None
        candidate.rank_score = 0
        candidate.status = "generated" if candidate.format_score > 0 else "format_invalid"
        candidate.verify_result = None
        candidate.verify_failure = None


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


def _recovered_format_score(candidate: Candidate, evidence: list[EvidenceItem]) -> float:
    assistant_messages = [
        str(message.get("content") or "")
        for message in candidate.proposer_trajectory
        if message.get("role") == "assistant"
    ]
    if assistant_messages and ("<question>" in assistant_messages[-1] or "<answer>" in assistant_messages[-1]):
        assistant_messages = assistant_messages[:-1]
    think_reward = challenger_think_reward(assistant_messages)

    # A successfully extracted evidence bundle proves the tool-call component:
    # hop 1 receives the base tool reward, while later hops have exactly n - 1
    # paired search calls and responses.
    evidence_text = "\n\n".join(item.content for item in evidence)
    return challenger_format_score(
        question=candidate.question,
        answer=candidate.reference_answer,
        think_reward=think_reward,
        tool_reward=1.0,
        answer_reward=challenger_answer_reward(evidence_text, candidate.reference_answer),
    )


def _reset_candidate_after_snapshot_repair(candidate: Candidate) -> None:
    candidate.rubric_evaluation = []
    candidate.rubric_raw_output = ""
    candidate.rubric_failure = None
    candidate.rank_score = 0
    candidate.verify_result = None
    candidate.verify_failure = None


def repair_candidate_snapshot(
    candidates_path: str | Path,
    *,
    backup_path: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild derived candidate fields from a persisted generation trajectory."""
    snapshot_path = Path(candidates_path)
    backup = (
        Path(backup_path)
        if backup_path is not None
        else snapshot_path.with_suffix(snapshot_path.suffix + ".before_repair")
    )
    if backup.exists():
        raise FileExistsError(f"candidate snapshot backup already exists: {backup}")

    status_counts: Counter[str] = Counter()
    candidate_count = 0
    recovered_count = 0
    temporary_path: Path | None = None
    try:
        with (
            snapshot_path.open(encoding="utf-8") as source,
            tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=snapshot_path.parent,
                prefix=f".{snapshot_path.name}.",
                suffix=".repair.tmp",
                delete=False,
            ) as destination,
        ):
            temporary_path = Path(destination.name)
            for line_index, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                candidate = Candidate.model_validate_json(line)
                candidate_count += 1
                try:
                    trajectory, evidence = extract_evidence_bundle(
                        candidate.source_document,
                        candidate.proposer_trajectory,
                        hop_count=candidate.hop_count,
                    )
                    candidate.proposer_trajectory = trajectory
                    candidate.evidence_bundle = evidence
                    candidate.format_score = _recovered_format_score(candidate, evidence)
                    candidate.format_failure = None
                    if candidate.format_score > 0:
                        candidate.status = "generated"
                        recovered_count += 1
                    else:
                        candidate.status = "format_invalid"
                        candidate.format_failure = {
                            "error_type": "FormatScoreError",
                            "reason": "recovered proposer format score is zero",
                        }
                except ValueError as error:
                    candidate.format_score = 0
                    candidate.status = "format_invalid"
                    candidate.format_failure = {
                        "error_type": type(error).__name__,
                        "reason": str(error),
                    }
                _reset_candidate_after_snapshot_repair(candidate)
                status_counts[candidate.status] += 1
                destination.write(
                    json.dumps(candidate.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"
                )
            destination.flush()
            os.fsync(destination.fileno())

        backup.parent.mkdir(parents=True, exist_ok=True)
        os.link(snapshot_path, backup)
        os.replace(temporary_path, snapshot_path)
        temporary_path = None
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return {
        "candidate_count": candidate_count,
        "recovered_count": recovered_count,
        "invalid_count": candidate_count - recovered_count,
        "status_counts": dict(status_counts),
    }


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
            if candidate.status == "format_invalid":
                failure_reasons["format_invalid"] += 1
            if candidate.status in {"verify_passed", "verify_failed", "verify_error"}:
                verified_rank_counts[rank] += 1
            result = candidate.verify_result
            if candidate.status == "verify_error":
                failure_reasons["invocation_error"] += 1
            elif result and not result.passed:
                if not result.with_evidence_succeeded:
                    failure_reasons["with_evidence_all_incorrect"] += 1
                if not result.question_only_all_incorrect:
                    failure_reasons["question_only_answered_correctly"] += 1
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
