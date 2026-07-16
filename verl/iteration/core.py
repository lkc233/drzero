from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_SKILLS = 12
MAX_RUBRICS = 12
MAX_SKILL_INSTRUCTION_CHARS = 1_000
MAX_SKILLS_PROMPT_CHARS = 8_000
MAX_RUBRIC_TEXT_CHARS = 2_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Skill(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    instruction: str = Field(min_length=1, max_length=MAX_SKILL_INSTRUCTION_CHARS)
    evidence: str = Field(min_length=1, max_length=2_000)


class Rubric(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=256)
    description: str = Field(min_length=1, max_length=MAX_RUBRIC_TEXT_CHARS)
    score_1_anchor: str = Field(min_length=1, max_length=MAX_RUBRIC_TEXT_CHARS)
    score_3_anchor: str = Field(min_length=1, max_length=MAX_RUBRIC_TEXT_CHARS)
    score_5_anchor: str = Field(min_length=1, max_length=MAX_RUBRIC_TEXT_CHARS)


class RubricEvaluation(StrictModel):
    rubric_id: str
    score: int = Field(ge=1, le=5)
    reason: str = Field(min_length=1)


class EvidenceItem(StrictModel):
    evidence_id: str
    kind: Literal["seed_document", "search_result"]
    query: str = ""
    content: str = Field(min_length=1)
    source: str
    trajectory_index: int = Field(ge=0)


class CandidateJudgment(StrictModel):
    candidate_index: int = Field(ge=0)
    semantically_equivalent: bool
    reason: str


class VerifierSample(StrictModel):
    sample_index: int = Field(ge=0)
    raw_response: str
    extracted_answer: str
    latency_seconds: float = Field(ge=0)


class VerifyResult(StrictModel):
    evidence_support: bool
    question_is_determinate: bool
    candidate_judgments: list[CandidateJudgment]
    passed: bool
    reason: str
    verifier_samples: list[VerifierSample] = Field(default_factory=list)
    judge_raw_output: str = ""
    latency_seconds: float = Field(default=0, ge=0)
    failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_pass_contract(self) -> VerifyResult:
        expected = (
            self.evidence_support
            and self.question_is_determinate
            and any(item.semantically_equivalent for item in self.candidate_judgments)
        )
        if self.passed != expected:
            raise ValueError("passed must equal the three-condition verify contract")
        return self


class Candidate(StrictModel):
    candidate_id: str
    iteration: int = Field(ge=0)
    doc_id: str
    hop_count: int = Field(ge=1)
    source_document: str = Field(min_length=1)
    proposer_trajectory: list[dict[str, Any]]
    evidence_bundle: list[EvidenceItem]
    question: str
    reference_answer: str
    format_score: float = Field(ge=0, le=1)
    format_failure: dict[str, Any] | None = None
    rubric_evaluation: list[RubricEvaluation] = Field(default_factory=list)
    rubric_raw_output: str = ""
    rubric_failure: dict[str, Any] | None = None
    rank_score: float = Field(default=0, ge=0, le=1)
    generation_index: int = Field(default=0, ge=0)
    status: Literal[
        "generated",
        "format_invalid",
        "ranked",
        "rubric_error",
        "verify_passed",
        "verify_failed",
        "verify_error",
        "not_verified",
    ] = "generated"
    verify_result: VerifyResult | None = None
    verify_failure: dict[str, Any] | None = None


class KeepoutResult(StrictModel):
    candidate_id: str
    doc_id: str
    question: str
    reference_answer: str
    trajectory: list[dict[str, Any]]
    model_answer: str
    judge_result: dict[str, Any]
    judge_raw_output: str = ""
    correct: bool


class TrajectorySummary(StrictModel):
    candidate_id: str
    correct: bool
    outcome_stage: str
    root_causes: list[str] = Field(min_length=1)
    related_rubric_ids: list[str] = Field(min_length=1)
    evidence_quotes: list[str] = Field(min_length=1)
    actionable_improvements: list[str] = Field(min_length=1)


class StageRecord(StrictModel):
    status: Literal["pending", "running", "failed", "completed"] = "pending"
    attempt: int = Field(default=0, ge=0)
    started_at: str | None = None
    completed_at: str | None = None
    manifest_path: str | None = None
    artifact_paths: list[str] = Field(default_factory=list)
    error: str | None = None


class ModelReferences(StrictModel):
    proposer: str
    solver_before: str
    solver_after: str = ""


class IterationState(StrictModel):
    schema_version: Literal[1] = 1
    iteration: int = Field(ge=0)
    status: Literal["running", "failed", "completed"] = "running"
    models: ModelReferences
    skills: list[Skill]
    rubrics: list[Rubric]
    config_snapshot: dict[str, Any]
    stages: dict[str, StageRecord] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_dynamic_state(self) -> IterationState:
        dynamic_config = self.config_snapshot.get("dynamic_state", {})
        max_skills = int(dynamic_config.get("max_skills", MAX_SKILLS))
        max_rubrics = int(dynamic_config.get("max_rubrics", MAX_RUBRICS))
        max_retries = int(dynamic_config.get("max_retries", 3))
        if max_retries < 1:
            raise ValueError("dynamic_state.max_retries must be positive")
        validate_skills(self.skills, max_items=max_skills)
        validate_rubrics(self.rubrics, max_items=max_rubrics)
        reject_plaintext_secrets(self.config_snapshot)
        return self


def initial_skills() -> list[Skill]:
    return [
        Skill(
            id="skill-evidence-chain",
            instruction="Construct a complete, verifiable evidence chain from the seed document to the final answer.",
            evidence="Initial iteration requirement.",
        ),
        Skill(
            id="skill-required-hops",
            instruction="Make every hop necessary to reach the final answer; no intermediate relation may be skipped.",
            evidence="Initial iteration requirement.",
        ),
        Skill(
            id="skill-unambiguous",
            instruction="Remove ambiguity from the question, evidence relations, and canonical reference answer.",
            evidence="Initial iteration requirement.",
        ),
    ]


def initial_rubrics() -> list[Rubric]:
    return [
        Rubric(
            id="rubric-evidence-support",
            name="Evidence support",
            description="The full evidence chain supports both the question premises and reference answer.",
            score_1_anchor="Key claims or the answer are unsupported or contradicted by the evidence.",
            score_3_anchor=(
                "The main answer is supported but one relation is weak, indirect, or incompletely documented."
            ),
            score_5_anchor=(
                "Every required relation and the canonical answer are explicitly supported by traceable evidence."
            ),
        ),
        Rubric(
            id="rubric-answer-uniqueness",
            name="Answer uniqueness",
            description="The evidence leads to one clear and canonical answer.",
            score_1_anchor=(
                "Multiple materially different answers fit the question or the requested answer type is unclear."
            ),
            score_3_anchor="A likely answer exists but normalization, scope, or wording leaves minor ambiguity.",
            score_5_anchor="Exactly one canonical concise answer follows from the question and evidence.",
        ),
        Rubric(
            id="rubric-hop-necessity",
            name="Multi-hop necessity",
            description="Every hop is necessary and contributes to reaching the final answer.",
            score_1_anchor="The answer is directly available or one or more hops are decorative or bypassable.",
            score_3_anchor="The chain is multi-hop but one transition can plausibly be skipped or inferred directly.",
            score_5_anchor="Each transition depends on the preceding result and all hops are required.",
        ),
        Rubric(
            id="rubric-retrievability",
            name="Search solvability",
            description="A solver can retrieve the required evidence using the normal search protocol.",
            score_1_anchor=(
                "Required facts are inaccessible, obscure without usable clues, or depend on unavailable sources."
            ),
            score_3_anchor=(
                "The chain is retrievable but queries require fragile wording or evidence is difficult to identify."
            ),
            score_5_anchor="Each hop provides a specific clue that reliably supports a productive search query.",
        ),
        Rubric(
            id="rubric-discrimination",
            name="Capability discrimination",
            description=(
                "The problem distinguishes solver capability instead of being trivial or effectively unsolvable."
            ),
            score_1_anchor="The problem is trivial, answer-leaking, arbitrary, or not realistically solvable.",
            score_3_anchor="The problem requires some reasoning but has limited depth or discriminative power.",
            score_5_anchor="The problem is challenging yet solvable and rewards robust multi-hop search and reasoning.",
        ),
    ]


def validate_skills(skills: list[Skill], *, max_items: int = MAX_SKILLS) -> None:
    if not 1 <= max_items <= MAX_SKILLS:
        raise ValueError(f"max_skills must be between 1 and {MAX_SKILLS}")
    if not skills:
        raise ValueError("skills must not be empty")
    if len(skills) > max_items:
        raise ValueError(f"skills exceed the maximum of {max_items}")
    ids = [skill.id for skill in skills]
    if len(ids) != len(set(ids)):
        raise ValueError("skill ids must be unique")
    prompt_length = sum(len(skill.instruction) + len(skill.evidence) for skill in skills)
    if prompt_length > MAX_SKILLS_PROMPT_CHARS:
        raise ValueError(f"serialized skills exceed {MAX_SKILLS_PROMPT_CHARS} characters")


def validate_rubrics(rubrics: list[Rubric], *, max_items: int = MAX_RUBRICS) -> None:
    if not 1 <= max_items <= MAX_RUBRICS:
        raise ValueError(f"max_rubrics must be between 1 and {MAX_RUBRICS}")
    if not rubrics:
        raise ValueError("rubrics must not be empty")
    if len(rubrics) > max_items:
        raise ValueError(f"rubrics exceed the maximum of {max_items}")
    ids = [rubric.id for rubric in rubrics]
    if len(ids) != len(set(ids)):
        raise ValueError("rubric ids must be unique")


def reject_plaintext_secrets(value: Any, path: str = "config_snapshot") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            sensitive = (
                lowered in {"api_key", "password", "secret", "access_token"}
                or lowered.endswith("_api_key")
                or lowered.endswith("_password")
                or lowered.endswith("_secret")
            )
            if sensitive and item is not None and item != "":
                raise ValueError(f"plaintext secret is forbidden in iteration state: {path}.{key}")
            reject_plaintext_secrets(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_plaintext_secrets(item, f"{path}[{index}]")


def normalize_document(document: str) -> str:
    return " ".join(document.split())


def stable_document_id(document: str, source_id: str | int | None = None) -> str:
    if source_id is not None and str(source_id).strip():
        return str(source_id).strip()
    normalized = normalize_document(document)
    if not normalized:
        raise ValueError("cannot derive doc_id from an empty document")
    return f"sha256:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"


def _parse_tool_call(raw: str | dict[str, Any]) -> str:
    payload = raw if isinstance(raw, dict) else json.loads(raw.strip())
    if "function" in payload:
        payload = {
            "name": payload["function"].get("name"),
            "arguments": payload["function"].get("arguments", {}),
        }
    if payload.get("name") != "search":
        raise ValueError("evidence extraction encountered a non-search tool call")
    arguments = payload.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    queries = arguments.get("query_list")
    if queries is None and isinstance(arguments.get("query"), str):
        queries = [arguments["query"]]
    valid_queries = isinstance(queries, list) and queries and all(
        isinstance(item, str) and item.strip() for item in queries
    )
    if not valid_queries:
        raise ValueError("search tool call is missing a non-empty query list")
    return "\n".join(item.strip() for item in queries)


def _parse_tool_response(raw: str | dict[str, Any]) -> tuple[str, str]:
    payload: Any = raw
    if isinstance(raw, str):
        stripped = raw.strip()
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = stripped
    if isinstance(payload, dict):
        content = payload.get("result") or payload.get("content")
        source = payload.get("source") or payload.get("url") or "search"
    else:
        content = payload
        source = "search"
    if not isinstance(content, str) or not content.strip():
        raise ValueError("search tool response is missing result content")
    return content.strip(), str(source)


def normalize_trajectory(trajectory: str | list[Any]) -> list[dict[str, Any]]:
    if isinstance(trajectory, list):
        normalized: list[dict[str, Any]] = []
        for item in trajectory:
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            elif hasattr(item, "__dict__") and not isinstance(item, dict):
                item = vars(item)
            if not isinstance(item, dict):
                raise ValueError("trajectory entries must be mappings")
            normalized.append(json.loads(json.dumps(item, default=str)))
        return normalized

    message_pattern = re.compile(r"<\|im_start\|>(assistant|user|system)\n(.*?)<\|im_end\|>", re.DOTALL)
    messages = [{"role": role, "content": content} for role, content in message_pattern.findall(trajectory)]
    if messages:
        return messages
    return [{"role": "raw", "content": trajectory}]


def extract_evidence_bundle(
    source_document: str,
    trajectory: str | list[Any],
    *,
    hop_count: int,
) -> tuple[list[dict[str, Any]], list[EvidenceItem]]:
    normalized_trajectory = normalize_trajectory(trajectory)
    calls: list[tuple[int, str]] = []
    responses: list[tuple[int, str | dict[str, Any]]] = []

    for index, message in enumerate(normalized_trajectory):
        raw_content = message.get("content")
        content = str(raw_content or "")
        role = message.get("role")
        if role in {"assistant", "raw"}:
            tool_calls = message.get("tool_calls")
            if tool_calls:
                for tool_call in tool_calls:
                    calls.append((index, _parse_tool_call(tool_call)))
            for raw_call in re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL):
                calls.append((index, _parse_tool_call(raw_call)))
        if role == "tool":
            responses.append((index, raw_content))
        elif role == "raw":
            for raw_response in re.findall(r"<tool_response>(.*?)</tool_response>", content, re.DOTALL):
                responses.append((index, raw_response))
        elif role == "user" and len(responses) < len(calls):
            for raw_response in re.findall(r"<tool_response>(.*?)</tool_response>", content, re.DOTALL):
                if len(responses) == len(calls):
                    break
                responses.append((index, raw_response))

    expected_searches = max(0, hop_count - 1)
    if len(calls) != expected_searches:
        raise ValueError(f"expected {expected_searches} search calls, found {len(calls)}")
    if len(responses) != len(calls):
        raise ValueError(f"search call/response mismatch: {len(calls)} calls and {len(responses)} responses")

    evidence = [
        EvidenceItem(
            evidence_id="evidence-0",
            kind="seed_document",
            content=source_document,
            source="seed_document",
            trajectory_index=0,
        )
    ]
    for evidence_index, ((call_index, query), (response_index, response)) in enumerate(
        zip(calls, responses, strict=True),
        start=1,
    ):
        if response_index < call_index:
            raise ValueError("tool response appears before its search call")
        content, source = _parse_tool_response(response)
        evidence.append(
            EvidenceItem(
                evidence_id=f"evidence-{evidence_index}",
                kind="search_result",
                query=query,
                content=content,
                source=source,
                trajectory_index=response_index,
            )
        )
    return normalized_trajectory, evidence


def normalized_rubric_mean(evaluations: list[RubricEvaluation], rubrics: list[Rubric] | None = None) -> float:
    if not evaluations:
        raise ValueError("rubric evaluations must not be empty")
    ids = [evaluation.rubric_id for evaluation in evaluations]
    if len(ids) != len(set(ids)):
        raise ValueError("rubric evaluation ids must be unique")
    if rubrics is not None and set(ids) != {rubric.id for rubric in rubrics}:
        raise ValueError("rubric evaluations must cover every active rubric exactly once")
    return sum((evaluation.score - 1) / 4 for evaluation in evaluations) / len(evaluations)


def resolve_rubric_scores(
    evaluation_groups: list[list[RubricEvaluation] | None],
    *,
    neutral_score: float = 0.5,
) -> tuple[list[float], list[bool]]:
    if not 0 <= neutral_score <= 1:
        raise ValueError("neutral rubric score must be in [0, 1]")
    normalized_scores = [
        normalized_rubric_mean(evaluations) if evaluations is not None else None
        for evaluations in evaluation_groups
    ]
    valid_scores = [score for score in normalized_scores if score is not None]
    fallback_score = sum(valid_scores) / len(valid_scores) if valid_scores else neutral_score
    scores = [score if score is not None else fallback_score for score in normalized_scores]
    failures = [evaluations is None for evaluations in evaluation_groups]
    return scores, failures


def candidate_rank_score(format_score: float, evaluations: list[RubricEvaluation]) -> float:
    if not 0 <= format_score <= 1:
        raise ValueError("format score must be in [0, 1]")
    return 0.5 * format_score + 0.5 * normalized_rubric_mean(evaluations)


def proposer_reward_components(
    format_score: float,
    difficulty_score: float,
    evaluations: list[RubricEvaluation],
    *,
    format_weight: float = 0.5,
    difficulty_weight: float = 1.0,
    rubric_weight: float = 0.5,
    rubric_score_override: float | None = None,
) -> dict[str, float]:
    rubric_score = (
        rubric_score_override
        if rubric_score_override is not None
        else normalized_rubric_mean(evaluations)
        if evaluations
        else 0.0
    )
    if not 0 <= rubric_score <= 1:
        raise ValueError("rubric score must be in [0, 1]")
    weighted_format = format_weight * format_score
    weighted_difficulty = difficulty_weight * difficulty_score
    weighted_rubric = rubric_weight * rubric_score
    return {
        "score": weighted_format + weighted_difficulty + weighted_rubric,
        "format_score": format_score,
        "difficulty_score": difficulty_score,
        "rubric_score": rubric_score,
        "weighted_format_score": weighted_format,
        "weighted_difficulty_score": weighted_difficulty,
        "weighted_rubric_score": weighted_rubric,
    }


def stable_group_split(
    candidates: list[Candidate],
    *,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[list[Candidate], list[Candidate], dict[str, Any]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be strictly between zero and one")
    groups: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        groups.setdefault(candidate.doc_id, []).append(candidate)

    train: list[Candidate] = []
    keepout: list[Candidate] = []
    assignments: dict[str, str] = {}
    denominator = 2**256
    for doc_id in sorted(groups):
        digest = hashlib.sha256(f"{seed}:{doc_id}".encode()).hexdigest()
        bucket = int(digest, 16) / denominator
        split = "train" if bucket < train_ratio else "keepout"
        assignments[doc_id] = split
        (train if split == "train" else keepout).extend(groups[doc_id])

    if not train:
        raise ValueError("stable group split produced an empty solver train set")
    if not keepout:
        raise ValueError("stable group split produced an empty keepout set")
    manifest = {
        "seed": seed,
        "train_ratio": train_ratio,
        "assignments": assignments,
        "train_doc_count": sum(value == "train" for value in assignments.values()),
        "keepout_doc_count": sum(value == "keepout" for value in assignments.values()),
        "train_candidate_ids": [candidate.candidate_id for candidate in train],
        "keepout_candidate_ids": [candidate.candidate_id for candidate in keepout],
    }
    return train, keepout, manifest


def canonical_hash(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dynamic_state_hash(state: IterationState) -> str:
    return canonical_hash(
        {
            "iteration": state.iteration,
            "models": {
                "proposer": state.models.proposer,
                "solver_before": state.models.solver_before,
            },
            "skills": [item.model_dump(mode="json") for item in state.skills],
            "rubrics": [item.model_dump(mode="json") for item in state.rubrics],
            "config_snapshot": state.config_snapshot,
        }
    )


def diff_by_id(before: list[BaseModel], after: list[BaseModel]) -> dict[str, Any]:
    before_by_id = {item.id: item.model_dump(mode="json") for item in before}
    after_by_id = {item.id: item.model_dump(mode="json") for item in after}
    return {
        "added": [after_by_id[item_id] for item_id in sorted(after_by_id.keys() - before_by_id.keys())],
        "removed": [before_by_id[item_id] for item_id in sorted(before_by_id.keys() - after_by_id.keys())],
        "modified": [
            {"before": before_by_id[item_id], "after": after_by_id[item_id]}
            for item_id in sorted(before_by_id.keys() & after_by_id.keys())
            if before_by_id[item_id] != after_by_id[item_id]
        ],
    }


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary_path = Path(handle.name)
    os.replace(temporary_path, destination)


class StateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> IterationState:
        with self.path.open(encoding="utf-8") as handle:
            return IterationState.model_validate(json.load(handle))

    def save(self, state: IterationState) -> None:
        state.updated_at = utc_now()
        validated = IterationState.model_validate(state.model_dump(mode="json"))
        atomic_write_json(self.path, validated)

    def initialize(
        self,
        *,
        iteration: int,
        proposer: str,
        solver_before: str,
        config_snapshot: dict[str, Any],
        skills: list[Skill] | None = None,
        rubrics: list[Rubric] | None = None,
    ) -> IterationState:
        if self.path.exists():
            raise FileExistsError(f"iteration state already exists: {self.path}")
        state = IterationState(
            iteration=iteration,
            models=ModelReferences(proposer=proposer, solver_before=solver_before),
            skills=skills or initial_skills(),
            rubrics=rubrics or initial_rubrics(),
            config_snapshot=config_snapshot,
        )
        self.save(state)
        return state
