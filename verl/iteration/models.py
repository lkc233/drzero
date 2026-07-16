from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal

import httpx
from pydantic import Field

from verl.custom_reward.format_scoring import normalize_answer
from verl.iteration.core import (
    Candidate,
    KeepoutResult,
    Rubric,
    RubricEvaluation,
    Skill,
    StrictModel,
    TrajectorySummary,
    VerifierSample,
    VerifyResult,
    candidate_rank_score,
    diff_by_id,
    resolve_rubric_scores,
    validate_rubrics,
    validate_skills,
)
from verl.prompts import (
    ANSWER_JUDGE_PROMPT,
    ANSWER_PATTERN,
    DEFAULT_SOLVER_PREFIX,
    GLOBAL_ANALYSIS_PROMPT,
    QUESTION_ONLY_VERIFIER_PROMPT,
    RUBRIC_EVALUATION_PROMPT,
    RUBRICS_UPDATE_PROMPT,
    SKILLS_UPDATE_PROMPT,
    TRAJECTORY_ANALYSIS_PROMPT,
    VERIFIER_PROMPT,
)

logger = logging.getLogger(__name__)


class ModelCallError(RuntimeError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


class EndpointConfig(StrictModel):
    model_name: str
    base_url: str
    api_key_env: str | None = None
    timeout_seconds: float = Field(default=120, gt=0)
    max_retries: int = Field(default=3, ge=1)
    max_concurrency: int = Field(default=32, ge=1)
    disable_thinking: bool = True


class ClientMetrics(StrictModel):
    calls: int = 0
    retries: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency_seconds: float = 0


class RubricEvaluationResponse(StrictModel):
    evaluations: list[RubricEvaluation]


class SemanticJudgment(StrictModel):
    semantically_equivalent: bool
    reason: str


class UpdateDecision(StrictModel):
    id: str
    action: Literal["added", "retained", "modified", "removed"]
    reason: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)


class SkillsResponse(StrictModel):
    skills: list[Skill]
    decisions: list[UpdateDecision]


class RubricsResponse(StrictModel):
    rubrics: list[Rubric]
    decisions: list[UpdateDecision]


class AnalysisReport(StrictModel):
    problem_frequencies: dict[str, int]
    success_patterns: list[str]
    failure_patterns: list[str]
    related_rubric_ids: list[str]
    representative_cases: list[str]
    actionable_improvements: list[str]


def validate_model_reference(configured: str, expected: str, *, role: str) -> None:
    if not expected:
        raise ValueError(f"state does not define the expected {role} model")
    if configured != expected:
        raise ValueError(
            f"{role} model does not match iteration state: {configured!r} != {expected!r}"
        )


def validate_update_decisions(
    before: list[StrictModel],
    after: list[StrictModel],
    decisions: list[UpdateDecision],
) -> None:
    before_by_id = {item.id: item.model_dump(mode="json") for item in before}
    after_by_id = {item.id: item.model_dump(mode="json") for item in after}
    decision_by_id = {item.id: item for item in decisions}
    expected_ids = before_by_id.keys() | after_by_id.keys()
    if len(decision_by_id) != len(decisions) or decision_by_id.keys() != expected_ids:
        raise ModelCallError("update decisions must cover every before/after id exactly once")
    for item_id in expected_ids:
        if item_id not in before_by_id:
            expected_action = "added"
        elif item_id not in after_by_id:
            expected_action = "removed"
        elif before_by_id[item_id] == after_by_id[item_id]:
            expected_action = "retained"
        else:
            expected_action = "modified"
        if decision_by_id[item_id].action != expected_action:
            raise ModelCallError(
                f"update decision for {item_id} must be {expected_action}, "
                f"got {decision_by_id[item_id].action}"
            )


def validate_evidence_references(
    decisions: list[UpdateDecision],
    evidence_context: dict[str, Any],
) -> None:
    for decision in decisions:
        for reference in decision.evidence_refs:
            if not reference or not reference.startswith("/"):
                raise ModelCallError("update evidence_refs must be non-root JSON Pointers")
            current: Any = evidence_context
            try:
                for raw_token in reference[1:].split("/"):
                    if re.search(r"~(?:[^01]|$)", raw_token):
                        raise ValueError("invalid JSON Pointer escape")
                    token = raw_token.replace("~1", "/").replace("~0", "~")
                    if isinstance(current, list):
                        if not re.fullmatch(r"(?:0|[1-9][0-9]*)", token):
                            raise ValueError("invalid JSON Pointer array index")
                        current = current[int(token)]
                    else:
                        current = current[token]
            except (KeyError, IndexError, TypeError, ValueError) as error:
                raise ModelCallError(
                    f"update decision {decision.id} has an unresolved evidence reference: {reference}"
                ) from error


def _chat_completions_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if not url.endswith("/v1"):
        url += "/v1"
    return url + "/chat/completions"


def _extract_json(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start_candidates = [position for position in (stripped.find("{"), stripped.find("[")) if position >= 0]
        if not start_candidates:
            raise
        start = min(start_candidates)
        end = max(stripped.rfind("}"), stripped.rfind("]"))
        if end < start:
            raise
        return json.loads(stripped[start : end + 1])


class OpenAICompatibleClient:
    """Small strict client that never persists or logs API-key values."""

    def __init__(self, config: EndpointConfig):
        self.config = config
        self.metrics = ClientMetrics()
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._client = httpx.AsyncClient(timeout=config.timeout_seconds, trust_env=False)

    async def __aenter__(self) -> OpenAICompatibleClient:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        await self.close()
        return False

    async def close(self) -> None:
        await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env:
            api_key = os.getenv(self.config.api_key_env)
            if not api_key:
                raise ModelCallError(f"required API key environment variable is unset: {self.config.api_key_env}")
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def complete_message(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0,
        max_tokens: int = 2_048,
        json_mode: bool = False,
    ) -> tuple[dict[str, Any], str, float]:
        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools is not None:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.config.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        started = time.monotonic()
        last_error: Exception | None = None
        async with self._semaphore:
            for attempt in range(self.config.max_retries):
                self.metrics.calls += 1
                try:
                    response = await self._client.post(
                        _chat_completions_url(self.config.base_url),
                        headers=self._headers(),
                        json=payload,
                    )
                    response.raise_for_status()
                    body = response.json()
                    message = body["choices"][0]["message"]
                    raw = message.get("content") or ""
                    usage = body.get("usage") or {}
                    self.metrics.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                    self.metrics.completion_tokens += int(usage.get("completion_tokens") or 0)
                    latency = time.monotonic() - started
                    self.metrics.total_latency_seconds += latency
                    return message, raw, latency
                except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
                    last_error = error
                    if attempt + 1 < self.config.max_retries:
                        self.metrics.retries += 1
                        await asyncio.sleep(2**attempt)
            self.metrics.failures += 1
        raise ModelCallError(
            f"model call failed after {self.config.max_retries} attempts: {last_error}"
        ) from last_error

    async def complete_text(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0,
        max_tokens: int = 2_048,
    ) -> tuple[str, float]:
        _, raw, latency = await self.complete_message(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return raw, latency

    async def complete_structured(
        self,
        prompt: str,
        response_model: type[StrictModel],
    ) -> tuple[StrictModel, str, float]:
        last_error: Exception | None = None
        raw_outputs: list[str] = []
        for parse_attempt in range(self.config.max_retries):
            _, raw, latency = await self.complete_message(
                [{"role": "user", "content": prompt}],
                temperature=0,
                json_mode=True,
            )
            raw_outputs.append(raw)
            try:
                return response_model.model_validate(_extract_json(raw)), raw, latency
            except (json.JSONDecodeError, ValueError) as error:
                last_error = error
                if parse_attempt + 1 < self.config.max_retries:
                    self.metrics.retries += 1
        self.metrics.failures += 1
        raise ModelCallError(
            f"structured output validation failed: {last_error}",
            details={"raw_outputs": raw_outputs},
        ) from last_error

    async def complete_json(self, prompt: str) -> tuple[dict[str, Any], str, float]:
        _, raw, latency = await self.complete_message(
            [{"role": "user", "content": prompt}],
            temperature=0,
            json_mode=True,
        )
        parsed = _extract_json(raw)
        if not isinstance(parsed, dict):
            raise ModelCallError("expected a JSON object")
        return parsed, raw, latency


def extract_answer(response: str) -> str:
    matches = re.findall(ANSWER_PATTERN, response, re.DOTALL)
    return matches[-1].strip() if matches else response.strip()


class ProblemVerifier:
    def __init__(
        self,
        solver_client: OpenAICompatibleClient,
        *,
        samples: int = 3,
    ):
        if samples != 3:
            raise ValueError("the verifier contract requires exactly 3 solver samples")
        self.solver_client = solver_client
        self.samples = samples

    async def verify(self, candidate: Candidate) -> VerifyResult:
        started = time.monotonic()
        evidence_json = json.dumps(
            [item.model_dump(mode="json") for item in candidate.evidence_bundle],
            ensure_ascii=False,
        )
        with_evidence_prompt = VERIFIER_PROMPT.format(
            evidence_bundle=evidence_json,
            question=candidate.question,
        )
        question_only_prompt = QUESTION_ONLY_VERIFIER_PROMPT.format(question=candidate.question)

        async def sample_once(
            prompt: str,
            sample_index: int,
            prompt_mode: Literal["with_evidence", "question_only"],
        ) -> VerifierSample:
            raw, latency = await self.solver_client.complete_text(
                [{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            if "<tool_call>" in raw:
                raise ModelCallError(
                    "verifier attempted a forbidden tool call",
                    details={
                        "prompt_mode": prompt_mode,
                        "sample_index": sample_index,
                        "raw_output": raw,
                    },
                )
            extracted_answer = extract_answer(raw)
            return VerifierSample(
                sample_index=sample_index,
                raw_response=raw,
                extracted_answer=extracted_answer,
                correct=normalize_answer(extracted_answer)
                == normalize_answer(candidate.reference_answer),
                latency_seconds=latency,
            )

        outcomes = await asyncio.gather(
            *(
                sample_once(with_evidence_prompt, index, "with_evidence")
                for index in range(self.samples)
            ),
            *(
                sample_once(question_only_prompt, index, "question_only")
                for index in range(self.samples)
            ),
            return_exceptions=True,
        )
        with_evidence_outcomes = outcomes[: self.samples]
        question_only_outcomes = outcomes[self.samples :]
        with_evidence_samples = sorted(
            [outcome for outcome in with_evidence_outcomes if isinstance(outcome, VerifierSample)],
            key=lambda item: item.sample_index,
        )
        question_only_samples = sorted(
            [outcome for outcome in question_only_outcomes if isinstance(outcome, VerifierSample)],
            key=lambda item: item.sample_index,
        )
        sample_failures = [
            {
                "prompt_mode": "with_evidence" if index < self.samples else "question_only",
                "sample_index": index % self.samples,
                "error_type": type(outcome).__name__,
                "reason": str(outcome),
                "details": getattr(outcome, "details", {}),
            }
            for index, outcome in enumerate(outcomes)
            if isinstance(outcome, BaseException)
        ]
        if sample_failures:
            raise ModelCallError(
                "one or more verifier solver samples failed",
                details={
                    "with_evidence_samples": [
                        item.model_dump(mode="json") for item in with_evidence_samples
                    ],
                    "question_only_samples": [
                        item.model_dump(mode="json") for item in question_only_samples
                    ],
                    "sample_failures": sample_failures,
                },
            )
        return VerifyResult.from_samples(
            with_evidence_samples=with_evidence_samples,
            question_only_samples=question_only_samples,
            latency_seconds=time.monotonic() - started,
        )


async def evaluate_rubrics(
    candidate: Candidate,
    rubrics: list[Rubric],
    judge_client: OpenAICompatibleClient,
    *,
    max_attempts: int = 3,
) -> tuple[list[RubricEvaluation], str]:
    if max_attempts < 1:
        raise ValueError("rubric evaluation max_attempts must be positive")
    prompt = RUBRIC_EVALUATION_PROMPT.format(
        rubrics=json.dumps([item.model_dump(mode="json") for item in rubrics], ensure_ascii=False),
        source_document=candidate.source_document,
        trajectory=json.dumps(candidate.proposer_trajectory, ensure_ascii=False),
        question=candidate.question,
        reference_answer=candidate.reference_answer,
    )
    expected = {rubric.id for rubric in rubrics}
    attempt_prompt = prompt
    failures: list[dict[str, Any]] = []
    last_error: ModelCallError | None = None
    for attempt in range(1, max_attempts + 1):
        response_model, raw, _ = await judge_client.complete_structured(
            attempt_prompt,
            RubricEvaluationResponse,
        )
        response = RubricEvaluationResponse.model_validate(response_model)
        actual = {evaluation.rubric_id for evaluation in response.evaluations}
        if expected == actual and len(actual) == len(response.evaluations):
            return response.evaluations, raw

        last_error = ModelCallError(
            "rubric response must cover every active rubric exactly once",
            details={"raw_output": raw},
        )
        failures.append(
            {
                "attempt": attempt,
                "error_type": type(last_error).__name__,
                "reason": str(last_error),
                "raw_output": raw,
                "details": last_error.details,
            }
        )
        if attempt < max_attempts:
            attempt_prompt = (
                prompt
                + "\n\nThe previous response was invalid. Return exactly one evaluation for each "
                + f"rubric id in this list: {json.dumps(sorted(expected), ensure_ascii=False)}."
            )
    raise ModelCallError(
        f"rubric evaluation failed after {max_attempts} attempts",
        details={"attempts": failures},
    ) from last_error


async def evaluate_rubric_rewards(
    candidates: list[Candidate],
    rubrics: list[Rubric],
    judge_client: OpenAICompatibleClient,
) -> tuple[list[list[RubricEvaluation]], list[float], list[bool]]:
    """Evaluate and score only candidates that passed the proposer format gate."""
    evaluations_by_candidate: list[list[RubricEvaluation]] = [[] for _ in candidates]
    scores_by_candidate = [0.0 for _ in candidates]
    failures_by_candidate = [False for _ in candidates]
    eligible = [
        (index, candidate)
        for index, candidate in enumerate(candidates)
        if candidate.format_score > 0
    ]
    if not eligible:
        return evaluations_by_candidate, scores_by_candidate, failures_by_candidate

    outputs = await asyncio.gather(
        *(evaluate_rubrics(candidate, rubrics, judge_client) for _, candidate in eligible),
        return_exceptions=True,
    )
    evaluation_groups: list[list[RubricEvaluation] | None] = []
    for (_, candidate), output in zip(eligible, outputs, strict=True):
        if isinstance(output, BaseException):
            evaluation_groups.append(None)
            logger.warning(
                "Rubric evaluation failed for %s: %s; details=%s",
                candidate.candidate_id,
                output,
                json.dumps(getattr(output, "details", {}), ensure_ascii=False, default=str),
            )
        else:
            evaluations, _ = output
            evaluation_groups.append(evaluations)

    eligible_scores, eligible_failures = resolve_rubric_scores(evaluation_groups)
    for (index, _), evaluations, score, failed in zip(
        eligible,
        evaluation_groups,
        eligible_scores,
        eligible_failures,
        strict=True,
    ):
        evaluations_by_candidate[index] = evaluations or []
        scores_by_candidate[index] = score
        failures_by_candidate[index] = failed
    return evaluations_by_candidate, scores_by_candidate, failures_by_candidate


async def rank_and_verify_candidates(
    candidates: list[Candidate],
    rubrics: list[Rubric],
    judge_client: OpenAICompatibleClient,
    verifier: ProblemVerifier,
) -> tuple[Candidate | None, list[Candidate], dict[str, str]]:
    raw_rubric_outputs: dict[str, str] = {}
    for candidate in candidates:
        if candidate.format_score <= 0:
            candidate.status = "format_invalid"
            continue
        started = time.monotonic()
        try:
            rubric_evaluations, raw_output = await evaluate_rubrics(candidate, rubrics, judge_client)
            candidate.rubric_evaluation = rubric_evaluations
            candidate.rubric_raw_output = raw_output
            candidate.rank_score = candidate_rank_score(candidate.format_score, rubric_evaluations)
            candidate.status = "ranked"
            raw_rubric_outputs[candidate.candidate_id] = raw_output
        except Exception as error:
            candidate.status = "rubric_error"
            candidate.rubric_failure = {
                "error_type": type(error).__name__,
                "reason": str(error),
                "latency_seconds": time.monotonic() - started,
                "details": getattr(error, "details", {}),
            }
            raise

    ordered = sorted(candidates, key=lambda item: (-item.rank_score, item.generation_index))
    selected: Candidate | None = None
    for index, candidate in enumerate(ordered):
        if candidate.format_score <= 0:
            continue
        started = time.monotonic()
        try:
            result = await verifier.verify(candidate)
        except Exception as error:
            candidate.status = "verify_error"
            candidate.verify_failure = {
                "error_type": type(error).__name__,
                "reason": str(error),
                "latency_seconds": time.monotonic() - started,
                "details": getattr(error, "details", {}),
            }
            logger.warning(
                "Verify failed for %s: %s; details=%s",
                candidate.candidate_id,
                error,
                json.dumps(getattr(error, "details", {}), ensure_ascii=False, default=str),
            )
            continue
        candidate.verify_result = result
        candidate.status = "verify_passed" if result.passed else "verify_failed"
        if result.passed:
            selected = candidate
            for remaining in ordered[index + 1 :]:
                if remaining.format_score > 0:
                    remaining.status = "not_verified"
            break
    return selected, ordered, raw_rubric_outputs


async def rank_and_verify_candidate_groups(
    candidate_groups: list[list[Candidate]],
    rubrics: list[Rubric],
    judge_client: OpenAICompatibleClient,
    verifier: ProblemVerifier,
    *,
    max_concurrency: int,
    on_group_complete: Callable[[list[Candidate]], Awaitable[None]] | None = None,
) -> list[Candidate | None]:
    """Rank and verify independent document groups concurrently in stable input order."""
    if max_concurrency < 1:
        raise ValueError("candidate group concurrency must be positive")
    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_group(group: list[Candidate]) -> Candidate | None:
        async with semaphore:
            try:
                try:
                    winner, _, _ = await rank_and_verify_candidates(
                        group,
                        rubrics,
                        judge_client,
                        verifier,
                    )
                    return winner
                except Exception as error:
                    logger.warning(
                        "Candidate group verify failed for %s: %s; details=%s",
                        [candidate.candidate_id for candidate in group],
                        error,
                        json.dumps(
                            getattr(error, "details", {}),
                            ensure_ascii=False,
                            default=str,
                        ),
                    )
                    return None
            finally:
                if on_group_complete is not None:
                    await on_group_complete(group)

    return list(await asyncio.gather(*(process_group(group) for group in candidate_groups)))


SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Searches for relevant information based on a list of semantic queries.",
        "parameters": {
            "type": "object",
            "properties": {
                "query_list": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["query_list"],
        },
    },
}


class SearchRolloutClient:
    def __init__(
        self,
        model_client: OpenAICompatibleClient,
        *,
        retrieval_service_url: str,
        topk: int = 3,
        max_turns: int = 5,
    ):
        self.model_client = model_client
        self.retrieval_service_url = retrieval_service_url
        self.topk = topk
        self.max_turns = max_turns

    async def run(self, question: str) -> tuple[list[dict[str, Any]], str]:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": DEFAULT_SOLVER_PREFIX.format(question=question.strip())}
        ]
        for _ in range(self.max_turns):
            try:
                message, content, _ = await self.model_client.complete_message(
                    messages,
                    tools=[SEARCH_TOOL_SCHEMA],
                    temperature=0,
                )
            except Exception as error:
                raise ModelCallError(
                    "solver keepout rollout model call failed",
                    details={
                        "trajectory": messages,
                        "error_type": type(error).__name__,
                        "reason": str(error),
                        "model_call_details": getattr(error, "details", {}),
                    },
                ) from error
            assistant_message = {
                key: value
                for key, value in message.items()
                if key in {"role", "content", "tool_calls"} and value is not None
            }
            assistant_message.setdefault("role", "assistant")
            messages.append(assistant_message)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return messages, extract_answer(content)
            for tool_call in tool_calls:
                function = tool_call.get("function") or {}
                if function.get("name") != "search":
                    raise ModelCallError(
                        "solver requested an unsupported tool",
                        details={"trajectory": messages, "tool_call": tool_call},
                    )
                arguments = function.get("arguments") or {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError as error:
                        raise ModelCallError(
                            "solver search call returned invalid JSON arguments",
                            details={
                                "trajectory": messages,
                                "tool_call": tool_call,
                                "raw_arguments": arguments,
                            },
                        ) from error
                queries = arguments.get("query_list") if isinstance(arguments, dict) else None
                if not isinstance(queries, list) or not queries:
                    raise ModelCallError(
                        "solver search call omitted query_list",
                        details={"trajectory": messages, "tool_call": tool_call},
                    )
                from verl.tools.utils.search_r1_like_utils import perform_single_search_batch

                try:
                    result_text, _ = await asyncio.to_thread(
                        perform_single_search_batch,
                        retrieval_service_url=self.retrieval_service_url,
                        query_list=queries,
                        topk=self.topk,
                    )
                except Exception as error:
                    raise ModelCallError(
                        "solver keepout retrieval call failed",
                        details={
                            "trajectory": messages,
                            "queries": queries,
                            "error_type": type(error).__name__,
                            "reason": str(error),
                        },
                    ) from error
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "name": "search",
                        "content": result_text,
                    }
                )
        raise ModelCallError(
            "solver keepout rollout exceeded max_turns without a final answer",
            details={"trajectory": messages},
        )


class KeepoutEvaluator:
    def __init__(self, rollout_client: SearchRolloutClient, judge_client: OpenAICompatibleClient):
        self.rollout_client = rollout_client
        self.judge_client = judge_client

    async def evaluate_one(self, candidate: Candidate) -> KeepoutResult:
        trajectory, model_answer = await self.rollout_client.run(candidate.question)
        prompt = ANSWER_JUDGE_PROMPT.format(
            question=candidate.question,
            reference_answer=candidate.reference_answer,
            model_answer=model_answer,
        )
        try:
            result_model, raw_judge, _ = await self.judge_client.complete_structured(
                prompt,
                SemanticJudgment,
            )
        except Exception as error:
            raise ModelCallError(
                "keepout answer judge failed",
                details={
                    "candidate_id": candidate.candidate_id,
                    "trajectory": trajectory,
                    "model_answer": model_answer,
                    "judge_error": {
                        "error_type": type(error).__name__,
                        "reason": str(error),
                        "details": getattr(error, "details", {}),
                    },
                },
            ) from error
        judgment = SemanticJudgment.model_validate(result_model)
        return KeepoutResult(
            candidate_id=candidate.candidate_id,
            doc_id=candidate.doc_id,
            question=candidate.question,
            reference_answer=candidate.reference_answer,
            trajectory=trajectory,
            model_answer=model_answer,
            judge_result=judgment.model_dump(mode="json"),
            judge_raw_output=raw_judge,
            correct=judgment.semantically_equivalent,
        )

    async def evaluate_all(self, candidates: list[Candidate]) -> tuple[list[KeepoutResult], float]:
        results = list(await asyncio.gather(*(self.evaluate_one(candidate) for candidate in candidates)))
        if not results:
            raise ValueError("keepout set must not be empty")
        accuracy = sum(result.correct for result in results) / len(results)
        return results, accuracy


class TrajectoryAnalyzer:
    def __init__(self, client: OpenAICompatibleClient, *, chunk_size: int = 50):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.client = client
        self.chunk_size = chunk_size

    async def analyze_one(
        self,
        result: KeepoutResult,
        rubrics: list[Rubric],
    ) -> dict[str, Any]:
        prompt = TRAJECTORY_ANALYSIS_PROMPT.format(
            rubrics=json.dumps([item.model_dump(mode="json") for item in rubrics], ensure_ascii=False),
            record=result.model_dump_json(),
        )
        summary_model, raw, _ = await self.client.complete_structured(prompt, TrajectorySummary)
        summary = TrajectorySummary.model_validate(summary_model)
        if summary.candidate_id != result.candidate_id or summary.correct != result.correct:
            raise ModelCallError(
                "trajectory summary identity/correctness does not match its source record",
                details={"input": result.model_dump(mode="json"), "raw_output": raw},
            )
        rubric_ids = {rubric.id for rubric in rubrics}
        unknown_rubrics = set(summary.related_rubric_ids) - rubric_ids
        if unknown_rubrics:
            raise ModelCallError(
                f"trajectory summary references unknown rubrics: {sorted(unknown_rubrics)}",
                details={"input": result.model_dump(mode="json"), "raw_output": raw},
            )
        source_text = json.dumps(result.trajectory, ensure_ascii=False)
        missing_quotes = [quote for quote in summary.evidence_quotes if quote not in source_text]
        if missing_quotes:
            raise ModelCallError(
                "trajectory summary contains evidence quotes absent from its source record",
                details={"input": result.model_dump(mode="json"), "raw_output": raw},
            )
        return {"input": result.model_dump(mode="json"), "raw_output": raw, "summary": summary.model_dump(mode="json")}

    async def analyze_all(
        self,
        results: list[KeepoutResult],
        rubrics: list[Rubric],
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        item_records = []
        for result in results:
            item_records.append(await self.analyze_one(result, rubrics))
            if progress_callback:
                progress_callback({"status": "running", "items": item_records, "chunks": []})
        chunk_records = []
        summaries = [item["summary"] for item in item_records]
        for start in range(0, len(summaries), self.chunk_size):
            chunk = summaries[start : start + self.chunk_size]
            prompt = GLOBAL_ANALYSIS_PROMPT.format(summaries=json.dumps(chunk, ensure_ascii=False))
            report_model, raw, _ = await self.client.complete_structured(prompt, AnalysisReport)
            report = AnalysisReport.model_validate(report_model).model_dump(mode="json")
            chunk_records.append({"start": start, "input": chunk, "raw_output": raw, "report": report})
            if progress_callback:
                progress_callback({"status": "running", "items": item_records, "chunks": chunk_records})
        final_prompt = GLOBAL_ANALYSIS_PROMPT.format(
            summaries=json.dumps([item["report"] for item in chunk_records], ensure_ascii=False)
        )
        global_model, raw_global, _ = await self.client.complete_structured(final_prompt, AnalysisReport)
        global_report = AnalysisReport.model_validate(global_model).model_dump(mode="json")
        completed = {
            "status": "completed",
            "items": item_records,
            "chunks": chunk_records,
            "global": {
                "input": [item["report"] for item in chunk_records],
                "raw_output": raw_global,
                "report": global_report,
            },
        }
        if progress_callback:
            progress_callback(completed)
        return completed


class DynamicStateUpdater:
    def __init__(
        self,
        client: OpenAICompatibleClient,
        *,
        max_skills: int = 12,
        max_rubrics: int = 12,
        max_retries: int | None = None,
    ):
        self.client = client
        self.max_skills = max_skills
        self.max_rubrics = max_rubrics
        self.max_retries = 1 if max_retries is None else max_retries
        if self.max_retries < 1:
            raise ValueError("max_retries must be positive")

    async def _complete_validated_update(
        self,
        prompt: str,
        response_model: type[StrictModel],
        validator: Callable[[Any], tuple[list[Any], list[UpdateDecision]]],
        *,
        label: str,
    ) -> tuple[list[Any], list[UpdateDecision], str]:
        failures = []
        attempt_prompt = prompt
        for attempt in range(1, self.max_retries + 1):
            raw_output: str | None = None
            try:
                model, raw_output, _ = await self.client.complete_structured(
                    attempt_prompt,
                    response_model,
                )
                items, decisions = validator(model)
                return items, decisions, raw_output
            except Exception as error:
                failures.append(
                    {
                        "attempt": attempt,
                        "error_type": type(error).__name__,
                        "reason": str(error),
                        "raw_output": raw_output,
                        "details": getattr(error, "details", {}),
                    }
                )
                attempt_prompt = (
                    f"{prompt}\n\nThe previous {label} update was invalid: {error}. "
                    "Return a corrected response satisfying every schema and evidence-reference constraint."
                )
        raise ModelCallError(
            f"{label} update failed validation after {self.max_retries} attempts",
            details={"attempts": failures},
        )

    async def update_skills(
        self,
        *,
        skills: list[Skill],
        rubrics: list[Rubric],
        rubric_evidence: Any,
        verify_evidence: Any,
        keepout_evidence: Any,
    ) -> tuple[list[Skill], list[UpdateDecision], str]:
        skills_prompt = SKILLS_UPDATE_PROMPT.format(
            skills=json.dumps([item.model_dump(mode="json") for item in skills], ensure_ascii=False),
            rubric_evidence=json.dumps(
                {
                    "rubrics": [item.model_dump(mode="json") for item in rubrics],
                    "evaluations": rubric_evidence,
                },
                ensure_ascii=False,
            ),
            verify_evidence=json.dumps(verify_evidence, ensure_ascii=False),
            keepout_evidence=json.dumps(keepout_evidence, ensure_ascii=False),
        )
        skills_prompt = skills_prompt.replace(
            "Use at most 12 skills",
            f"Use at most {self.max_skills} skills",
        )
        def validate(model: Any) -> tuple[list[Any], list[UpdateDecision]]:
            response = SkillsResponse.model_validate(model)
            next_skills = response.skills
            validate_skills(next_skills, max_items=self.max_skills)
            validate_update_decisions(skills, next_skills, response.decisions)
            validate_evidence_references(
                response.decisions,
                {
                    "current_skills": [item.model_dump(mode="json") for item in skills],
                    "current_rubrics": [item.model_dump(mode="json") for item in rubrics],
                    "rubric_evidence": rubric_evidence,
                    "verify_evidence": verify_evidence,
                    "keepout_evidence": keepout_evidence,
                },
            )
            return next_skills, response.decisions

        next_skills, decisions, skills_raw = await self._complete_validated_update(
            skills_prompt,
            SkillsResponse,
            validate,
            label="skills",
        )
        return next_skills, decisions, skills_raw

    async def update_rubrics(
        self,
        *,
        rubrics: list[Rubric],
        next_skills: list[Skill],
        rubric_evidence: Any,
        verify_evidence: Any,
        keepout_evidence: Any,
    ) -> tuple[list[Rubric], list[UpdateDecision], str]:
        rubrics_prompt = RUBRICS_UPDATE_PROMPT.format(
            rubrics=json.dumps([item.model_dump(mode="json") for item in rubrics], ensure_ascii=False),
            skills=json.dumps([item.model_dump(mode="json") for item in next_skills], ensure_ascii=False),
            rubric_evidence=json.dumps(rubric_evidence, ensure_ascii=False),
            verify_evidence=json.dumps(verify_evidence, ensure_ascii=False),
            keepout_evidence=json.dumps(keepout_evidence, ensure_ascii=False),
        )
        rubrics_prompt = rubrics_prompt.replace(
            "Use at most 12 rubrics",
            f"Use at most {self.max_rubrics} rubrics",
        )
        def validate(model: Any) -> tuple[list[Any], list[UpdateDecision]]:
            response = RubricsResponse.model_validate(model)
            next_rubrics = response.rubrics
            validate_rubrics(next_rubrics, max_items=self.max_rubrics)
            validate_update_decisions(rubrics, next_rubrics, response.decisions)
            validate_evidence_references(
                response.decisions,
                {
                    "current_rubrics": [item.model_dump(mode="json") for item in rubrics],
                    "next_skills": [item.model_dump(mode="json") for item in next_skills],
                    "rubric_evidence": rubric_evidence,
                    "verify_evidence": verify_evidence,
                    "keepout_evidence": keepout_evidence,
                },
            )
            return next_rubrics, response.decisions

        next_rubrics, decisions, rubrics_raw = await self._complete_validated_update(
            rubrics_prompt,
            RubricsResponse,
            validate,
            label="rubrics",
        )
        return next_rubrics, decisions, rubrics_raw

    async def update(
        self,
        *,
        skills: list[Skill],
        rubrics: list[Rubric],
        rubric_evidence: Any,
        verify_evidence: Any,
        keepout_evidence: Any,
    ) -> dict[str, Any]:
        next_skills, skill_decisions, skills_raw = await self.update_skills(
            skills=skills,
            rubrics=rubrics,
            rubric_evidence=rubric_evidence,
            verify_evidence=verify_evidence,
            keepout_evidence=keepout_evidence,
        )
        next_rubrics, rubric_decisions, rubrics_raw = await self.update_rubrics(
            rubrics=rubrics,
            next_skills=next_skills,
            rubric_evidence=rubric_evidence,
            verify_evidence=verify_evidence,
            keepout_evidence=keepout_evidence,
        )
        return {
            "skills": next_skills,
            "rubrics": next_rubrics,
            "skills_diff": diff_by_id(skills, next_skills),
            "rubrics_diff": diff_by_id(rubrics, next_rubrics),
            "skills_decisions": skill_decisions,
            "rubrics_decisions": rubric_decisions,
            "raw_outputs": {"skills": skills_raw, "rubrics": rubrics_raw},
        }


RubricEvaluator = Callable[
    [Candidate, list[Rubric], OpenAICompatibleClient],
    Awaitable[tuple[list[RubricEvaluation], str]],
]
