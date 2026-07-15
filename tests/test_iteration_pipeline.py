import asyncio
import json
import os
import re
import sys

import pandas as pd
import pytest

from verl.iteration.core import (
    Candidate,
    CandidateJudgment,
    Rubric,
    RubricEvaluation,
    Skill,
    StateStore,
    TrajectorySummary,
    VerifyResult,
    candidate_rank_score,
    diff_by_id,
    extract_evidence_bundle,
    initial_rubrics,
    initial_skills,
    normalized_rubric_mean,
    proposer_reward_components,
    resolve_rubric_scores,
    stable_document_id,
    stable_group_split,
    validate_rubrics,
    validate_skills,
)
from verl.iteration.dataset import should_use_proposer_iteration_prompt
from verl.iteration.generation import write_generation_datasets
from verl.iteration.models import (
    AnalysisReport,
    DynamicStateUpdater,
    KeepoutEvaluator,
    ModelCallError,
    ProblemVerifier,
    RubricEvaluationResponse,
    RubricsResponse,
    SemanticJudgment,
    SkillsResponse,
    TrajectoryAnalyzer,
    UpdateDecision,
    VerifyDecision,
    evaluate_rubric_rewards,
    evaluate_rubrics,
    rank_and_verify_candidates,
    validate_evidence_references,
    validate_model_reference,
)
from verl.iteration.orchestrator import (
    STAGE_ORDER,
    IterationLock,
    IterationOrchestrator,
    _command_file_contracts,
    training_data_reference,
    validate_checkpoint_state_reference,
    validate_training_data_reference,
    write_checkpoint_state_reference,
    write_training_data_reference,
)
from verl.prompts import build_challenger_prompt
from verl.trainer.ppo.reward import compute_reward


def trajectory(query: str = "bridge entity") -> str:
    return (
        "<|im_start|>user\nsource document: Seed evidence<|im_end|>"
        "<|im_start|>assistant\n<think>search</think>"
        f'<tool_call>{{"name":"search","arguments":{{"query_list":["{query}"]}}}}</tool_call>'
        "<|im_end|><|im_start|>user\n"
        '<tool_response>{"result":"Bridge evidence supports Final"}</tool_response>'
        "<|im_end|><|im_start|>assistant\n"
        "<question>Which final entity?</question><answer>Final</answer><|im_end|>"
    )


def make_candidate(index: int, *, format_score: float = 1.0, doc_id: str | None = None) -> Candidate:
    normalized, evidence = extract_evidence_bundle("Seed evidence", trajectory(), hop_count=2)
    return Candidate(
        candidate_id=f"candidate-{index}",
        iteration=0,
        doc_id=doc_id or f"doc-{index}",
        hop_count=2,
        source_document="Seed evidence",
        proposer_trajectory=normalized,
        evidence_bundle=evidence,
        question=f"q-{index}",
        reference_answer="Final",
        format_score=format_score,
        generation_index=index,
    )


def test_stable_doc_id_and_prompt_injection_are_deterministic():
    assert stable_document_id(" A  document\n") == stable_document_id("A document")
    assert stable_document_id("anything", source_id="source-7") == "source-7"
    prompt = build_challenger_prompt(hops=2, document="seed", skills=initial_skills())
    positions = [prompt.index(skill.id) for skill in initial_skills()]
    assert positions == sorted(positions)
    assert "n = 2" in prompt


def test_proposer_prompt_dataset_is_selected_only_for_proposer_training():
    config = {
        "use_proposer_iteration_prompt": False,
        "proposer_iteration_state_path": "/tmp/state.json",
    }
    assert not should_use_proposer_iteration_prompt(config, is_train=True, phase="solver_train")
    assert should_use_proposer_iteration_prompt(config, is_train=True, phase="proposer_train")
    assert not should_use_proposer_iteration_prompt(config, is_train=False, phase="proposer_train")
    validate_model_reference("solver-0", "solver-0", role="verify solver")
    with pytest.raises(ValueError, match="does not match"):
        validate_model_reference("wrong", "solver-0", role="verify solver")


def test_evidence_extraction_is_complete_and_strict():
    normalized, evidence = extract_evidence_bundle("Seed evidence", trajectory(), hop_count=2)
    assert len(normalized) == 4
    assert [item.kind for item in evidence] == ["seed_document", "search_result"]
    assert evidence[1].query == "bridge entity"
    assert evidence[1].content == "Bridge evidence supports Final"

    broken = trajectory().replace("<tool_response>", "<broken>").replace("</tool_response>", "</broken>")
    with pytest.raises(ValueError, match="mismatch"):
        extract_evidence_bundle("Seed evidence", broken, hop_count=2)


def test_rubric_scoring_and_reward_components():
    evaluations = [
        {"rubric_id": "r1", "score": 1, "reason": "low"},
        {"rubric_id": "r2", "score": 5, "reason": "high"},
    ]
    parsed = [RubricEvaluation.model_validate(item) for item in evaluations]
    assert normalized_rubric_mean(parsed) == pytest.approx(0.5)
    assert candidate_rank_score(0.8, parsed) == pytest.approx(0.65)
    weighted = proposer_reward_components(0.8, 0.25, parsed)
    assert weighted["score"] == pytest.approx(0.9)
    legacy = proposer_reward_components(0.8, 0.25, [], rubric_weight=0)
    assert legacy["score"] == pytest.approx(0.65)


def test_failed_rubric_evaluations_use_the_valid_batch_mean():
    low = [RubricEvaluation(rubric_id="r1", score=1, reason="low")]
    high = [RubricEvaluation(rubric_id="r1", score=5, reason="high")]

    scores, failures = resolve_rubric_scores([low, None, high])

    assert scores == pytest.approx([0.0, 0.5, 1.0])
    assert failures == [False, True, False]
    failed_reward = proposer_reward_components(
        0.8,
        0.25,
        [],
        rubric_weight=0.5,
        rubric_score_override=scores[1],
    )
    assert failed_reward["score"] == pytest.approx(0.9)
    assert failed_reward["format_score"] == pytest.approx(0.8)
    assert failed_reward["difficulty_score"] == pytest.approx(0.25)


def test_all_failed_rubric_evaluations_use_a_neutral_score():
    scores, failures = resolve_rubric_scores([None, None])

    assert scores == pytest.approx([0.5, 0.5])
    assert failures == [True, True]


def test_compute_reward_does_not_repeat_a_failed_batch():
    calls = 0

    def failing_reward_fn(data, return_dict=False):
        nonlocal calls
        calls += 1
        raise RuntimeError("reward failed")

    with pytest.raises(RuntimeError, match="reward failed"):
        compute_reward(object(), failing_reward_fn)

    assert calls == 1


def test_compute_reward_accepts_a_tensor_result_without_repeating_the_batch():
    calls = 0
    reward_tensor = object()

    def tensor_reward_fn(data, return_dict=False):
        nonlocal calls
        del data, return_dict
        calls += 1
        return reward_tensor

    result, extra_info = compute_reward(object(), tensor_reward_fn)

    assert result is reward_tensor
    assert extra_info == {}
    assert calls == 1


def test_compute_reward_accepts_a_dict_result_with_extra_info():
    calls = 0
    reward_tensor = object()
    reward_extra_info = {"rubric_score": [0.75]}

    def dict_reward_fn(data, return_dict=False):
        nonlocal calls
        del data, return_dict
        calls += 1
        return {
            "reward_tensor": reward_tensor,
            "reward_extra_info": reward_extra_info,
        }

    result, extra_info = compute_reward(object(), dict_reward_fn)

    assert result is reward_tensor
    assert extra_info is reward_extra_info
    assert calls == 1


def test_rubric_rewards_only_evaluate_format_valid_candidates():
    rubrics = initial_rubrics()[:2]
    candidates = [
        make_candidate(0, format_score=0.0),
        make_candidate(1, format_score=0.5),
    ]

    class CountingClient:
        def __init__(self):
            self.prompts = []

        async def complete_structured(self, prompt, response_model):
            self.prompts.append(prompt)
            payload = {
                "evaluations": [
                    {"rubric_id": rubric.id, "score": 5, "reason": "supported"}
                    for rubric in rubrics
                ]
            }
            return response_model.model_validate(payload), json.dumps(payload), 0.0

    client = CountingClient()
    evaluations, scores, failures = asyncio.run(
        evaluate_rubric_rewards(candidates, rubrics, client)
    )

    assert len(client.prompts) == 1
    assert "Question: q-1" in client.prompts[0]
    assert evaluations[0] == []
    assert scores == pytest.approx([0.0, 1.0])
    assert failures == [False, False]
    invalid_reward = proposer_reward_components(
        candidates[0].format_score,
        0.0,
        evaluations[0],
        rubric_score_override=scores[0],
    )
    assert invalid_reward["rubric_score"] == 0.0
    assert invalid_reward["score"] == 0.0


def test_rubric_rewards_isolate_permanent_failures_and_report_them(caplog):
    rubrics = initial_rubrics()[:2]
    candidates = [make_candidate(index) for index in range(3)]

    class PartiallyFailingClient:
        async def complete_structured(self, prompt, response_model):
            score = 1 if "Question: q-0" in prompt else 5
            active_rubrics = rubrics[:1] if "Question: q-1" in prompt else rubrics
            payload = {
                "evaluations": [
                    {"rubric_id": rubric.id, "score": score, "reason": "judge result"}
                    for rubric in active_rubrics
                ]
            }
            return response_model.model_validate(payload), json.dumps(payload), 0.0

    with caplog.at_level("WARNING", logger="verl.iteration.models"):
        evaluations, scores, failures = asyncio.run(
            evaluate_rubric_rewards(candidates, rubrics, PartiallyFailingClient())
        )

    assert [len(items) for items in evaluations] == [2, 0, 2]
    assert scores == pytest.approx([0.0, 0.5, 1.0])
    assert failures == [False, True, False]
    assert "Rubric evaluation failed for candidate-1" in caplog.text
    assert "raw_output" in caplog.text


def test_rubric_evaluation_retries_incomplete_coverage():
    rubrics = initial_rubrics()[:2]

    class CoverageRetryClient:
        def __init__(self):
            self.calls = 0
            self.prompts = []

        async def complete_structured(self, prompt, response_model):
            self.calls += 1
            self.prompts.append(prompt)
            evaluations = [
                {"rubric_id": rubrics[0].id, "score": 3, "reason": "first"},
            ]
            if self.calls == 2:
                evaluations.append(
                    {"rubric_id": rubrics[1].id, "score": 4, "reason": "second"}
                )
            payload = {"evaluations": evaluations}
            return payload, json.dumps(payload), 0.0

    client = CoverageRetryClient()
    evaluations, _ = asyncio.run(
        evaluate_rubrics(make_candidate(0), rubrics, client, max_attempts=2)
    )

    assert client.calls == 2
    expected_ids = json.dumps(sorted(rubric.id for rubric in rubrics), ensure_ascii=False)
    assert f"rubric id in this list: {expected_ids}." in client.prompts[1]
    assert {item.rubric_id for item in evaluations} == {item.id for item in rubrics}


def test_rubric_evaluation_rejects_duplicate_ids():
    rubrics = initial_rubrics()[:2]

    class DuplicateRubricClient:
        async def complete_structured(self, prompt, response_model):
            del prompt
            payload = {
                "evaluations": [
                    {"rubric_id": rubrics[0].id, "score": 3, "reason": "first"},
                    {"rubric_id": rubrics[0].id, "score": 4, "reason": "duplicate"},
                ]
            }
            return response_model.model_validate(payload), json.dumps(payload), 0.0

    with pytest.raises(ModelCallError, match="failed after 1 attempts") as raised:
        asyncio.run(
            evaluate_rubrics(
                make_candidate(0),
                rubrics,
                DuplicateRubricClient(),
                max_attempts=1,
            )
        )

    assert raised.value.details["attempts"][0]["reason"] == (
        "rubric response must cover every active rubric exactly once"
    )


def test_rubric_evaluation_does_not_outer_retry_transport_failures():
    class TransportFailureClient:
        def __init__(self):
            self.calls = 0

        async def complete_structured(self, prompt, response_model):
            del prompt, response_model
            self.calls += 1
            raise ModelCallError("transport retries exhausted")

    client = TransportFailureClient()
    with pytest.raises(ModelCallError, match="transport retries exhausted"):
        asyncio.run(
            evaluate_rubrics(
                make_candidate(0),
                initial_rubrics()[:2],
                client,
                max_attempts=3,
            )
        )

    assert client.calls == 1


def test_dynamic_schema_limits_and_verify_contract():
    duplicate_skills = [
        Skill(id="same", instruction="one", evidence="e"),
        Skill(id="same", instruction="two", evidence="e"),
    ]
    with pytest.raises(ValueError, match="unique"):
        validate_skills(duplicate_skills)
    with pytest.raises(ValueError, match="maximum of 2"):
        validate_skills(initial_skills(), max_items=2)
    with pytest.raises(ValueError, match="between 1 and 12"):
        validate_skills(initial_skills(), max_items=13)
    with pytest.raises(ValueError, match="8000"):
        validate_skills(
            [
                Skill(
                    id=f"long-{index}",
                    instruction="i" * 1_000,
                    evidence="e" * 2_000,
                )
                for index in range(3)
            ]
        )
    with pytest.raises(ValueError, match="maximum"):
        validate_rubrics(
            [
                Rubric(
                    id=f"r-{index}",
                    name="r",
                    description="d",
                    score_1_anchor="one",
                    score_3_anchor="three",
                    score_5_anchor="five",
                )
                for index in range(13)
            ]
        )
    for evidence_support, question_is_determinate, equivalent in (
        (False, True, True),
        (True, False, True),
        (True, True, False),
    ):
        with pytest.raises(ValueError, match="three-condition"):
            VerifyResult(
                evidence_support=evidence_support,
                question_is_determinate=question_is_determinate,
                candidate_judgments=[
                    CandidateJudgment(
                        candidate_index=0,
                        semantically_equivalent=equivalent,
                        reason="same",
                    )
                ],
                passed=True,
                reason="invalid",
            )

    next_skills = initial_skills()
    next_skills[0].instruction = "updated"
    skill_diff = diff_by_id(initial_skills(), next_skills)
    assert [item["after"]["id"] for item in skill_diff["modified"]] == [next_skills[0].id]


def test_stable_group_split_never_leaks_doc_ids():
    candidates = [
        make_candidate(index, doc_id=f"doc-{index // 2}")
        for index in range(80)
    ]
    train, keepout, manifest = stable_group_split(candidates, train_ratio=0.9, seed=17)
    assert {item.doc_id for item in train}.isdisjoint({item.doc_id for item in keepout})
    _, _, repeated = stable_group_split(list(reversed(candidates)), train_ratio=0.9, seed=17)
    assert manifest["assignments"] == repeated["assignments"]


def test_generation_writes_mutually_exclusive_train_and_keepout_parquets(tmp_path):
    candidates = [make_candidate(index) for index in range(80)]
    source_rows = [
        pd.Series(
            {
                "prompt": [{"role": "user", "content": "proposer"}],
                "reward_model": {"ground_truth": {"target": None}},
                "metadata": {"doc_id": candidate.doc_id},
                "data_source": "search_zero_2",
            }
        )
        for candidate in candidates
    ]
    train_path = tmp_path / "train.parquet"
    keepout_path = tmp_path / "keepout.parquet"
    manifest = write_generation_datasets(
        source_rows,
        candidates,
        train_path=str(train_path),
        keepout_path=str(keepout_path),
        split_manifest_path=str(tmp_path / "split.json"),
        train_ratio=0.9,
        split_seed=17,
    )
    train = pd.read_parquet(train_path)
    keepout = pd.read_parquet(keepout_path)
    train_doc_ids = {item["doc_id"] for item in train.metadata}
    keepout_doc_ids = {item["doc_id"] for item in keepout.metadata}
    assert train_doc_ids.isdisjoint(keepout_doc_ids)
    assert len(train) + len(keepout) == len(candidates)
    assert manifest["train_doc_count"] == len(train_doc_ids)
    Candidate.model_validate_json(keepout.iloc[0].metadata["candidate_json"])


class FakeSolver:
    def __init__(self):
        self.calls = 0

    async def complete_text(self, messages, *, temperature=0, max_tokens=2048):
        del messages, temperature, max_tokens
        self.calls += 1
        return "<answer>Final</answer>", 0.01


class FakeJudge:
    def __init__(self, rubrics, pass_index=2):
        self.rubrics = rubrics
        self.pass_index = pass_index

    async def complete_structured(self, prompt, response_model):
        if response_model is RubricEvaluationResponse:
            payload = {
                "evaluations": [
                    {"rubric_id": rubric.id, "score": 5, "reason": "supported"}
                    for rubric in self.rubrics
                ]
            }
        elif response_model is VerifyDecision:
            passed = self.pass_index is not None and f"Question: q-{self.pass_index}" in prompt
            payload = {
                "evidence_support": True,
                "question_is_determinate": True,
                "candidate_judgments": [
                    {
                        "candidate_index": index,
                        "semantically_equivalent": passed and index == 0,
                        "reason": "match" if passed and index == 0 else "different",
                    }
                    for index in range(3)
                ],
                "passed": passed,
                "reason": "decision",
            }
        else:
            raise AssertionError(response_model)
        return response_model.model_validate(payload), json.dumps(payload), 0.01


def test_candidates_are_ranked_then_verified_until_first_pass():
    rubrics = initial_rubrics()
    candidates = [make_candidate(index, format_score=1 - index * 0.1) for index in range(5)]
    solver = FakeSolver()
    judge = FakeJudge(rubrics)
    with pytest.raises(ValueError, match="exactly 3"):
        ProblemVerifier(solver, judge, samples=2)
    verifier = ProblemVerifier(solver, judge, samples=3)
    winner, ordered, _ = asyncio.run(
        rank_and_verify_candidates(candidates, rubrics, judge, verifier)
    )
    assert winner.candidate_id == "candidate-2"
    assert [item.status for item in ordered] == [
        "verify_failed",
        "verify_failed",
        "verify_passed",
        "not_verified",
        "not_verified",
    ]
    assert solver.calls == 9


def test_all_semantic_verify_failures_return_no_selection_without_masking_errors():
    rubrics = initial_rubrics()
    candidates = [make_candidate(index, format_score=1 - index * 0.1) for index in range(5)]
    solver = FakeSolver()
    judge = FakeJudge(rubrics, pass_index=None)
    verifier = ProblemVerifier(solver, judge, samples=3)
    winner, ordered, _ = asyncio.run(
        rank_and_verify_candidates(candidates, rubrics, judge, verifier)
    )
    assert winner is None
    assert all(item.status == "verify_failed" for item in ordered)
    assert solver.calls == 15


class FakeUpdaterClient:
    def __init__(self):
        self.skills_seen_by_rubrics = False

    async def complete_structured(self, prompt, response_model):
        if response_model is SkillsResponse:
            payload = {
                "skills": [
                    {"id": "skill-next", "instruction": "new instruction", "evidence": "failure pattern"}
                ],
                "decisions": [
                    {
                        "id": skill.id,
                        "action": "removed",
                        "reason": "superseded",
                        "evidence_refs": ["/keepout_evidence/analysis/failure_patterns/0"],
                    }
                    for skill in initial_skills()
                ]
                + [
                    {
                        "id": "skill-next",
                        "action": "added",
                        "reason": "addresses failure",
                        "evidence_refs": ["/keepout_evidence/analysis/failure_patterns/0"],
                    }
                ],
            }
        elif response_model is RubricsResponse:
            self.skills_seen_by_rubrics = "new instruction" in prompt
            payload = {
                "rubrics": [
                    {
                        "id": "rubric-next",
                        "name": "next",
                        "description": "next rubric",
                        "score_1_anchor": "bad",
                        "score_3_anchor": "middle",
                        "score_5_anchor": "good",
                    }
                ],
                "decisions": [
                    {
                        "id": rubric.id,
                        "action": "removed",
                        "reason": "superseded",
                        "evidence_refs": ["/keepout_evidence/analysis/failure_patterns/0"],
                    }
                    for rubric in initial_rubrics()
                ]
                + [
                    {
                        "id": "rubric-next",
                        "action": "added",
                        "reason": "matches next skill",
                        "evidence_refs": ["/next_skills/0/id"],
                    }
                ],
            }
        else:
            raise AssertionError(response_model)
        return response_model.model_validate(payload), json.dumps(payload), 0.01


def test_skills_update_precedes_rubrics_and_new_skills_are_used():
    client = FakeUpdaterClient()
    result = asyncio.run(
        DynamicStateUpdater(client).update(
            skills=initial_skills(),
            rubrics=initial_rubrics(),
            rubric_evidence=[],
            verify_evidence=[],
            keepout_evidence={"analysis": {"failure_patterns": ["pattern"]}},
        )
    )
    assert result["skills"][0].id == "skill-next"
    assert result["rubrics"][0].id == "rubric-next"
    assert client.skills_seen_by_rubrics


def test_update_retries_post_validation_and_json_pointers_are_strict():
    class RetryClient(FakeUpdaterClient):
        def __init__(self):
            super().__init__()
            self.calls = 0

        async def complete_structured(self, prompt, response_model):
            model, raw, latency = await super().complete_structured(prompt, response_model)
            if response_model is SkillsResponse:
                self.calls += 1
                if self.calls == 1:
                    payload = model.model_dump(mode="json")
                    payload["decisions"][0]["evidence_refs"] = ["/keepout_evidence/~2invalid"]
                    model = response_model.model_validate(payload)
            return model, raw, latency

    client = RetryClient()
    skills, _, _ = asyncio.run(
        DynamicStateUpdater(client, max_retries=2).update_skills(
            skills=initial_skills(),
            rubrics=initial_rubrics(),
            rubric_evidence=[],
            verify_evidence=[],
            keepout_evidence={"analysis": {"failure_patterns": ["pattern"]}},
        )
    )
    assert skills[0].id == "skill-next"
    assert client.calls == 2

    decision = UpdateDecision(
        id="skill",
        action="retained",
        reason="evidence",
        evidence_refs=["/evidence/a~1b"],
    )
    validate_evidence_references([decision], {"evidence": {"a/b": "value"}})
    decision.evidence_refs = ["//evidence"]
    with pytest.raises(ModelCallError, match="unresolved"):
        validate_evidence_references([decision], {"evidence": "value"})
    decision.evidence_refs = ["/evidence/-1"]
    with pytest.raises(ModelCallError, match="unresolved"):
        validate_evidence_references([decision], {"evidence": ["value"]})


def test_verifier_failure_retains_completed_sample_outputs():
    class PartiallyFailingSolver(FakeSolver):
        async def complete_text(self, messages, *, temperature=0, max_tokens=2048):
            self.calls += 1
            if self.calls == 2:
                return "<tool_call>forbidden</tool_call>", 0.01
            return f"<answer>answer-{self.calls}</answer>", 0.01

    solver = PartiallyFailingSolver()
    verifier = ProblemVerifier(solver, FakeJudge(initial_rubrics()), samples=3)
    with pytest.raises(ModelCallError) as raised:
        asyncio.run(verifier.verify(make_candidate(0)))
    assert len(raised.value.details["verifier_samples"]) == 2
    assert raised.value.details["sample_failures"][0]["details"]["raw_output"] == (
        "<tool_call>forbidden</tool_call>"
    )


class FullRoundClient:
    async def complete_text(self, messages, *, temperature=0, max_tokens=2048):
        del messages, temperature, max_tokens
        return "<answer>Final</answer>", 0.01

    async def complete_structured(self, prompt, response_model):
        if response_model is RubricEvaluationResponse:
            payload = {
                "evaluations": [
                    {"rubric_id": rubric.id, "score": 5, "reason": "supported"}
                    for rubric in initial_rubrics()
                ]
            }
        elif response_model is VerifyDecision:
            payload = {
                "evidence_support": True,
                "question_is_determinate": True,
                "candidate_judgments": [
                    {
                        "candidate_index": index,
                        "semantically_equivalent": index == 0,
                        "reason": "match" if index == 0 else "alternative",
                    }
                    for index in range(3)
                ],
                "passed": True,
                "reason": "supported",
            }
        elif response_model is SemanticJudgment:
            payload = {"semantically_equivalent": True, "reason": "same answer"}
        elif response_model is TrajectorySummary:
            candidate_id = re.search(r'"candidate_id":"([^"]+)"', prompt).group(1)
            payload = {
                "candidate_id": candidate_id,
                "correct": True,
                "outcome_stage": "answered",
                "root_causes": ["effective retrieval"],
                "related_rubric_ids": [initial_rubrics()[0].id],
                "evidence_quotes": ["Final"],
                "actionable_improvements": ["retain the successful pattern"],
            }
        elif response_model is AnalysisReport:
            payload = {
                "problem_frequencies": {"effective retrieval": 1},
                "success_patterns": ["effective retrieval"],
                "failure_patterns": [],
                "related_rubric_ids": [initial_rubrics()[0].id],
                "representative_cases": ["candidate"],
                "actionable_improvements": ["retain the successful pattern"],
            }
        elif response_model is SkillsResponse:
            payload = {
                "skills": [item.model_dump(mode="json") for item in initial_skills()],
                "decisions": [
                    {
                        "id": item.id,
                        "action": "retained",
                        "reason": "supported by keepout success",
                        "evidence_refs": ["/keepout_evidence/accuracy"],
                    }
                    for item in initial_skills()
                ],
            }
        elif response_model is RubricsResponse:
            payload = {
                "rubrics": [item.model_dump(mode="json") for item in initial_rubrics()],
                "decisions": [
                    {
                        "id": item.id,
                        "action": "retained",
                        "reason": "still discriminative",
                        "evidence_refs": ["/keepout_evidence/accuracy"],
                    }
                    for item in initial_rubrics()
                ],
            }
        else:
            raise AssertionError(response_model)
        return response_model.model_validate(payload), json.dumps(payload), 0.01


class FakeSearchRollout:
    async def run(self, question):
        del question
        return [{"role": "assistant", "content": "<answer>Final</answer>"}], "Final"


def test_mock_full_round_flows_through_verify_keepout_analysis_and_updates():
    async def run_round():
        rubrics = initial_rubrics()
        client = FullRoundClient()
        verifier = ProblemVerifier(client, client, samples=3)
        selected = []
        for doc_index in range(30):
            group = [
                make_candidate(
                    doc_index * 5 + generation_index,
                    format_score=1 - generation_index * 0.1,
                    doc_id=f"doc-{doc_index}",
                )
                for generation_index in range(5)
            ]
            winner, _, _ = await rank_and_verify_candidates(group, rubrics, client, verifier)
            selected.append(winner)
        _, keepout, _ = stable_group_split(selected, train_ratio=0.9, seed=17)
        results, accuracy = await KeepoutEvaluator(FakeSearchRollout(), client).evaluate_all(keepout)
        analysis = await TrajectoryAnalyzer(client, chunk_size=2).analyze_all(results, rubrics)
        updates = await DynamicStateUpdater(client).update(
            skills=initial_skills(),
            rubrics=rubrics,
            rubric_evidence=[],
            verify_evidence={"accepted": len(selected)},
            keepout_evidence={"accuracy": accuracy, "analysis": analysis},
        )
        return selected, keepout, results, accuracy, analysis, updates

    selected, keepout, results, accuracy, analysis, updates = asyncio.run(run_round())
    assert len(selected) == 30
    assert len(results) == len(keepout)
    assert accuracy == 1
    assert len(analysis["items"]) == len(keepout)
    assert updates["skills"][0].id == initial_skills()[0].id
    assert updates["rubrics"][0].id == initial_rubrics()[0].id


def test_state_store_and_checkpoint_reference_are_atomic_and_validated(tmp_path):
    with pytest.raises(ValueError, match="plaintext secret"):
        StateStore(tmp_path / "unsafe.json").initialize(
            iteration=0,
            proposer="p",
            solver_before="s",
            config_snapshot={"meta_model": {"api_key": "do-not-store"}},
        )
    state_path = tmp_path / "iterations" / "iter_0" / "state.json"
    store = StateStore(state_path)
    state = store.initialize(
        iteration=0,
        proposer="proposer-0",
        solver_before="solver-0",
        config_snapshot={"seed": 42},
    )
    checkpoint = tmp_path / "checkpoint"
    write_checkpoint_state_reference(checkpoint, state_path)
    validate_checkpoint_state_reference(checkpoint, state_path)
    state.models.solver_after = "solver-1"
    store.save(state)
    validate_checkpoint_state_reference(checkpoint, state_path)
    state.skills[0].instruction = "changed"
    store.save(state)
    with pytest.raises(ValueError, match="mismatch"):
        validate_checkpoint_state_reference(checkpoint, state_path)

    train_path = tmp_path / "train.parquet"
    train_path.write_bytes(b"training-v1")
    write_training_data_reference(checkpoint, str(train_path))
    validate_training_data_reference(checkpoint, str(train_path))
    train_path.write_bytes(b"training-v2")
    with pytest.raises(ValueError, match="receipt"):
        validate_training_data_reference(checkpoint, str(train_path))


def test_iteration_lock_recovers_stale_owner_but_rejects_active_owner(tmp_path):
    state_path = tmp_path / "state.json"
    lock_path = state_path.with_suffix(".lock")
    lock_path.write_text(
        json.dumps({"pid": os.getpid(), "process_start_ticks": "stale"}),
        encoding="utf-8",
    )
    with IterationLock(state_path):
        assert lock_path.exists()
        with pytest.raises(RuntimeError, match="already locked"):
            with IterationLock(state_path):
                pass
    assert not lock_path.exists()


def test_shell_command_file_contracts_include_nested_scripts_and_configs(tmp_path):
    script = tmp_path / "stage.py"
    config = tmp_path / "stage.yaml"
    script.write_text("print('first')", encoding="utf-8")
    config.write_text("seed: 1", encoding="utf-8")
    command = ["bash", "-lc", f"python {script} --config={config}"]
    before = _command_file_contracts(command, tmp_path)
    contracted_paths = {item["path"] for item in before}
    assert str(script.resolve()) in contracted_paths
    assert str(config.resolve()) in contracted_paths
    script.write_text("print('other')", encoding="utf-8")
    after = _command_file_contracts(command, tmp_path)
    assert before != after


def test_mock_iteration_orchestrator_runs_once_and_resumes(tmp_path):
    state_path = tmp_path / "iter_0" / "state.json"
    StateStore(state_path).initialize(
        iteration=0,
        proposer="proposer-0",
        solver_before="solver-0",
        config_snapshot={"seed": 42},
    )
    skills_path = tmp_path / "next_skills.json"
    rubrics_path = tmp_path / "next_rubrics.json"
    skills_path.write_text(
        json.dumps({"skills": [item.model_dump(mode="json") for item in initial_skills()]}),
        encoding="utf-8",
    )
    rubrics_path.write_text(
        json.dumps({"rubrics": [item.model_dump(mode="json") for item in initial_rubrics()]}),
        encoding="utf-8",
    )
    train_path = tmp_path / "solver_train.parquet"
    keepout_path = tmp_path / "keepout.parquet"
    split_path = tmp_path / "split.json"
    pd.DataFrame([{"metadata": {"candidate_id": "train-1"}}]).to_parquet(train_path, index=False)
    pd.DataFrame([{"metadata": {"candidate_id": "keepout-1"}}]).to_parquet(keepout_path, index=False)
    split_path.write_text(
        json.dumps(
            {
                "train_candidate_ids": ["train-1"],
                "keepout_candidate_ids": ["keepout-1"],
            }
        ),
        encoding="utf-8",
    )

    stages = {}
    for stage in STAGE_ORDER[:-1]:
        artifact = tmp_path / f"{stage}.done"
        command = [
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(artifact)!r}).write_text('ok')",
        ]
        stages[stage] = {"command": command, "artifacts": [str(artifact)]}
        if stage == "solver_train":
            solver_checkpoint = tmp_path / "solver_checkpoint"
            receipt = json.dumps(training_data_reference(str(train_path)))
            solver_command = [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    f"d=Path({str(solver_checkpoint)!r}); d.mkdir(parents=True, exist_ok=True); "
                    f"(d/'training_data_ref.json').write_text({receipt!r})"
                ),
                str(train_path),
            ]
            stages[stage].update(
                {
                    "command": solver_command,
                    "artifacts": [str(solver_checkpoint)],
                    "checkpoint_dir": str(solver_checkpoint),
                    "train_data_path": str(train_path),
                    "keepout_path": str(keepout_path),
                    "split_manifest_path": str(split_path),
                }
            )
        if stage == "convert_solver":
            stages[stage]["solver_after"] = "solver-1"
    next_state_path = tmp_path / "iter_1" / "state.json"
    stages["finalize"] = {
        "command": None,
        "artifacts": [],
        "next_state_path": str(next_state_path),
        "skills_path": str(skills_path),
        "rubrics_path": str(rubrics_path),
        "proposer_after": "proposer-1",
        "solver_after": "solver-1",
    }
    config_path = tmp_path / "orchestrator.json"
    config_path.write_text(
        json.dumps(
            {
                "state_path": str(state_path),
                "run_dir": str(tmp_path / "run"),
                "working_directory": str(tmp_path),
                "stages": stages,
            }
        ),
        encoding="utf-8",
    )

    successful_proposer_command = stages["proposer_train"]["command"]
    stages["proposer_train"]["command"] = [sys.executable, "-c", "raise SystemExit(7)"]
    config_path.write_text(
        json.dumps(
            {
                "state_path": str(state_path),
                "run_dir": str(tmp_path / "run"),
                "working_directory": str(tmp_path),
                "stages": stages,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="proposer_train failed"):
        IterationOrchestrator(config_path).run()
    stage_dir = tmp_path / "run" / "stages" / "proposer_train"
    assert (stage_dir / "attempt-1.log").exists()
    assert (stage_dir / "manifest-attempt-1.json").exists()

    stages["proposer_train"]["command"] = successful_proposer_command
    config_path.write_text(
        json.dumps(
            {
                "state_path": str(state_path),
                "run_dir": str(tmp_path / "run"),
                "working_directory": str(tmp_path),
                "stages": stages,
            }
        ),
        encoding="utf-8",
    )
    completed = IterationOrchestrator(config_path).run()
    assert completed.status == "completed"
    assert (stage_dir / "attempt-2.log").exists()
    assert (stage_dir / "manifest-attempt-2.json").exists()
    next_state = StateStore(next_state_path).load()
    assert next_state.iteration == 1
    assert next_state.models.solver_before == "solver-1"
    next_state.status = "completed"
    StateStore(next_state_path).save(next_state)
    resumed = IterationOrchestrator(config_path).run()
    assert resumed.status == "completed"
    proposer_artifact = tmp_path / "proposer_train.done"
    artifact_stat = proposer_artifact.stat()
    proposer_artifact.write_text("no", encoding="utf-8")
    os.utime(proposer_artifact, ns=(artifact_stat.st_atime_ns, artifact_stat.st_mtime_ns))
    with pytest.raises(RuntimeError, match="invalid stage manifest"):
        IterationOrchestrator(config_path).run()
