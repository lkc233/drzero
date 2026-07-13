from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

from verl.iteration.core import (
    Candidate,
    KeepoutResult,
    Skill,
    StateStore,
    atomic_write_json,
    diff_by_id,
)
from verl.iteration.generation import atomic_write_jsonl
from verl.iteration.models import (
    DynamicStateUpdater,
    EndpointConfig,
    KeepoutEvaluator,
    ModelCallError,
    OpenAICompatibleClient,
    SearchRolloutClient,
    TrajectoryAnalyzer,
    validate_model_reference,
)
from verl.iteration.orchestrator import IterationOrchestrator


def _config(path: str) -> dict[str, Any]:
    result = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(result, dict):
        raise ValueError("configuration must be a mapping")
    return result


def _endpoint(config: dict[str, Any], key: str) -> EndpointConfig:
    value: Any = config
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"configuration is missing {key}")
        value = value[part]
    return EndpointConfig.model_validate(value)


def _read_json_or_jsonl(path: str) -> Any:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        if source.suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        return json.load(handle)


def _candidate_evidence(path: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = [Candidate.model_validate(item) for item in _read_json_or_jsonl(path)]
    rubric_evidence = [
        {
            "candidate_id": candidate.candidate_id,
            "doc_id": candidate.doc_id,
            "evaluations": [item.model_dump(mode="json") for item in candidate.rubric_evaluation],
        }
        for candidate in candidates
    ]
    verify_evidence = [
        {
            "candidate_id": candidate.candidate_id,
            "doc_id": candidate.doc_id,
            "status": candidate.status,
            "verify_result": candidate.verify_result.model_dump(mode="json") if candidate.verify_result else None,
        }
        for candidate in candidates
    ]
    return rubric_evidence, verify_evidence


def init_state(args: argparse.Namespace) -> None:
    config_snapshot = _config(args.config_snapshot) if args.config_snapshot else {}
    state = StateStore(args.state).initialize(
        iteration=args.iteration,
        proposer=args.proposer,
        solver_before=args.solver,
        config_snapshot=config_snapshot,
    )
    print(json.dumps({"state": str(Path(args.state).resolve()), "iteration": state.iteration}))


def run_iteration(args: argparse.Namespace) -> None:
    state = IterationOrchestrator(args.config).run()
    print(json.dumps({"iteration": state.iteration, "status": state.status}))


def _metadata_candidate(row: pd.Series) -> Candidate:
    metadata = row.get("metadata")
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if not isinstance(metadata, dict) or "candidate_json" not in metadata:
        raise ValueError(
            "keepout parquet is missing metadata.candidate_json; regenerate it with the iteration pipeline"
        )
    return Candidate.model_validate_json(metadata["candidate_json"])


async def _keepout_eval(args: argparse.Namespace) -> None:
    state = StateStore(args.state).load()
    if not state.models.solver_after:
        raise ValueError("state.models.solver_after must be recorded before keepout evaluation")
    config = _config(args.config)
    frame = pd.read_parquet(args.keepout)
    candidates = [_metadata_candidate(row) for _, row in frame.iterrows()]
    solver_config = _endpoint(config, "keepout_eval.solver_model")
    validate_model_reference(
        solver_config.model_name,
        state.models.solver_after,
        role="keepout solver",
    )
    meta_config = _endpoint(config, "meta_model")
    retrieval_url = config.get("keepout_eval", {}).get("retrieval_service_url")
    if not retrieval_url:
        raise ValueError("configuration requires keepout_eval.retrieval_service_url")
    async with OpenAICompatibleClient(solver_config) as solver_client:
        async with OpenAICompatibleClient(meta_config) as meta_client:
            rollout = SearchRolloutClient(
                solver_client,
                retrieval_service_url=retrieval_url,
                topk=int(config.get("keepout_eval", {}).get("topk", 3)),
                max_turns=int(config.get("keepout_eval", {}).get("max_turns", 5)),
            )
            evaluator = KeepoutEvaluator(rollout, meta_client)
            results_by_index: dict[int, KeepoutResult] = {}
            failures_by_index: dict[int, dict[str, Any]] = {}
            failure_path = f"{args.output}.failures.json"

            async def evaluate_indexed(index: int, candidate: Candidate):
                try:
                    return index, await evaluator.evaluate_one(candidate), None
                except Exception as error:
                    return (
                        index,
                        None,
                        {
                            "candidate_id": candidate.candidate_id,
                            "error_type": type(error).__name__,
                            "reason": str(error),
                            "details": getattr(error, "details", {}),
                        },
                    )

            tasks = [
                asyncio.create_task(evaluate_indexed(index, candidate))
                for index, candidate in enumerate(candidates)
            ]
            for completed in asyncio.as_completed(tasks):
                index, result, failure = await completed
                if result is not None:
                    results_by_index[index] = result
                    partial = [results_by_index[key] for key in sorted(results_by_index)]
                    atomic_write_jsonl(args.output, [item.model_dump(mode="json") for item in partial])
                else:
                    failures_by_index[index] = failure
                atomic_write_json(
                    failure_path,
                    {
                        "status": "running",
                        "completed_candidate_ids": [
                            results_by_index[key].candidate_id for key in sorted(results_by_index)
                        ],
                        "failures": [failures_by_index[key] for key in sorted(failures_by_index)],
                    },
                )
            if failures_by_index:
                atomic_write_json(
                    failure_path,
                    {
                        "status": "failed",
                        "completed_candidate_ids": [
                            results_by_index[key].candidate_id for key in sorted(results_by_index)
                        ],
                        "failures": [failures_by_index[key] for key in sorted(failures_by_index)],
                        "solver_model_call_metrics": solver_client.metrics.model_dump(mode="json"),
                        "meta_model_call_metrics": meta_client.metrics.model_dump(mode="json"),
                    },
                )
                raise ModelCallError(
                    "one or more keepout evaluations failed",
                    details={
                        "failure_artifact": failure_path,
                        "failures": [failures_by_index[key] for key in sorted(failures_by_index)],
                    },
                )
            results = [results_by_index[index] for index in range(len(candidates))]
            if not results:
                raise ValueError("keepout set must not be empty")
            accuracy = sum(result.correct for result in results) / len(results)
            metrics = {
                "solver": solver_client.metrics.model_dump(mode="json"),
                "meta": meta_client.metrics.model_dump(mode="json"),
            }
    atomic_write_jsonl(args.output, [result.model_dump(mode="json") for result in results])
    summary_path = args.summary or str(Path(args.output).with_suffix(".summary.json"))
    atomic_write_json(
        summary_path,
        {
            "count": len(results),
            "correct": sum(result.correct for result in results),
            "accuracy": accuracy,
            "model_call_metrics": metrics,
        },
    )


def keepout_eval(args: argparse.Namespace) -> None:
    asyncio.run(_keepout_eval(args))


async def _analyze(args: argparse.Namespace) -> None:
    state = StateStore(args.state).load()
    config = _config(args.config)
    results = [KeepoutResult.model_validate(item) for item in _read_json_or_jsonl(args.results)]
    partial_analysis: dict[str, Any] = {"status": "running", "items": [], "chunks": []}

    def persist_progress(value: dict[str, Any]) -> None:
        partial_analysis.clear()
        partial_analysis.update(value)
        atomic_write_json(args.output, partial_analysis)

    async with OpenAICompatibleClient(_endpoint(config, "meta_model")) as client:
        analyzer = TrajectoryAnalyzer(
            client,
            chunk_size=int(config.get("trajectory_analysis", {}).get("chunk_size", 50)),
        )
        try:
            analysis = await analyzer.analyze_all(
                results,
                state.rubrics,
                progress_callback=persist_progress,
            )
        except Exception as error:
            partial_analysis.update(
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "error_details": getattr(error, "details", {}),
                    "model_call_metrics": client.metrics.model_dump(mode="json"),
                }
            )
            atomic_write_json(args.output, partial_analysis)
            raise
        analysis["model_call_metrics"] = client.metrics.model_dump(mode="json")
    atomic_write_json(args.output, analysis)


def analyze(args: argparse.Namespace) -> None:
    asyncio.run(_analyze(args))


async def _update_skills(args: argparse.Namespace) -> None:
    state = StateStore(args.state).load()
    config = _config(args.config)
    rubric_evidence, verify_evidence = _candidate_evidence(args.candidates)
    verify_evidence = {
        "candidates": verify_evidence,
        "generation_summary": _read_json_or_jsonl(args.generation_summary),
    }
    keepout_evidence = {
        "analysis": _read_json_or_jsonl(args.analysis),
        "keepout_summary": _read_json_or_jsonl(args.keepout_summary),
    }
    async with OpenAICompatibleClient(_endpoint(config, "meta_model")) as client:
        dynamic_config = config.get("dynamic_state", {})
        updater = DynamicStateUpdater(
            client,
            max_skills=int(dynamic_config.get("max_skills", 12)),
            max_rubrics=int(dynamic_config.get("max_rubrics", 12)),
            max_retries=int(dynamic_config.get("max_retries", 3)),
        )
        try:
            skills, decisions, raw = await updater.update_skills(
                skills=state.skills,
                rubrics=state.rubrics,
                rubric_evidence=rubric_evidence,
                verify_evidence=verify_evidence,
                keepout_evidence=keepout_evidence,
            )
        except Exception as error:
            atomic_write_json(
                args.output,
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "error_details": getattr(error, "details", {}),
                    "model_call_metrics": client.metrics.model_dump(mode="json"),
                },
            )
            raise
        metrics = client.metrics.model_dump(mode="json")
    atomic_write_json(
        args.output,
        {
            "status": "completed",
            "skills": [item.model_dump(mode="json") for item in skills],
            "diff": diff_by_id(state.skills, skills),
            "decisions": [item.model_dump(mode="json") for item in decisions],
            "raw_output": raw,
            "model_call_metrics": metrics,
        },
    )


def update_skills(args: argparse.Namespace) -> None:
    asyncio.run(_update_skills(args))


async def _update_rubrics(args: argparse.Namespace) -> None:
    state = StateStore(args.state).load()
    config = _config(args.config)
    skills_payload = _read_json_or_jsonl(args.skills)
    skill_items = skills_payload.get("skills", skills_payload)
    next_skills = [Skill.model_validate(item) for item in skill_items]
    rubric_evidence, verify_evidence = _candidate_evidence(args.candidates)
    verify_evidence = {
        "candidates": verify_evidence,
        "generation_summary": _read_json_or_jsonl(args.generation_summary),
    }
    keepout_evidence = {
        "analysis": _read_json_or_jsonl(args.analysis),
        "keepout_summary": _read_json_or_jsonl(args.keepout_summary),
    }
    async with OpenAICompatibleClient(_endpoint(config, "meta_model")) as client:
        dynamic_config = config.get("dynamic_state", {})
        updater = DynamicStateUpdater(
            client,
            max_skills=int(dynamic_config.get("max_skills", 12)),
            max_rubrics=int(dynamic_config.get("max_rubrics", 12)),
            max_retries=int(dynamic_config.get("max_retries", 3)),
        )
        try:
            rubrics, decisions, raw = await updater.update_rubrics(
                rubrics=state.rubrics,
                next_skills=next_skills,
                rubric_evidence=rubric_evidence,
                verify_evidence=verify_evidence,
                keepout_evidence=keepout_evidence,
            )
        except Exception as error:
            atomic_write_json(
                args.output,
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "error_details": getattr(error, "details", {}),
                    "model_call_metrics": client.metrics.model_dump(mode="json"),
                },
            )
            raise
        metrics = client.metrics.model_dump(mode="json")
    atomic_write_json(
        args.output,
        {
            "status": "completed",
            "rubrics": [item.model_dump(mode="json") for item in rubrics],
            "diff": diff_by_id(state.rubrics, rubrics),
            "decisions": [item.model_dump(mode="json") for item in decisions],
            "raw_output": raw,
            "model_call_metrics": metrics,
        },
    )


def update_rubrics(args: argparse.Namespace) -> None:
    asyncio.run(_update_rubrics(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dr.Zero iteration orchestration utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init-state")
    init.add_argument("--state", required=True)
    init.add_argument("--iteration", type=int, required=True)
    init.add_argument("--proposer", required=True)
    init.add_argument("--solver", required=True)
    init.add_argument("--config-snapshot")
    init.set_defaults(func=init_state)

    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True)
    run.set_defaults(func=run_iteration)

    evaluate = subparsers.add_parser("keepout-eval")
    evaluate.add_argument("--state", required=True)
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--keepout", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--summary")
    evaluate.set_defaults(func=keepout_eval)

    analysis = subparsers.add_parser("analyze")
    analysis.add_argument("--state", required=True)
    analysis.add_argument("--config", required=True)
    analysis.add_argument("--results", required=True)
    analysis.add_argument("--output", required=True)
    analysis.set_defaults(func=analyze)

    skills = subparsers.add_parser("update-skills")
    skills.add_argument("--state", required=True)
    skills.add_argument("--config", required=True)
    skills.add_argument("--candidates", required=True)
    skills.add_argument("--analysis", required=True)
    skills.add_argument("--generation-summary", required=True)
    skills.add_argument("--keepout-summary", required=True)
    skills.add_argument("--output", required=True)
    skills.set_defaults(func=update_skills)

    rubrics = subparsers.add_parser("update-rubrics")
    rubrics.add_argument("--state", required=True)
    rubrics.add_argument("--config", required=True)
    rubrics.add_argument("--candidates", required=True)
    rubrics.add_argument("--analysis", required=True)
    rubrics.add_argument("--generation-summary", required=True)
    rubrics.add_argument("--keepout-summary", required=True)
    rubrics.add_argument("--skills", required=True)
    rubrics.add_argument("--output", required=True)
    rubrics.set_defaults(func=update_rubrics)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
