# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""

import asyncio
import os
import socket
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from pprint import pprint

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import hydra
import pandas as pd
import ray
from omegaconf import OmegaConf
from tqdm import trange

from verl.iteration.core import StateStore, atomic_write_json, candidate_rank_score
from verl.iteration.generation import (
    append_candidate_progress,
    build_candidate,
    build_candidate_snapshot_contract,
    build_generation_summary,
    candidate_group_is_complete,
    compact_candidate_progress,
    load_candidates_with_progress,
    persist_candidates,
    reset_candidate_group,
    reset_candidate_progress,
    resolve_generation_phase,
    validate_candidate_snapshot_manifest,
    write_candidate_snapshot_manifest,
    write_generation_datasets,
)
from verl.iteration.models import (
    EndpointConfig,
    OpenAICompatibleClient,
    ProblemVerifier,
    evaluate_rubrics,
    rank_and_verify_candidate_groups,
    validate_model_reference,
)
from verl.prompts import DEFAULT_SOLVER_PREFIX, build_challenger_prompt
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import wrap_worker_cls_with_env
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker


def _release_generation_workers(worker_group: RayWorkerGroup, resource_pool: RayResourcePool) -> None:
    """Release rollout actors and their placement groups before verification."""
    for worker in worker_group._workers:
        ray.kill(worker, no_restart=True)
    for placement_group in resource_pool.pgs or []:
        ray.util.remove_placement_group(placement_group)
    # Actor teardown is asynchronous; give CUDA contexts time to disappear before
    # the serial verify server claims one of the same physical GPUs.
    time.sleep(5)


def _start_local_verify_server(config) -> tuple[subprocess.Popen | None, object | None]:
    server_config = config.verify.get("local_server", {})
    if not bool(server_config.get("enabled", False)):
        return None, None

    endpoint = EndpointConfig.model_validate(OmegaConf.to_container(config.verify.solver_model, resolve=True))
    model_path = str(server_config.get("model_path") or endpoint.model_name)
    port = int(server_config.get("port", 8001))
    log_path = Path(str(server_config.get("log_path", "logs/verify_solver.log")))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(server_config.get("gpu_devices", "2"))
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model",
        model_path,
        "--served-model-name",
        endpoint.model_name,
        "--port",
        str(port),
        "--tp-size",
        str(int(server_config.get("tp_size", 1))),
        "--mem-fraction-static",
        str(float(server_config.get("mem_fraction_static", 0.8))),
        "--tool-call-parser",
        "qwen25",
        "--log-level",
        "error",
    ]
    process = subprocess.Popen(command, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
    models_url = endpoint.base_url.rstrip("/") + "/v1/models"
    deadline = time.monotonic() + float(server_config.get("startup_timeout_seconds", 300))
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_handle.close()
            raise RuntimeError(f"local verify server exited with code {process.returncode}; see {log_path}")
        try:
            with urllib.request.urlopen(models_url, timeout=5) as response:
                if response.status == 200:
                    return process, log_handle
        except Exception:
            time.sleep(2)
    process.terminate()
    process.wait(timeout=30)
    log_handle.close()
    raise TimeoutError(f"local verify server did not become ready at {models_url}; see {log_path}")


def _stop_local_verify_server(process: subprocess.Popen | None, log_handle: object | None) -> None:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    if log_handle is not None:
        log_handle.close()


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    phase = resolve_generation_phase(config.data.get("phase", "all"))
    if phase == "verify":
        _main_task(config)
        return

    owns_ray = not ray.is_initialized()
    if owns_ray:
        # this is for local ray cluster
        temp_dir = os.environ.get("DRZERO_RAY_TMPDIR", "/tmp/drzero-ray")
        ray.init(
            num_cpus=config.ray_init.num_cpus,
            _temp_dir=temp_dir,
            _node_ip_address=os.environ.get("DRZERO_RAY_NODE_IP", "127.0.0.2"),
        )

    try:
        ray.get(main_task.remote(config))
    finally:
        if owns_ray:
            ray.shutdown()


@ray.remote(num_cpus=1)
def main_task(config):
    return _main_task(config)


def _main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    phase = resolve_generation_phase(config.data.get("phase", "all"))

    dataset = pd.read_parquet(config.data.path)
    if config.data.partition is not None:
        partition_length = len(dataset) // 5
        partition = int(config.data.partition)
        assert 0 < partition <= 5
        start = partition_length * (partition - 1)
        end = len(dataset) if partition == 5 else partition_length * partition
        dataset = dataset.iloc[start:end]
        print(f"Using partition {partition}/5, from {start} to {end}")

    iteration_config = config.get("iteration", {})
    state_path = iteration_config.get("state_path")
    verify_config = config.get("verify", {})
    verify_enabled = bool(verify_config.get("enabled", True))
    if verify_enabled and not state_path:
        raise ValueError("verify is enabled but iteration.state_path is not configured")
    state = StateStore(state_path).load() if state_path else None
    if phase != "all" and state is None:
        raise ValueError("split generate/verify phases require iteration.state_path")
    if state and verify_enabled:
        configured_solver = str(verify_config.get("solver_model", {}).get("model_name"))
        validate_model_reference(
            configured_solver,
            state.models.solver_before,
            role="verify solver",
        )

    metadata_rows = [dict(item or {}) for item in dataset.metadata.tolist()]
    if state:
        chat_lst = []
        all_hops = []
        for metadata in metadata_rows:
            required = {"doc_id", "source_document", "hop_count"}
            missing = sorted(required - metadata.keys())
            if missing:
                raise ValueError(
                    "legacy proposer data is missing structured metadata "
                    f"{missing}; regenerate it with process_train.py"
                )
            chat_lst.append(
                [
                    {
                        "role": "user",
                        "content": build_challenger_prompt(
                            hops=int(metadata["hop_count"]),
                            document=metadata["source_document"],
                            skills=state.skills,
                        ),
                    }
                ]
            )
            all_hops.append(int(metadata["hop_count"]))
    else:
        chat_lst = [
            chat.tolist() if hasattr(chat, "tolist") else list(chat)
            for chat in dataset[config.data.prompt_key].tolist()
        ]
        all_hops = [int(x.split("_")[-1]) for x in dataset.data_source]

    total_samples = len(dataset)
    config_batch_size = config.data.batch_size

    candidate_count = int(iteration_config.get("candidate_count_per_document", 5))
    rollout_count = int(config.actor_rollout_ref.rollout.n)
    if state and rollout_count != candidate_count:
        raise ValueError(
            f"generation must produce exactly {candidate_count} candidates per document, got rollout.n={rollout_count}"
        )

    output_path = Path(config.data.output_path)
    candidates_path = config.data.get("candidates_path") or str(output_path.with_suffix(".candidates.jsonl"))
    candidate_progress_path = config.data.get("candidate_progress_path") or str(
        Path(candidates_path).with_name(Path(candidates_path).stem + "_progress.jsonl")
    )
    candidate_manifest_path = config.data.get("candidate_manifest_path") or str(
        Path(candidates_path).with_name(Path(candidates_path).stem + "_manifest.json")
    )
    snapshot_contract = build_candidate_snapshot_contract(
        state=state,
        metadata_rows=metadata_rows,
        candidate_count=candidate_count,
        model_path=config.actor_rollout_ref.model.path,
        rollout_config=OmegaConf.to_container(config.actor_rollout_ref.rollout, resolve=True),
        verification_config={
            "enabled": verify_enabled,
            "solver_samples": int(verify_config.get("solver_samples", 3)),
            "solver_model": OmegaConf.to_container(verify_config.get("solver_model"), resolve=True),
            "judge_model": OmegaConf.to_container(
                verify_config.get("judge_model") or config.meta_model,
                resolve=True,
            ),
        },
        tool_config_path=config.actor_rollout_ref.rollout.multi_turn.tool_config_path,
    ) if state else None
    candidate_snapshot_exists = Path(candidates_path).exists()
    resume_requested = bool(config.data.get("resume_candidates", True))
    if phase == "generate" and resume_requested and candidate_snapshot_exists:
        validate_candidate_snapshot_manifest(candidate_manifest_path, snapshot_contract)
        print(f"Generation snapshot already exists at {candidates_path}; skipping generation")
        return
    if phase == "verify" and not candidate_snapshot_exists:
        raise ValueError(f"verify phase requires an existing candidate snapshot: {candidates_path}")
    resume_candidates = bool(
        state
        and candidate_snapshot_exists
        and (resume_requested or phase == "verify")
    )

    candidate_groups = []
    source_rows = []
    if resume_candidates:
        validate_candidate_snapshot_manifest(candidate_manifest_path, snapshot_contract)
        restored_candidates = load_candidates_with_progress(candidates_path, candidate_progress_path)
        expected_candidate_count = total_samples * candidate_count
        if len(restored_candidates) != expected_candidate_count:
            raise ValueError(
                "candidate snapshot does not match the selected dataset partition: "
                f"expected {expected_candidate_count}, got {len(restored_candidates)}"
            )
        for group_index in range(total_samples):
            start = group_index * candidate_count
            group = restored_candidates[start : start + candidate_count]
            expected_doc_id = str(metadata_rows[group_index]["doc_id"])
            if {candidate.doc_id for candidate in group} != {expected_doc_id}:
                raise ValueError(
                    f"candidate snapshot document mismatch at group {group_index}: expected {expected_doc_id}"
                )
            if sorted(candidate.generation_index for candidate in group) != list(range(candidate_count)):
                raise ValueError(f"candidate snapshot has invalid generation indices at group {group_index}")
            candidate_groups.append(group)
            source_rows.append(dataset.iloc[group_index])
        num_batch = 0
        print(f"Resuming verify from {candidates_path} with {len(restored_candidates)} candidates")
    else:
        import numpy as np
        import torch

        import verl.utils.torch_functional as verl_F
        from verl import DataProto
        from verl.custom_reward.reward_function import compute_challenger_format_scores
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
        from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
        from verl.utils import hf_tokenizer
        from verl.utils.fs import copy_to_local
        from verl.utils.hdfs_io import makedirs
        from verl.utils.model import compute_position_id_with_mask
        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        resource_pool = RayResourcePool(
            process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes
        )
        ray_cls_with_init = RayClassWithInitArgs(
            cls=ray.remote(wrap_worker_cls_with_env(ActorRolloutRefWorker)),
            config=config.actor_rollout_ref,
            role="rollout"
        )

        wg = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=ray_cls_with_init,
            device_name=config.trainer.device,
        )
        wg.init_model()
        ckpt_path = config.get("ckpt_path")
        if ckpt_path:
            wg.load_checkpoint(
                os.path.join(ckpt_path, "actor"),
                del_local_after_load=False,
            )
        num_batch = -(-total_samples // config_batch_size)

    legacy_questions, legacy_answers, legacy_contexts = [], [], []
    for batch_idx in trange(num_batch):
        gen_batch = defaultdict(list)
        batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
        gen_batch["hops"] = all_hops[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
        for idx, messages in enumerate(batch_chat_lst):
            row_dict = {}
            raw_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            model_inputs = tokenizer(
                raw_prompt, return_tensors="pt", add_special_tokens=False
            )
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=config.actor_rollout_ref.rollout.prompt_length,
                pad_token_id=tokenizer.pad_token_id,
                left_pad=True,
                truncation="error",
            )
            position_ids = compute_position_id_with_mask(attention_mask)

            row_dict["index"] = str(batch_idx)+"_"+str(idx)
            row_dict["raw_prompt"] = messages
            row_dict["full_prompts"] = raw_prompt
            row_dict["input_ids"] = input_ids[0]
            row_dict["attention_mask"] = attention_mask[0]
            row_dict["position_ids"] = position_ids[0]

            row_dict["tools_kwargs"] = {
                "search": {
                    "create_kwargs": {
                        "data_source": "search_zero",
                        "ground_truth": "",
                        "question": "",
                    }
                }
            }
            row_dict["interaction_kwargs"] = {}

            for key, value in row_dict.items():
                gen_batch[key].append(value)

        for key in gen_batch:
            if isinstance(gen_batch[key][0], torch.Tensor):
                gen_batch[key] = torch.stack(gen_batch[key])
            else:
                gen_batch[key] = np.array(gen_batch[key])

        gen_batch = DataProto.from_single_dict(gen_batch)
        gen_batch = gen_batch.repeat(repeat_times=config.actor_rollout_ref.rollout.n, interleave=True)
        gen_batch, pad_size = pad_dataproto_to_divisor(gen_batch, wg.world_size)
        
        output_padded = wg.generate_sequences(gen_batch)
        outputs = unpad_dataproto(output_padded, pad_size=pad_size)
        structured_messages = outputs.non_tensor_batch.get("messages")

        for i in range(0, len(outputs), config.actor_rollout_ref.rollout.n):
            cur_batch = outputs.batch[i:i+config.actor_rollout_ref.rollout.n]
            cur_hops = gen_batch.non_tensor_batch["hops"][i:i+config.actor_rollout_ref.rollout.n]

            raw_messages = [
                tokenizer.decode(cur_batch["input_ids"][i]) for i in range(len(cur_batch))
            ]
            trajectories = (
                [item["messages"] for item in structured_messages[i : i + len(cur_batch)]]
                if structured_messages is not None
                else raw_messages
            )
            responses = [
                tokenizer.decode(
                    cur_batch["responses"][i], skip_special_tokens=True,
                ) for i in range(len(cur_batch))
            ]

            format_scores, raw_qs, raw_ans = compute_challenger_format_scores(
                raw_messages, responses, cur_hops, return_qa=True,
            )
            document_offset = i // rollout_count
            source_index = batch_idx * config_batch_size + document_offset
            if state:
                metadata = metadata_rows[source_index]
                group = []
                for candidate_index in range(rollout_count):
                    candidate = build_candidate(
                        state=state,
                        metadata=metadata,
                        hop_count=int(cur_hops[candidate_index]),
                        trajectory=trajectories[candidate_index],
                        response=responses[candidate_index],
                        question=raw_qs[candidate_index].strip(),
                        reference_answer=raw_ans[candidate_index].strip(),
                        format_score=float(format_scores[candidate_index]),
                        generation_index=candidate_index,
                    )
                    group.append(candidate)
                candidate_groups.append(group)
                source_rows.append(dataset.iloc[source_index])
            else:
                sample_idx = int(np.argmax(format_scores))
                legacy_questions.append(" ".join(raw_qs[sample_idx].split()[:50]))
                legacy_answers.append(raw_ans[sample_idx])
                legacy_contexts.append(raw_messages[sample_idx])

    if not resume_candidates:
        _release_generation_workers(wg, resource_pool)

    if not state:
        for idx, (question, answer, context) in enumerate(
            zip(legacy_questions, legacy_answers, legacy_contexts, strict=True)
        ):
            dataset.at[dataset.index[idx], "prompt"] = [
                {"role": "user", "content": DEFAULT_SOLVER_PREFIX.format(question=question.strip())}
            ]
            reward_model = dict(dataset.iloc[idx].reward_model)
            ground_truth = dict(reward_model["ground_truth"])
            ground_truth["target"] = [answer]
            reward_model["ground_truth"] = ground_truth
            dataset.at[dataset.index[idx], "reward_model"] = reward_model
            dataset.at[dataset.index[idx], "metadata"] = {"raw_context": context}
        output_dir = os.path.dirname(config.data.output_path)
        makedirs(output_dir, exist_ok=True)
        dataset.to_parquet(config.data.output_path)
        return

    keepout_path = config.data.get("keepout_path") or str(output_path.with_name(output_path.stem + "_keepout.parquet"))
    split_manifest_path = config.data.get("split_manifest_path") or str(
        output_path.with_name(output_path.stem + "_split_manifest.json")
    )
    all_candidates = [candidate for group in candidate_groups for candidate in group]
    completed_groups_reused = (
        sum(candidate_group_is_complete(group, verify_enabled=verify_enabled) for group in candidate_groups)
        if resume_candidates
        else 0
    )
    if not resume_candidates:
        persist_candidates(candidates_path, all_candidates)
        write_candidate_snapshot_manifest(candidate_manifest_path, snapshot_contract)
        reset_candidate_progress(candidate_progress_path)
        if phase == "generate":
            print(f"Generated and persisted {len(all_candidates)} candidates to {candidates_path}")
            return

    async def _rank_and_verify_all():
        meta_config = EndpointConfig.model_validate(
            OmegaConf.to_container(config.meta_model, resolve=True)
        )
        selected = []
        selected_rows = []
        async with OpenAICompatibleClient(meta_config) as meta_client:
            if not verify_enabled:
                for group_index, group in enumerate(candidate_groups):
                    if candidate_group_is_complete(group, verify_enabled=False):
                        eligible = [candidate for candidate in group if candidate.format_score > 0]
                        if eligible:
                            selected.append(max(eligible, key=lambda item: (item.rank_score, -item.generation_index)))
                            selected_rows.append(source_rows[group_index])
                        continue
                    reset_candidate_group(group)
                    for candidate in group:
                        if candidate.format_score <= 0:
                            continue
                        try:
                            scores, raw_output = await evaluate_rubrics(candidate, state.rubrics, meta_client)
                            candidate.rubric_evaluation = scores
                            candidate.rubric_raw_output = raw_output
                            candidate.rank_score = candidate_rank_score(candidate.format_score, scores)
                            candidate.status = "not_verified"
                        except Exception as error:
                            candidate.status = "rubric_error"
                            candidate.rubric_failure = {
                                "error_type": type(error).__name__,
                                "reason": str(error),
                                "details": getattr(error, "details", {}),
                            }
                            append_candidate_progress(candidate_progress_path, group)
                            raise
                    eligible = [candidate for candidate in group if candidate.format_score > 0]
                    if eligible:
                        selected.append(max(eligible, key=lambda item: (item.rank_score, -item.generation_index)))
                        selected_rows.append(source_rows[group_index])
                    append_candidate_progress(candidate_progress_path, group)
                return selected_rows, selected, {
                    "meta": meta_client.metrics.model_dump(mode="json"),
                    "solver": {},
                }

            solver_config = EndpointConfig.model_validate(
                OmegaConf.to_container(config.verify.solver_model, resolve=True)
            )
            async with OpenAICompatibleClient(solver_config) as solver_client:
                verifier = ProblemVerifier(
                    solver_client,
                    samples=int(config.verify.solver_samples),
                )
                winners_by_group: list = [None] * len(candidate_groups)
                pending_groups = []
                pending_group_indices = []
                for group_index, group in enumerate(candidate_groups):
                    if candidate_group_is_complete(group, verify_enabled=True):
                        winners_by_group[group_index] = next(
                            (candidate for candidate in group if candidate.status == "verify_passed"),
                            None,
                        )
                        continue
                    reset_candidate_group(group)
                    pending_groups.append(group)
                    pending_group_indices.append(group_index)

                progress_lock = asyncio.Lock()

                async def persist_completed_group(group):
                    async with progress_lock:
                        await asyncio.to_thread(
                            append_candidate_progress,
                            candidate_progress_path,
                            group,
                        )

                pending_winners = await rank_and_verify_candidate_groups(
                    pending_groups,
                    state.rubrics,
                    meta_client,
                    verifier,
                    max_concurrency=int(config.verify.get("max_group_concurrency", 8)),
                    on_group_complete=persist_completed_group,
                )
                for group_index, winner in zip(
                    pending_group_indices,
                    pending_winners,
                    strict=True,
                ):
                    winners_by_group[group_index] = winner
                for group_index, winner in enumerate(winners_by_group):
                    if winner is not None:
                        selected.append(winner)
                        selected_rows.append(source_rows[group_index])
            metrics = {
                "meta": meta_client.metrics.model_dump(mode="json"),
                "solver": solver_client.metrics.model_dump(mode="json"),
            }
        return selected_rows, selected, metrics

    verify_server, verify_log = _start_local_verify_server(config)
    try:
        selected_source_rows, selected_candidates, model_call_metrics = asyncio.run(_rank_and_verify_all())
    finally:
        _stop_local_verify_server(verify_server, verify_log)
    model_call_metrics["resume"] = {
        "resumed_snapshot": resume_candidates,
        "completed_groups_reused": completed_groups_reused,
        "call_metrics_scope": "current_process_only",
    }
    compact_candidate_progress(candidates_path, candidate_progress_path, all_candidates)
    split_manifest = write_generation_datasets(
        selected_source_rows,
        selected_candidates,
        train_path=str(output_path),
        keepout_path=keepout_path,
        split_manifest_path=split_manifest_path,
        train_ratio=float(iteration_config.get("solver_train_ratio", 0.9)),
        split_seed=int(iteration_config.get("split_seed", 42)),
    )
    summary_path = config.data.get("generation_summary_path") or str(
        output_path.with_name(output_path.stem + "_generation_summary.json")
    )
    atomic_write_json(
        summary_path,
        build_generation_summary(
            candidate_groups,
            selected_candidates,
            split_manifest,
            model_call_metrics=model_call_metrics,
        ),
    )


if __name__ == "__main__":
    main()
