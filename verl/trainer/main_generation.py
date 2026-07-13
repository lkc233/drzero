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
from collections import defaultdict
from pathlib import Path
from pprint import pprint

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import hydra
import numpy as np
import pandas as pd
import ray
import torch
from omegaconf import OmegaConf
from tqdm import trange

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.custom_reward.reward_function import compute_challenger_format_scores
from verl.iteration.core import StateStore, atomic_write_json, candidate_rank_score
from verl.iteration.generation import (
    build_candidate,
    build_generation_summary,
    persist_candidates,
    write_generation_datasets,
)
from verl.iteration.models import (
    EndpointConfig,
    OpenAICompatibleClient,
    ProblemVerifier,
    evaluate_rubrics,
    rank_and_verify_candidates,
    validate_model_reference,
)
from verl.prompts import DEFAULT_SOLVER_PREFIX, build_challenger_prompt
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        temp_dir = os.path.join(os.getcwd(), "tmp/ray")
        if len(temp_dir) > 64:
            temp_dir = "/tmp/ray"
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
            _temp_dir=temp_dir,
        )

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    local_path = copy_to_local(
        config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
    )
    trust_remote_code = config.data.get("trust_remote_code", False)

    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    resource_pool = RayResourcePool(
        process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes
    )
    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(ActorRolloutRefWorker),
        config=config.actor_rollout_ref,
        role="rollout"
    )

    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )
    wg.init_model()
    wg.load_checkpoint(
        os.path.join(config.ckpt_path, "actor"), del_local_after_load=False,
    )

    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    num_batch = -(-total_samples // config_batch_size)

    candidate_count = int(iteration_config.get("candidate_count_per_document", 5))
    rollout_count = int(config.actor_rollout_ref.rollout.n)
    if state and rollout_count != candidate_count:
        raise ValueError(
            f"generation must produce exactly {candidate_count} candidates per document, got rollout.n={rollout_count}"
        )

    candidate_groups = []
    source_rows = []
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

        for i in range(0, len(outputs), config.actor_rollout_ref.rollout.n):
            cur_batch = outputs.batch[i:i+config.actor_rollout_ref.rollout.n]
            cur_hops = gen_batch.non_tensor_batch["hops"][i:i+config.actor_rollout_ref.rollout.n]

            raw_messages = [
                tokenizer.decode(cur_batch["input_ids"][i]) for i in range(len(cur_batch))
            ]
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
                group = [
                    build_candidate(
                        state=state,
                        metadata=metadata,
                        hop_count=int(cur_hops[candidate_index]),
                        trajectory=raw_messages[candidate_index],
                        response=responses[candidate_index],
                        question=raw_qs[candidate_index].strip(),
                        reference_answer=raw_ans[candidate_index].strip(),
                        format_score=float(format_scores[candidate_index]),
                        generation_index=candidate_index,
                    )
                    for candidate_index in range(rollout_count)
                ]
                candidate_groups.append(group)
                source_rows.append(dataset.iloc[source_index])
            else:
                sample_idx = int(np.argmax(format_scores))
                legacy_questions.append(" ".join(raw_qs[sample_idx].split()[:50]))
                legacy_answers.append(raw_ans[sample_idx])
                legacy_contexts.append(raw_messages[sample_idx])

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

    output_path = Path(config.data.output_path)
    candidates_path = config.data.get("candidates_path") or str(output_path.with_suffix(".candidates.jsonl"))
    keepout_path = config.data.get("keepout_path") or str(output_path.with_name(output_path.stem + "_keepout.parquet"))
    split_manifest_path = config.data.get("split_manifest_path") or str(
        output_path.with_name(output_path.stem + "_split_manifest.json")
    )
    all_candidates = [candidate for group in candidate_groups for candidate in group]
    persist_candidates(candidates_path, all_candidates)

    async def _rank_and_verify_all():
        meta_config = EndpointConfig.model_validate(
            OmegaConf.to_container(config.meta_model, resolve=True)
        )
        selected = []
        selected_rows = []
        async with OpenAICompatibleClient(meta_config) as meta_client:
            if not verify_enabled:
                for group_index, group in enumerate(candidate_groups):
                    for candidate in group:
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
                            raise
                        finally:
                            persist_candidates(candidates_path, all_candidates)
                    selected.append(max(group, key=lambda item: (item.rank_score, -item.generation_index)))
                    selected_rows.append(source_rows[group_index])
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
                    meta_client,
                    samples=int(config.verify.solver_samples),
                )
                for group_index, group in enumerate(candidate_groups):
                    try:
                        winner, _, _ = await rank_and_verify_candidates(
                            group,
                            state.rubrics,
                            meta_client,
                            verifier,
                        )
                    finally:
                        persist_candidates(candidates_path, all_candidates)
                    if winner is not None:
                        selected.append(winner)
                        selected_rows.append(source_rows[group_index])
            metrics = {
                "meta": meta_client.metrics.model_dump(mode="json"),
                "solver": solver_client.metrics.model_dump(mode="json"),
            }
        return selected_rows, selected, metrics

    selected_source_rows, selected_candidates, model_call_metrics = asyncio.run(_rank_and_verify_all())
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
