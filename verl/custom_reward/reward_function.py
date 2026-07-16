import json
import logging
import os
import re
import time
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.custom_reward.format_scoring import (
    challenger_answer_reward,
    challenger_format_score,
    challenger_think_reward,
    normalize_answer,
)
from verl.custom_reward.reward_rollout import MultiTurnRewardRollout
from verl.iteration.core import (
    Candidate,
    EvidenceItem,
    StateStore,
    extract_evidence_bundle,
    normalize_trajectory,
    proposer_reward_components,
)
from verl.iteration.models import (
    EndpointConfig,
    OpenAICompatibleClient,
    evaluate_rubric_rewards,
    validate_model_reference,
)
from verl.prompts import (
    ANSWER_PATTERN,
    ASSISTANT_PATTERN,
    DEFAULT_SOLVER_PREFIX,
    QUESTION_PATTERN,
    SOURCE_PATTERN,
    TOOL_CALL_PATTERN,
    TOOL_RESPONSE_PATTERN,
)
from verl.utils.asyncio_utils import get_or_create_event_loop
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SOLVER_PROMPT_MAX_LENGTH = 512


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def count_valid_tool_calls(response):
    valid_count = 0
    tool_matches = re.findall(TOOL_CALL_PATTERN, response, re.DOTALL)
    for match in tool_matches:
        try:
            parsed = json.loads(match.strip())
            if (
                isinstance(parsed, dict)
                and parsed.get("name") == "search"
                and isinstance(parsed.get("arguments"), dict)
                and isinstance(parsed["arguments"].get("query_list"), list)
                and all(isinstance(q, str) for q in parsed["arguments"]["query_list"])
            ):
                valid_count += 1
        except (json.JSONDecodeError, TypeError):
            continue

    return valid_count


def compute_difficulty_score(binary_list):
    assert len(binary_list) > 1
    n = len(binary_list) 
    num_ones = sum(binary_list)
    if num_ones == 0 or num_ones == n:
        return 0.0
    score = (n - num_ones) / (n - 1)    
    return score


def compute_challenger_format_scores(raw_messages, responses, hops, return_qa=False):
    assistant_turns = [re.findall(ASSISTANT_PATTERN, m, re.DOTALL) for m in raw_messages]
    think_rewards = [challenger_think_reward(messages) for messages in assistant_turns]
    
    raw_docs, tool_rewards = [], []
    for messages, response, hop in zip(raw_messages, responses, hops, strict=True):
        tool_calls = count_valid_tool_calls(response)
        doc = re.findall(SOURCE_PATTERN, messages, re.DOTALL)[0]
        doc_matches = re.findall(TOOL_RESPONSE_PATTERN, messages, re.DOTALL)
        
        expected_tool_calls = max(0, int(hop) - 1)
        reward = float(tool_calls == len(doc_matches) == expected_tool_calls)
        if reward and tool_calls > 0:
            try:
                doc += "\n\n" + "\n\n".join([json.loads(x)["result"] for x in doc_matches])
            except (json.JSONDecodeError, KeyError, TypeError):
                logger.error(f"Error parsing tool response: {doc_matches}")

        raw_docs.append(doc)
        tool_rewards.append(reward)

    raw_ans, ans_rewards = [], []
    for doc, text in zip(raw_docs, responses, strict=True):
        ans_matches = re.findall(ANSWER_PATTERN, text, re.DOTALL)
        ans = ans_matches[-1].strip() if ans_matches else ""
        raw_ans.append(ans)
        ans_rewards.append(challenger_answer_reward(doc, ans))
    
    raw_qs = []
    for text in responses:
        q_matches = re.findall(QUESTION_PATTERN, text, re.DOTALL)
        raw_qs.append(q_matches[-1].strip() if q_matches else "")

    final_scores = [
        challenger_format_score(
            question=question,
            answer=answer,
            think_reward=think_reward,
            tool_reward=tool_reward,
            answer_reward=answer_reward,
        )
        for question, answer, think_reward, tool_reward, answer_reward in zip(
            raw_qs,
            raw_ans,
            think_rewards,
            tool_rewards,
            ans_rewards,
            strict=True,
        )
    ]
    # print(
    #     f"Think rewards: Avg {np.mean(think_rewards)}, Max {np.max(think_rewards)}\n"
    #     f"Tool rewards: Avg {np.mean(tool_rewards)}, Max {np.max(tool_rewards)}\n"
    #     f"Answer rewards: Avg {np.mean(ans_rewards)}, Max {np.max(ans_rewards)}\n"
    #     f"Question rewards: Avg {np.mean(integrity_rewards)}, Max {np.max(integrity_rewards)}\n"
    #     f"Final format rewards: Avg {np.mean(final_scores)}, Max {np.max(final_scores)}\n"
    # )

    if return_qa:
        return final_scores, raw_qs, raw_ans
    else:
        return final_scores


def compute_challenger_score_batch(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    reward_started = time.monotonic()
    timings = {}
    batch = kwargs["data"]
    processing_class = kwargs["processing_class"]
    assert "qwen" in type(processing_class).__name__.lower()

    rollout_config = deepcopy(kwargs["config"])
    rollout_config["prompt_length"] = SOLVER_PROMPT_MAX_LENGTH

    hops = [int(x.split('_')[-1]) for x in batch.non_tensor_batch["data_source"]]
    raw_messages = [
        processing_class.decode(batch.batch["input_ids"][i]) for i in range(len(batch))
    ]
    responses = [
        processing_class.decode(
            batch.batch["responses"][i], skip_special_tokens=True,
        ) for i in range(len(batch))
    ]

    format_started = time.monotonic()
    format_scores, raw_qs, raw_ans = compute_challenger_format_scores(
        raw_messages, responses, hops, return_qa=True,
    )
    timings["format"] = time.monotonic() - format_started

    solver_prepare_started = time.monotonic()
    gen_batch_ids, gen_batch = [], defaultdict(list)
    for idx, (s, q, a) in enumerate(zip(format_scores, raw_qs, raw_ans, strict=True)):
        if s > 0:
            gen_batch_ids.append(idx)
            row_dict, messages = {}, [
                {"role": "user", "content": DEFAULT_SOLVER_PREFIX.format(question=q.strip())}
            ]

            raw_prompt = processing_class.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            model_inputs = processing_class(
                raw_prompt, return_tensors="pt", add_special_tokens=False
            )
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=rollout_config["prompt_length"],
                pad_token_id=processing_class.pad_token_id,
                left_pad=True,
                truncation="left",  # in case of very long questions
            )
            position_ids = compute_position_id_with_mask(attention_mask)

            row_dict["index"] = idx
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

    group_size = kwargs["reward_rollout_n"]
    timings["solver_prepare"] = time.monotonic() - solver_prepare_started
    loop = get_or_create_event_loop()
    solver_scores = []
    grouped_preds = []
    if gen_batch_ids:
        for key in gen_batch:
            if isinstance(gen_batch[key][0], torch.Tensor):
                gen_batch[key] = torch.stack(gen_batch[key])
            else:
                gen_batch[key] = np.array(gen_batch[key])

        gen_batch = DataProto.from_single_dict(gen_batch)
        gen_batch = gen_batch.repeat(repeat_times=group_size, interleave=True)

        async def _compute_with_rollout():
            async with MultiTurnRewardRollout(
                config=rollout_config,
                processing_class=processing_class,
                model_name=kwargs["model_name"],
                base_url=kwargs["base_url"],
                request_concurrency=int(kwargs.get("solver_max_concurrency", 64)),
            ) as reward_rollout:
                gen_batch_output = await reward_rollout.generate_sequences(gen_batch)
                return gen_batch_output

        solver_started = time.monotonic()
        outputs = loop.run_until_complete(_compute_with_rollout())
        timings["solver_rollout"] = time.monotonic() - solver_started
        extracted_preds = []
        last_turn_responses = [
            x["messages"][-1].content for x in outputs.non_tensor_batch["messages"]
        ]
        for raw_response in last_turn_responses:
            extracted = re.findall(ANSWER_PATTERN, raw_response, re.DOTALL)
            if len(extracted) > 0:
                extracted_preds.append(extracted[-1].strip())
            else:
                extracted_preds.append(raw_response.strip())

        grouped_preds = [
            extracted_preds[i:i + group_size] for i in range(0, len(extracted_preds), group_size)
        ]
        ground_truths = [x for idx, x in enumerate(raw_ans) if idx in gen_batch_ids]
        for preds, gt in zip(grouped_preds, ground_truths, strict=True):
            em_scores = [em_check(pred, gt) for pred in preds]
            solver_scores.append(compute_difficulty_score(em_scores))

    difficulty_scores = [0.0] * len(format_scores)
    for idx, score in zip(gen_batch_ids, solver_scores, strict=True):
        difficulty_scores[idx] = score

    reward_config = kwargs.get("proposer_reward", {})
    format_weight = float(reward_config.get("format_weight", 0.5))
    difficulty_weight = float(reward_config.get("difficulty_weight", 1.0))
    rubric_weight = float(reward_config.get("rubric_weight", 0.0))
    rubric_evaluations = [[] for _ in format_scores]
    rubric_score_overrides = [None for _ in format_scores]
    rubric_evaluation_failures = [False for _ in format_scores]

    if rubric_weight:
        rubric_prepare_started = time.monotonic()
        iteration_state_path = kwargs.get("iteration_state_path")
        meta_model_config = kwargs.get("meta_model")
        if not iteration_state_path or not meta_model_config:
            raise ValueError("rubric reward requires iteration_state_path and meta_model configuration")
        state = StateStore(iteration_state_path).load()
        validate_model_reference(
            kwargs["model_name"],
            state.models.solver_before,
            role="reward rollout solver",
        )
        endpoint_config = EndpointConfig.model_validate(dict(meta_model_config))
        candidates = []
        for idx, (message, response, hop, question, answer, format_score) in enumerate(
            zip(raw_messages, responses, hops, raw_qs, raw_ans, format_scores, strict=True)
        ):
            source_matches = re.findall(SOURCE_PATTERN, message, re.DOTALL)
            if not source_matches:
                raise ValueError("proposer reward input is missing the source document")
            source_document = source_matches[0].strip()
            # The prompt contains illustrative tool markup; only the generated
            # response is an executed proposer trajectory.
            trajectory_text = response
            try:
                trajectory, evidence = extract_evidence_bundle(
                    source_document,
                    trajectory_text,
                    hop_count=hop,
                )
            except ValueError:
                trajectory = normalize_trajectory(trajectory_text)
                evidence = [
                    EvidenceItem(
                        evidence_id="evidence-0",
                        kind="seed_document",
                        content=source_document,
                        source="seed_document",
                        trajectory_index=0,
                    )
                ]
            candidates.append(
                Candidate(
                    candidate_id=f"reward-{idx}",
                    iteration=state.iteration,
                    doc_id=f"reward-{idx}",
                    hop_count=hop,
                    source_document=source_document,
                    proposer_trajectory=trajectory,
                    evidence_bundle=evidence,
                    question=question,
                    reference_answer=answer,
                    format_score=float(format_score),
                    generation_index=idx,
                )
            )

        async def _compute_rubric_evaluations():
            async with OpenAICompatibleClient(endpoint_config) as meta_client:
                return await evaluate_rubric_rewards(
                    candidates,
                    state.rubrics,
                    meta_client,
                )

        timings["rubric_prepare"] = time.monotonic() - rubric_prepare_started
        rubric_started = time.monotonic()
        (
            rubric_evaluations,
            rubric_score_overrides,
            rubric_evaluation_failures,
        ) = loop.run_until_complete(_compute_rubric_evaluations())
        timings["rubric_evaluation"] = time.monotonic() - rubric_started

    scoring_started = time.monotonic()
    final_scores = [
        proposer_reward_components(
            float(format_score),
            float(difficulty_score),
            evaluations,
            format_weight=format_weight,
            difficulty_weight=difficulty_weight,
            rubric_weight=rubric_weight,
            rubric_score_override=rubric_score,
        )
        for format_score, difficulty_score, evaluations, rubric_score in zip(
            format_scores,
            difficulty_scores,
            rubric_evaluations,
            rubric_score_overrides,
            strict=True,
        )
    ]
    for score, failed in zip(final_scores, rubric_evaluation_failures, strict=True):
        score["rubric_evaluation_failed"] = float(failed)
    timings["scoring"] = time.monotonic() - scoring_started
    timings["total"] = time.monotonic() - reward_started

    example_idx = gen_batch_ids[0] if gen_batch_ids else 0
    example_solver_responses = grouped_preds[0] if grouped_preds else []
    print(
        f"🚀 Raw format rewards: Avg {np.mean(format_scores)}, Max {np.max(format_scores)}\n"
        f"🚀 Difficulty rewards: Avg {np.mean(difficulty_scores)}, Max {np.max(difficulty_scores)}\n"
        f"🚀 Rubric rewards: Avg {np.mean([item['rubric_score'] for item in final_scores])}, "
        f"Max {np.max([item['rubric_score'] for item in final_scores])}\n"
        f"🚨 Rubric evaluation failures: {sum(rubric_evaluation_failures)}\n"
        f"⏱️ Reward timings (seconds): {json.dumps({key: round(value, 3) for key, value in timings.items()}, sort_keys=True)}\n"
        f"🚀 Final rewards: Avg {np.mean([item['score'] for item in final_scores])}, "
        f"Max {np.max([item['score'] for item in final_scores])}\n"
        f"🧑‍🏫 Challenger question: {raw_qs[example_idx]}\n"
        f"🧑‍🎓 Challenger answer: {raw_ans[example_idx]}\n"
        f"🙋‍♂️ Solver responses: {example_solver_responses}"
    )
    return final_scores
