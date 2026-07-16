import asyncio
import os
from types import SimpleNamespace

import httpx
from omegaconf import OmegaConf

from verl.custom_reward.reward_rollout import gather_with_concurrency
from verl.custom_reward.reward_function import compute_challenger_format_scores
from verl.iteration.models import EndpointConfig, OpenAICompatibleClient
from verl.single_controller.ray.base import wrap_worker_cls_with_env
from verl.trainer.ppo.ray_trainer import compute_rollout_timing_metrics
from verl.tools.schemas import OpenAIFunctionToolSchema


def test_standalone_worker_receives_environment_before_base_init(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    class WorkerBase:
        def __init__(self, marker):
            self.world_size_seen_in_init = os.environ.get("WORLD_SIZE")
            self.marker = marker

        def _configure_before_init(self, name, rank):
            self.register_center_call = (name, rank)

    wrapped = wrap_worker_cls_with_env(WorkerBase)
    worker = wrapped(
        "ready",
        _verl_worker_env={"WORLD_SIZE": "6", "WG_PREFIX": "smoke", "RANK": "0"},
    )

    assert worker.world_size_seen_in_init == "6"
    assert worker.register_center_call == ("smoke_register_center", 0)
    assert worker.marker == "ready"


def test_local_model_client_disables_thinking_and_environment_proxy():
    captured = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(__import__("json").loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok":true}'}}]},
        )

    async def run():
        client = OpenAICompatibleClient(
            EndpointConfig(model_name="local", base_url="http://127.0.0.1:8000")
        )
        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
        try:
            await client.complete_message([{"role": "user", "content": "json"}], json_mode=True)
        finally:
            await client.close()

    asyncio.run(run())

    assert captured["response_format"] == {"type": "json_object"}
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}


def test_reward_rollout_concurrency_is_bounded():
    active = 0
    maximum = 0

    async def work(item):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        return item

    result = asyncio.run(gather_with_concurrency(range(20), 3, work))

    assert result == list(range(20))
    assert maximum == 3


def test_search_tool_array_schema_preserves_item_type():
    raw = OmegaConf.to_container(
        OmegaConf.load("config/search_tool_config.yaml"), resolve=True
    )["tools"][0]["tool_schema"]
    schema = OpenAIFunctionToolSchema.model_validate(raw).model_dump(exclude_none=True)

    assert schema["function"]["parameters"]["properties"]["query_list"]["items"] == {
        "type": "string"
    }


def test_proposer_format_reward_rejects_extra_search_call():
    prompt = (
        "<|im_start|>user\nsource document: seed<|im_end|>"
        "<|im_start|>assistant\n<think>first</think><|im_end|>"
        "<|im_start|>user\n<tool_response>{\"result\":\"one\"}</tool_response><|im_end|>"
        "<|im_start|>assistant\n<think>second</think><|im_end|>"
        "<|im_start|>user\n<tool_response>{\"result\":\"answer\"}</tool_response><|im_end|>"
    )
    response = (
        '<tool_call>{"name":"search","arguments":{"query_list":["one"]}}</tool_call>'
        '<tool_call>{"name":"search","arguments":{"query_list":["two"]}}</tool_call>'
        "<question>Which result?</question><answer>answer</answer>"
    )

    correct_hop_score = compute_challenger_format_scores([prompt], [response], [3])[0]
    extra_call_score = compute_challenger_format_scores([prompt], [response], [2])[0]

    assert correct_hop_score > 0
    assert extra_call_score == 0


def test_rollout_timing_metrics_separate_model_and_retriever_latency():
    output = SimpleNamespace(
        non_tensor_batch={
            "rollout_metrics": [
                {
                    "rollout_timing": {
                        "model_seconds": 8.0,
                        "retriever_seconds": 2.0,
                        "overhead_seconds": 1.0,
                        "total_seconds": 11.0,
                    },
                    "search": [{"latency_seconds": 2.0}],
                },
                {
                    "rollout_timing": {
                        "model_seconds": 4.0,
                        "retriever_seconds": 1.0,
                        "overhead_seconds": 1.0,
                        "total_seconds": 6.0,
                    },
                    "search": [{"latency_seconds": 0.4}, {"latency_seconds": 0.6}],
                },
            ]
        }
    )

    metrics = compute_rollout_timing_metrics(output)

    assert metrics["generation/model_latency_share"] == 12 / 17
    assert metrics["generation/retriever_latency_share"] == 3 / 17
    assert metrics["generation/retriever_request_count"] == 3
    assert metrics["generation/retriever_request_seconds_p99"] > 1.9
