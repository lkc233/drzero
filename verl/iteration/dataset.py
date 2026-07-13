from __future__ import annotations

import json
from typing import Any

from verl.iteration.core import StateStore, dynamic_state_hash
from verl.prompts import build_challenger_prompt
from verl.utils.dataset.rl_dataset import RLHFDataset


def should_use_proposer_iteration_prompt(config, *, is_train: bool, phase: str | None = None) -> bool:
    return is_train and (
        bool(config.get("use_proposer_iteration_prompt", False))
        or phase == "proposer_train"
    )


class ProposerIterationDataset(RLHFDataset):
    """RLHF dataset that rebuilds proposer prompts from one frozen iteration state."""

    def __init__(self, data_files, tokenizer, config, processor=None):
        state_path = config.get("proposer_iteration_state_path")
        if not state_path:
            raise ValueError("data.proposer_iteration_state_path is required for proposer training")
        self.iteration_state = StateStore(state_path).load()
        self.iteration_state_hash = dynamic_state_hash(self.iteration_state)
        super().__init__(data_files=data_files, tokenizer=tokenizer, config=config, processor=processor)

    @staticmethod
    def _metadata(example: dict[str, Any]) -> dict[str, Any]:
        metadata = example.get("metadata")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        if not isinstance(metadata, dict):
            raise ValueError("proposer data requires structured metadata")
        return metadata

    def _build_messages(self, example: dict):
        metadata = self._metadata(example)
        source_document = metadata.get("source_document")
        hop_count = metadata.get("hop_count")
        doc_id = metadata.get("doc_id")
        if not source_document or not doc_id or not isinstance(hop_count, int):
            raise ValueError("proposer metadata requires doc_id, source_document, and integer hop_count")
        prompt = build_challenger_prompt(
            hops=hop_count,
            document=source_document,
            skills=self.iteration_state.skills,
        )
        return [{"role": "user", "content": prompt}]
