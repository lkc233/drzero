# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

source "$(dirname "${BASH_SOURCE[0]}")/.venv/bin/activate"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir ./checkpoints/dr-zero/solver_iter1_ratio4321_grpo_group5_qwen3-4b-instruct-2507/global_step_50/actor \
    --target_dir ./checkpoints/dr-zero/solver_iter1_ratio4321_grpo_group5_qwen3-4b-instruct-2507/solver_iter1_hf