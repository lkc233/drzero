#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

hop_ratio="${HOP_RATIO:-4321}"
rounds="${ROUNDS:-3}"
export TRAIN_GPU_DEVICES="${TRAIN_GPU_DEVICES:-2,3,4,5,6,7}"
export GENERATION_GPU_DEVICES="${GENERATION_GPU_DEVICES:-2,3,4,5,6,7}"
export WANDB_MODE="${WANDB_MODE:-offline}"
# Ray's local dashboard/runtime-env agents communicate over loopback and the
# node address.  Bypass the cluster HTTP proxy for those local control-plane
# requests; otherwise actor creation is sent to the compliance gateway.
local_hosts="127.0.0.1,localhost,::1,$(hostname),$(hostname -i | tr ' ' ',')"
export NO_PROXY="${local_hosts},${NO_PROXY:-}"
export no_proxy="$NO_PROXY"

curl --fail --silent http://127.0.0.1:8000/v1/models >/dev/null
curl --fail --silent http://127.0.0.1:8020/docs >/dev/null

for iteration in $(seq 1 "$rounds"); do
    echo "[$(date -Is)] Starting iteration $iteration/$rounds"
    bash "iter${iteration}_challenger.sh" "$hop_ratio"
    bash "iter${iteration}_gen_data.sh" "$hop_ratio"
    bash "iter${iteration}_solver.sh" "$hop_ratio"
    bash convert.sh "$iteration" "$hop_ratio"
    # Training, generation/verify, and update are intentionally serial, so all
    # six training GPUs (2-7) can be reused by every phase.
    bash update_iteration_state.sh "$iteration" "$hop_ratio"
    echo "[$(date -Is)] Completed iteration $iteration/$rounds"
done
