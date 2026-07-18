#!/usr/bin/env bash

run_id="${RUN_ID:-}"
if [[ -n "$run_id" && ! "$run_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: invalid RUN_ID: $run_id" >&2
    return 2
fi
export DRZERO_RUN_SUFFIX="${run_id:+_run-${run_id}}"
export DRZERO_ITERATION_ROOT="${run_id:+./runs/${run_id}/}iterations"
export DRZERO_LOG_DIR="${run_id:+./logs/${run_id}}"
export DRZERO_LOG_DIR="${DRZERO_LOG_DIR:-./logs}"
