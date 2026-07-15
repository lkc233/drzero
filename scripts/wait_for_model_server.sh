#!/usr/bin/env bash
set -euo pipefail

url="${1:?models URL is required}"
model_name="${2-}"
server_pid="${3:-0}"
timeout_seconds="${4:-600}"
deadline=$((SECONDS + timeout_seconds))

while (( SECONDS < deadline )); do
    if [ "$server_pid" != "0" ] && ! kill -0 "$server_pid" 2>/dev/null; then
        echo "ERROR: model server process $server_pid exited before $url became ready" >&2
        exit 1
    fi
    if response="$(curl --fail --silent --show-error "$url" 2>/dev/null)" \
        && { [ -z "$model_name" ] || grep -Fq "\"id\":\"$model_name\"" <<<"$response"; }; then
        echo "Server ready${model_name:+: $model_name} at $url"
        exit 0
    fi
    sleep 5
done

echo "ERROR: timed out after ${timeout_seconds}s waiting for $model_name at $url" >&2
exit 1
