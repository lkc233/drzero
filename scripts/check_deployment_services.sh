#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$root/scripts/load_deployment_config.sh"
scope="${1:-all}"
timeout_seconds="${DRZERO_SERVICE_STARTUP_TIMEOUT:-600}"
case "$scope" in
    all|judge|retriever) ;;
    *)
        echo "ERROR: unknown service-check scope: $scope (expected all, judge, or retriever)" >&2
        exit 2
        ;;
esac

models_url() {
    local base="${1%/}"
    if [[ "$base" == */v1 ]]; then
        printf '%s/models\n' "$base"
    else
        printf '%s/v1/models\n' "$base"
    fi
}

chat_url() {
    local base="${1%/}"
    if [[ "$base" == */v1 ]]; then
        printf '%s/chat/completions\n' "$base"
    else
        printf '%s/v1/chat/completions\n' "$base"
    fi
}

check_model() {
    local role="$1" base="$2" model="$3" api_key_env="${4:-}"
    local url auth_args=() payload chat_response
    url="$(models_url "$base")"
    if [[ -n "$api_key_env" ]]; then
        if [[ -z "${!api_key_env:-}" ]]; then
            echo "ERROR: $role requires unset API key variable: $api_key_env" >&2
            return 1
        fi
        auth_args=(-H "Authorization: Bearer ${!api_key_env}")
    fi
    local response
    response="$(curl --fail --silent --show-error "${auth_args[@]}" "$url")"
    if ! python3 -c '
import json, sys
body = json.load(sys.stdin)
expected = sys.argv[1]
raise SystemExit(0 if any(item.get("id") == expected for item in body.get("data", [])) else 1)
' "$model" <<<"$response"; then
        echo "ERROR: $role endpoint $url does not advertise model $model" >&2
        return 1
    fi
    payload="$(python3 -c '
import json, sys
print(json.dumps({
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": "Return JSON with ok set to true."}],
    "temperature": 0,
    "max_tokens": 32,
    "stream": False,
    "response_format": {"type": "json_object"},
    "chat_template_kwargs": {"enable_thinking": False},
}))
' "$model")"
    chat_response="$(curl --fail --silent --show-error "${auth_args[@]}" \
        -H 'Content-Type: application/json' -d "$payload" "$(chat_url "$base")")"
    if ! python3 -c '
import json, sys
body = json.load(sys.stdin)
content = body["choices"][0]["message"]["content"]
parsed = json.loads(content)
raise SystemExit(0 if parsed.get("ok") is True else 1)
' <<<"$chat_response"; then
        echo "ERROR: $role endpoint does not satisfy the structured chat-completion contract at $base" >&2
        return 1
    fi
    echo "[deployment] $role ready: $model at $base"
}

check_retriever() {
    local payload response
    payload='{"queries":["deployment health check"],"topk":1,"return_scores":true}'
    response="$(curl --fail --silent --show-error \
        -H 'Content-Type: application/json' -d "$payload" "$DRZERO_RETRIEVER_URL")"
    if ! python3 -c '
import json, sys
body = json.load(sys.stdin)
result = body.get("result")
valid = isinstance(result, list) and len(result) == 1 and isinstance(result[0], list)
raise SystemExit(0 if valid else 1)
' <<<"$response"; then
        echo "ERROR: Retriever returned an unexpected response from $DRZERO_RETRIEVER_URL" >&2
        return 1
    fi
    echo "[deployment] Retriever ready at $DRZERO_RETRIEVER_URL"
}

wait_until_ready() {
    local role="$1"
    shift
    local deadline=$((SECONDS + timeout_seconds)) output=""
    while (( SECONDS < deadline )); do
        if output="$("$@" 2>&1)"; then
            printf '%s\n' "$output"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: timed out after ${timeout_seconds}s waiting for $role" >&2
    [[ -z "$output" ]] || printf '%s\n' "$output" >&2
    return 1
}

if [[ "$scope" == "all" || "$scope" == "retriever" ]]; then
    wait_until_ready "Retriever at $DRZERO_RETRIEVER_URL" check_retriever
fi

if [[ "$scope" == "all" || "$scope" == "judge" ]]; then
    wait_until_ready "judge at $DRZERO_META_BASE_URL" \
        check_model "judge" "$DRZERO_META_BASE_URL" "$DRZERO_META_MODEL" "${DRZERO_META_API_KEY_ENV:-}"
    if [[ "$DRZERO_UPDATER_BASE_URL/$DRZERO_UPDATER_MODEL/${DRZERO_UPDATER_API_KEY_ENV:-}" != \
          "$DRZERO_META_BASE_URL/$DRZERO_META_MODEL/${DRZERO_META_API_KEY_ENV:-}" ]]; then
        wait_until_ready "updater at $DRZERO_UPDATER_BASE_URL" \
            check_model "updater" "$DRZERO_UPDATER_BASE_URL" "$DRZERO_UPDATER_MODEL" \
                "${DRZERO_UPDATER_API_KEY_ENV:-}"
    fi
fi
