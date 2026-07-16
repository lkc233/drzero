#!/usr/bin/env bash
# Shared deployment initialization for iteration entry points.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "ERROR: source scripts/init_deployment.sh [all|judge|retriever]" >&2
    exit 2
fi

_drzero_init_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$_drzero_init_root/scripts/load_deployment_config.sh"
bash "$_drzero_init_root/scripts/check_deployment_services.sh" "${1:-all}"
unset _drzero_init_root
