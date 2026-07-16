from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOADER = ROOT / "scripts" / "load_deployment_config.sh"
CHECKER = ROOT / "scripts" / "check_deployment_services.sh"


def test_remote_profile_separates_service_ownership_from_endpoints() -> None:
    command = (
        f"source {LOADER}; "
        "printf '%s|%s|%s' \"$MANAGE_RETRIEVER\" \"$MANAGE_JUDGE\" \"$DRZERO_META_BASE_URL\""
    )
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT,
        env={
            **os.environ,
            "DRZERO_DEPLOY_CONFIG": str(ROOT / "deploy" / "remote-services.env.example"),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == "false|false|http://judge.internal:8000"


def test_explicit_missing_profile_is_an_error(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", "-c", f"source {LOADER}"],
        cwd=ROOT,
        env={**os.environ, "DRZERO_DEPLOY_CONFIG": str(tmp_path / "missing.env")},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "deployment config does not exist" in result.stderr


def test_service_checker_validates_remote_contracts(tmp_path: Path) -> None:
    curl = tmp_path / "curl"
    curl.write_text(
        "#!/usr/bin/env bash\n"
        "case \"$*\" in\n"
        "  *retrieve*) printf '%s' '{\"result\":[[\"document\"]]}' ;;\n"
        "  *chat/completions*) printf '%s' '{\"choices\":[{\"message\":{\"content\":\"{\\\"ok\\\":true}\"}}]}' ;;\n"
        "  *) printf '%s' '{\"data\":[{\"id\":\"remote-judge\"}]}' ;;\n"
        "esac\n"
    )
    curl.chmod(0o755)
    profile = tmp_path / "remote.env"
    profile.write_text(
        "MANAGE_RETRIEVER=false\n"
        "MANAGE_JUDGE=false\n"
        "DRZERO_RETRIEVER_URL=http://retriever/retrieve\n"
        "DRZERO_META_BASE_URL=http://judge\n"
        "DRZERO_META_MODEL=remote-judge\n"
        "DRZERO_UPDATER_BASE_URL=http://judge\n"
        "DRZERO_UPDATER_MODEL=remote-judge\n"
    )
    result = subprocess.run(
        ["bash", str(CHECKER), "all"],
        cwd=ROOT,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "DRZERO_DEPLOY_CONFIG": str(profile),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Retriever ready" in result.stdout
    assert "judge ready" in result.stdout


def test_service_checker_rejects_unknown_scope() -> None:
    result = subprocess.run(
        ["bash", str(CHECKER), "judeg"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "unknown service-check scope" in result.stderr
