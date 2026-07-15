from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "setup_qwen36_judge.sh"


def _fake_curl(tmp_path: Path, response: str) -> dict[str, str]:
    curl = tmp_path / "curl"
    curl.write_text(f"#!/usr/bin/env bash\nprintf '%s' '{response}'\n")
    curl.chmod(0o755)
    return {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}


def test_check_accepts_expected_local_model(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--check"],
        env=_fake_curl(tmp_path, '{"data":[{"id":"Qwen/Qwen3.6-35B-A3B"}]}'),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Local judge/updater is ready" in result.stdout


def test_check_rejects_wrong_local_model(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--check"],
        env=_fake_curl(tmp_path, '{"data":[{"id":"some-other-model"}]}'),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "unavailable or does not serve" in result.stderr
