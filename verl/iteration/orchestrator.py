from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import uuid
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from verl.iteration.core import (
    IterationState,
    ModelReferences,
    Rubric,
    Skill,
    StageRecord,
    StateStore,
    atomic_write_json,
    canonical_hash,
    dynamic_state_hash,
    utc_now,
)

STAGE_ORDER = (
    "proposer_train",
    "generation_verify_split",
    "solver_train",
    "convert_solver",
    "keepout_eval",
    "trajectory_analysis",
    "update_skills",
    "update_rubrics",
    "finalize",
)


class IterationLock(AbstractContextManager):
    def __init__(self, state_path: str | Path):
        self.path = Path(state_path).with_suffix(".lock")
        self.token = uuid.uuid4().hex
        self.acquired = False

    @staticmethod
    def _process_start_ticks(pid: int) -> str | None:
        try:
            fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            return None
        return fields[21] if len(fields) > 21 else None

    def _existing_owner_is_active(self) -> bool:
        try:
            raw = self.path.read_text(encoding="utf-8").strip()
            owner = json.loads(raw) if raw.startswith("{") else {"pid": int(raw)}
            pid = int(owner["pid"])
        except (FileNotFoundError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return False
        if owner.get("hostname") not in {None, socket.gethostname()}:
            return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        expected_start = owner.get("process_start_ticks")
        return expected_start is None or expected_start == self._process_start_ticks(pid)

    def __enter__(self) -> IterationLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        owner = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "process_start_ticks": self._process_start_ticks(os.getpid()),
            "token": self.token,
        }
        temporary_path = self.path.with_name(f".{self.path.name}.{self.token}.tmp")
        temporary_fd = os.open(temporary_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(temporary_fd, json.dumps(owner, sort_keys=True).encode())
            os.fsync(temporary_fd)
        finally:
            os.close(temporary_fd)
        try:
            for _ in range(2):
                try:
                    os.link(temporary_path, self.path)
                    self.acquired = True
                    break
                except FileExistsError as error:
                    try:
                        stale_stat = self.path.stat()
                        stale_identity = (stale_stat.st_dev, stale_stat.st_ino)
                    except FileNotFoundError:
                        continue
                    if self._existing_owner_is_active():
                        raise RuntimeError(f"iteration is already locked: {self.path}") from error
                    try:
                        current_stat = self.path.stat()
                        current_identity = (current_stat.st_dev, current_stat.st_ino)
                    except FileNotFoundError:
                        continue
                    if current_identity == stale_identity:
                        self.path.unlink(missing_ok=True)
        finally:
            temporary_path.unlink(missing_ok=True)
        if not self.acquired:
            raise RuntimeError(f"could not acquire iteration lock: {self.path}")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        try:
            owner = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            owner = {}
        if owner.get("token") == self.token:
            self.path.unlink(missing_ok=True)
        self.acquired = False
        return False


def _artifact_snapshot(paths: list[str]) -> list[dict[str, Any]]:
    snapshots = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"declared stage artifact does not exist: {path}")
        snapshot = {
            "path": str(path.resolve()),
            "kind": "directory" if path.is_dir() else "file",
        }
        if path.is_dir():
            entries = sorted(path.rglob("*"), key=lambda entry: entry.relative_to(path).as_posix())
            files = [entry for entry in entries if entry.is_file() and not entry.is_symlink()]
            digest = hashlib.sha256()
            for entry in entries:
                relative = entry.relative_to(path).as_posix()
                if entry.is_symlink():
                    contract = f"symlink:{relative}:{os.readlink(entry)}"
                elif entry.is_dir():
                    contract = f"directory:{relative}"
                elif entry.is_file():
                    contract = f"file:{relative}:{entry.stat().st_size}:{_file_sha256(entry)}"
                else:
                    contract = f"other:{relative}"
                digest.update(contract.encode())
                digest.update(b"\0")
            snapshot.update(
                {
                    "entry_count": len(entries),
                    "file_count": len(files),
                    "recursive_size": sum(entry.stat().st_size for entry in files),
                    "sha256": digest.hexdigest(),
                }
            )
        else:
            snapshot.update({"size": path.stat().st_size, "sha256": _file_sha256(path)})
        snapshots.append(snapshot)
    return snapshots


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_contract(path: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"required input file does not exist: {source}")
    return {
        "path": str(source.resolve()),
        "size": source.stat().st_size,
        "sha256": _file_sha256(source),
    }


def _command_file_contracts(command: list[str] | None, working_directory: str | Path) -> list[dict[str, Any]]:
    token_groups = [command or []]
    if command:
        shell_name = Path(command[0]).name
        for index, token in enumerate(command[:-1]):
            if shell_name in {"bash", "dash", "sh", "zsh"} and token in {"-c", "-lc"}:
                try:
                    token_groups.append(shlex.split(command[index + 1]))
                except ValueError:
                    pass

    contracts = []
    seen_paths = set()
    for tokens in token_groups:
        for index, token in enumerate(tokens):
            candidates = [token]
            if "=" in token:
                candidates.append(token.split("=", 1)[1])
            if index == 0:
                executable = shutil.which(token)
                if executable:
                    candidates.append(executable)
            for candidate in candidates:
                try:
                    path = Path(candidate)
                    if not path.is_absolute():
                        path = Path(working_directory) / path
                    if path.is_file():
                        resolved = str(path.resolve())
                        if resolved not in seen_paths:
                            contracts.append(_file_contract(resolved))
                            seen_paths.add(resolved)
                except (OSError, ValueError):
                    continue
    return sorted(contracts, key=lambda item: item["path"])


def _parquet_candidate_ids(path: str) -> list[str]:
    import pandas as pd

    frame = pd.read_parquet(path, columns=["metadata"])
    candidate_ids = []
    for metadata in frame["metadata"]:
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        if not isinstance(metadata, dict) or not metadata.get("candidate_id"):
            raise ValueError(f"parquet row is missing metadata.candidate_id: {path}")
        candidate_ids.append(str(metadata["candidate_id"]))
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"parquet contains duplicate candidate ids: {path}")
    return candidate_ids


def training_data_reference(data_files: str | list[str]) -> dict[str, Any]:
    paths = [data_files] if isinstance(data_files, str) else list(data_files)
    return {"files": [_file_contract(path) for path in paths]}


def write_training_data_reference(checkpoint_dir: str | Path, data_files: str | list[str]) -> Path:
    destination = Path(checkpoint_dir) / "training_data_ref.json"
    atomic_write_json(destination, training_data_reference(data_files))
    return destination


def validate_training_data_reference(checkpoint_dir: str | Path, data_files: str | list[str]) -> None:
    reference_path = Path(checkpoint_dir) / "training_data_ref.json"
    if not reference_path.exists():
        raise FileNotFoundError(f"checkpoint is missing training_data_ref.json: {checkpoint_dir}")
    with reference_path.open(encoding="utf-8") as handle:
        actual = json.load(handle)
    if actual != training_data_reference(data_files):
        raise ValueError("checkpoint training-data receipt does not match configured train files")


def checkpoint_state_reference(state: IterationState, state_path: str | Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "iteration": state.iteration,
        "state_path": str(Path(state_path).resolve()),
        "dynamic_state_hash": dynamic_state_hash(state),
        "skills_hash": canonical_hash([item.model_dump(mode="json") for item in state.skills]),
        "rubrics_hash": canonical_hash([item.model_dump(mode="json") for item in state.rubrics]),
        "models": {
            "proposer": state.models.proposer,
            "solver_before": state.models.solver_before,
        },
        "config_hash": canonical_hash(state.config_snapshot),
    }


def write_checkpoint_state_reference(checkpoint_dir: str | Path, state_path: str | Path) -> Path:
    state = StateStore(state_path).load()
    destination = Path(checkpoint_dir) / "iteration_state_ref.json"
    atomic_write_json(destination, checkpoint_state_reference(state, state_path))
    return destination


def validate_checkpoint_state_reference(checkpoint_dir: str | Path, state_path: str | Path) -> None:
    reference_path = Path(checkpoint_dir) / "iteration_state_ref.json"
    if not reference_path.exists():
        raise FileNotFoundError(f"checkpoint is missing iteration_state_ref.json: {checkpoint_dir}")
    with reference_path.open(encoding="utf-8") as handle:
        actual = json.load(handle)
    state = StateStore(state_path).load()
    expected = checkpoint_state_reference(state, state_path)
    for key in ("iteration", "dynamic_state_hash", "skills_hash", "rubrics_hash", "models", "config_hash"):
        if actual.get(key) != expected[key]:
            raise ValueError(f"checkpoint iteration state mismatch for {key}")


class IterationOrchestrator:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        raw_config = OmegaConf.load(self.config_path)
        self.config = OmegaConf.to_container(raw_config, resolve=True)
        if not isinstance(self.config, dict):
            raise ValueError("orchestrator config must be a mapping")
        state_path = self.config.get("state_path")
        if not state_path:
            raise ValueError("orchestrator config requires state_path")
        self.store = StateStore(state_path)
        self.state_path = Path(state_path)
        self.run_dir = Path(self.config.get("run_dir") or self.state_path.parent)
        self.working_directory = Path(self.config.get("working_directory") or os.getcwd())
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _stage_config(self, stage: str) -> dict[str, Any]:
        stages = self.config.get("stages") or {}
        config = stages.get(stage)
        if config is None:
            if stage == "finalize":
                return {"command": None, "artifacts": []}
            raise ValueError(f"orchestrator config is missing required stage: {stage}")
        if not isinstance(config, dict):
            raise ValueError(f"stage config must be a mapping: {stage}")
        return config

    @staticmethod
    def _command(config: dict[str, Any]) -> list[str] | None:
        command = config.get("command")
        if command is None:
            return None
        if isinstance(command, str):
            return shlex.split(command)
        if isinstance(command, list) and command and all(isinstance(item, str) for item in command):
            return command
        raise ValueError("stage command must be a non-empty string/list or null")

    def _completed_stage_is_valid(self, stage: str, record: StageRecord) -> bool:
        if record.status != "completed" or not record.manifest_path:
            return False
        manifest_path = Path(record.manifest_path)
        if not manifest_path.exists():
            return False
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        try:
            current_artifacts = _artifact_snapshot([item["path"] for item in manifest.get("artifacts", [])])
        except (FileNotFoundError, KeyError):
            return False
        config = self._stage_config(stage)
        command = self._command(config)
        valid = (
            manifest.get("stage") == stage
            and manifest.get("status") == "completed"
            and manifest.get("command") == command
            and manifest.get("command_inputs")
            == _command_file_contracts(command, self.working_directory)
            and manifest.get("stage_config_hash") == canonical_hash(config)
            and manifest.get("state_hash") == dynamic_state_hash(self.store.load())
            and manifest.get("artifacts") == current_artifacts
        )
        if not valid:
            return False
        if stage == "solver_train":
            try:
                input_contract = self._solver_input_contract(
                    config,
                    self._command(config),
                )
                receipt = self._solver_checkpoint_receipt(config, input_contract)
                return (
                    manifest.get("input_contract") == input_contract
                    and manifest.get("checkpoint_receipt") == receipt
                )
            except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
                return False
        if stage == "finalize":
            next_state_path = manifest.get("next_state_path")
            next_state_hash = manifest.get("next_state_hash")
            if not next_state_path or not Path(next_state_path).exists():
                return False
            return dynamic_state_hash(StateStore(next_state_path).load()) == next_state_hash
        return True

    def _solver_input_contract(
        self,
        config: dict[str, Any],
        command: list[str] | None,
    ) -> dict[str, Any]:
        required = {"train_data_path", "keepout_path", "split_manifest_path"}
        missing = sorted(required - config.keys())
        if missing:
            raise ValueError(f"solver_train stage is missing input contract fields: {missing}")
        if command is None:
            raise ValueError("solver_train requires a command")
        train_path = str(Path(config["train_data_path"]).resolve())
        keepout_path = str(Path(config["keepout_path"]).resolve())
        if train_path == keepout_path:
            raise ValueError("solver train and keepout paths must be different")
        command_text = " ".join(command)
        if train_path not in command_text and str(config["train_data_path"]) not in command_text:
            raise ValueError("solver command does not reference the declared train_data_path")
        if keepout_path in command_text or str(config["keepout_path"]) in command_text:
            raise ValueError("solver command must not reference the keepout path")
        with Path(config["split_manifest_path"]).open(encoding="utf-8") as handle:
            split_manifest = json.load(handle)
        train_ids = set(split_manifest.get("train_candidate_ids") or [])
        keepout_ids = set(split_manifest.get("keepout_candidate_ids") or [])
        if not train_ids or not keepout_ids:
            raise ValueError("split manifest must contain non-empty train and keepout candidate ids")
        if train_ids & keepout_ids:
            raise ValueError("split manifest leaks candidate ids across train and keepout")
        parquet_train_ids = set(_parquet_candidate_ids(config["train_data_path"]))
        parquet_keepout_ids = set(_parquet_candidate_ids(config["keepout_path"]))
        if parquet_train_ids != train_ids:
            raise ValueError("solver train parquet candidate ids do not match the split manifest")
        if parquet_keepout_ids != keepout_ids:
            raise ValueError("keepout parquet candidate ids do not match the split manifest")
        return {
            "train_data": _file_contract(config["train_data_path"]),
            "keepout_data": _file_contract(config["keepout_path"]),
            "split_manifest": _file_contract(config["split_manifest_path"]),
            "train_candidate_count": len(train_ids),
            "keepout_candidate_count": len(keepout_ids),
        }

    @staticmethod
    def _solver_checkpoint_receipt(config: dict[str, Any], input_contract: dict[str, Any]) -> dict[str, Any]:
        checkpoint_dir = config.get("checkpoint_dir")
        if not checkpoint_dir:
            raise ValueError("solver_train stage must declare checkpoint_dir")
        receipt_path = Path(checkpoint_dir) / "training_data_ref.json"
        if not receipt_path.exists():
            raise FileNotFoundError(f"solver checkpoint is missing training_data_ref.json: {checkpoint_dir}")
        with receipt_path.open(encoding="utf-8") as handle:
            receipt = json.load(handle)
        expected = {"files": [input_contract["train_data"]]}
        if receipt != expected:
            raise ValueError("solver checkpoint training-data receipt does not match the declared train parquet")
        return {"path": str(receipt_path.resolve()), "content": receipt}

    def _finalize_iteration(self, state: IterationState, config: dict[str, Any]) -> dict[str, str]:
        required = {"next_state_path", "skills_path", "rubrics_path", "proposer_after", "solver_after"}
        missing = sorted(required - config.keys())
        if missing:
            raise ValueError(f"finalize stage is missing configuration: {missing}")
        with Path(config["skills_path"]).open(encoding="utf-8") as handle:
            skills_payload = json.load(handle)
        with Path(config["rubrics_path"]).open(encoding="utf-8") as handle:
            rubrics_payload = json.load(handle)
        skill_items = skills_payload.get("skills", []) if isinstance(skills_payload, dict) else skills_payload
        rubric_items = rubrics_payload.get("rubrics", []) if isinstance(rubrics_payload, dict) else rubrics_payload
        next_skills = [Skill.model_validate(item) for item in skill_items]
        next_rubrics = [Rubric.model_validate(item) for item in rubric_items]
        if state.models.solver_after and state.models.solver_after != str(config["solver_after"]):
            raise ValueError("finalize solver_after does not match the converted solver recorded in state")
        state.models.solver_after = str(config["solver_after"])
        next_state = IterationState(
            iteration=state.iteration + 1,
            models=ModelReferences(
                proposer=str(config["proposer_after"]),
                solver_before=str(config["solver_after"]),
            ),
            skills=next_skills,
            rubrics=next_rubrics,
            config_snapshot=state.config_snapshot,
            artifacts={"previous_iteration_state": str(self.state_path.resolve())},
        )
        next_store = StateStore(config["next_state_path"])
        if next_store.path.exists():
            existing = next_store.load()
            if dynamic_state_hash(existing) != dynamic_state_hash(next_state):
                raise ValueError("existing next iteration state does not match finalized dynamic state")
        else:
            next_store.save(next_state)
        state.artifacts["next_iteration_state"] = str(next_store.path.resolve())
        return {
            "next_state_path": str(next_store.path.resolve()),
            "next_state_hash": dynamic_state_hash(next_store.load()),
        }

    def _run_stage(self, state: IterationState, stage: str) -> None:
        config = self._stage_config(stage)
        command = self._command(config)
        artifacts = [str(Path(path)) for path in config.get("artifacts", [])]
        stage_dir = self.run_dir / "stages" / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        previous = state.stages.get(stage)
        attempt = (previous.attempt if previous else 0) + 1
        log_path = stage_dir / f"attempt-{attempt}.log"
        manifest_path = stage_dir / f"manifest-attempt-{attempt}.json"
        record = StageRecord(
            status="running",
            attempt=attempt,
            started_at=utc_now(),
            manifest_path=str(manifest_path),
            artifact_paths=artifacts,
        )
        state.stages[stage] = record
        state.status = "running"
        self.store.save(state)

        return_code = 0
        error: str | None = None
        stage_details: dict[str, Any] = {}
        try:
            if stage == "solver_train":
                stage_details["input_contract"] = self._solver_input_contract(config, command)
            if command is not None:
                environment = os.environ.copy()
                environment["DRZERO_ITERATION_STATE"] = str(self.state_path.resolve())
                environment["DRZERO_ITERATION_PHASE"] = stage
                environment["DRZERO_SOLVER_BEFORE"] = state.models.solver_before
                if state.models.solver_after:
                    environment["DRZERO_SOLVER_AFTER"] = state.models.solver_after
                with log_path.open("w", encoding="utf-8") as log:
                    completed = subprocess.run(
                        command,
                        cwd=self.working_directory,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                return_code = completed.returncode
                if return_code:
                    raise RuntimeError(f"stage command exited with code {return_code}")
            if stage == "solver_train":
                stage_details["checkpoint_receipt"] = self._solver_checkpoint_receipt(
                    config,
                    stage_details["input_contract"],
                )
            elif stage == "finalize":
                stage_details.update(self._finalize_iteration(state, config))
            elif stage == "convert_solver":
                solver_after = config.get("solver_after")
                if not solver_after:
                    raise ValueError("convert_solver stage must declare solver_after")
                state.models.solver_after = str(solver_after)
            artifact_manifest = _artifact_snapshot(artifacts)
            completed_at = utc_now()
            manifest = {
                "schema_version": 1,
                "iteration": state.iteration,
                "stage": stage,
                "attempt": attempt,
                "command": command,
                "command_inputs": _command_file_contracts(command, self.working_directory),
                "stage_config_hash": canonical_hash(config),
                "state_hash": dynamic_state_hash(state),
                "started_at": record.started_at,
                "completed_at": completed_at,
                "return_code": return_code,
                "log_path": str(log_path) if command is not None else None,
                "artifacts": artifact_manifest,
                "status": "completed",
                **stage_details,
            }
            atomic_write_json(manifest_path, manifest)
            record.status = "completed"
            record.completed_at = completed_at
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[orchestrator] {error}\n")
            record.status = "failed"
            record.completed_at = utc_now()
            record.error = error
            state.status = "failed"
            atomic_write_json(
                manifest_path,
                {
                    "schema_version": 1,
                    "iteration": state.iteration,
                    "stage": stage,
                    "attempt": attempt,
                    "command": command,
                    "command_inputs": _command_file_contracts(command, self.working_directory),
                    "stage_config_hash": canonical_hash(config),
                    "state_hash": dynamic_state_hash(state),
                    "started_at": record.started_at,
                    "completed_at": record.completed_at,
                    "return_code": return_code,
                    "log_path": str(log_path) if command is not None else None,
                    "artifacts": [],
                    "status": "failed",
                    "error": error,
                    **stage_details,
                },
            )
        finally:
            state.stages[stage] = record
            self.store.save(state)
        if error:
            raise RuntimeError(f"iteration stage {stage} failed: {error}")

    def run(self) -> IterationState:
        with IterationLock(self.state_path):
            state = self.store.load()
            if state.status == "completed":
                for stage in STAGE_ORDER:
                    record = state.stages.get(stage)
                    if not record or not self._completed_stage_is_valid(stage, record):
                        raise RuntimeError(f"completed iteration has an invalid stage manifest: {stage}")
                return state
            for stage in STAGE_ORDER:
                existing = state.stages.get(stage)
                if existing and existing.status == "completed":
                    if not self._completed_stage_is_valid(stage, existing):
                        raise RuntimeError(f"completed stage has invalid or missing artifacts: {stage}")
                    continue
                self._run_stage(state, stage)
            state.status = "completed"
            self.store.save(state)
            return state
