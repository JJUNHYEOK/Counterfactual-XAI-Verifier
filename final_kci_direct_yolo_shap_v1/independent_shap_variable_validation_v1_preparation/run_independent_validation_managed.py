from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import secrets
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import psutil


REPOSITORY = Path(r"C:\Users\lab\Counterfactual-XAI-Verifier")
WORKTREE = REPOSITORY / ".k"
DIRECT = WORKTREE / "final_kci_direct_yolo_shap_v1"
PREPARATION = DIRECT / "independent_shap_variable_validation_v1_preparation"
OUTPUT = DIRECT / "independent_shap_variable_validation_run_001_lab"
WORKER = PREPARATION / "independent_validation_worker.py"
PYTHON = REPOSITORY / ".venv" / "Scripts" / "python.exe"
PROTECTED_BASELINE = PREPARATION / "protected_hashes_before.csv"
EXPECTED_ACCOUNT = r"desktop-cmoipge\lab"
STAGE_TIMEOUTS = {
    "preflight": 120,
    "module_import": 120,
    "matlab_startup": 300,
    "detector_startup": 180,
    "render": 300,
    "yolo_inference": 120,
    "scoring": 60,
    "condition": 420,
    "matlab_shutdown": 60,
    "finalization": 300,
}
OVERALL_TIMEOUT_SECONDS = 14400


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    write_json(temporary, value)
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def current_account() -> str:
    completed = subprocess.run(["whoami"], capture_output=True, text=True, timeout=10, check=True)
    return completed.stdout.strip().lower()


def matlab_snapshot() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "ppid", "name", "exe", "create_time", "cmdline"]):
        try:
            if str(process.info.get("name") or "").lower() != "matlab.exe":
                continue
            rows.append({
                "pid": process.info["pid"], "ppid": process.info.get("ppid"),
                "name": process.info.get("name"), "executable": process.info.get("exe"),
                "create_time": process.info.get("create_time"),
                "command_line": json.dumps(process.info.get("cmdline") or [], ensure_ascii=False),
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return sorted(rows, key=lambda row: int(row["pid"]))


def protected_snapshot(path: Path) -> tuple[list[dict[str, Any]], bool]:
    rows = []
    for item in read_csv(PROTECTED_BASELINE):
        target = Path(item["path"])
        expected = item["expected_sha256"].upper()
        actual = sha256(target) if target.is_file() else ""
        rows.append({
            "path": str(target), "expected_sha256": expected,
            "actual_sha256": actual, "exists": target.is_file(), "matches": actual == expected,
        })
    write_csv(path, rows, ["path", "expected_sha256", "actual_sha256", "exists", "matches"])
    return rows, len(rows) == 129 and all(row["matches"] for row in rows)


def next_attempt_dir() -> Path:
    root = OUTPUT / "supervisor_attempts"
    root.mkdir(parents=True, exist_ok=True)
    existing = [int(path.name.split("_")[-1]) for path in root.glob("attempt_[0-9][0-9][0-9]") if path.is_dir()]
    number = max(existing, default=0) + 1
    path = root / f"attempt_{number:03d}"
    path.mkdir(parents=False, exist_ok=False)
    return path


def load_events() -> list[dict[str, Any]]:
    path = OUTPUT / "stage_events.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def active_timeout() -> tuple[str, float, int] | None:
    events = load_events()
    if not events:
        return None
    last = events[-1]
    stage = str(last.get("stage", ""))
    if last.get("status") != "started" or stage not in STAGE_TIMEOUTS:
        return None
    started = datetime.fromisoformat(str(last["at_utc"])).astimezone(timezone.utc)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    return stage, elapsed, STAGE_TIMEOUTS[stage]


def child_identity(process: subprocess.Popen[str]) -> dict[str, Any]:
    observed = psutil.Process(process.pid)
    return {
        "pid": process.pid, "create_time": observed.create_time(),
        "executable": observed.exe(), "command_line": observed.cmdline(),
    }


def mock_cleanup_eligibility(*, direct_handle_owned: bool, launch: dict[str, Any], observed: dict[str, Any]) -> tuple[bool, str]:
    """Pure decision helper used only by MATLAB-free safety tests."""
    if not direct_handle_owned:
        return False, "no direct Popen handle"
    required = ("pid", "create_time", "executable", "command_line")
    if not all(key in launch and key in observed for key in required):
        return False, "identity information incomplete"
    if int(launch["pid"]) != int(observed["pid"]):
        return False, "PID mismatch"
    if abs(float(launch["create_time"]) - float(observed["create_time"])) > 0.1:
        return False, "creation-time mismatch"
    try:
        if Path(str(launch["executable"])).resolve() != Path(str(observed["executable"])).resolve():
            return False, "executable mismatch"
    except (OSError, ValueError):
        return False, "executable identity invalid"
    launch_command = [str(value).lower() for value in launch["command_line"]]
    observed_command = [str(value).lower() for value in observed["command_line"]]
    if launch_command != observed_command:
        return False, "command-line mismatch"
    if str(WORKER.resolve()).lower() not in observed_command:
        return False, "worker path fingerprint absent"
    return True, "direct Popen identity matched"


def identity_matches(process: subprocess.Popen[str], launch: dict[str, Any]) -> tuple[bool, str]:
    if process.pid != int(launch.get("pid", -1)):
        return False, "Popen PID mismatch"
    if not all(key in launch for key in ("create_time", "executable", "command_line")):
        return False, "launch identity incomplete"
    try:
        observed = psutil.Process(process.pid)
        if abs(observed.create_time() - float(launch["create_time"])) > 0.1:
            return False, "create time mismatch; possible PID reuse"
        if Path(observed.exe()).resolve() != PYTHON.resolve() or Path(observed.exe()).resolve() != Path(str(launch["executable"])).resolve():
            return False, "executable mismatch"
        command = [str(value).lower() for value in observed.cmdline()]
        if str(WORKER.resolve()).lower() not in command:
            return False, "worker path is absent from command line"
        if command != [str(value).lower() for value in launch["command_line"]]:
            return False, "command line changed"
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as exc:
        return False, f"identity unavailable: {type(exc).__name__}"
    return True, "direct Popen identity reverified"


def stop_direct_worker(process: subprocess.Popen[str], launch: dict[str, Any]) -> dict[str, Any]:
    matched, reason = identity_matches(process, launch)
    record = {
        "pid": process.pid, "identity_reverified": matched, "reason": reason,
        "terminate_called": False, "kill_called": False,
        "other_processes_targeted": False, "matlab_processes_targeted": False,
    }
    if not matched:
        return record
    process.terminate()
    record["terminate_called"] = True
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        matched_again, reason_again = identity_matches(process, launch)
        record["second_identity_reverified"] = matched_again
        record["second_reason"] = reason_again
        if matched_again:
            process.kill()
            process.wait(timeout=10)
            record["kill_called"] = True
    return record


def relay(stream: TextIO, path: Path) -> None:
    with path.open("w", encoding="utf-8", buffering=1) as handle:
        for line in iter(stream.readline, ""):
            handle.write(line)
            print(line, end="", flush=True)


def run(resume: bool) -> int:
    account = current_account()
    if account != EXPECTED_ACCOUNT:
        raise RuntimeError(f"account gate failed before MATLAB start: {account}")
    if os.environ.get("MATLAB_PREFDIR"):
        raise RuntimeError("MATLAB_PREFDIR is defined; default lab preferences would not be inherited")
    if resume:
        if not OUTPUT.is_dir():
            raise RuntimeError("resume was requested but the locked output folder does not exist")
        if (OUTPUT / "gate_results.json").is_file():
            raise RuntimeError("completed gate result already exists; resume is not allowed")
    else:
        if OUTPUT.exists():
            raise FileExistsError(f"one-shot result folder already exists: {OUTPUT}")
        OUTPUT.mkdir(parents=True, exist_ok=False)
    before = matlab_snapshot()
    if before:
        raise RuntimeError(f"pre-existing MATLAB process detected; nothing terminated: {[row['pid'] for row in before]}")

    attempt = next_attempt_dir()
    columns = ["pid", "ppid", "name", "executable", "create_time", "command_line"]
    write_csv(attempt / "processes_before.csv", before, columns)
    protected_before, protected_ok = protected_snapshot(attempt / "protected_hashes_before.csv")
    if not protected_ok:
        raise RuntimeError("protected-file preflight mismatch")

    child_env = dict(os.environ)
    child_env["DIRECT_SHAP_INDEPENDENT_VALIDATION_POPEN_TOKEN"] = secrets.token_hex(32)
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [str(PYTHON), str(WORKER)] + (["--resume"] if resume else [])
    process = subprocess.Popen(
        command, cwd=str(WORKTREE), env=child_env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="replace", bufsize=1,
    )
    assert process.stdout is not None and process.stderr is not None
    time.sleep(0.1)
    launch = child_identity(process)
    write_json(attempt / "supervisor_launch.json", {
        "schema_version": "1.0", "started_at_utc": utc_now(), "account": account,
        "command": command, "cwd": str(WORKTREE), "resume": resume,
        "direct_popen_identity": launch, "MATLAB_PREFDIR_present": False,
        "token_value_recorded": False, "automatic_retry": False,
        "overall_timeout_seconds": OVERALL_TIMEOUT_SECONDS,
        "stage_timeouts_seconds": STAGE_TIMEOUTS,
    })
    stdout_thread = threading.Thread(target=relay, args=(process.stdout, attempt / "python_stdout.log"), daemon=True)
    stderr_thread = threading.Thread(target=relay, args=(process.stderr, attempt / "python_stderr.log"), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    started = time.monotonic()
    timeout_record: dict[str, Any] | None = None
    while process.poll() is None:
        elapsed = time.monotonic() - started
        if elapsed > OVERALL_TIMEOUT_SECONDS:
            timeout_record = {
                "type": "overall", "limit_seconds": OVERALL_TIMEOUT_SECONDS,
                "elapsed_seconds": elapsed, "direct_worker_stop": stop_direct_worker(process, launch),
            }
            break
        active = active_timeout()
        if active is not None:
            stage, stage_elapsed, limit = active
            if stage_elapsed > limit:
                timeout_record = {
                    "type": "stage", "stage": stage, "limit_seconds": limit,
                    "elapsed_seconds": stage_elapsed, "direct_worker_stop": stop_direct_worker(process, launch),
                }
                break
        time.sleep(0.5)

    if timeout_record is not None:
        write_json(attempt / "supervisor_timeout.json", timeout_record)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)

    after = matlab_snapshot()
    deadline = time.monotonic() + 30.0
    while after and time.monotonic() < deadline:
        time.sleep(1.0)
        after = matlab_snapshot()
    write_csv(attempt / "processes_after.csv", after, columns)
    protected_after, protected_after_ok = protected_snapshot(attempt / "protected_hashes_after.csv")
    registry_path = OUTPUT / "completion_registry.json"
    registry = read_json(registry_path) if registry_path.is_file() else {}
    completed = int(registry.get("completed_logical_evaluations", 0))
    result_count = len(list((OUTPUT / "condition_results").glob("*.json"))) if (OUTPUT / "condition_results").is_dir() else 0
    manifest_count = len(list((OUTPUT / "condition_completion_manifests").glob("*.json"))) if (OUTPUT / "condition_completion_manifests").is_dir() else 0
    gate_path = OUTPUT / "gate_results.json"
    evidence_complete = completed == 110 and result_count == 110 and manifest_count == 110 and gate_path.is_file()
    technical_complete = process.poll() == 0 and timeout_record is None and evidence_complete and not after and protected_after_ok
    final = {
        "schema_version": "1.0", "completed_at_utc": utc_now(),
        "status": "COMPLETE" if technical_complete else "TECHNICAL_INCOMPLETE",
        "scientific_gate_status": read_json(gate_path).get("overall_status") if technical_complete else None,
        "process_exit_code": process.poll(), "timeout": timeout_record,
        "completed_conditions": completed, "result_files": result_count,
        "completion_manifests": manifest_count, "matlab_residual_count": len(after),
        "matlab_processes_terminated_by_supervisor": 0, "unrelated_processes_terminated": 0,
        "direct_worker_stop_only": timeout_record.get("direct_worker_stop") if timeout_record else None,
        "protected_file_count": len(protected_after), "protected_all_unchanged": protected_after_ok,
        "automatic_retry": False, "resume": resume,
        "RF_or_other_surrogate_used": False, "virtual_map_used": False,
        "boundary_search_executed": False,
    }
    write_json(attempt / "supervisor_attempt_status.json", final)
    write_json_atomic(OUTPUT / "supervisor_status_index.json", {
        "latest_attempt": attempt.name, "latest_attempt_status_path": str((attempt / "supervisor_attempt_status.json").resolve()),
        "status": final["status"], "updated_at_utc": utc_now(),
    })
    if technical_complete:
        write_json(OUTPUT / "supervisor_final_status.json", final)
    else:
        write_json(OUTPUT / f"supervisor_technical_incomplete_{attempt.name}.json", final)
    return 0 if technical_complete else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    return run(arguments.resume)


if __name__ == "__main__":
    raise SystemExit(main())
