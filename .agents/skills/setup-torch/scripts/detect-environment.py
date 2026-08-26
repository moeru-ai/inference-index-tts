#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
from collections.abc import Callable
from typing import TypedDict


class CommandResult(TypedDict):
    available: bool
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    error: str | None


class NvidiaInfo(TypedDict):
    available: bool
    detected: bool
    driver_version: str | None
    driver_cuda_max: str | None
    gpus: list[str]
    interpretation: str
    summary: CommandResult
    gpu_query: CommandResult


class EnvironmentInfo(TypedDict):
    os: str
    architecture: str
    python_version: str
    apple_silicon: bool
    candidate_backend: str
    verification_required: bool
    candidate_reason: str
    nvidia_smi: NvidiaInfo
    nvtop: CommandResult


CommandRunner = Callable[[str, list[str]], CommandResult]


def empty_command_result(
    *,
    available: bool,
    error: str | None = None,
) -> CommandResult:
    return {
        "available": available,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "timed_out": False,
        "error": error,
    }


def normalize_stream(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace").strip()
    return value.strip() if value is not None else ""


def run_command(executable: str, arguments: list[str]) -> CommandResult:
    if shutil.which(executable) is None:
        return empty_command_result(available=False)
    try:
        completed = subprocess.run(
            [executable, *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "available": True,
            "returncode": None,
            "stdout": normalize_stream(error.stdout),
            "stderr": normalize_stream(error.stderr),
            "timed_out": True,
            "error": f"command timed out after {error.timeout} seconds",
        }
    except OSError as error:
        return empty_command_result(available=True, error=str(error))
    return {
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "timed_out": False,
        "error": None,
    }


def label_value(label: str, text: str) -> str | None:
    match = re.search(rf"(?:^|[\s|]){re.escape(label)}:\s*([^\s|]+)", text, flags=re.IGNORECASE)
    return match.group(1) if match is not None else None


def reported_version(text: str, *, current_label: str, legacy_label: str) -> str | None:
    current = label_value(current_label, text)
    return current if current is not None else label_value(legacy_label, text)


def detect_environment(
    *,
    system: str | None = None,
    machine: str | None = None,
    python_version: str | None = None,
    command_runner: CommandRunner | None = None,
) -> EnvironmentInfo:
    detected_system = system or platform.system()
    detected_machine = machine or platform.machine()
    detected_python = python_version or platform.python_version()
    runner = command_runner or run_command

    nvidia_status = runner("nvidia-smi", [])
    summary_responded = nvidia_status["available"] and nvidia_status["returncode"] == 0
    if summary_responded:
        gpu_status = runner("nvidia-smi", ["--query-gpu=name", "--format=csv,noheader"])
    else:
        gpu_status = empty_command_result(
            available=nvidia_status["available"],
            error="not run because the nvidia-smi summary probe did not succeed",
        )

    if gpu_status["returncode"] == 0:
        gpus = [line.strip() for line in gpu_status["stdout"].splitlines() if line.strip()]
    else:
        gpus = []
    nvidia_detected = summary_responded and bool(gpus)

    if nvidia_detected:
        interpretation = (
            "The reported CUDA UMD/legacy CUDA value is the latest/maximum driver-supported CUDA level; "
            "it is not proof that a matching CUDA toolkit is installed. GPU enumeration succeeded, "
            "but Torch runtime verification is still required."
        )
    elif summary_responded:
        interpretation = (
            "Any reported CUDA UMD/legacy CUDA value is the latest/maximum driver-supported CUDA level; "
            "it is not proof that a matching CUDA toolkit is installed. nvidia-smi responded, but GPU enumeration "
            "is unconfirmed; this does not prove that no NVIDIA GPU exists. Check host passthrough, query support, "
            "and driver installation."
        )
    else:
        interpretation = (
            "nvidia-smi was unavailable or failed; this does not prove that no NVIDIA GPU exists. "
            "Check host passthrough, PATH, and driver installation."
        )

    normalized_system = detected_system.casefold()
    normalized_machine = detected_machine.casefold()
    apple_silicon = normalized_system == "darwin" and normalized_machine in {"arm64", "aarch64"}
    if nvidia_detected:
        backend = "cuda"
        candidate_reason = (
            "nvidia-smi enumerated at least one NVIDIA GPU; verify the CUDA runtime through Torch after installation."
        )
    elif apple_silicon:
        backend = "mps"
        candidate_reason = "Apple Silicon was detected; verify the MPS runtime through Torch after installation."
    else:
        backend = "cpu"
        candidate_reason = (
            "CPU is the safe fallback because no accelerator was confirmed; "
            "verify the Torch runtime after installation."
        )

    return {
        "os": detected_system,
        "architecture": detected_machine,
        "python_version": detected_python,
        "apple_silicon": apple_silicon,
        "candidate_backend": backend,
        "verification_required": True,
        "candidate_reason": candidate_reason,
        "nvidia_smi": {
            "available": nvidia_status["available"],
            "detected": nvidia_detected,
            "driver_version": reported_version(
                nvidia_status["stdout"],
                current_label="KMD Version",
                legacy_label="Driver Version",
            ),
            "driver_cuda_max": reported_version(
                nvidia_status["stdout"],
                current_label="CUDA UMD Version",
                legacy_label="CUDA Version",
            ),
            "gpus": gpus,
            "interpretation": interpretation,
            "summary": nvidia_status,
            "gpu_query": gpu_status,
        },
        "nvtop": runner("nvtop", ["--version"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect host information relevant to PyTorch setup.")
    parser.add_argument("--json", action="store_true", help="Emit formatted JSON")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = detect_environment()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
