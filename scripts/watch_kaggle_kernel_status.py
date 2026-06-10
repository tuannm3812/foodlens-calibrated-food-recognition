#!/usr/bin/env python3
"""Watch a Kaggle kernel and wait until it completes.

Examples:
  python3 scripts/watch_kaggle_kernel_status.py tuannm3823/foodlens-a4-convnext-tiny-ft-hr-320
  python3 scripts/watch_kaggle_kernel_status.py <kernel_slug> --interval-seconds 120 --max-attempts 50
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import os
import shutil
from pathlib import Path
from typing import Any


RUNNING_STATES = {
    "running",
    "queued",
    "starting",
    "submitted",
    "preparing",
}

DONE_STATES = {"completed", "complete", "ready", "finished", "stopped"}
FAILED_STATES = {
    "error",
    "failed",
    "aborted",
    "timeout",
    "cancelled",
    "canceled",
    "cancel_acknowledged",
    "cancel_acknowledge",
    "cancel",
}


def resolve_kaggle_command() -> str:
    candidates = [
        os.getenv("KAGGLE_BIN"),
        shutil.which("kaggle"),
        str(Path.home() / ".local" / "bin" / "kaggle"),
        str(Path.home() / "Library" / "Python" / "3.9" / "bin" / "kaggle"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate

    raise FileNotFoundError(
        "Kaggle CLI not found. Install `kaggle` and authenticate before running this script."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Poll Kaggle kernel status and optionally download outputs when complete."
    )
    parser.add_argument("kernel_slug", help="Kernel slug, e.g. tuannm3823/foodlens-a4-convnext-tiny-ft-hr-320")
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=60,
        help="Sleep interval between checks.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Maximum status checks; 0 means unlimited.",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/kaggle_watch_output",
        help="Directory to place downloaded outputs.",
    )
    parser.add_argument(
        "--download-on-complete",
        action="store_true",
        help="Download kernel outputs when status is complete.",
    )
    return parser.parse_args()


def run_command(cmd: list[str]) -> str:
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout.strip()
    if not output:
        output = completed.stderr.strip()
    return output


def normalize_status_token(value: str) -> str:
    status = value.strip().lower()
    if status.startswith("kernelworkerstatus."):
        status = status.split(".", 1)[1]
    return status.replace("-", "_")


def parse_status(raw_output: str) -> str:
    raw = raw_output.strip()
    if not raw:
        return "unknown"

    try:
        parsed = json.loads(raw)
        status = parsed.get("status") or parsed.get("kernelStatus") or parsed.get("state")
        if isinstance(status, str):
            return normalize_status_token(status)
    except json.JSONDecodeError:
        pass

    lowered = raw.lower()
    # Kaggle output can vary; try to find stable status-like tokens.
    for token in (
        "completed",
        "complete",
        "ready",
        "running",
        "queued",
        "starting",
        "error",
        "failed",
        "aborted",
        "cancelled",
        "canceled",
        "cancel_acknowledged",
        "prepare",
    ):
        if token in lowered:
            return token
    return "unknown"


def normalize_status(status: str) -> str:
    value = status.strip().lower()
    return value


def is_running(status: str) -> bool:
    return normalize_status(status) in RUNNING_STATES or normalize_status(status) == "in_progress"


def is_done(status: str) -> bool:
    return normalize_status(status) in DONE_STATES


def is_failed(status: str) -> bool:
    return normalize_status(status) in FAILED_STATES


def wait_for_completion(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attempt = 0
    while True:
        attempt += 1
        kaggle_cmd = resolve_kaggle_command()
        status_raw = run_command([kaggle_cmd, "kernels", "status", args.kernel_slug])
        status = parse_status(status_raw)
        status_text = normalize_status(status)
        print(f"Attempt {attempt}: status={status_text or 'unknown'}")
        if status_raw:
            print(status_raw)

        if is_done(status_text):
            print(f"Kernel {args.kernel_slug} completed.")
            if args.download_on_complete:
                print(f"Downloading outputs to {output_dir}")
                out = run_command(
                    [
                        kaggle_cmd,
                        "kernels",
                        "output",
                        args.kernel_slug,
                        "-p",
                        str(output_dir),
                    ]
                )
                print(out)
            return 0

        if is_failed(status_text):
            print(f"Kernel {args.kernel_slug} failed with status: {status_text}")
            return 2

        if status_text == "unknown":
            print("Could not parse kernel status confidently.")

        if args.max_attempts and attempt >= args.max_attempts:
            print(f"Maximum attempts reached ({args.max_attempts}). Exiting.")
            return 1

        print(f"Not complete yet. Sleeping {args.interval_seconds}s...")
        time.sleep(args.interval_seconds)


def main() -> int:
    args = parse_args()
    try:
        return wait_for_completion(args)
    except FileNotFoundError:
        print("Error: Kaggle CLI not found. Install `kaggle` and authenticate before running this script.")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
