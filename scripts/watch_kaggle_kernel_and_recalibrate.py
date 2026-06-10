#!/usr/bin/env python3
"""Watch a Kaggle kernel, download outputs, and run decision-layer recalibration."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
import shutil
import os


DONE_STATES = {
    "completed",
    "complete",
    "ready",
    "finished",
    "stopped",
}

FAILED_STATES = {
    "error",
    "failed",
    "aborted",
    "timeout",
}

CANCEL_STATES = {
    "cancelled",
    "canceled",
    "cancel_acknowledged",
    "cancel_acknowledge",
    "cancel",
}

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_RECALIBRATION_SCRIPT = (
    REPO_ROOT / "kaggle" / "accuracy_phase1" / "recalibrate_decision_layer.py"
)


@dataclass(frozen=True)
class PollResult:
    status: str
    raw: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Poll Kaggle kernel status, download outputs on completion, and rerun "
            "decision-layer recalibration for a configured split."
        )
    )
    parser.add_argument(
        "--kernel-slug",
        required=True,
        help="Kernel slug, e.g. tuannm3823/foodlens-a4-convnext-tiny-ft-hr-320",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Result folder name used by the Kaggle notebook, e.g. a4_convnext_tiny_full_finetune_320",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("val", "test"),
        help="Split to use for recalibration.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=60,
        help="Seconds between Kaggle status checks.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Maximum status checks; 0 means unlimited.",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/kaggle_a4_check",
        help="Destination directory for kernel output download.",
    )
    parser.add_argument(
        "--recalibration-script",
        default=str(DEFAULT_RECALIBRATION_SCRIPT),
        help="Decision-layer recalibration script path.",
    )
    parser.add_argument(
        "--python-exec",
        default=None,
        help="Python executable for recalibration (defaults to sys.executable).",
    )
    parser.add_argument(
        "--hard-classes-file",
        default=None,
        help="Optional hard-classes file path passed to recalibration.",
    )
    parser.add_argument(
        "--confusion-pairs-file",
        default=None,
        help="Optional confusion pairs file path passed to recalibration.",
    )
    parser.add_argument(
        "--class-report-file",
        default=None,
        help="Optional class report file path passed to recalibration.",
    )
    parser.add_argument(
        "--recalibration-output-dir",
        default=None,
        help=(
            "Override recalibration output directory. Defaults to <results_dir>/<split>_decision_layer."
        ),
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Pass --no-zip to recalibration script.",
    )
    parser.add_argument(
        "--continue-on-cancel",
        action="store_true",
        help=(
            "Keep polling when status is cancelled so a rerun can be picked up "
            "automatically."
        ),
    )
    parser.add_argument(
        "--max-confusion-pairs",
        type=int,
        default=40,
        help="max_confusion_pairs passed to recalibration script.",
    )
    return parser.parse_args()


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
        "Kaggle CLI not found. Install and authenticate the `kaggle` command first."
    )


def run_command(cmd: Iterable[str]) -> str:
    completed = subprocess.run(
        list(cmd),
        capture_output=True,
        text=True,
        check=False,
    )
    output = (completed.stdout + "\n" + completed.stderr).strip()
    return output


def parse_status(raw_output: str) -> str:
    raw = raw_output.strip()
    if not raw:
        return "unknown"

    try:
        payload = json.loads(raw)
        status = payload.get("status") or payload.get("kernelStatus") or payload.get("state")
        if isinstance(status, str):
            normalized = status.strip().lower()
            if normalized.startswith("kernelworkerstatus."):
                normalized = normalized.split(".", 1)[1]
            return normalized.replace("-", "_")
    except (TypeError, ValueError):
        pass

    lowered = raw.lower()
    for token in (
        "completed",
        "complete",
        "ready",
        "running",
        "queued",
        "error",
        "failed",
        "aborted",
        "preparing",
        "cancelled",
        "canceled",
        "cancel_acknowledged",
    ):
        if token in lowered:
            return token

    return "unknown"


def check_status(kernel_slug: str) -> PollResult:
    raw = run_command([resolve_kaggle_command(), "kernels", "status", kernel_slug])
    return PollResult(parse_status(raw), raw)


def status_is_done(status: str) -> bool:
    return status in DONE_STATES


def status_is_failed(status: str) -> bool:
    return status in FAILED_STATES


def status_is_cancelled(status: str) -> bool:
    return status in CANCEL_STATES


def download_kernel_outputs(kernel_slug: str, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading kernel output to: {output_dir}")
    return run_command([resolve_kaggle_command(), "kernels", "output", kernel_slug, "-p", str(output_dir)])


def expected_run_dir(output_dir: Path, run_id: str, split: str) -> Optional[Path]:
    candidates = [
        output_dir / "results" / "accuracy_phase1" / run_id,
        output_dir / "results" / run_id,
        output_dir / run_id,
    ]
    for candidate in candidates:
        if (candidate / f"{split}_predictions.csv").exists():
            return candidate

    for predictions_path in output_dir.rglob(f"{split}_predictions.csv"):
        if predictions_path.parent.name == run_id:
            return predictions_path.parent

    return None


def build_recalibration_args(args: argparse.Namespace, results_dir: Path) -> list[str]:
    import sys

    python_exec = args.python_exec or sys.executable
    command: list[str] = [
        python_exec,
        str(Path(args.recalibration_script).resolve()),
        "--results-dir",
        str(results_dir),
        "--split",
        args.split,
        "--max-confusion-pairs",
        str(args.max_confusion_pairs),
    ]

    if args.recalibration_output_dir:
        command.extend(["--output-dir", args.recalibration_output_dir])
    if args.hard_classes_file:
        command.extend(["--hard-classes-file", args.hard_classes_file])
    if args.confusion_pairs_file:
        command.extend(["--confusion-pairs-file", args.confusion_pairs_file])
    if args.class_report_file:
        command.extend(["--class-report-file", args.class_report_file])
    if args.no_zip:
        command.append("--no-zip")
    return command


def validate_kaggle_prerequisites() -> None:
    _ = resolve_kaggle_command()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()

    try:
        validate_kaggle_prerequisites()
    except FileNotFoundError as exc:
        print(exc)
        return 3

    attempt = 0
    while True:
        attempt += 1
        poll = check_status(args.kernel_slug)
        status = poll.status
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"[{now}] attempt {attempt}: status={status}")
        if poll.raw:
            print(poll.raw)

        if status_is_done(status):
            print(f"Kernel {args.kernel_slug} reached done state: {status}")
            download_output_message = download_kernel_outputs(
                args.kernel_slug,
                output_dir,
            )
            print(download_output_message)
            break

        if status_is_failed(status):
            print(f"Kernel {args.kernel_slug} failed with status: {status}")
            return 2

        if status_is_cancelled(status):
            if args.continue_on_cancel:
                print(
                    f"Kernel {args.kernel_slug} is cancelled (status={status}); "
                    "waiting for rerun."
                )
            else:
                print(f"Kernel {args.kernel_slug} failed with status: {status}")
                return 2
                # Continue only when continue_on_cancel is true.
            if args.max_attempts and attempt >= args.max_attempts:
                print(
                    f"Stopped at attempt {attempt}; max attempts reached ({args.max_attempts}) "
                    "while status was cancelled."
                )
                return 1
            print(f"Not complete yet. Sleeping for {args.interval_seconds}s...")
            time.sleep(args.interval_seconds)
            continue

        if args.max_attempts and attempt >= args.max_attempts:
            print(f"Stopped at attempt {attempt}; max attempts reached ({args.max_attempts})")
            return 1

        print(f"Not complete yet. Sleeping for {args.interval_seconds}s...")
        time.sleep(args.interval_seconds)

    run_dir = expected_run_dir(output_dir, args.run_id, args.split)
    if run_dir is None:
        print(
            "Could not find the expected run directory after download. "
            f"Searched under {output_dir} for {args.run_id}/{args.split}_predictions.csv"
        )
        return 2

    print(f"Using run directory: {run_dir}")
    recalibrate_cmd = build_recalibration_args(args, run_dir)
    print(f"Running recalibration: {' '.join(recalibrate_cmd)}")
    result = run_command(recalibrate_cmd)
    if result:
        print(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
