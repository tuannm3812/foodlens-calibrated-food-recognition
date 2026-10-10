"""Re-score the A3b ConvNeXt-Tiny checkpoint to add per-class top-5 confidences.

The A3b accuracy-phase run (`kaggle/accuracy_phase1_a3b/`) is the current
accuracy leader (83.90% test top-1, 95.78% test top-5) but its
`test_predictions.csv` predates the predictions-CSV contract documented in
`docs/4_next_steps.md`: it records `top_5` as pipe-separated class *labels*
only, with no per-class confidences. `scripts/recalibrate_decision_layer.py`
needs `top_5_confidence` to derive `top_1_confidence`, `top_2_confidence`, and
the top1-top2 margin the decision policy depends on, and that information
cannot be recovered from the existing CSV.

This script **trains nothing**. It loads the already-trained
`convnext_tiny_continued_best.pth` checkpoint, reconstructs the exact A3b
model architecture and eval preprocessing, re-scores a split, and writes a
new CSV in the documented schema -- leaving the original
`<split>_predictions.csv` untouched (it is an immutable run record; see
`docs/0_coding_standards.md`, "kaggle/*/ holds immutable run records").

Confidences are **temperature-scaled** to match what `app/backend/inference.py`
actually serves in production (`softmax(logits / temperature, dim=1)`, with
`temperature` read from `calibration.json`). `scripts/recalibrate_decision_layer.py`
does not apply temperature itself -- it fits decision thresholds directly on
whatever confidences the predictions CSV already contains -- so if this script
emitted raw, un-scaled softmax, those thresholds would be fitted on a
different distribution than the one the backend serves, and would be
systematically wrong once deployed. By default the temperature is read from
`calibration.json` in `--results-dir`; it can be overridden with
`--temperature` (pass `1.0` to deliberately opt out of scaling). Top-1/top-5
accuracy are unaffected by which temperature is used, because temperature
scaling is a monotonic rescaling of the logits, so argmax and rank order are
unchanged -- the self-check below still reproduces the run's recorded
metrics regardless of temperature.

Because a mismatched preprocessing pipeline or class ordering would silently
produce confidences for a *different* model, this script always ends with a
self-check: it compares the top-1/top-5 accuracy it achieves against the
`<split>_metrics.csv` recorded by the original training run (when present)
and exits non-zero if they disagree by more than 0.05 percentage points.

Usage:
    python rescore_predictions.py \\
        --results-dir results/accuracy_phase1/a3b_convnext_tiny_continued_224 \\
        --data-dir /path/to/food-101 \\
        --split test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

NUM_CLASSES = 101
IMAGE_SIZE = (224, 224)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
TOP_K = 5
ACCURACY_TOLERANCE_PCT = 0.05

REQUIRED_OUTPUT_COLUMNS = [
    "path",
    "true_label",
    "pred_label",
    "confidence",
    "is_correct",
    "top_5",
    "top_5_confidence",
    "temperature",
]

EVAL_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ]
)


class AccuracyMismatchError(RuntimeError):
    """Raised when achieved accuracy does not match the run's recorded metrics."""


class MissingTemperatureError(RuntimeError):
    """Raised when no temperature is available and none was requested explicitly."""


# --------------------------------------------------------------------------
# Pure helpers (no model, no filesystem I/O beyond what is passed in) --
# these are what tests/test_rescore_predictions.py exercises directly.
# --------------------------------------------------------------------------


def remap_manifest_path(kaggle_path: str, data_dir: Path) -> Path:
    """Remap a Kaggle-absolute manifest path to a local `--data-dir` path.

    Manifest paths look like
    `/kaggle/input/datasets/kmader/food41/images/<class>/<id>.jpg`. Only the
    last two segments (class directory name and filename) are meaningful
    once the dataset has been copied elsewhere, so this takes the last two
    POSIX segments of `kaggle_path` and joins them onto `data_dir`.

    Args:
        kaggle_path: An absolute path as recorded in `<split>_manifest.csv`.
        data_dir: Local root of the Food-101 images.

    Returns:
        The corresponding path under `data_dir`.

    Raises:
        ValueError: If `kaggle_path` has fewer than two path segments.
    """
    parts = PurePosixPath(kaggle_path).parts
    if len(parts) < 2:
        raise ValueError(
            f"Cannot remap manifest path with fewer than two segments: {kaggle_path!r}"
        )
    class_name, filename = parts[-2], parts[-1]
    return data_dir / class_name / filename


def find_missing_paths(paths: Sequence[Path]) -> list[Path]:
    """Return the subset of `paths` that do not exist on disk."""
    return [path for path in paths if not path.exists()]


def ensure_paths_exist(paths: Sequence[Path]) -> None:
    """Fail early and clearly if any resolved image path is missing.

    Args:
        paths: Resolved local image paths to check.

    Raises:
        FileNotFoundError: Naming how many files are missing and one example,
            rather than letting the dataset silently score a partial set (or
            crash mid-run on the first bad path deep inside a worker process).
    """
    missing = find_missing_paths(paths)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(paths)} image file(s) are missing on disk.\n"
            f"  example missing path: {missing[0]}\n"
            "Check that --data-dir (or FOODLENS_DATA_DIR) points at the Food-101 "
            "image root, i.e. a directory of `<class_name>/<id>.jpg` files, and "
            "that the dataset has been fully extracted there."
        )


def load_class_names(results_dir: Path) -> list[str]:
    """Load the run's class ordering, which the checkpoint's outputs are bound to.

    Args:
        results_dir: The run directory that should contain `class_names.json`.

    Returns:
        Class names in the exact order the checkpoint's output layer uses.

    Raises:
        FileNotFoundError: If `class_names.json` is absent. Class ordering
            must come from this file, never from sorting the manifest --
            the checkpoint's output indices are bound to the order recorded
            when the run was trained, and that order cannot be reconstructed
            after the fact.
    """
    path = results_dir / "class_names.json"
    if not path.exists():
        raise FileNotFoundError(
            f"class_names.json not found in {results_dir}. "
            "The checkpoint's output indices are bound to this run's original "
            "class ordering, which cannot be recovered by sorting the manifest -- "
            "restore or regenerate class_names.json before rescoring."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_temperature(results_dir: Path, explicit: float | None) -> tuple[float, str]:
    """Resolve the temperature to divide logits by before softmax.

    This must match `app/backend/inference.py`'s `softmax(logits / temperature,
    dim=1)` exactly, because `scripts/recalibrate_decision_layer.py` fits
    decision thresholds directly on whatever confidences this script emits --
    it does not apply temperature itself. A threshold fitted on raw softmax
    would be systematically wrong against the temperature-scaled confidences
    production actually serves.

    Args:
        results_dir: Run directory that should contain `calibration.json`.
        explicit: An explicit `--temperature` override, or None to read
            `calibration.json`.

    Returns:
        A `(temperature, source)` pair, where `source` is either
        `"--temperature flag"` or the path to `calibration.json`, for
        logging provenance.

    Raises:
        MissingTemperatureError: If `explicit` is None and `calibration.json`
            is absent or lacks a `temperature` key. This never silently
            falls back to 1.0 -- a silent 1.0 is exactly the bug this
            argument exists to fix.
    """
    if explicit is not None:
        return explicit, "--temperature flag"

    calibration_path = results_dir / "calibration.json"
    if not calibration_path.exists():
        raise MissingTemperatureError(
            f"calibration.json not found in {results_dir} and no --temperature "
            "was given. Confidences must be scaled by the same temperature "
            "app/backend/inference.py serves in production, so this refuses to "
            "silently default to 1.0. Pass --temperature explicitly (use 1.0 "
            "to deliberately opt out of scaling), or restore calibration.json."
        )

    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if "temperature" not in calibration:
        raise MissingTemperatureError(
            f"{calibration_path} has no 'temperature' key and no --temperature "
            "was given. Refusing to silently default to 1.0 -- pass "
            "--temperature explicitly (use 1.0 to deliberately opt out of "
            "scaling), or restore the key in calibration.json."
        )
    return float(calibration["temperature"]), str(calibration_path)


def format_topk_columns(top_labels: Sequence[str], top_scores: Sequence[float]) -> tuple[str, str]:
    """Format rank-aligned top-k labels and scores as pipe-separated strings.

    Args:
        top_labels: Class names, most confident first.
        top_scores: Per-class confidences, in the same order as `top_labels`.

    Returns:
        A `(top_5, top_5_confidence)` pair of pipe-separated strings, with
        `top_5[i]` and `top_5_confidence[i]` referring to the same class.

    Raises:
        ValueError: If `top_labels` and `top_scores` differ in length.
    """
    if len(top_labels) != len(top_scores):
        raise ValueError(
            f"top_labels ({len(top_labels)}) and top_scores ({len(top_scores)}) "
            "must be the same length"
        )
    top_5 = "|".join(top_labels)
    top_5_confidence = "|".join(f"{score:.8f}" for score in top_scores)
    return top_5, top_5_confidence


def build_prediction_row(
    path: str,
    true_label: str,
    top_labels: Sequence[str],
    top_scores: Sequence[float],
    temperature: float = 1.0,
) -> dict[str, object]:
    """Build one output row in the documented predictions-CSV schema.

    Args:
        path: The (original, un-remapped) manifest image path.
        true_label: Ground-truth class name.
        top_labels: Top-k predicted class names, most confident first.
        top_scores: Top-k per-class confidences, rank-aligned with `top_labels`.
        temperature: The temperature `top_scores` were already scaled by
            (i.e. computed as `softmax(logits / temperature)`), recorded
            per row so the CSV is self-describing about which scaling
            produced it.

    Returns:
        A dict with exactly the keys in `REQUIRED_OUTPUT_COLUMNS`.
    """
    top_5, top_5_confidence = format_topk_columns(top_labels, top_scores)
    pred_label = top_labels[0]
    return {
        "path": path,
        "true_label": true_label,
        "pred_label": pred_label,
        "confidence": float(top_scores[0]),
        "is_correct": pred_label == true_label,
        "top_5": top_5,
        "top_5_confidence": top_5_confidence,
        "temperature": float(temperature),
    }


def accuracy_from_rows(rows: Sequence[dict[str, object]]) -> tuple[float, float]:
    """Compute top-1 and top-5 accuracy (as percentages) from built rows."""
    if not rows:
        return 0.0, 0.0
    top1_hits = sum(1 for row in rows if row["is_correct"])
    top5_hits = sum(
        1 for row in rows if row["true_label"] in str(row["top_5"]).split("|")
    )
    total = len(rows)
    return 100.0 * top1_hits / total, 100.0 * top5_hits / total


def load_recorded_metrics(metrics_path: Path) -> tuple[float, float] | None:
    """Load `(top_1_accuracy, top_5_accuracy)` from a `<split>_metrics.csv`.

    Returns:
        The recorded percentages, or None if `metrics_path` does not exist.
    """
    if not metrics_path.exists():
        return None
    frame = pd.read_csv(metrics_path)
    row = frame.iloc[0]
    return float(row["top_1_accuracy"]), float(row["top_5_accuracy"])


def check_accuracy_matches_recorded(
    achieved_top1_pct: float,
    achieved_top5_pct: float,
    recorded: tuple[float, float] | None,
) -> None:
    """Self-check: verify achieved accuracy matches the run's recorded metrics.

    Args:
        achieved_top1_pct: Top-1 accuracy this script achieved, as a percentage.
        achieved_top5_pct: Top-5 accuracy this script achieved, as a percentage.
        recorded: `(top_1_accuracy, top_5_accuracy)` from `<split>_metrics.csv`,
            or None when no recorded metrics file exists (in which case the
            check is skipped -- there is nothing to compare against).

    Raises:
        AccuracyMismatchError: If either accuracy differs from the recorded
            value by more than `ACCURACY_TOLERANCE_PCT` percentage points. A
            mismatch means the preprocessing or class ordering is wrong, and
            these confidences are not this model's.
    """
    if recorded is None:
        print(
            "SELF-CHECK SKIPPED: no <split>_metrics.csv found to compare against."
        )
        return

    recorded_top1, recorded_top5 = recorded
    top1_diff = abs(achieved_top1_pct - recorded_top1)
    top5_diff = abs(achieved_top5_pct - recorded_top5)

    if top1_diff <= ACCURACY_TOLERANCE_PCT and top5_diff <= ACCURACY_TOLERANCE_PCT:
        print(
            "SELF-CHECK PASSED: achieved top-1="
            f"{achieved_top1_pct:.4f}% (recorded {recorded_top1:.4f}%), "
            f"top-5={achieved_top5_pct:.4f}% (recorded {recorded_top5:.4f}%), "
            f"within {ACCURACY_TOLERANCE_PCT} pp tolerance."
        )
        return

    raise AccuracyMismatchError(
        "SELF-CHECK FAILED: achieved accuracy does not match the run's recorded "
        f"metrics -- top-1={achieved_top1_pct:.4f}% vs recorded {recorded_top1:.4f}% "
        f"(diff {top1_diff:.4f} pp), top-5={achieved_top5_pct:.4f}% vs recorded "
        f"{recorded_top5:.4f}% (diff {top5_diff:.4f} pp), tolerance is "
        f"{ACCURACY_TOLERANCE_PCT} pp. A mismatch means the preprocessing or "
        "class ordering is wrong, and these confidences are not this model's."
    )


# --------------------------------------------------------------------------
# Model construction and scoring (needs torch + the real checkpoint/images).
# --------------------------------------------------------------------------


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Build the exact A3b ConvNeXt-Tiny architecture, uninitialized.

    Mirrors `build_convnext_tiny()` in
    `kaggle/accuracy_phase1_a3b/foodlens_accuracy_phase1_a3b.py` exactly:
    `torchvision.models.convnext_tiny(weights=None)` with `classifier[2]`
    replaced by a 3-layer MLP head. Constructing anything else would make
    the re-scored confidences describe a different model.
    """
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes),
    )
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load `checkpoint_path`'s state dict into `model` in place."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def detect_device(requested: str | None) -> torch.device:
    """Resolve the compute device: explicit choice, else cuda > mps > cpu."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class RescoreDataset(Dataset):
    """Loads images from resolved local paths and applies the eval transform."""

    def __init__(self, paths: Sequence[Path], transform: transforms.Compose) -> None:
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image)


def resolve_manifest_paths(manifest: pd.DataFrame, data_dir: Path | None) -> list[Path]:
    """Resolve manifest paths to local paths, remapping when `data_dir` is given."""
    if data_dir is not None:
        return [remap_manifest_path(path, data_dir) for path in manifest["path"]]
    return [Path(path) for path in manifest["path"]]


def resolve_data_dir(cli_data_dir: str | None) -> Path | None:
    """Resolve `--data-dir`, falling back to the `FOODLENS_DATA_DIR` env var."""
    candidate = cli_data_dir or os.environ.get("FOODLENS_DATA_DIR")
    if not candidate:
        return None
    return Path(candidate).expanduser().resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments, mirroring `scripts/recalibrate_decision_layer.py`'s style."""
    parser = argparse.ArgumentParser(
        description=(
            "Re-score the trained A3b ConvNeXt-Tiny checkpoint on a split, "
            "emitting per-class top-5 confidences in the documented "
            "predictions-CSV schema. Trains nothing."
        )
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Run directory holding the checkpoint, class_names.json and <split>_manifest.csv",
    )
    parser.add_argument("--split", default="test", help="Manifest split to re-score")
    parser.add_argument(
        "--checkpoint",
        default="convnext_tiny_continued_best.pth",
        help="Checkpoint filename (or path) within --results-dir",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Root of the local Food-101 images. Falls back to FOODLENS_DATA_DIR. "
            "If neither is set, manifest paths are used as-is (this will fail "
            "fast off Kaggle)."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default=None,
        help="Force a torch device (e.g. cpu, cuda, mps); default auto-detects cuda > mps > cpu",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output CSV path; defaults to <results-dir>/<split>_predictions_rescored.csv. "
            "Must not equal <split>_predictions.csv -- that file is an immutable run record."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Temperature to divide logits by before softmax, matching "
            "app/backend/inference.py's softmax(logits / temperature, dim=1). "
            "Defaults to the 'temperature' key in <results-dir>/calibration.json; "
            "pass 1.0 explicitly to deliberately opt out of scaling. There is "
            "no silent default -- if calibration.json is missing or lacks the "
            "key and this flag is not given, the script fails with a clear error."
        ),
    )
    return parser.parse_args(argv)


def rescore(args: argparse.Namespace) -> int:
    """Run the full re-score: load, score, write, self-check. Returns an exit code."""
    results_dir = Path(args.results_dir).expanduser().resolve()

    manifest_path = results_dir / f"{args.split}_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)

    original_predictions_path = results_dir / f"{args.split}_predictions.csv"
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else results_dir / f"{args.split}_predictions_rescored.csv"
    )
    if output_path == original_predictions_path.resolve():
        raise ValueError(
            f"--output must not overwrite the original run record {original_predictions_path}. "
            "Choose a different filename."
        )

    class_names = load_class_names(results_dir)

    temperature, temperature_source = resolve_temperature(results_dir, args.temperature)
    print(f"Temperature: {temperature} (source: {temperature_source})")

    data_dir = resolve_data_dir(args.data_dir)
    resolved_paths = resolve_manifest_paths(manifest, data_dir)
    ensure_paths_exist(resolved_paths)

    device = detect_device(args.device)
    print(f"Device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = results_dir / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(len(class_names))
    load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)
    model.eval()

    dataset = RescoreDataset(resolved_paths, EVAL_TRANSFORMS)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    rows: list[dict[str, object]] = []
    seen = 0
    total = len(manifest)
    print(f"Scoring {total:,} images from split={args.split!r}")

    with torch.no_grad():
        for batch in loader:
            images = batch.to(device, non_blocking=True)
            logits = model(images)
            probabilities = torch.softmax(logits / temperature, dim=1)
            top_scores, top_indices = torch.topk(probabilities, TOP_K, dim=1)

            for row_offset in range(images.size(0)):
                manifest_row = manifest.iloc[seen + row_offset]
                top_labels = [class_names[idx] for idx in top_indices[row_offset].tolist()]
                top_scores_list = top_scores[row_offset].tolist()
                rows.append(
                    build_prediction_row(
                        path=str(manifest_row["path"]),
                        true_label=str(manifest_row["label"]),
                        top_labels=top_labels,
                        top_scores=top_scores_list,
                        temperature=temperature,
                    )
                )
            seen += images.size(0)
            if seen % (args.batch_size * 10) < args.batch_size or seen == total:
                print(f"  scored {seen:,}/{total:,}")

    predictions_df = pd.DataFrame(rows, columns=REQUIRED_OUTPUT_COLUMNS)
    predictions_df.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")

    top1_pct, top5_pct = accuracy_from_rows(rows)
    print(f"Achieved top-1 accuracy: {top1_pct:.4f}%")
    print(f"Achieved top-5 accuracy: {top5_pct:.4f}%")

    metrics_path = results_dir / f"{args.split}_metrics.csv"
    recorded = load_recorded_metrics(metrics_path)
    check_accuracy_matches_recorded(top1_pct, top5_pct, recorded)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return rescore(args)
    except (FileNotFoundError, ValueError, MissingTemperatureError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except AccuracyMismatchError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
