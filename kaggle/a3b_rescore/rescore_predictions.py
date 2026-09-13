"""Re-score a trained checkpoint to add per-class top-5 confidences.

This directory started out A3b-specific but is no longer: it now supports
any run built on the project's two standard architectures (ConvNeXt-Tiny via
`--arch convnext_tiny`, the default, and ResNet50 via `--arch resnet50`), so
that different models can be pushed through the *identical* re-score /
recalibrate pipeline and compared like for like. The A3b accuracy-phase run
(`kaggle/accuracy_phase1_a3b/`) is what motivated it: its `test_predictions.csv`
predates the predictions-CSV contract documented in `docs/4_next_steps.md`,
recording `top_5` as pipe-separated class *labels* only, with no per-class
confidences. `scripts/recalibrate_decision_layer.py` needs `top_5_confidence`
to derive `top_1_confidence`, `top_2_confidence`, and the top1-top2 margin the
decision policy depends on, and that information cannot be recovered from the
existing CSV.

This script **trains nothing**. It loads an already-trained checkpoint,
reconstructs the exact model architecture and eval preprocessing for
`--arch`, re-scores a split, and writes a new CSV in the documented schema --
leaving the original `<split>_predictions.csv` untouched (it is an immutable
run record; see `docs/0_coding_standards.md`, "kaggle/*/ holds immutable run
records").

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
self-check: it compares the top-1/top-5 accuracy it achieves against a
recorded value and exits non-zero if they disagree by more than 0.05
percentage points. That recorded value comes from, in priority order: (1)
`<split>_metrics.csv` in `--results-dir`, when present -- this always wins,
even if `--expected-top1`/`--expected-top5` are also given; (2)
`--expected-top1`/`--expected-top5`, for run directories with no metrics CSV
of their own (e.g. a checkpoint whose published figures live elsewhere, such
as hardcoded constants in another script). If neither is available, the
script says so explicitly and skips the check rather than silently treating
it as passed. The output CSV is written to a temporary file first and only
atomically renamed into place after the self-check passes (or is skipped), so
a mismatch leaves no complete-looking output file at all -- not a
partially-written one, and not one left over from before the self-check ran.

Usage:
    python rescore_predictions.py \\
        --results-dir results/accuracy_phase1/a3b_convnext_tiny_continued_224 \\
        --data-dir /path/to/food-101 \\
        --split test

    python rescore_predictions.py \\
        --results-dir results/accuracy_phase1/champion_resnet50_ft_v2 \\
        --arch resnet50 \\
        --checkpoint app/artifacts/resnet50_ft_v2_best.pth \\
        --data-dir /path/to/food-101 \\
        --split test \\
        --expected-top1 78.28 --expected-top5 92.65
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
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


class InvalidTemperatureError(RuntimeError):
    """Raised when a temperature is non-finite or non-positive.

    Zero, negative, NaN and infinity are all rejected: they are divided
    directly into the logits before softmax, and zero/negative/NaN/infinity
    all silently corrupt the resulting confidences (division by zero,
    negative "confidences", or a NaN/inf propagating through every row)
    rather than raising anywhere near the point where the bad value was
    introduced.
    """


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
        InvalidTemperatureError: If the resolved temperature (from either
            source) is zero, negative, NaN, or infinite.
    """
    if explicit is not None:
        validate_temperature(explicit, "--temperature flag")
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
    temperature = float(calibration["temperature"])
    validate_temperature(temperature, str(calibration_path))
    return temperature, str(calibration_path)


def validate_temperature(temperature: float, source: str) -> None:
    """Reject a temperature that would silently corrupt confidences.

    `temperature` is divided directly into logits before softmax
    (`softmax(logits / temperature, dim=1)`). Zero raises a division error
    deep inside torch with no context about which run or flag caused it;
    negative values silently flip the ranking softmax is supposed to
    preserve; NaN or infinity propagate a NaN/degenerate distribution
    through every row without ever raising. Rejecting all four here, right
    where the value was resolved, names the source and the offending value
    instead.

    Args:
        temperature: The resolved temperature value.
        source: Where it came from (`"--temperature flag"` or a
            `calibration.json` path), for the error message.

    Raises:
        InvalidTemperatureError: If `temperature` is not finite or not
            strictly positive.
    """
    if not math.isfinite(temperature):
        raise InvalidTemperatureError(
            f"Invalid temperature {temperature!r} from {source}: must be a "
            "finite number (got NaN or infinity). Dividing logits by this "
            "would silently corrupt every row's confidences rather than "
            "raise anywhere near the source of the bad value."
        )
    if temperature <= 0:
        raise InvalidTemperatureError(
            f"Invalid temperature {temperature!r} from {source}: must be "
            "strictly positive. Zero would divide by zero; a negative value "
            "would silently invert the confidence ranking softmax is "
            "supposed to preserve."
        )


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


class InvalidExpectedMetricsError(ValueError):
    """Raised when only one of --expected-top1/--expected-top5 is given."""


def resolve_recorded_metrics(
    metrics_path: Path,
    expected_top1_pct: float | None,
    expected_top5_pct: float | None,
) -> tuple[tuple[float, float] | None, str | None]:
    """Resolve what to self-check achieved accuracy against, with a fallback.

    Priority order:
        1. `<split>_metrics.csv` at `metrics_path`, when present -- this
           always wins, even if `--expected-top1`/`--expected-top5` were
           also given, because it is the run's own recorded ground truth.
        2. `--expected-top1`/`--expected-top5`, for run directories that
           have no metrics CSV of their own (e.g. a production checkpoint
           whose published figures live elsewhere).
        3. Neither -- the caller must say so clearly and skip the check
           rather than silently treating it as passed.

    Args:
        metrics_path: Path to the run's `<split>_metrics.csv`.
        expected_top1_pct: `--expected-top1` value, or None.
        expected_top5_pct: `--expected-top5` value, or None.

    Returns:
        A `(recorded, source)` pair. `recorded` is `(top1, top5)` or None
        (nothing to compare against); `source` names where it came from, or
        is None alongside a `recorded` of None.

    Raises:
        InvalidExpectedMetricsError: If exactly one of `expected_top1_pct`/
            `expected_top5_pct` is given. Both or neither -- a lone value
            would silently compare only half of the self-check.
    """
    if (expected_top1_pct is None) != (expected_top5_pct is None):
        raise InvalidExpectedMetricsError(
            "--expected-top1 and --expected-top5 must be given together "
            f"(got expected_top1={expected_top1_pct!r}, expected_top5={expected_top5_pct!r}). "
            "A lone value would silently self-check only half the metric."
        )

    recorded = load_recorded_metrics(metrics_path)
    if recorded is not None:
        return recorded, str(metrics_path)

    if expected_top1_pct is not None and expected_top5_pct is not None:
        return (expected_top1_pct, expected_top5_pct), "--expected-top1/--expected-top5"

    return None, None


def check_accuracy_matches_recorded(
    achieved_top1_pct: float,
    achieved_top5_pct: float,
    recorded: tuple[float, float] | None,
) -> None:
    """Self-check: verify achieved accuracy matches the run's recorded metrics.

    Args:
        achieved_top1_pct: Top-1 accuracy this script achieved, as a percentage.
        achieved_top5_pct: Top-5 accuracy this script achieved, as a percentage.
        recorded: `(top_1_accuracy, top_5_accuracy)` to compare against --
            from `<split>_metrics.csv` or the `--expected-top1`/
            `--expected-top5` fallback (see `resolve_recorded_metrics`) --
            or None when neither is available (in which case the check is
            skipped -- there is nothing to compare against).

    Raises:
        AccuracyMismatchError: If either accuracy differs from the recorded
            value by more than `ACCURACY_TOLERANCE_PCT` percentage points. A
            mismatch means the preprocessing or class ordering is wrong, and
            these confidences are not this model's.
    """
    if recorded is None:
        print(
            "SELF-CHECK SKIPPED: no <split>_metrics.csv found in --results-dir "
            "and no --expected-top1/--expected-top5 given -- nothing to "
            "compare achieved accuracy against."
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


def write_predictions_if_accuracy_matches(
    predictions_df: pd.DataFrame,
    output_path: Path,
    achieved_top1_pct: float,
    achieved_top5_pct: float,
    recorded: tuple[float, float] | None,
) -> None:
    """Write `predictions_df` to `output_path` only if the self-check passes.

    Writes to a temporary file in `output_path`'s own directory first, runs
    `check_accuracy_matches_recorded`, and only then atomically renames the
    temp file into place with `os.replace` (atomic within a filesystem,
    which using the same directory guarantees). On a self-check failure the
    temp file is removed and `output_path` is left untouched -- no
    complete-looking artifact is left behind that belongs to the wrong
    model. Previously the CSV was written *before* the self-check, so a
    mismatch still left the file in place despite exiting non-zero.

    Args:
        predictions_df: The re-scored predictions to write.
        output_path: Final destination for the predictions CSV.
        achieved_top1_pct: Top-1 accuracy this run achieved, as a percentage.
        achieved_top5_pct: Top-5 accuracy this run achieved, as a percentage.
        recorded: `(top_1_accuracy, top_5_accuracy)` from `<split>_metrics.csv`,
            or None if no recorded metrics file exists.

    Raises:
        AccuracyMismatchError: If the self-check fails. `output_path` is
            guaranteed not to exist (or to be left as it was, if something
            already existed there before this call) when this is raised.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        predictions_df.to_csv(temp_path, index=False)
        check_accuracy_matches_recorded(achieved_top1_pct, achieved_top5_pct, recorded)
        os.replace(temp_path, output_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------
# Model construction and scoring (needs torch + the real checkpoint/images).
# --------------------------------------------------------------------------


SUPPORTED_ARCHS = ("convnext_tiny", "resnet50")


def make_classifier_head(num_classes: int, in_features: int) -> nn.Module:
    """The project-standard 3-layer MLP head, shared by both architectures.

    Identical to both `build_convnext_tiny()` in
    `kaggle/accuracy_phase1_a3b/foodlens_accuracy_phase1_a3b.py` and
    `make_classifier_head()` in `app/backend/inference.py`: only the
    backbone and the attribute it replaces (`classifier[2]` for ConvNeXt,
    `fc` for ResNet50) differ between architectures.
    """
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes),
    )


def build_model(num_classes: int = NUM_CLASSES, arch: str = "convnext_tiny") -> nn.Module:
    """Build an uninitialized model for `arch`.

    `arch="convnext_tiny"` mirrors `build_convnext_tiny()` in
    `kaggle/accuracy_phase1_a3b/foodlens_accuracy_phase1_a3b.py` exactly:
    `torchvision.models.convnext_tiny(weights=None)` with `classifier[2]`
    replaced by the project-standard 3-layer MLP head.

    `arch="resnet50"` mirrors `load_runtime()` in
    `app/backend/inference.py` exactly: `torchvision.models.resnet50(weights=None)`
    with `fc` replaced by the same 3-layer MLP head (only the backbone and
    the replaced attribute differ from the ConvNeXt path).

    Constructing anything else would make the re-scored confidences describe
    a different model.

    Raises:
        ValueError: If `arch` is not one of `SUPPORTED_ARCHS`.
    """
    if arch == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = make_classifier_head(num_classes, model.classifier[2].in_features)
        return model
    if arch == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = make_classifier_head(num_classes, model.fc.in_features)
        return model
    raise ValueError(f"Unsupported --arch {arch!r}; must be one of {SUPPORTED_ARCHS}")


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
        "--arch",
        default="convnext_tiny",
        choices=SUPPORTED_ARCHS,
        help=(
            "Model architecture to reconstruct before loading --checkpoint. "
            "'convnext_tiny' (default) mirrors the A3b accuracy-phase model; "
            "'resnet50' mirrors app/backend/inference.py's production "
            "champion. Both use the identical 3-layer MLP classifier head."
        ),
    )
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
    parser.add_argument(
        "--expected-top1",
        type=float,
        default=None,
        help=(
            "Fallback top-1 accuracy (percentage) to self-check achieved "
            "accuracy against, used only when --results-dir has no "
            "<split>_metrics.csv. If <split>_metrics.csv is present it always "
            "wins over this flag. Must be given together with --expected-top5; "
            "if neither a metrics CSV nor this flag pair is available, the "
            "self-check is skipped and the script says so explicitly."
        ),
    )
    parser.add_argument(
        "--expected-top5",
        type=float,
        default=None,
        help="Fallback top-5 accuracy (percentage); see --expected-top1.",
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

    model = build_model(len(class_names), arch=args.arch)
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

    top1_pct, top5_pct = accuracy_from_rows(rows)
    print(f"Achieved top-1 accuracy: {top1_pct:.4f}%")
    print(f"Achieved top-5 accuracy: {top5_pct:.4f}%")

    metrics_path = results_dir / f"{args.split}_metrics.csv"
    recorded, recorded_source = resolve_recorded_metrics(
        metrics_path, args.expected_top1, args.expected_top5
    )
    if recorded is not None:
        print(f"Self-check target: top-1={recorded[0]}%, top-5={recorded[1]}% (source: {recorded_source})")

    write_predictions_if_accuracy_matches(
        predictions_df,
        output_path,
        top1_pct,
        top5_pct,
        recorded,
    )
    print(f"Wrote {output_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return rescore(args)
    except (
        FileNotFoundError,
        ValueError,
        MissingTemperatureError,
        InvalidTemperatureError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except AccuracyMismatchError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
