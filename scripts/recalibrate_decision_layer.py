"""Recalibrate decision-layer thresholds for a finished classifier run.

This utility expects a run directory containing per-split prediction and report
artifacts from `foodlens_accuracy_phase1_a*` experiments, and emits a small
artifact bundle matching the app contract:

- decision_policy.json
- hard_classes.json
- confusion_pairs.json
- decision_policy.csv
- decision_policy_search.csv
- decision_band_metrics_fit.csv
- decision_band_metrics_eval.csv
- decision_examples_<band>.csv

The threshold grid search, hard classes and confusion pairs are derived from
the fit split (`--fit-split`, default `val`) alone; the selected policy is
then frozen and evaluated exactly once on the eval split (`--eval-split`,
default `test`). Both splits' band metrics are written, clearly labelled, so
the gap between them is visible rather than hidden -- selecting thresholds
and reporting them on the same split leaks that split's labels into policy
selection and produces optimistic numbers (see
docs/superpowers/specs/2026-09-14-decision-layer-methodology-design.md,
section 3.2). The single-split `--split` flag is kept as a deprecated alias
for "fit and evaluate on the same split" and prints a warning explaining why
that leaks.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# This script shares the production decision-routing rule with
# `app/backend/decision_rules.py` rather than vendoring its own copy -- that
# is the fix for the offline/production decision-function divergence this
# script used to have (see docs/superpowers/specs/
# 2026-09-14-decision-layer-methodology-design.md, section 3.1). The repo
# root must be on `sys.path` for that import to resolve when this script is
# invoked directly (its usual invocation is `python
# scripts/recalibrate_decision_layer.py ...` from the repo root, but Python
# does not add the repo root to `sys.path` in that case -- only the script's
# own directory).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from app.backend.decision_rules import route_decision
except ImportError as exc:  # pragma: no cover - exercised only outside the repo checkout
    raise SystemExit(
        "Could not import app.backend.decision_rules.route_decision.\n"
        "This script deliberately does not vendor a copy of the production "
        "decision-routing rule -- it shares app/backend/decision_rules.py so "
        "the offline policy stays executable at inference time. Run this "
        "script from within the foodlens-calibrated-food-recognition repo "
        "checkout (so the repo root containing `app/` is importable), rather "
        "than from a copied or partial checkout.\n"
        f"Original import error: {exc}"
    ) from exc

# Columns `build_features()` reads off the predictions frame, after
# `normalize_actual_col()` has already renamed the actual/predicted-label
# aliases (true_label/actual_label/label -> "actual",
# pred_label/predicted_label/prediction/class_name/predicted_class ->
# "predicted") and, if missing, derived `is_correct` from them. Anything still
# missing at this point cannot be recovered by this script.
REQUIRED_PREDICTION_COLUMNS = {
    "actual": "ground-truth label per row (accepts the aliases above)",
    "predicted": "model's top-1 predicted label (accepts the aliases above)",
    "is_correct": "whether predicted == actual; feeds decision-band accuracy metrics",
    "top_5": (
        "pipe-separated top-5 predicted labels, rank-aligned with "
        "top_5_confidence; used to check whether the true label is in the top-5"
    ),
    "top_5_confidence": (
        "pipe-separated per-class confidences for the top-5 labels, "
        "rank-aligned with top_5; used to derive top_1_confidence, "
        "top_2_confidence and the top_1_top_2_margin the decision policy "
        "depends on"
    ),
}


class PredictionSchemaError(ValueError):
    """Raised when a predictions CSV lacks a column build_features() needs."""


def validate_predictions_schema(predictions: pd.DataFrame, predictions_path: Path) -> None:
    """Fail fast, with an actionable message, before any analysis work runs.

    `build_features()` used to hit a bare `KeyError: 'top_5_confidence'` deep
    inside pandas -- after hard-class and confusion-pair analysis had already
    run -- when pointed at a run whose predictions predate this contract.
    """
    missing = [name for name in REQUIRED_PREDICTION_COLUMNS if name not in predictions.columns]
    if not missing:
        return

    found = ", ".join(str(col) for col in predictions.columns) or "(none)"
    raise PredictionSchemaError(
        "Predictions file is missing columns required by decision-layer "
        "recalibration.\n"
        f"  file: {predictions_path}\n"
        f"  missing columns: {', '.join(missing)}\n"
        f"  columns found: {found}\n"
        "Accuracy-phase training runs (kaggle/*) write `top_5` as "
        "pipe-separated labels only and do not record per-class confidences, "
        "which predates this script's documented predictions-CSV contract "
        "(see docs/4_next_steps.md). Regenerate this run's predictions with "
        "per-class top-5 confidences before recalibrating."
    )


AUTO_CONFIDENCE_GRID = tuple(np.round(np.arange(0.70, 0.96, 0.05), 2))
SUGGEST_CONFIDENCE_GRID = tuple(np.round(np.arange(0.35, 0.76, 0.05), 2))
MARGIN_GRID = tuple(np.round(np.arange(0.05, 0.51, 0.05), 2))
MIN_AUTO_ACCEPT_ACCURACY = 0.90
MIN_SUGGEST_ACCURACY = 0.80
ZIP_NAME = "decision_layer_artifacts.zip"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive decision-layer policy and export app-facing payloads "
            "from FoodLens phase-1 prediction outputs."
        )
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Run directory with prediction/report artifacts",
    )
    parser.add_argument(
        "--split",
        default=None,
        choices=("val", "test"),
        help=(
            "Deprecated alias for --fit-split and --eval-split both set to "
            "the same split -- i.e. 'fit thresholds and evaluate them on the "
            "same data'. That leaks the eval split's labels into policy "
            "selection, producing optimistic metrics; a warning is printed "
            "naming this whenever --split is used. Kept only because "
            "runbook commands using it are already in circulation -- prefer "
            "--fit-split/--eval-split. Cannot be combined with --fit-split, "
            "--eval-split, --fit-predictions-file or --eval-predictions-file."
        ),
    )
    parser.add_argument(
        "--fit-split",
        default=None,
        choices=("val", "test"),
        help=(
            "Split the threshold grid search, hard classes and confusion "
            "pairs are derived from. Default 'val'. The eval split's labels "
            "are never consulted while fitting."
        ),
    )
    parser.add_argument(
        "--eval-split",
        default=None,
        choices=("val", "test"),
        help=(
            "Split the frozen policy (selected from --fit-split alone) is "
            "evaluated on, exactly once. Default 'test'."
        ),
    )
    parser.add_argument(
        "--predictions-file",
        default=None,
        help=(
            "Path to a predictions CSV overriding the `<split>_predictions.csv` "
            "convention for the deprecated --split flag -- e.g. a re-scored "
            "artifact from kaggle/a3b_rescore/rescore_predictions.py, which "
            "deliberately writes `<split>_predictions_rescored.csv` rather "
            "than the conventional name so it never clobbers the immutable "
            "run record. --results-dir still supplies the other artifacts "
            "(class report, confusion pairs, output location). Only valid "
            "with --split; use --fit-predictions-file/--eval-predictions-file "
            "with --fit-split/--eval-split."
        ),
    )
    parser.add_argument(
        "--fit-predictions-file",
        default=None,
        help=(
            "Predictions CSV for --fit-split, overriding "
            "`<fit-split>_predictions.csv`. Same override semantics as "
            "--predictions-file."
        ),
    )
    parser.add_argument(
        "--eval-predictions-file",
        default=None,
        help=(
            "Predictions CSV for --eval-split, overriding "
            "`<eval-split>_predictions.csv`. Same override semantics as "
            "--predictions-file."
        ),
    )
    parser.add_argument(
        "--hard-classes-file",
        default=None,
        help=(
            "Explicit override: a JSON list of hard-class names, used as-is. "
            "Must exist and contain at least one name -- an empty or missing "
            "file is an error, never a silent fallback. When omitted, hard "
            "classes are derived from --class-report-file or, failing that, "
            "directly from the fit split's own predictions."
        ),
    )
    parser.add_argument(
        "--confusion-pairs-file",
        default=None,
        help="Optional CSV with confusion pairs and optional counts",
    )
    parser.add_argument(
        "--class-report-file",
        default=None,
        help=(
            "Explicit override: a class-level report CSV (`class_name`, "
            "`f1-score` columns) to derive hard classes from -- the bottom "
            "10%% of classes by F1 (at least 5). Must exist and have both "
            "columns -- never a silent fallback. When omitted, hard classes "
            "are derived the same way directly from the fit split's own "
            "predictions, so two runs being compared always use the same "
            "derivation path regardless of which optional files happen to "
            "exist on disk."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory; defaults to <results-dir>/<split>_decision_layer "
            "when using the deprecated --split, or "
            "<results-dir>/decision_layer_fit_<fit-split>_eval_<eval-split> "
            "otherwise."
        ),
    )
    parser.add_argument(
        "--max-confusion-pairs",
        default=40,
        type=int,
        help="Maximum frequent confusion pairs included as review risk",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Skip creating a decision_layer_artifacts.zip file",
    )
    return parser.parse_args(argv)


def _float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None

    if math.isnan(parsed) or math.isinf(parsed):
        return None

    return parsed


def parse_float_list(value: object) -> list[float]:
    if value is None:
        return []

    text = str(value).strip()
    if not text:
        return []

    parts = [part.strip() for part in text.split("|")]
    return [parsed for part in parts if (parsed := _float(part)) is not None]


def parse_label_list(value: object) -> list[str]:
    if value is None:
        return []

    text = str(value).strip()
    if not text:
        return []

    return [part.strip() for part in text.split("|") if part.strip()]


def coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def normalize_actual_col(data: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    frame = data.copy()
    if "actual" not in frame.columns:
        for candidate in ("actual_label", "true_label", "label", "actual_label_name"):
            if candidate in frame.columns:
                frame = frame.rename(columns={candidate: "actual"})
                break
    if "predicted" not in frame.columns:
        for candidate in (
            "predicted_label",
            "pred_label",
            "prediction",
            "class_name",
            "predicted_class",
        ):
            if candidate in frame.columns:
                frame = frame.rename(columns={candidate: "predicted"})
                break

    for required in ("actual", "predicted"):
        if required not in frame.columns:
            raise ValueError(
                f"Prediction file must include a `{required}` column (or a compatible alias)."
            )

    if "is_correct" not in frame.columns and {
        "actual",
        "predicted",
    }.issubset(frame.columns):
        frame["is_correct"] = frame["actual"].astype(str) == frame["predicted"].astype(
            str
        )

    return frame, "actual", "predicted"


def normalize_class_report(class_report: pd.DataFrame) -> pd.DataFrame:
    if class_report.empty:
        return class_report

    if "Unnamed: 0" in class_report.columns:
        class_report = class_report.rename(columns={"Unnamed: 0": "class_name"})

    if "class_name" not in class_report.columns:
        first_col = class_report.columns[0]
        class_report = class_report.rename(columns={first_col: "class_name"})

    return class_report


class HardClassDerivationError(ValueError):
    """Raised when hard classes cannot be resolved without silently guessing.

    Both `--hard-classes-file` and `--class-report-file` are explicit,
    opt-in overrides: if the caller names one, it must exist and be usable,
    or this raises rather than quietly substituting a default. The default
    (no override given) path derives hard classes directly from the fit
    split's own predictions, which are always available -- so this should
    only be reachable if `fit_predictions` itself is malformed.
    """


def class_f1_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute a per-class F1 table directly from `predictions`.

    This mirrors the shape of a `<split>_class_report.csv` (an
    `sklearn.metrics.classification_report(..., output_dict=True)` table)
    closely enough to feed `select_hard_classes_by_f1()` -- precision,
    recall and F1 per class computed the standard way (one-vs-rest true/false
    positives and negatives) -- but computed straight from this script's own
    required `actual`/`predicted` columns instead of an optional side-file.
    That is the fix for the bug this function replaces: hard-class
    derivation no longer depends on whether a `val_class_report.csv` happens
    to be sitting in `--results-dir` (see docs/9_agent_log.md, the
    2026-09-14 Codex review, for the two comparison runs this silently
    diverged for -- 11 hard classes for one, 5 for the other).

    Every class appearing as either an actual or a predicted label is
    scored. Rows are `class_name`-sorted before `select_hard_classes_by_f1()`
    sorts by F1, so the tie-breaking on equal F1 scores is deterministic.

    Args:
        predictions: A predictions frame with `actual` and `predicted`
            columns (as produced by `normalize_actual_col()`).

    Returns:
        A `class_name`/`f1-score` frame, one row per class.
    """
    actual = predictions["actual"].astype(str)
    predicted = predictions["predicted"].astype(str)
    classes = sorted(set(actual) | set(predicted))

    rows = []
    for class_name in classes:
        is_actual = actual == class_name
        is_predicted = predicted == class_name
        true_positive = int((is_actual & is_predicted).sum())
        false_positive = int((is_predicted & ~is_actual).sum())
        false_negative = int((is_actual & ~is_predicted).sum())

        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else 0.0
        )
        f1_score = (
            2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        )
        rows.append({"class_name": class_name, "f1-score": f1_score})

    return pd.DataFrame(rows)


def select_hard_classes_by_f1(class_report: pd.DataFrame) -> set[str]:
    """Bottom 10% of classes by F1 (at least 5), from a class-level report.

    This is the one selection rule, shared by the explicit
    `--class-report-file` path and the default derive-from-fit-predictions
    path (`class_f1_table()`), so both apply identical tie-breaking
    (`DataFrame.sort_values(ascending=True)`'s stable ordering on ties) and
    rounding (`max(5, ceil(0.1 * n))`) -- the two paths can never disagree
    about what "bottom 10%" means, only about where the F1 numbers came from.

    Raises:
        HardClassDerivationError: If `class_report` lacks `class_name` or
            `f1-score` columns.
    """
    if not {"class_name", "f1-score"} <= set(class_report.columns):
        raise HardClassDerivationError(
            "Class report must have `class_name` and `f1-score` columns; "
            f"found {', '.join(str(c) for c in class_report.columns) or '(none)'}."
        )
    low_f1 = class_report.sort_values("f1-score", ascending=True)
    limit = max(5, math.ceil(0.1 * len(low_f1)))
    return set(low_f1.head(limit)["class_name"].astype(str).tolist())


def load_hard_classes(
    results_dir: Path,
    hard_classes_file: str | None,
    class_report_file: str | None,
    fit_predictions: pd.DataFrame,
) -> tuple[set[str], str]:
    """Resolve the hard-class set and where it came from.

    Precedence, each an explicit ask rather than a silent guess:

    1. `--hard-classes-file` -- an explicit JSON list of class names, used
       as-is.
    2. `--class-report-file` -- an explicit class-level report CSV, scored
       with `select_hard_classes_by_f1()`.
    3. Derived directly from `fit_predictions` via `class_f1_table()` --
       the fit split's own predictions are already required input to this
       script, so this path never depends on an optional file's presence,
       and two runs being compared always take it identically unless one of
       them explicitly opts into an override.

    Neither override silently falls back to the default derivation, and the
    default derivation never falls back to a fixed constant list: previously
    a missing/empty `--hard-classes-file` or a `--class-report-file` that
    did not exist both silently returned a hardcoded 5-class default
    (`AUTO_HARD_CLASSES`, now removed), which is exactly how two comparison
    runs ended up with different, silently-diverging hard-class sets (11 vs.
    5) despite the claim that they used the same pipeline -- see
    docs/9_agent_log.md, the 2026-09-14 Codex review.

    Returns:
        `(hard_classes, source)` -- `source` is a short, human-readable
        description of which of the three paths produced the result, for
        logging at run start.

    Raises:
        HardClassDerivationError: If `hard_classes_file`/`class_report_file`
            is given but missing, empty, or malformed, or if the default
            derivation is impossible because `fit_predictions` lacks
            `actual`/`predicted` columns.
    """
    if hard_classes_file:
        path = Path(hard_classes_file)
        if not path.is_absolute():
            path = results_dir / path
        if not path.exists():
            raise HardClassDerivationError(f"--hard-classes-file {path} does not exist.")
        text = path.read_text().strip()
        if not text:
            raise HardClassDerivationError(f"--hard-classes-file {path} is empty.")
        classes = {
            entry.strip().strip('"\'') for entry in json.loads(text) if isinstance(entry, str)
        }
        if not classes:
            raise HardClassDerivationError(
                f"--hard-classes-file {path} contained no class names."
            )
        return classes, f"--hard-classes-file {path}"

    if class_report_file:
        path = Path(class_report_file)
        if not path.is_absolute():
            path = results_dir / path
        if not path.exists():
            raise HardClassDerivationError(f"--class-report-file {path} does not exist.")
        report = normalize_class_report(pd.read_csv(path))
        return select_hard_classes_by_f1(report), f"--class-report-file {path}"

    if not {"actual", "predicted"} <= set(fit_predictions.columns):
        raise HardClassDerivationError(
            "Cannot derive hard classes: fit-split predictions have no "
            "`actual`/`predicted` columns, and neither --hard-classes-file "
            "nor --class-report-file was given. Pass one of those explicitly."
        )
    class_report = class_f1_table(fit_predictions)
    return (
        select_hard_classes_by_f1(class_report),
        "derived from fit-split predictions (bottom 10% by F1)",
    )


def load_confusion_pairs(
    predictions: pd.DataFrame,
    results_dir: Path,
    confusion_pairs_file: str | None,
    max_pairs: int,
) -> set[tuple[str, str]]:
    confusion_aliases = {
        "actual": "actual",
        "predicted": "predicted",
        "true_label": "actual",
        "actual_label": "actual",
        "pred_label": "predicted",
        "predicted_label": "predicted",
    }

    dataframe = None
    if confusion_pairs_file:
        path = Path(confusion_pairs_file)
        if not path.is_absolute():
            path = results_dir / path
        if path.exists():
            dataframe = pd.read_csv(path)
            rename_map = {
                source: target
                for source, target in confusion_aliases.items()
                if source in dataframe.columns and source != target
            }
            if rename_map:
                dataframe = dataframe.rename(columns=rename_map)

    if dataframe is None:
        wrong = predictions[~predictions["is_correct"]]
        if wrong.empty:
            return set()
        pair_frame = (
            wrong.groupby(["actual", "predicted"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values("count", ascending=False)
            .head(max(1, max_pairs))
        )
    else:
        if {"actual", "predicted"} <= set(dataframe.columns):
            pair_frame = dataframe.copy()
        elif {"true_label", "pred_label"} <= set(dataframe.columns):
            pair_frame = dataframe.rename(
                columns={"true_label": "actual", "pred_label": "predicted"}
            )
        elif {"actual_label", "predicted_label"} <= set(dataframe.columns):
            pair_frame = dataframe.rename(
                columns={"actual_label": "actual", "predicted_label": "predicted"}
            )
        else:
            pair_frame = dataframe.copy()

        for col in ("count", "support", "n"):
            if col in pair_frame.columns:
                pair_frame = pair_frame.sort_values(col, ascending=False)
                break

        pair_frame = pair_frame.head(max_pairs)

    pairs: set[tuple[str, str]] = set()
    for actual, predicted in pair_frame[["actual", "predicted"]].itertuples(
        index=False,
        name=None,
    ):
        pairs.add((str(actual), str(predicted)))
    return pairs


def assign_decision_band(
    row: pd.Series,
    auto_confidence: float,
    suggest_confidence: float,
    margin_threshold: float,
    hard_classes: set[str],
    confusion_pairs: set[tuple[str, str]],
) -> str:
    """Route one row through the shared production decision rule.

    This calls `route_decision` with only inference-time inputs -- the
    predicted label, top-1 confidence and top-1/top-2 margin -- exactly as
    production does. The actual label is not consulted here; it is used
    elsewhere in this script only after routing, to score how each band
    performed (see `decision_band_metrics`).
    """
    return route_decision(
        row["top_1_confidence"],
        row["top_1_top_2_margin"],
        row["predicted"],
        policy={
            "auto_confidence": auto_confidence,
            "suggest_confidence": suggest_confidence,
            "margin_threshold": margin_threshold,
        },
        hard_classes=hard_classes,
        confusion_pairs=confusion_pairs,
        mode="image",
    )


def decision_band_metrics(decision_df: pd.DataFrame) -> pd.DataFrame:
    total = max(1, len(decision_df))
    rows = []
    expected_bands = ("auto_accept", "suggest", "confirm", "review")

    for band in expected_bands:
        band_df = decision_df[decision_df["decision_band"] == band]
        if band_df.empty:
            rows.append(
                {
                    "decision_band": band,
                    "sample_count": 0,
                    "coverage": 0.0,
                    "top_1_accuracy": 0.0,
                    "top_5_contains_actual": 0.0,
                    "mean_confidence": 0.0,
                    "mean_margin": 0.0,
                }
            )
            continue

        rows.append(
            {
                "decision_band": band,
                "sample_count": int(len(band_df)),
                "coverage": len(band_df) / total,
                "top_1_accuracy": band_df["is_correct"].mean(),
                "top_5_contains_actual": band_df["top_5_contains_actual"].mean(),
                "mean_confidence": band_df["top_1_confidence"].mean(),
                "mean_margin": band_df["top_1_top_2_margin"].mean(),
            }
        )

    return pd.DataFrame(rows).sort_values("decision_band")


def evaluate_policy(
    features_df: pd.DataFrame,
    auto_confidence: float,
    suggest_confidence: float,
    margin_threshold: float,
    hard_classes: set[str],
    confusion_pairs: set[tuple[str, str]],
) -> dict[str, float | int]:
    scored = features_df.copy()
    scored["decision_band"] = scored.apply(
        assign_decision_band,
        axis=1,
        auto_confidence=auto_confidence,
        suggest_confidence=suggest_confidence,
        margin_threshold=margin_threshold,
        hard_classes=hard_classes,
        confusion_pairs=confusion_pairs,
    )
    metrics_df = decision_band_metrics(scored)

    metrics: dict[str, float | int] = {
        "auto_confidence": auto_confidence,
        "suggest_confidence": suggest_confidence,
        "margin_threshold": margin_threshold,
    }
    for _, row in metrics_df.iterrows():
        band = row["decision_band"]
        metrics[f"{band}_coverage"] = float(row["coverage"])
        metrics[f"{band}_top_1_accuracy"] = float(row["top_1_accuracy"])
        metrics[f"{band}_top_5_contains_actual"] = float(
            row["top_5_contains_actual"]
        )

    return metrics


def build_features(
    predictions: pd.DataFrame,
    hard_classes: set[str],
    confusion_pairs: set[tuple[str, str]],
) -> pd.DataFrame:
    actual_col = "actual"
    predicted_col = "predicted"

    features = predictions.copy()
    features["is_correct"] = features["is_correct"].map(coerce_bool)
    features["top_5_labels"] = features["top_5"].map(parse_label_list)
    features["top_5_confidences"] = features["top_5_confidence"].map(parse_float_list)

    features["top_1_confidence"] = features["top_5_confidences"].map(
        lambda values: float(values[0]) if len(values) > 0 else 0.0
    )
    features["top_2_confidence"] = features["top_5_confidences"].map(
        lambda values: float(values[1]) if len(values) > 1 else 0.0
    )
    features["top_1_top_2_margin"] = (
        features["top_1_confidence"] - features["top_2_confidence"]
    )

    actual_series = features[actual_col].astype(str)
    predicted_series = features[predicted_col].astype(str)
    features["is_hard_actual_class"] = actual_series.isin(hard_classes)
    features["is_hard_predicted_class"] = predicted_series.isin(hard_classes)
    features["is_hard_case"] = (
        features["is_hard_actual_class"] | features["is_hard_predicted_class"]
    )
    features["is_frequent_confusion_pair"] = [
        (str(actual), str(predicted)) in confusion_pairs
        for actual, predicted in zip(actual_series, predicted_series, strict=True)
    ]
    features["top_5_contains_actual"] = features.apply(
        lambda row: row[actual_col] in row["top_5_labels"], axis=1
    )

    return features


class ConflictingSplitArgumentsError(ValueError):
    """Raised when --split is mixed with --fit-split/--eval-split/*-predictions-file."""


DEPRECATED_SPLIT_WARNING = (
    "warning: --split is deprecated. It fits decision thresholds and "
    "evaluates them on the SAME split -- test labels (or val labels) leak "
    "into policy selection, so the reported band metrics are optimistic "
    "rather than a genuine generalisation estimate. Use --fit-split and "
    "--eval-split instead (defaults: --fit-split val --eval-split test)."
)


def resolve_predictions_path(
    results_dir: Path, split: str, predictions_file: str | None
) -> Path:
    """Resolve the predictions CSV for `split`.

    `predictions_file`, when given, is resolved like a normal CLI path
    argument (relative to the current working directory), not relative to
    `results_dir`: the whole point of this override is pointing at an
    artifact that lives outside the `<split>_predictions.csv` convention --
    e.g. a sibling `*_predictions_rescored.csv` file -- so it must not be
    forced back under `results_dir`.
    """
    if predictions_file:
        return Path(predictions_file).expanduser().resolve()
    return results_dir / f"{split}_predictions.csv"


def load_predictions_for_split(
    results_dir: Path, split: str, predictions_file: str | None
) -> pd.DataFrame:
    """Load and schema-validate the predictions CSV for one split."""
    predictions_path = resolve_predictions_path(results_dir, split, predictions_file)
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Missing prediction file: {predictions_path}. Re-run train script first."
        )

    predictions = normalize_actual_col(pd.read_csv(predictions_path))[0]
    validate_predictions_schema(predictions, predictions_path)
    return predictions


def search_policy_grid(
    features_df: pd.DataFrame,
    hard_classes: set[str],
    confusion_pairs: set[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.Series]:
    """Grid-search the policy thresholds against `features_df` alone.

    `features_df` must come from the fit split only -- this is the one place
    eval-split labels must never appear, per the master standard's leakage
    rule (docs/superpowers/specs/2026-09-14-decision-layer-methodology-design.md,
    section 3.2).

    Returns:
        `(policy_search_df, best_policy_row)`.
    """
    policy_rows = []
    for auto_confidence in AUTO_CONFIDENCE_GRID:
        for suggest_confidence in SUGGEST_CONFIDENCE_GRID:
            if suggest_confidence >= auto_confidence:
                continue
            for margin_threshold in MARGIN_GRID:
                policy_rows.append(
                    evaluate_policy(
                        features_df,
                        auto_confidence=auto_confidence,
                        suggest_confidence=suggest_confidence,
                        margin_threshold=margin_threshold,
                        hard_classes=hard_classes,
                        confusion_pairs=confusion_pairs,
                    )
                )

    policy_search_df = pd.DataFrame(policy_rows).fillna(0.0)
    policy_search_df["meets_auto_accuracy"] = (
        policy_search_df.get("auto_accept_top_1_accuracy", pd.Series(dtype=float))
        >= MIN_AUTO_ACCEPT_ACCURACY
    )
    policy_search_df["meets_suggest_accuracy"] = (
        policy_search_df.get("suggest_top_5_contains_actual", pd.Series(dtype=float))
        >= MIN_SUGGEST_ACCURACY
    )
    policy_search_df["policy_score"] = (
        2.0 * policy_search_df["auto_accept_coverage"].fillna(0.0)
        + policy_search_df["suggest_coverage"].fillna(0.0)
        - 0.5 * policy_search_df["review_coverage"].fillna(0.0)
    )

    eligible = policy_search_df[
        policy_search_df["meets_auto_accuracy"] & policy_search_df["meets_suggest_accuracy"]
    ].copy()
    if not eligible.empty:
        best_policy = eligible.sort_values("policy_score", ascending=False).iloc[0]
    else:
        best_policy = policy_search_df.sort_values("policy_score", ascending=False).iloc[0]

    return policy_search_df, best_policy


def score_split(
    features_df: pd.DataFrame,
    auto_confidence: float,
    suggest_confidence: float,
    margin_threshold: float,
    hard_classes: set[str],
    confusion_pairs: set[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a frozen policy to `features_df` and compute its band metrics."""
    scored = features_df.copy()
    scored["decision_band"] = scored.apply(
        assign_decision_band,
        axis=1,
        auto_confidence=auto_confidence,
        suggest_confidence=suggest_confidence,
        margin_threshold=margin_threshold,
        hard_classes=hard_classes,
        confusion_pairs=confusion_pairs,
    )
    return scored, decision_band_metrics(scored)


def run_analysis(
    results_dir: Path,
    fit_split: str,
    eval_split: str,
    hard_classes_file: str | None,
    confusion_pairs_file: str | None,
    class_report_file: str | None,
    output_dir: Path,
    max_confusion_pairs: int,
    skip_zip: bool,
    fit_predictions_file: str | None = None,
    eval_predictions_file: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit decision thresholds on `fit_split`, freeze them, and evaluate once on `eval_split`.

    The grid search, hard classes and confusion pairs are derived from
    `fit_split` alone. The frozen policy is then evaluated exactly once on
    `eval_split`. Both splits' band metrics are written and returned so the
    fit/eval generalisation gap is visible.

    Returns:
        `(band_metrics_fit, band_metrics_eval)`.
    """
    fit_predictions = load_predictions_for_split(results_dir, fit_split, fit_predictions_file)
    eval_predictions = load_predictions_for_split(results_dir, eval_split, eval_predictions_file)

    confusion_path = results_dir / f"{fit_split}_confusion_pairs.csv"

    # `class_report_file` is only ever what the caller explicitly passed --
    # unlike confusion pairs below, this is deliberately NOT auto-discovered
    # from a `<fit_split>_class_report.csv` convenience file in results_dir.
    # That auto-discovery used to be exactly how two runs being compared
    # silently diverged (one run directory happened to have the file, the
    # other didn't) -- see load_hard_classes()'s docstring and
    # docs/9_agent_log.md's 2026-09-14 Codex review. With no override, hard
    # classes are always derived the same way: from fit_predictions itself.
    hard_classes, hard_class_source = load_hard_classes(
        results_dir, hard_classes_file, class_report_file, fit_predictions
    )
    print(
        f"Hard classes in use ({len(hard_classes)}, source: {hard_class_source}): "
        f"{sorted(hard_classes)}"
    )

    if not confusion_pairs_file:
        confusion_pairs_file = str(confusion_path) if confusion_path.exists() else None

    # Derived from the fit split's predictions only -- never the eval split's.
    confusion_pairs = load_confusion_pairs(
        fit_predictions,
        results_dir,
        confusion_pairs_file,
        max_pairs=max_confusion_pairs,
    )

    features_fit = build_features(fit_predictions, hard_classes, confusion_pairs)
    features_eval = build_features(eval_predictions, hard_classes, confusion_pairs)

    policy_search_df, best_policy = search_policy_grid(features_fit, hard_classes, confusion_pairs)

    output_dir.mkdir(parents=True, exist_ok=True)

    auto_confidence = float(best_policy["auto_confidence"])
    suggest_confidence = float(best_policy["suggest_confidence"])
    margin_threshold = float(best_policy["margin_threshold"])

    final_fit, band_metrics_fit = score_split(
        features_fit,
        auto_confidence,
        suggest_confidence,
        margin_threshold,
        hard_classes,
        confusion_pairs,
    )
    final_eval, band_metrics_eval = score_split(
        features_eval,
        auto_confidence,
        suggest_confidence,
        margin_threshold,
        hard_classes,
        confusion_pairs,
    )

    policy_row = pd.DataFrame(
        [
            {
                "auto_confidence": auto_confidence,
                "suggest_confidence": suggest_confidence,
                "margin_threshold": margin_threshold,
                "min_auto_accept_accuracy": MIN_AUTO_ACCEPT_ACCURACY,
                "min_suggest_accuracy": MIN_SUGGEST_ACCURACY,
                "fit_split": fit_split,
                "eval_split": eval_split,
            }
        ]
    )

    policy_search_df.to_csv(output_dir / "decision_policy_search.csv", index=False)
    features_fit.to_csv(output_dir / "decision_features_fit.csv", index=False)
    features_eval.to_csv(output_dir / "decision_features_eval.csv", index=False)
    final_fit.to_csv(output_dir / "fit_predictions_with_decisions.csv", index=False)
    final_eval.to_csv(output_dir / "eval_predictions_with_decisions.csv", index=False)
    band_metrics_fit.to_csv(output_dir / "decision_band_metrics_fit.csv", index=False)
    band_metrics_eval.to_csv(output_dir / "decision_band_metrics_eval.csv", index=False)
    policy_row.to_csv(output_dir / "decision_policy.csv", index=False)

    (output_dir / "decision_policy.json").write_text(policy_row.to_json(orient="records", indent=2))

    confusion_records = [
        {"actual": actual, "predicted": predicted} for actual, predicted in sorted(confusion_pairs)
    ]
    (output_dir / "confusion_pairs.json").write_text(json.dumps(confusion_records, indent=2))
    (output_dir / "hard_classes.json").write_text(json.dumps(sorted(hard_classes), indent=2))

    confusion_df = pd.DataFrame(confusion_records)
    if not confusion_df.empty:
        confusion_df.to_csv(output_dir / "top_confusion_pairs.csv", index=False)

    # Examples are drawn from the eval split -- the frozen policy applied to
    # data it was not fitted on, which is what production behavior actually
    # looks like.
    for band in ("auto_accept", "suggest", "confirm", "review"):
        final_eval[final_eval["decision_band"] == band].head(25).to_csv(
            output_dir / f"decision_examples_{band}.csv",
            index=False,
        )

    if not skip_zip:
        zip_path = output_dir / ZIP_NAME
        artifact_files = [
            "decision_features_fit.csv",
            "decision_features_eval.csv",
            "decision_policy_search.csv",
            "fit_predictions_with_decisions.csv",
            "eval_predictions_with_decisions.csv",
            "decision_band_metrics_fit.csv",
            "decision_band_metrics_eval.csv",
            "decision_policy.csv",
            "top_confusion_pairs.csv",
            "confusion_pairs.json",
            "hard_classes.json",
            "decision_policy.json",
        ]
        with zipfile.ZipFile(zip_path, "w") as archive:
            for artifact_name in artifact_files:
                artifact_path = output_dir / artifact_name
                if artifact_path.exists():
                    archive.write(artifact_path, arcname=artifact_name)

    print(
        f"Decision policy recalibration complete (fit_split={fit_split}, "
        f"eval_split={eval_split})"
    )
    print(f"Artifacts in: {output_dir}")
    print(f"\nDecision band metrics -- fit split ({fit_split}), used to select thresholds:")
    print(band_metrics_fit.to_string(index=False))
    print(f"\nDecision band metrics -- eval split ({eval_split}), frozen policy evaluated once:")
    print(band_metrics_eval.to_string(index=False))

    return band_metrics_fit, band_metrics_eval


def resolve_splits(args: argparse.Namespace) -> tuple[str, str, str | None, str | None, bool]:
    """Resolve fit/eval splits and their predictions-file overrides from CLI args.

    Returns:
        `(fit_split, eval_split, fit_predictions_file, eval_predictions_file,
        used_deprecated_split)`.

    Raises:
        ConflictingSplitArgumentsError: If the deprecated --split is mixed
            with --fit-split/--eval-split/--fit-predictions-file/
            --eval-predictions-file, or if --predictions-file is given
            without --split.
    """
    if args.split is not None:
        if any(
            (
                args.fit_split is not None,
                args.eval_split is not None,
                args.fit_predictions_file is not None,
                args.eval_predictions_file is not None,
            )
        ):
            raise ConflictingSplitArgumentsError(
                "--split cannot be combined with --fit-split, --eval-split, "
                "--fit-predictions-file or --eval-predictions-file. Drop "
                "--split and use --fit-split/--eval-split instead, or drop "
                "the new flags to keep using deprecated --split."
            )
        print(DEPRECATED_SPLIT_WARNING, file=sys.stderr)
        return args.split, args.split, args.predictions_file, args.predictions_file, True

    if args.predictions_file is not None:
        raise ConflictingSplitArgumentsError(
            "--predictions-file is only valid together with the deprecated "
            "--split flag. Use --fit-predictions-file and/or "
            "--eval-predictions-file with --fit-split/--eval-split."
        )

    fit_split = args.fit_split or "val"
    eval_split = args.eval_split or "test"
    return fit_split, eval_split, args.fit_predictions_file, args.eval_predictions_file, False


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results_dir = Path(args.results_dir).expanduser().resolve()

    try:
        fit_split, eval_split, fit_predictions_file, eval_predictions_file, deprecated = (
            resolve_splits(args)
        )
    except ConflictingSplitArgumentsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    elif deprecated:
        output_dir = results_dir / f"{fit_split}_decision_layer"
    else:
        output_dir = results_dir / f"decision_layer_fit_{fit_split}_eval_{eval_split}"

    try:
        run_analysis(
            results_dir=results_dir,
            fit_split=fit_split,
            eval_split=eval_split,
            hard_classes_file=args.hard_classes_file,
            confusion_pairs_file=args.confusion_pairs_file,
            class_report_file=args.class_report_file,
            output_dir=output_dir,
            max_confusion_pairs=args.max_confusion_pairs,
            skip_zip=args.no_zip,
            fit_predictions_file=fit_predictions_file,
            eval_predictions_file=eval_predictions_file,
        )
    except (PredictionSchemaError, HardClassDerivationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
