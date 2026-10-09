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
- derivation_provenance.json (inputs and derivation settings; see
  `build_provenance()` and scripts/compare_provenance.py)

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
import hashlib
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

# Hard classes: the bottom HARD_CLASS_FRACTION of classes by F1, at least
# HARD_CLASS_MINIMUM. Both cutoffs below use an explicit total order, so the
# selected set depends only on the values, never on input row order. Among
# entries tied exactly at a cutoff the secondary key is an arbitrary but
# fixed choice; it carries no meaning beyond making the result reproducible.
HARD_CLASS_FRACTION = 0.1
HARD_CLASS_MINIMUM = 5
HARD_CLASS_TIE_BREAK = (
    "f1-score ascending, then normalised class name ascending (Unicode code "
    "point order); keep the first max(minimum, ceil(fraction * n_classes)). "
    "The name order is arbitrary among classes with equal F1 at the cutoff."
)
CONFUSION_PAIR_TIE_BREAK = (
    "count descending, then (actual, predicted) ascending (Unicode code point "
    "order); keep exactly the first max_pairs. The label order is arbitrary "
    "among pairs with equal count at the cutoff."
)


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
            "Must exist and decode to a non-empty list of non-empty, distinct "
            "strings (compared after removing surrounding whitespace) -- "
            "anything else is an error, never a silent fallback. When omitted, hard "
            "classes are derived from --class-report-file or, failing that, "
            "directly from the fit split's own predictions."
        ),
    )
    parser.add_argument(
        "--confusion-pairs-file",
        default=None,
        help=(
            "Explicit override: a CSV of confusion pairs (one actual and one "
            "predicted label column, optional positive-integer `count`). Must "
            "exist and be well-formed -- never a silent fallback. When "
            "omitted, pairs are derived from the fit split's own predictions; "
            "a `<fit-split>_confusion_pairs.csv` in --results-dir is ignored "
            "unless named here."
        ),
    )
    parser.add_argument(
        "--class-report-file",
        default=None,
        help=(
            "Explicit override: a class-level report CSV (`class_name`, "
            "`f1-score` columns) to derive hard classes from -- the bottom "
            "10%% of classes by F1 (at least 5). Must exist, have both "
            "columns, at least one row, distinct class names and finite "
            "F1 values within [0, 1] -- never a silent fallback. When omitted, hard classes "
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
        help=(
            "Number of most frequent confusion pairs included as review risk "
            "(fewer only if fewer exist). Ranked by count descending, then "
            "(actual, predicted) ascending, so pairs tied at the cutoff are "
            "chosen by label order -- arbitrary, but independent of row order."
        ),
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


def normalize_class_name(name: object) -> str:
    """The one normalisation applied to a class name read from any source.

    Surrounding whitespace is dropped; nothing else (case is significant,
    because names must match the predicted labels exactly). Duplicate and
    empty checks on overrides are made on this normalised form.
    """
    return str(name).strip()


def normalize_class_report(class_report: pd.DataFrame) -> pd.DataFrame:
    if len(class_report.columns) == 0:
        return class_report

    # An explicit `class_name` column wins over the unnamed index column a
    # `to_csv()` with the index leaves behind; renaming the index too would
    # produce two `class_name` columns.
    if "Unnamed: 0" in class_report.columns and "class_name" not in class_report.columns:
        class_report = class_report.rename(columns={"Unnamed: 0": "class_name"})

    if "class_name" not in class_report.columns:
        first_col = class_report.columns[0]
        class_report = class_report.rename(columns={first_col: "class_name"})

    return class_report


class DerivationInputError(ValueError):
    """Base class for a derived-input source that is missing or malformed.

    Every subclass is caught by `main()` and reported as a one-line
    `error: ...` with a non-zero exit, never a traceback. An explicitly named
    override file that cannot be used always raises one of these; it never
    silently falls back to deriving the input some other way.
    """


class HardClassDerivationError(DerivationInputError):
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
    scored. Tie-breaking on equal F1 is `select_hard_classes_by_f1()`'s job
    (by class name), so this table's row order does not matter.

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
    path (`class_f1_table()`), so both apply identical ordering and rounding
    (`max(HARD_CLASS_MINIMUM, ceil(HARD_CLASS_FRACTION * n))`) -- the two
    paths can never disagree about what "bottom 10%" means, only about where
    the F1 numbers came from.

    Ordering is F1 ascending, then normalised class name ascending (see
    `HARD_CLASS_TIE_BREAK`). The name is a semantic tie-break, not an
    appeal to sort stability: a stable sort would merely preserve whatever
    order the rows arrived in, and pandas' default `sort_values` is not
    stable anyway. With the name as secondary key, the selected set is the
    same for any permutation of the input rows. Which of several classes
    tied at the cutoff F1 gets in is still arbitrary -- it is just fixed.

    Raises:
        HardClassDerivationError: If `class_report` lacks `class_name` or
            `f1-score` columns, has no rows, has empty or duplicate
            (normalised) class names, or has an F1 that is not a finite
            number within [0, 1].
    """
    if not {"class_name", "f1-score"} <= set(class_report.columns):
        raise HardClassDerivationError(
            "Class report must have `class_name` and `f1-score` columns; "
            f"found {', '.join(str(c) for c in class_report.columns) or '(none)'}."
        )
    if class_report.empty:
        raise HardClassDerivationError(
            "Class report has a header but no class rows; at least one class is required."
        )

    names = [
        "" if pd.isna(name) else normalize_class_name(name)
        for name in class_report["class_name"]
    ]
    empty_rows = [index for index, name in enumerate(names) if not name]
    if empty_rows:
        raise HardClassDerivationError(
            f"Class report has empty `class_name` values (data rows {empty_rows[:5]})."
        )
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise HardClassDerivationError(
            "Class report has duplicate class names after normalisation "
            f"(surrounding whitespace removed): {duplicates[:5]}. Each class "
            "must appear once, or the bottom-k selection would silently pick "
            "fewer than k classes."
        )

    f1_scores = pd.to_numeric(class_report["f1-score"], errors="coerce")
    bad = [
        f"{name}={raw!r}"
        for name, raw, value in zip(names, class_report["f1-score"], f1_scores, strict=True)
        if pd.isna(value) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1
    ]
    if bad:
        raise HardClassDerivationError(
            "Class report `f1-score` values must be finite numbers within "
            f"[0, 1]; got {', '.join(bad[:5])}."
        )

    ordered = sorted(
        zip(f1_scores.astype(float), names, strict=True),
        key=lambda item: (item[0], item[1]),
    )
    limit = max(HARD_CLASS_MINIMUM, math.ceil(HARD_CLASS_FRACTION * len(ordered)))
    return {name for _, name in ordered[:limit]}


def parse_hard_classes_json(text: str, path: Path) -> set[str]:
    """Validate a `--hard-classes-file` body: a non-empty JSON list of names.

    Rejects, rather than coerces: malformed JSON; any top-level value that
    is not a list (a bare string used to be iterated into one-letter
    classes, and an object into its keys); an empty list; any entry that is
    not a string or is empty after normalisation; and duplicate names after
    normalisation.

    Raises:
        HardClassDerivationError: On any of the above.
    """
    if not text.strip():
        raise HardClassDerivationError(f"--hard-classes-file {path} is empty.")
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HardClassDerivationError(
            f"--hard-classes-file {path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(decoded, list):
        raise HardClassDerivationError(
            f"--hard-classes-file {path} must be a JSON list of class names; "
            f"got a JSON {type(decoded).__name__}."
        )
    if not decoded:
        raise HardClassDerivationError(
            f"--hard-classes-file {path} is an empty list; name at least one class."
        )
    non_strings = [entry for entry in decoded if not isinstance(entry, str)]
    if non_strings:
        raise HardClassDerivationError(
            f"--hard-classes-file {path} entries must all be strings; got "
            f"{non_strings[:5]!r}."
        )
    names = [normalize_class_name(entry) for entry in decoded]
    if any(not name for name in names):
        raise HardClassDerivationError(
            f"--hard-classes-file {path} contains an empty or whitespace-only class name."
        )
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise HardClassDerivationError(
            f"--hard-classes-file {path} names a class more than once "
            f"(after removing surrounding whitespace): {duplicates[:5]}."
        )
    return set(names)


def load_hard_classes(
    results_dir: Path,
    hard_classes_file: str | None,
    class_report_file: str | None,
    fit_predictions: pd.DataFrame,
) -> tuple[set[str], dict[str, str]]:
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
        `(hard_classes, source)` -- `source` has `kind` (`hard_classes_file`,
        `class_report_file` or `fit_predictions`) and, for a named file, its
        resolved `path`. It feeds the run's provenance record.

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
        return (
            parse_hard_classes_json(path.read_text(), path),
            {"kind": "hard_classes_file", "path": str(path)},
        )

    if class_report_file:
        path = Path(class_report_file)
        if not path.is_absolute():
            path = results_dir / path
        if not path.exists():
            raise HardClassDerivationError(f"--class-report-file {path} does not exist.")
        try:
            report = pd.read_csv(path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise HardClassDerivationError(
                f"--class-report-file {path} is not a readable CSV: {exc}"
            ) from exc
        try:
            selected = select_hard_classes_by_f1(normalize_class_report(report))
        except HardClassDerivationError as exc:
            raise HardClassDerivationError(f"--class-report-file {path}: {exc}") from exc
        return selected, {"kind": "class_report_file", "path": str(path)}

    if not {"actual", "predicted"} <= set(fit_predictions.columns):
        raise HardClassDerivationError(
            "Cannot derive hard classes: fit-split predictions have no "
            "`actual`/`predicted` columns, and neither --hard-classes-file "
            "nor --class-report-file was given. Pass one of those explicitly."
        )
    class_report = class_f1_table(fit_predictions)
    return select_hard_classes_by_f1(class_report), {"kind": "fit_predictions"}


class ConfusionPairsError(DerivationInputError):
    """Raised when a named `--confusion-pairs-file` is missing or malformed.

    Matches the hard-class override contract: a file the caller names must
    be usable, or the run stops. It never falls back to deriving pairs from
    the fit predictions instead.
    """


CONFUSION_PAIR_COLUMN_ALIASES = {
    "true_label": "actual",
    "actual_label": "actual",
    "pred_label": "predicted",
    "predicted_label": "predicted",
}
CONFUSION_PAIR_COUNT_COLUMNS = ("count", "support", "n")


def _is_positive_integer(value: object) -> bool:
    """True for a finite whole number >= 1 (`3` or `3.0`), as a pair count must be."""
    if pd.isna(value):
        return False
    number = float(value)
    return math.isfinite(number) and number >= 1 and number.is_integer()


def read_confusion_pairs_file(path: Path) -> pd.DataFrame:
    """Read and validate a named confusion-pair CSV into `actual`/`predicted`/`count`.

    The file must have exactly one actual-label column and exactly one
    predicted-label column (`actual`/`true_label`/`actual_label` and
    `predicted`/`pred_label`/`predicted_label`) -- two columns claiming the
    same role, such as `true_label` beside `actual_label`, are ambiguous and
    rejected rather than one being picked. It must also have at least one
    row, no empty label values and no duplicate pairs. A count column
    (`count`, `support` or `n`, first found wins) is optional; when present
    every value must be a positive integer (`3` and `3.0` are accepted; `0`,
    negatives, fractions, blanks and non-numbers are not). Without one, every
    pair counts as 1, so the cutoff is decided by the tie-break alone.

    Raises:
        ConfusionPairsError: If the file is missing, unreadable or fails any
            of the checks above.
    """
    if not path.exists():
        raise ConfusionPairsError(f"--confusion-pairs-file {path} does not exist.")
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ConfusionPairsError(
            f"--confusion-pairs-file {path} is not a readable CSV: {exc}"
        ) from exc

    for target in ("actual", "predicted"):
        claimants = [
            str(col)
            for col in frame.columns
            if col == target or CONFUSION_PAIR_COLUMN_ALIASES.get(col) == target
        ]
        if len(claimants) > 1:
            raise ConfusionPairsError(
                f"--confusion-pairs-file {path} has more than one `{target}` "
                f"label column ({', '.join(f'`{c}`' for c in claimants)}); keep "
                "exactly one, so the label source is not guessed."
            )
    frame = frame.rename(
        columns={
            source: target
            for source, target in CONFUSION_PAIR_COLUMN_ALIASES.items()
            if source in frame.columns
        }
    )
    if not {"actual", "predicted"} <= set(frame.columns):
        found = ", ".join(str(col) for col in frame.columns) or "(none)"
        raise ConfusionPairsError(
            f"--confusion-pairs-file {path} needs actual and predicted label "
            "columns (`actual`/`true_label`/`actual_label` and "
            f"`predicted`/`pred_label`/`predicted_label`); found {found}."
        )
    if frame.empty:
        raise ConfusionPairsError(
            f"--confusion-pairs-file {path} has a header but no pair rows."
        )

    actual = ["" if pd.isna(v) else normalize_class_name(v) for v in frame["actual"]]
    predicted = ["" if pd.isna(v) else normalize_class_name(v) for v in frame["predicted"]]
    if any(not name for name in actual + predicted):
        raise ConfusionPairsError(
            f"--confusion-pairs-file {path} has empty actual or predicted label values."
        )
    pairs = list(zip(actual, predicted, strict=True))
    duplicates = sorted({pair for pair in pairs if pairs.count(pair) > 1})
    if duplicates:
        raise ConfusionPairsError(
            f"--confusion-pairs-file {path} lists a pair more than once: {duplicates[:5]}."
        )

    count_col = next((c for c in CONFUSION_PAIR_COUNT_COLUMNS if c in frame.columns), None)
    if count_col is None:
        counts = pd.Series([1.0] * len(frame))
    else:
        counts = pd.to_numeric(frame[count_col], errors="coerce").reset_index(drop=True)
        bad = [
            f"{pair}={raw!r}"
            for pair, raw, value in zip(pairs, frame[count_col], counts, strict=True)
            if not _is_positive_integer(value)
        ]
        if bad:
            raise ConfusionPairsError(
                f"--confusion-pairs-file {path} `{count_col}` values must be "
                f"positive integers; got {', '.join(bad[:5])}."
            )

    return pd.DataFrame(
        {"actual": actual, "predicted": predicted, "count": counts.astype(float)}
    )


def count_confusion_pairs(predictions: pd.DataFrame) -> pd.DataFrame:
    """Count each `(actual, predicted)` error pair in `predictions`."""
    wrong = predictions[~predictions["is_correct"].map(coerce_bool)]
    if wrong.empty:
        return pd.DataFrame({"actual": [], "predicted": [], "count": []})
    return (
        wrong.assign(
            actual=wrong["actual"].astype(str),
            predicted=wrong["predicted"].astype(str),
        )
        .groupby(["actual", "predicted"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )


def load_confusion_pairs(
    predictions: pd.DataFrame,
    results_dir: Path,
    confusion_pairs_file: str | None,
    max_pairs: int,
) -> tuple[set[tuple[str, str]], dict[str, str]]:
    """Resolve the confusion-pair set and where it came from.

    Pairs are derived from `predictions` (the fit split) unless
    `confusion_pairs_file` names a file. Nothing is picked up because it
    happens to sit in `results_dir`: a `<fit_split>_confusion_pairs.csv`
    there is ignored unless named. A named file is validated by
    `read_confusion_pairs_file()` and never silently replaced by derivation.

    Returns:
        `(pairs, source)` -- `source` has `kind` (`fit_predictions` or
        `confusion_pairs_file`) and, for a named file, its resolved `path`.

    Raises:
        ConfusionPairsError: If a named file is missing or malformed.
    """
    if confusion_pairs_file:
        path = Path(confusion_pairs_file)
        if not path.is_absolute():
            path = results_dir / path
        pair_frame = read_confusion_pairs_file(path)
        source = {"kind": "confusion_pairs_file", "path": str(path)}
    else:
        pair_frame = count_confusion_pairs(predictions)
        source = {"kind": "fit_predictions"}

    return select_top_confusion_pairs(pair_frame, max_pairs), source


def select_top_confusion_pairs(pair_frame: pd.DataFrame, max_pairs: int) -> set[tuple[str, str]]:
    """Keep exactly `max_pairs` pairs (or all, if fewer) by `CONFUSION_PAIR_TIE_BREAK`.

    Sorted by count descending, then `(actual, predicted)` ascending. The
    label order only decides among pairs whose count ties at the cutoff, and
    that choice is arbitrary but fixed: the result is the same for any
    permutation of the input rows. Keeping every tied pair instead would
    make `--max-confusion-pairs` no longer a maximum.
    """
    ordered = sorted(
        (
            (float(count), str(actual), str(predicted))
            for actual, predicted, count in pair_frame[
                ["actual", "predicted", "count"]
            ].itertuples(index=False, name=None)
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    return {(actual, predicted) for _, actual, predicted in ordered[:max_pairs]}


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


PROVENANCE_FILE = "derivation_provenance.json"
PROVENANCE_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def build_provenance(
    *,
    fit_split: str,
    eval_split: str,
    fit_predictions_path: Path,
    eval_predictions_path: Path,
    hard_class_source: dict[str, str],
    confusion_pair_source: dict[str, str],
    max_confusion_pairs: int,
) -> dict[str, object]:
    """Record every input and derivation setting that shapes the policy.

    Deliberately limited to inputs and derivation settings, not general run
    management. Two runs are *compatible* (see scripts/compare_provenance.py)
    when everything matches except the model-specific `path`/`sha256` of the
    prediction files and override sources -- those differ between models by
    construction and are recorded for traceability, not for equality.
    """
    if hard_class_source["kind"] == "fit_predictions":
        hard_source_path = fit_predictions_path
    else:
        hard_source_path = Path(hard_class_source["path"])
    if hard_class_source["kind"] == "hard_classes_file":
        hard_algorithm = "explicit_list"
        hard_parameters: dict[str, object] = {}
        hard_tie_break = "not applicable: the named list is used as-is"
    else:
        hard_algorithm = "bottom_fraction_by_f1"
        hard_parameters = {"fraction": HARD_CLASS_FRACTION, "minimum": HARD_CLASS_MINIMUM}
        hard_tie_break = HARD_CLASS_TIE_BREAK

    if confusion_pair_source["kind"] == "fit_predictions":
        pair_source_path = fit_predictions_path
        pair_algorithm = "most_frequent_fit_errors"
    else:
        pair_source_path = Path(confusion_pair_source["path"])
        pair_algorithm = "top_named_pairs_by_count"

    routing_module = route_decision.__module__
    routing_file = Path(sys.modules[routing_module].__file__ or "")
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "generator": "scripts/recalibrate_decision_layer.py",
        "splits": {"fit": fit_split, "eval": eval_split},
        "predictions": {
            "fit": _file_record(fit_predictions_path),
            "eval": _file_record(eval_predictions_path),
        },
        "hard_classes": {
            "source_kind": hard_class_source["kind"],
            "source": _file_record(hard_source_path),
            "algorithm": hard_algorithm,
            "parameters": hard_parameters,
            "tie_break": hard_tie_break,
        },
        "confusion_pairs": {
            "source_kind": confusion_pair_source["kind"],
            "source": _file_record(pair_source_path),
            "algorithm": pair_algorithm,
            "parameters": {"max_pairs": max_confusion_pairs},
            "tie_break": CONFUSION_PAIR_TIE_BREAK,
        },
        "routing": {
            "module": routing_module,
            "function": route_decision.__name__,
            "mode": "image",
            "module_sha256": sha256_file(routing_file) if routing_file.is_file() else None,
        },
        "policy_search": {
            "auto_confidence_grid": [float(v) for v in AUTO_CONFIDENCE_GRID],
            "suggest_confidence_grid": [float(v) for v in SUGGEST_CONFIDENCE_GRID],
            "margin_grid": [float(v) for v in MARGIN_GRID],
            "constraint": "suggest_confidence < auto_confidence",
            "min_auto_accept_accuracy": MIN_AUTO_ACCEPT_ACCURACY,
            "min_suggest_accuracy": MIN_SUGGEST_ACCURACY,
            "objective": "2 * auto_accept_coverage + suggest_coverage - 0.5 * review_coverage",
        },
    }


def format_provenance_summary(provenance: dict[str, object]) -> str:
    """A short human-readable digest of `provenance` for the run log."""
    splits = provenance["splits"]
    predictions = provenance["predictions"]
    hard = provenance["hard_classes"]
    pairs = provenance["confusion_pairs"]
    routing = provenance["routing"]
    return "\n".join(
        [
            "Derivation provenance:",
            f"  splits: fit={splits['fit']} eval={splits['eval']}",
            f"  fit predictions:  {predictions['fit']['path']} "
            f"(sha256 {predictions['fit']['sha256'][:12]})",
            f"  eval predictions: {predictions['eval']['path']} "
            f"(sha256 {predictions['eval']['sha256'][:12]})",
            f"  hard classes: {hard['source_kind']} -> {hard['algorithm']} "
            f"{hard['parameters']}",
            f"  confusion pairs: {pairs['source_kind']} -> {pairs['algorithm']} "
            f"{pairs['parameters']}",
            f"  routing: {routing['module']}.{routing['function']}",
        ]
    )


def report_unnamed_sidecars(
    results_dir: Path,
    fit_split: str,
    hard_classes_file: str | None,
    confusion_pairs_file: str | None,
    class_report_file: str | None,
) -> None:
    """Print one informational line per conventional sidecar that is present but unused.

    This changes nothing: the run derives the input from the fit predictions
    either way. It only tells the reader that a file which looks like an
    input was deliberately not read, so nobody assumes it was.
    """
    sidecars = []
    if not confusion_pairs_file:
        sidecars.append((f"{fit_split}_confusion_pairs.csv", "--confusion-pairs-file"))
    if not hard_classes_file and not class_report_file:
        sidecars.append((f"{fit_split}_class_report.csv", "--class-report-file"))
    for name, flag in sidecars:
        if (results_dir / name).exists():
            print(
                f"note: {results_dir / name} is present but ignored; the input is "
                f"derived from the fit-split predictions. Name it with {flag} to use it."
            )


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

    # `class_report_file` is only ever what the caller explicitly passed --
    # unlike confusion pairs below, this is deliberately NOT auto-discovered
    # from a `<fit_split>_class_report.csv` convenience file in results_dir.
    # That auto-discovery used to be exactly how two runs being compared
    # silently diverged (one run directory happened to have the file, the
    # other didn't) -- see load_hard_classes()'s docstring and
    # docs/9_agent_log.md's 2026-09-14 Codex review. With no override, hard
    # classes are always derived the same way: from fit_predictions itself.
    report_unnamed_sidecars(
        results_dir, fit_split, hard_classes_file, confusion_pairs_file, class_report_file
    )
    if max_confusion_pairs < 1:
        raise ConfusionPairsError(
            f"--max-confusion-pairs must be at least 1; got {max_confusion_pairs}."
        )
    hard_classes, hard_class_source = load_hard_classes(
        results_dir, hard_classes_file, class_report_file, fit_predictions
    )
    print(
        f"Hard classes in use ({len(hard_classes)}, source: {hard_class_source['kind']}"
        + (f" {hard_class_source['path']}" if "path" in hard_class_source else "")
        + f"): {sorted(hard_classes)}"
    )

    # Derived from the fit split's predictions only -- never the eval split's
    # -- unless --confusion-pairs-file names a file. A conventional
    # `<fit_split>_confusion_pairs.csv` sitting in results_dir is NOT
    # auto-discovered: that is how the A3b and champion comparison runs ended
    # up on different derivation paths (docs/9_agent_log.md, the 2026-09-22
    # Codex review). It is mentioned below and otherwise ignored.
    confusion_pairs, confusion_pair_source = load_confusion_pairs(
        fit_predictions,
        results_dir,
        confusion_pairs_file,
        max_pairs=max_confusion_pairs,
    )
    print(
        f"Confusion pairs in use ({len(confusion_pairs)}, source: "
        f"{confusion_pair_source['kind']}"
        + (f" {confusion_pair_source['path']}" if "path" in confusion_pair_source else "")
        + ")"
    )

    provenance = build_provenance(
        fit_split=fit_split,
        eval_split=eval_split,
        fit_predictions_path=resolve_predictions_path(
            results_dir, fit_split, fit_predictions_file
        ),
        eval_predictions_path=resolve_predictions_path(
            results_dir, eval_split, eval_predictions_file
        ),
        hard_class_source=hard_class_source,
        confusion_pair_source=confusion_pair_source,
        max_confusion_pairs=max_confusion_pairs,
    )
    print(format_provenance_summary(provenance))

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
    (output_dir / PROVENANCE_FILE).write_text(json.dumps(provenance, indent=2) + "\n")

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
            PROVENANCE_FILE,
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
    except (PredictionSchemaError, DerivationInputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
