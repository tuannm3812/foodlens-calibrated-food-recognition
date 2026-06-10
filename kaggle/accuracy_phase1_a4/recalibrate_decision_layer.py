"""Recalibrate decision-layer thresholds for a finished classifier run.

This utility expects a run directory containing per-split prediction and report
artifacts from `foodlens_accuracy_phase1_a*` experiments, and emits a small
artifact bundle matching the app contract:

- decision_policy.json
- hard_classes.json
- confusion_pairs.json
- decision_policy.csv
- decision_policy_search.csv
- decision_band_metrics.csv
- decision_examples_<band>.csv
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


AUTO_HARD_CLASSES = {
    "chocolate_mousse",
    "steak",
    "pork_chop",
    "bread_pudding",
    "tuna_tartare",
}


AUTO_CONFIDENCE_GRID = tuple(np.round(np.arange(0.70, 0.96, 0.05), 2))
SUGGEST_CONFIDENCE_GRID = tuple(np.round(np.arange(0.35, 0.76, 0.05), 2))
MARGIN_GRID = tuple(np.round(np.arange(0.05, 0.51, 0.05), 2))
MIN_AUTO_ACCEPT_ACCURACY = 0.90
MIN_SUGGEST_ACCURACY = 0.80
ZIP_NAME = "decision_layer_artifacts.zip"


def parse_args() -> argparse.Namespace:
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
        default="test",
        choices=("val", "test"),
        help="Split used for recalibration",
    )
    parser.add_argument(
        "--hard-classes-file",
        default=None,
        help="Optional CSV/JSON with explicit hard-class names",
    )
    parser.add_argument(
        "--confusion-pairs-file",
        default=None,
        help="Optional CSV with confusion pairs and optional counts",
    )
    parser.add_argument(
        "--class-report-file",
        default=None,
        help="Optional class-report CSV for deriving hard classes",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory; defaults to <results-dir>/<split>_decision_layer"
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
    return parser.parse_args()


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


def load_hard_classes(
    results_dir: Path,
    hard_classes_file: str | None,
    class_report_file: str | None = None,
) -> set[str]:
    if hard_classes_file:
        path = Path(hard_classes_file)
        if not path.is_absolute():
            path = results_dir / path
        text = path.read_text().strip()
        if not text:
            return set(AUTO_HARD_CLASSES)
        return {
            entry.strip().strip('"\'') for entry in json.loads(text) if isinstance(entry, str)
        } or set(AUTO_HARD_CLASSES)

    if not class_report_file:
        return set(AUTO_HARD_CLASSES)

    path = Path(class_report_file)
    if not path.is_absolute():
        path = results_dir / path

    if not path.exists():
        return set(AUTO_HARD_CLASSES)

    report = pd.read_csv(path)
    report = normalize_class_report(report)

    if {
        "class_name",
        "f1-score",
    } <= set(report.columns):
        low_f1 = report.sort_values("f1-score", ascending=True)
        limit = max(5, math.ceil(0.1 * len(low_f1)))
        return set(low_f1.head(limit)["class_name"].astype(str).tolist())

    if "class_name" in report.columns:
        return set(report["class_name"].astype(str).head(8).tolist())

    return set(AUTO_HARD_CLASSES)


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
) -> str:
    if row["is_frequent_confusion_pair"]:
        return "review"

    if row["is_hard_case"] and row["top_1_confidence"] < auto_confidence:
        return "confirm"

    if (
        row["top_1_confidence"] >= auto_confidence
        and row["top_1_top_2_margin"] >= margin_threshold
        and not row["is_hard_case"]
    ):
        return "auto_accept"

    if row["top_1_confidence"] >= suggest_confidence and row["top_5_contains_actual"]:
        return "suggest"

    return "confirm"


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
) -> dict[str, float | int]:
    scored = features_df.copy()
    scored["decision_band"] = scored.apply(
        assign_decision_band,
        axis=1,
        auto_confidence=auto_confidence,
        suggest_confidence=suggest_confidence,
        margin_threshold=margin_threshold,
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
        for actual, predicted in zip(actual_series, predicted_series)
    ]
    features["top_5_contains_actual"] = features.apply(
        lambda row: row[actual_col] in row["top_5_labels"], axis=1
    )

    return features


def run_analysis(
    results_dir: Path,
    split: str,
    hard_classes_file: str | None,
    confusion_pairs_file: str | None,
    class_report_file: str | None,
    output_dir: Path,
    max_confusion_pairs: int,
    skip_zip: bool,
) -> None:
    predictions_path = results_dir / f"{split}_predictions.csv"
    class_report_path = results_dir / f"{split}_class_report.csv"
    confusion_path = results_dir / f"{split}_confusion_pairs.csv"

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Missing prediction file: {predictions_path}. Re-run train script first."
        )

    predictions = normalize_actual_col(pd.read_csv(predictions_path))[0]
    if not class_report_file:
        class_report_file = str(class_report_path) if class_report_path.exists() else None
    hard_classes = load_hard_classes(results_dir, hard_classes_file, class_report_file)

    if not confusion_pairs_file:
        confusion_pairs_file = (
            str(confusion_path) if confusion_path.exists() else None
        )

    confusion_pairs = load_confusion_pairs(
        predictions,
        results_dir,
        confusion_pairs_file,
        max_pairs=max_confusion_pairs,
    )

    features_df = build_features(predictions, hard_classes, confusion_pairs)
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
        policy_search_df["meets_auto_accuracy"]
        & policy_search_df["meets_suggest_accuracy"]
    ].copy()
    if not eligible.empty:
        best_policy = eligible.sort_values("policy_score", ascending=False).iloc[0]
    else:
        best_policy = policy_search_df.sort_values("policy_score", ascending=False).iloc[0]

    output_dir.mkdir(parents=True, exist_ok=True)

    auto_confidence = float(best_policy["auto_confidence"])
    suggest_confidence = float(best_policy["suggest_confidence"])
    margin_threshold = float(best_policy["margin_threshold"])

    final = features_df.copy()
    final["decision_band"] = final.apply(
        assign_decision_band,
        axis=1,
        auto_confidence=auto_confidence,
        suggest_confidence=suggest_confidence,
        margin_threshold=margin_threshold,
    )

    band_metrics = decision_band_metrics(final)

    policy_row = pd.DataFrame(
        [
            {
                "auto_confidence": auto_confidence,
                "suggest_confidence": suggest_confidence,
                "margin_threshold": margin_threshold,
                "min_auto_accept_accuracy": MIN_AUTO_ACCEPT_ACCURACY,
                "min_suggest_accuracy": MIN_SUGGEST_ACCURACY,
            }
        ]
    )

    policy_search_df.to_csv(output_dir / "decision_policy_search.csv", index=False)
    features_df.to_csv(output_dir / "decision_features.csv", index=False)
    final.to_csv(output_dir / f"{split}_predictions_with_decisions.csv", index=False)
    band_metrics.to_csv(output_dir / "decision_band_metrics.csv", index=False)
    policy_row.to_csv(output_dir / "decision_policy.csv", index=False)

    (output_dir / "decision_policy.json").write_text(
        policy_row.to_json(orient="records", indent=2)
    )

    confusion_records = [
        {"actual": actual, "predicted": predicted}
        for actual, predicted in sorted(confusion_pairs)
    ]
    (output_dir / "confusion_pairs.json").write_text(
        json.dumps(confusion_records, indent=2)
    )
    (output_dir / "hard_classes.json").write_text(
        json.dumps(sorted(hard_classes), indent=2)
    )

    confusion_df = pd.DataFrame(confusion_records)
    if not confusion_df.empty:
        confusion_df.to_csv(output_dir / "top_confusion_pairs.csv", index=False)

    for band in ("auto_accept", "suggest", "confirm", "review"):
        final[final["decision_band"] == band].head(25).to_csv(
            output_dir / f"decision_examples_{band}.csv",
            index=False,
        )

    if not skip_zip:
        zip_path = output_dir / ZIP_NAME
        artifact_files = [
            "decision_features.csv",
            "decision_policy_search.csv",
            f"{split}_predictions_with_decisions.csv",
            "decision_band_metrics.csv",
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

    print(f"Decision policy recalibration complete for split={split}")
    print(f"Artifacts in: {output_dir}")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else results_dir / f"{args.split}_decision_layer"
    )

    run_analysis(
        results_dir=results_dir,
        split=args.split,
        hard_classes_file=args.hard_classes_file,
        confusion_pairs_file=args.confusion_pairs_file,
        class_report_file=args.class_report_file,
        output_dir=output_dir,
        max_confusion_pairs=args.max_confusion_pairs,
        skip_zip=args.no_zip,
    )


if __name__ == "__main__":
    main()
