"""Measure the served model's latency and its errors on whole images and crops.

Runs the backend's own inference functions in this process, against the real
``app/artifacts/`` (or ``FOODLENS_ARTIFACT_DIR``), on a fixed, seeded sample of
the Food-101 test split. It answers the question Codex set before the next
training run (docs/9_agent_log.md, 2026-10-12 and 2026-10-15): is the limit
detection, cropping, classification or calibration?

What it records, per image and summarised:

- **Latency**, batch size 1, on this machine's device: cold costs (model load,
  first classification, first detection) and warm p50/p95/mean for decoding,
  classifier-only, classifier plus decision, detector-only, and the full
  multi-food path (detector plus crop classification).
- **Whole-image errors**: top-1/top-5 against the Food-101 label, and accuracy
  per decision band under the deployed policy.
- **Crop errors**: how many regions the detector proposed, how often it found
  none (the whole-image fallback), and whether the highest-confidence crop, or
  any crop, carries the true label. Food-101 images hold one labelled dish, so
  a crop that loses the label is a cropping or detection loss, not a second
  food.

Nothing is written to the artifact directory. Results go to ``--output-dir``
(default ``results/runtime_measurement_<UTC stamp>/``, untracked).

Usage:
    python scripts/measure_runtime.py --food101-root data/food-101 --per-class 2
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BANDS = ("auto_accept", "suggest", "confirm", "review")


# --- Pure helpers (no torch), tested in tests/test_measure_runtime.py ----------


def select_sample(test_list: Iterable[str], per_class: int, seed: int) -> list[str]:
    """A fixed sample of ``per_class`` entries per class from Food-101's ``meta/test.txt``.

    Entries are ``<class>/<image id>``. Classes and entries are sorted before a
    seeded draw, so the same seed and file give the same sample on any machine.
    """
    by_class: dict[str, list[str]] = defaultdict(list)
    for line in test_list:
        entry = line.strip()
        if entry:
            by_class[entry.split("/", 1)[0]].append(entry)
    rng = random.Random(seed)
    sample: list[str] = []
    for label in sorted(by_class):
        entries = sorted(by_class[label])
        sample += sorted(rng.sample(entries, min(per_class, len(entries))))
    return sample


def percentile(values: Sequence[float], q: float) -> float:
    """The ``q``-th percentile (0-100) by linear interpolation between ranks."""
    if not values:
        raise ValueError("percentile of no values")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def timing_summary(samples_ms: Sequence[float]) -> dict[str, float | int]:
    """Count, p50, p95, mean and max of a list of millisecond timings."""
    return {
        "n": len(samples_ms),
        "p50_ms": round(percentile(samples_ms, 50), 2),
        "p95_ms": round(percentile(samples_ms, 95), 2),
        "mean_ms": round(statistics.fmean(samples_ms), 2),
        "max_ms": round(max(samples_ms), 2),
    }


def rate(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


def accuracy_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Whole-image and crop accuracy, by decision band, from per-image rows."""
    total = len(rows)
    by_band: dict[str, dict[str, Any]] = {}
    for band in BANDS:
        in_band = [row for row in rows if row["whole_band"] == band]
        by_band[band] = {
            "n": len(in_band),
            "share": rate(len(in_band), total),
            "top1_accuracy": rate(sum(row["whole_top1_correct"] for row in in_band), len(in_band)),
            "top5_containment": rate(
                sum(row["whole_top5_correct"] for row in in_band), len(in_band)
            ),
        }
    detected = [row for row in rows if row.get("crop_count") is not None]
    real_crops = [row for row in detected if not row["detector_fallback"]]
    crop_bands = Counter(row["best_crop_band"] for row in detected)
    return {
        "images": total,
        "whole_image": {
            "top1_accuracy": rate(sum(row["whole_top1_correct"] for row in rows), total),
            "top5_accuracy": rate(sum(row["whole_top5_correct"] for row in rows), total),
            "by_band": by_band,
        },
        "crops": {
            "images": len(detected),
            "detector_fallback_rate": rate(
                sum(row["detector_fallback"] for row in detected), len(detected)
            ),
            "mean_regions": (
                round(statistics.fmean(row["crop_count"] for row in detected), 2)
                if detected
                else None
            ),
            "best_crop_top1_accuracy": rate(
                sum(row["best_crop_top1_correct"] for row in detected), len(detected)
            ),
            "any_crop_top1_accuracy": rate(
                sum(row["any_crop_top1_correct"] for row in detected), len(detected)
            ),
            # Only images where the detector proposed real regions: the effect
            # of cropping itself, without the whole-image fallback mixed in.
            "with_regions": {
                "images": len(real_crops),
                "whole_top1_accuracy": rate(
                    sum(row["whole_top1_correct"] for row in real_crops), len(real_crops)
                ),
                "best_crop_top1_accuracy": rate(
                    sum(row["best_crop_top1_correct"] for row in real_crops), len(real_crops)
                ),
            },
            "whole_right_best_crop_wrong": sum(
                row["whole_top1_correct"] and not row["best_crop_top1_correct"]
                for row in detected
            ),
            "whole_wrong_best_crop_right": sum(
                not row["whole_top1_correct"] and row["best_crop_top1_correct"]
                for row in detected
            ),
            "best_crop_bands": {band: crop_bands.get(band, 0) for band in BANDS},
        },
    }


def top_confusions(rows: Sequence[Mapping[str, Any]], limit: int = 15) -> list[dict[str, Any]]:
    """The most frequent (true, predicted) whole-image errors."""
    counts = Counter(
        (row["label"], row["whole_top1"]) for row in rows if not row["whole_top1_correct"]
    )
    return [
        {"actual": actual, "predicted": predicted, "count": count}
        for (actual, predicted), count in counts.most_common(limit)
    ]


# --- Measurement (needs torch, the artifacts and, for crops, ultralytics) -------


def timed(function: Callable[..., Any], *args: Any) -> tuple[Any, float]:
    """``function(*args)`` and how long it took, in milliseconds."""
    start = time.perf_counter()
    value = function(*args)
    return value, (time.perf_counter() - start) * 1000


def hardware() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(getattr(torch.backends, "mps", None))
        and torch.backends.mps.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
    return info


def measure(args: argparse.Namespace) -> dict[str, Any]:
    from app.backend import inference
    from app.backend.decision import build_decision
    from app.backend.imaging import build_full_image_region, open_rgb_image

    root = args.food101_root
    sample = select_sample(
        (root / "meta" / "test.txt").read_text(encoding="utf-8").splitlines(),
        args.per_class,
        args.seed,
    )
    images = [(entry, root / "images" / f"{entry}.jpg") for entry in sample]
    missing = [str(path) for _, path in images if not path.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} sampled images are missing, e.g. {missing[0]}")

    cold: dict[str, float] = {}
    runtime, cold["load_runtime_ms"] = timed(inference.load_runtime)
    first_image = open_rgb_image(images[0][1].read_bytes())
    _, cold["first_classification_ms"] = timed(
        inference.classify_pil_image, first_image, runtime
    )
    detector_error = None
    if not args.skip_detector:
        try:
            _, cold["first_detection_ms"] = timed(
                inference.detect_candidate_regions, first_image
            )
        except Exception as exc:  # reported, never hidden: crops are then skipped
            detector_error = f"{type(exc).__name__}: {exc}"
    run_detector = not args.skip_detector and detector_error is None

    def route(predictions: list[Any]) -> str:
        return build_decision(
            "image",
            predictions,
            policy=runtime["policy"],
            hard_classes=runtime["hard_classes"],
            confusion_pairs=runtime["confusion_pairs"],
        ).band

    timings: dict[str, list[float]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for index, (entry, path) in enumerate(images):
        label = entry.split("/", 1)[0]
        image_bytes = path.read_bytes()
        image, ms = timed(open_rgb_image, image_bytes)
        timings["decode"].append(ms)
        predictions, ms = timed(inference.classify_pil_image, image, runtime)
        timings["classifier_only"].append(ms)
        _, ms = timed(inference.build_prediction_response, image, runtime)
        timings["classifier_with_decision"].append(ms)
        labels = [prediction.class_name for prediction in predictions]
        row: dict[str, Any] = {
            "image": entry,
            "label": label,
            "width": image.size[0],
            "height": image.size[1],
            "whole_top1": labels[0],
            "whole_confidence": round(predictions[0].confidence, 4),
            "whole_top1_correct": labels[0] == label,
            "whole_top5_correct": label in labels,
            "whole_band": route(predictions),
            "crop_count": None,
        }
        if run_detector:
            detections, ms = timed(inference.detect_candidate_regions, image)
            timings["detector_only"].append(ms)
            fallback = not detections
            regions = detections or [build_full_image_region(image)]
            response, ms = timed(inference.build_multi_food_response, image, regions, runtime)
            timings["multi_food_crops"].append(ms)
            timings["detector_plus_crops"].append(timings["detector_only"][-1] + ms)
            crops = response.predictions
            best = max(crops, key=lambda crop: crop.detector.confidence)
            row.update(
                {
                    "crop_count": len(crops),
                    "detector_fallback": fallback,
                    "best_crop_top1": best.foodlens.top_label,
                    "best_crop_area_ratio": round(best.detector.crop_area_ratio, 4),
                    "best_crop_top1_correct": best.foodlens.top_label == label,
                    # Crops are routed by the multi-food policy, not the
                    # deployed one; this records the band the app shows.
                    "best_crop_band": best.foodlens.decision_band,
                    "any_crop_top1_correct": any(
                        crop.foodlens.top_label == label for crop in crops
                    ),
                }
            )
        rows.append(row)
        if (index + 1) % 50 == 0:
            print(f"{index + 1}/{len(images)} images", file=sys.stderr)

    identity = runtime["model_identity"]
    layer = inference.decision_layer_status()
    return {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "hardware": hardware(),
        "device": str(runtime["device"]),
        "batch_size": 1,
        "sample": {
            "split": "Food-101 meta/test.txt",
            "per_class": args.per_class,
            "seed": args.seed,
            "images": len(images),
        },
        "model": {
            key: identity.get(key)
            for key in ("architecture", "checkpoint", "model_name", "checkpoint_sha256")
        },
        "temperature": runtime["temperature"],
        "class_names_sha256": runtime["class_names_sha256"],
        "decision_layer": {
            "policy": runtime["policy"],
            "fingerprint": layer.get("fingerprint"),
            "hard_class_count": len(runtime["hard_classes"]),
            "confusion_pair_count": len(runtime["confusion_pairs"]),
        },
        "detector": {
            "weights": inference.detector_weights_path(),
            "ran": run_detector,
            "error": detector_error,
        },
        "cold_ms": {key: round(value, 2) for key, value in cold.items()},
        "warm_ms": {stage: timing_summary(values) for stage, values in timings.items()},
        "accuracy": accuracy_summary(rows),
        "top_confusions": top_confusions(rows),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--food101-root",
        type=Path,
        required=True,
        help="Food-101 directory holding meta/test.txt and images/",
    )
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--skip-detector", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.per_class < 1:
        parser.error("--per-class must be at least 1")

    sys.path.insert(0, str(REPO_ROOT))
    result = measure(args)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir or REPO_ROOT / "results" / f"runtime_measurement_{stamp}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "measurement.json").write_text(json.dumps(result, indent=2) + "\n")
    summary = {key: value for key, value in result.items() if key != "rows"}
    print(json.dumps(summary, indent=2))
    print(f"wrote {output / 'measurement.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
