"""Measure the served model's latency and its errors on whole images and crops.

Runs the backend's own inference functions in this process, against the real
``app/artifacts/`` (or ``FOODLENS_ARTIFACT_DIR``), on a fixed, seeded sample of
one of the served run's own splits. It answers the question Codex set before the next
training run (docs/9_agent_log.md, 2026-10-12 and 2026-10-15): is the limit
detection, cropping, classification or calibration?

What it records, per image and summarised:

- **Latency**, batch size 1, on this machine's device: cold costs (model load,
  first classification, first detection) and warm p50/p95/mean for decoding,
  classifier-only, classifier plus decision, detector-only, and the full
  multi-food path (detector plus crop classification).
- **Whole-image errors**: top-1/top-5 against the true label, and accuracy
  per decision band under the deployed policy.
- **Crop errors**: how many regions the detector proposed, how often it found
  none (the whole-image fallback), and whether the highest-confidence crop, or
  any crop, carries the true label. Every crop's detector label, box, area and
  classification is saved per image, so crops can be labelled individually
  later: a crop scored against the dish label is only a proxy, since it may
  hold a side item.
- **Confident false accepts**: wrong answers routed to auto-accept, for whole
  images and for best crops.
- **HTTP request time**, optionally (``--api-url``): each image posted to a
  running API's whole-image and multi-food endpoints, timed end to end.

**The split must be the served run's own.** A3b trained on a stratified
80/10/10 split of all 101,000 images (seed 42), not on Food-101's official
``meta/test.txt``, so most official test images were in its training set. A
sample from that file measured 90.6% top-1 against the recorded 83.90%. Pass
one of the run's manifests (``path,label``); each row's file is found by label
and file name under ``--images-dir``, whatever machine wrote the path. Use
``val_manifest.csv`` to choose what to change next, and keep
``test_manifest.csv`` for a final report, so no decision is tuned on test labels.

Nothing is written to the artifact directory. Results go to ``--output-dir``
(default ``results/runtime_measurement_<UTC stamp>/``, untracked).

Usage:
    python scripts/measure_runtime.py --images-dir data/food41/images \
        --manifest results/accuracy_phase1/a3b_convnext_tiny_continued_224/val_manifest.csv \
        --api-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import statistics
import sys
import time
import urllib.request
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BANDS = ("auto_accept", "suggest", "confirm", "review")


# --- Pure helpers (no torch), tested in tests/test_measure_runtime.py ----------


def manifest_entries(rows: Iterable[Mapping[str, str]]) -> list[str]:
    """``<label>/<file name>`` for each row of a run's ``path,label`` manifest.

    The recorded path is the training machine's; only its file name is kept.
    """
    return [f"{row['label']}/{Path(row['path']).name}" for row in rows]


def select_sample(test_list: Iterable[str], per_class: int, seed: int) -> list[str]:
    """A fixed sample of ``per_class`` entries per class.

    Entries are ``<class>/<file name>``. Classes and entries are sorted before a
    seeded draw, so the same seed and manifest give the same sample anywhere.
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
    false_accepts = sum(
        row["whole_band"] == "auto_accept" and not row["whole_top1_correct"] for row in rows
    )
    detected = [row for row in rows if row.get("crop_count") is not None]
    real_crops = [row for row in detected if not row["detector_fallback"]]
    crop_bands = Counter(row["best_crop_band"] for row in detected)
    return {
        "images": total,
        "whole_image": {
            "top1_accuracy": rate(sum(row["whole_top1_correct"] for row in rows), total),
            "top5_accuracy": rate(sum(row["whole_top5_correct"] for row in rows), total),
            "by_band": by_band,
            "confident_false_accepts": false_accepts,
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
            "best_crop_confident_false_accepts": sum(
                row["best_crop_band"] == "auto_accept" and not row["best_crop_top1_correct"]
                for row in detected
            ),
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


def crop_record(crop: Any) -> dict[str, Any]:
    """One crop's detector proposal and classification, for later labelling."""
    box = crop.bbox
    return {
        "detection_index": crop.detection_index,
        "detector_label": crop.detector.label,
        "proposal_role": crop.detector.proposal_role,
        "detector_confidence": round(crop.detector.confidence, 4),
        "area_ratio": round(crop.detector.crop_area_ratio, 4),
        "box": [box.x1, box.y1, box.x2, box.y2],
        "top1": crop.foodlens.top_label,
        "top1_confidence": round(crop.foodlens.top_confidence, 4),
        "band": crop.foodlens.decision_band,
    }


def multipart_body(file_name: str, data: bytes) -> tuple[bytes, str]:
    """A ``multipart/form-data`` body with one ``file`` field, and its type."""
    boundary = uuid.uuid4().hex
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{file_name}"\r\n'.encode(),
            b"Content-Type: image/jpeg\r\n\r\n",
            data,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    return body, f"multipart/form-data; boundary={boundary}"


def post_image(url: str, file_name: str, data: bytes) -> dict[str, Any]:
    body, content_type = multipart_body(file_name, data)
    request = urllib.request.Request(
        url, data=body, method="POST", headers={"Content-Type": content_type}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read())


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

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        entries = manifest_entries(csv.DictReader(handle))
    sample = select_sample(entries, args.per_class, args.seed)
    images = [(entry, args.images_dir / entry) for entry in sample]
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
    api_url = args.api_url.rstrip("/") if args.api_url else None
    api_status = get_json(f"{api_url}/runtime/status") if api_url else None
    api_warmup: dict[str, Any] = {}
    if api_url:
        # One request per endpoint first, so a lazily loaded server is warm
        # and its served model and status are on record before timing.
        first_bytes = images[0][1].read_bytes()
        whole = post_image(f"{api_url}/predict/image", images[0][1].name, first_bytes)
        api_warmup["image_model_name"] = whole.get("model_name")
        api_warmup["image_artifact_status"] = whole.get("artifact_status")
        if run_detector:
            multi = post_image(
                f"{api_url}/predict/multi-food/image", images[0][1].name, first_bytes
            )
            api_warmup["multi_food_detector_status"] = multi.get("detector_status")
            api_warmup["multi_food_artifact_status"] = multi.get("artifact_status")

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
                    "crops": [crop_record(crop) for crop in crops],
                }
            )
        if api_url:
            _, ms = timed(post_image, f"{api_url}/predict/image", path.name, image_bytes)
            timings["http_image"].append(ms)
            if run_detector:
                _, ms = timed(
                    post_image, f"{api_url}/predict/multi-food/image", path.name, image_bytes
                )
                timings["http_multi_food"].append(ms)
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
            "manifest": str(args.manifest),
            "manifest_rows": len(entries),
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
        "api": (
            {"url": api_url, "runtime_status": api_status, "warmup": api_warmup}
            if api_url
            else None
        ),
        "cold_ms": {key: round(value, 2) for key, value in cold.items()},
        "warm_ms": {stage: timing_summary(values) for stage, values in timings.items()},
        "accuracy": accuracy_summary(rows),
        "top_confusions": top_confusions(rows),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="directory of class folders holding the images",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="one of the served run's manifests (path,label): val to decide, test to report",
    )
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--skip-detector", action="store_true")
    parser.add_argument(
        "--api-url",
        help="a running API (e.g. http://127.0.0.1:8000) to time end-to-end requests against",
    )
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
