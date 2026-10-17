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
  any crop, carries the true label. "Best crop" is the one the app shows first:
  the frontend sorts regions by ``foodlens.top_confidence`` (a stable sort, so
  ties keep the response order). The detector-confidence pick is recorded
  alongside as a separate strategy. Every crop's detector label, box, area and
  classification is saved per image, so crops can be labelled individually
  later: a crop scored against the dish label is only a proxy, since it may
  hold a side item.
- **Confident false accepts**: wrong answers routed to auto-accept, for whole
  images and for best crops.
- **HTTP request time**, optionally (``--api-url``): each image posted to a
  running API's whole-image and multi-food endpoints, timed end to end. After
  one warm-up request per endpoint, the API's loaded model, temperature, class
  order and decision fingerprint must match this process's runtime, or the run
  stops. Each timed response is checked too; one that fell back to a demo or
  reports another model is counted as invalid and left out of the timings.

``--reanalyze measurement.json`` recomputes the crop strategies from a saved
run's per-crop records without running inference again.

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


def ui_best(crops: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """The crop the app shows first: highest classifier confidence, first on ties."""
    return max(crops, key=lambda crop: crop["top1_confidence"])


def detector_best(crops: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """The crop with the highest detector confidence, first on ties."""
    return max(crops, key=lambda crop: crop["detector_confidence"])


def strategy_summary(
    rows: Sequence[Mapping[str, Any]], min_area: float = 0.2
) -> dict[str, Any]:
    """Top-1 against the image label under each way of picking one answer.

    Uses each row's saved ``crops``. The ``*_min_area`` strategies drop crops
    smaller than ``min_area`` of the image and use the whole image when none
    remain. These are agreements with the dish label, not crop accuracy.
    """
    detected = [row for row in rows if row.get("crops")]

    def correct(row: Mapping[str, Any], crop: Mapping[str, Any] | None) -> bool:
        if crop is None:
            return bool(row["whole_top1_correct"])
        return crop["top1"] == row["label"]

    def large(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        return [crop for crop in row["crops"] if crop["area_ratio"] >= min_area]

    picks = {
        "whole_image": lambda row: None,
        "ui_ranked": lambda row: ui_best(row["crops"]),
        "detector_ranked": lambda row: detector_best(row["crops"]),
        "ui_ranked_min_area": lambda row: ui_best(large(row)) if large(row) else None,
        "detector_ranked_min_area": (
            lambda row: detector_best(large(row)) if large(row) else None
        ),
    }
    summary: dict[str, Any] = {"images": len(detected), "min_area": min_area}
    for name, pick in picks.items():
        hits = sum(correct(row, pick(row)) for row in detected)
        summary[name] = {"correct": hits, "top1": rate(hits, len(detected))}
    summary["ui_and_detector_pick_differ"] = sum(
        ui_best(row["crops"]) is not detector_best(row["crops"]) for row in detected
    )
    return summary


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
            "detector_ranked_best_crop_top1_accuracy": rate(
                sum(row["detector_best_crop_top1_correct"] for row in detected), len(detected)
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
    """One crop's detector proposal and classification, for later labelling.

    Scores and area keep full precision: ``--reanalyze`` picks crops from
    them, and rounding could turn two close scores into a tie the live pick
    never saw.
    """
    box = crop.bbox
    return {
        "detection_index": crop.detection_index,
        "detector_label": crop.detector.label,
        "proposal_role": crop.detector.proposal_role,
        "detector_confidence": crop.detector.confidence,
        "area_ratio": crop.detector.crop_area_ratio,
        "box": [box.x1, box.y1, box.x2, box.y2],
        "top1": crop.foodlens.top_label,
        "top1_confidence": crop.foodlens.top_confidence,
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


IDENTITY_KEYS = ("architecture", "model_name", "checkpoint_sha256")


def api_identity_problems(
    status: Mapping[str, Any], expected: Mapping[str, Any], *, multi_food: bool
) -> list[str]:
    """How a warmed API's /runtime/status differs from the measured runtime."""
    model = status.get("model") or {}
    problems = []
    if model.get("source") != "loaded_runtime":
        problems.append(f"model.source is {model.get('source')!r}, not 'loaded_runtime'")
    for key in (*IDENTITY_KEYS, "temperature", "class_names_sha256"):
        if model.get(key) != expected[key]:
            problems.append(f"model.{key}: API {model.get(key)!r}, measured {expected[key]!r}")
    fingerprint = (status.get("decision_layer") or {}).get("fingerprint")
    if fingerprint != expected["fingerprint"]:
        problems.append(
            f"decision_layer.fingerprint: API {fingerprint!r}, measured {expected['fingerprint']!r}"
        )
    mode = (status.get("multi_food") or {}).get("mode")
    if multi_food and mode != "live_yolo_classifier":
        problems.append(f"multi_food.mode is {mode!r}, not 'live_yolo_classifier'")
    return problems


def response_problem(
    endpoint: str, body: Mapping[str, Any], expected: Mapping[str, Any]
) -> str | None:
    """Why a response is not live A3b inference, or None when it is.

    A multi-food response may fall back to the whole image when the detector
    finds nothing: that is still the measured classifier, so it counts.
    """
    name_key = "model_name" if endpoint == "image" else "model"
    allowed_fallbacks = {None} if endpoint == "image" else {None, "no_detector_regions"}
    if body.get("artifact_status") != "ready":
        return f"artifact_status {body.get('artifact_status')!r}"
    if body.get("fallback_reason") not in allowed_fallbacks:
        return f"fallback_reason {body.get('fallback_reason')!r}"
    if body.get(name_key) != expected["model_name"]:
        return f"{name_key} {body.get(name_key)!r}"
    if body.get("temperature") != expected["temperature"]:
        return f"temperature {body.get('temperature')!r}"
    if endpoint == "multi_food" and not str(body.get("detector_status", "")).startswith(
        "live_yolo"
    ):
        return f"detector_status {body.get('detector_status')!r}"
    return None


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
    api_status = None
    api_invalid: list[dict[str, str]] = []
    expected = {
        **{key: runtime["model_identity"].get(key) for key in IDENTITY_KEYS},
        "temperature": runtime["temperature"],
        "class_names_sha256": runtime["class_names_sha256"],
        "fingerprint": inference.decision_layer_status().get("fingerprint"),
    }
    endpoints = {"image": "/predict/image"}
    if run_detector:
        endpoints["multi_food"] = "/predict/multi-food/image"
    if api_url:
        # Warm each endpoint once, so a lazily loading server has loaded its
        # model, then require it to be the runtime measured here.
        first_bytes = images[0][1].read_bytes()
        for endpoint, route_path in endpoints.items():
            body = post_image(f"{api_url}{route_path}", images[0][1].name, first_bytes)
            problem = response_problem(endpoint, body, expected)
            if problem:
                raise SystemExit(f"API warm-up on {route_path} is not live A3b: {problem}")
        api_status = get_json(f"{api_url}/runtime/status")
        problems = api_identity_problems(api_status, expected, multi_food=run_detector)
        if problems:
            raise SystemExit("API serves a different runtime: " + "; ".join(problems))

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
            # The app's pick: highest classifier confidence, first on ties.
            best = max(crops, key=lambda crop: crop.foodlens.top_confidence)
            by_detector = max(crops, key=lambda crop: crop.detector.confidence)
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
                    "detector_best_crop_top1": by_detector.foodlens.top_label,
                    "detector_best_crop_top1_correct": by_detector.foodlens.top_label == label,
                    "any_crop_top1_correct": any(
                        crop.foodlens.top_label == label for crop in crops
                    ),
                    "crops": [crop_record(crop) for crop in crops],
                }
            )
        if api_url:
            for endpoint, route_path in endpoints.items():
                body, ms = timed(post_image, f"{api_url}{route_path}", path.name, image_bytes)
                problem = response_problem(endpoint, body, expected)
                if problem:
                    api_invalid.append({"image": entry, "endpoint": endpoint, "problem": problem})
                else:
                    timings[f"http_{endpoint}"].append(ms)
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
            {
                "url": api_url,
                "runtime_status_after_warmup": api_status,
                "invalid_responses": api_invalid,
            }
            if api_url
            else None
        ),
        "cold_ms": {key: round(value, 2) for key, value in cold.items()},
        "warm_ms": {stage: timing_summary(values) for stage, values in timings.items()},
        "accuracy": accuracy_summary(rows),
        "strategies": strategy_summary(rows, args.min_area),
        "top_confusions": top_confusions(rows),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--reanalyze",
        type=Path,
        help="a saved measurement.json: print its crop strategies and exit",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="directory of class folders holding the images",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
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
    parser.add_argument("--min-area", type=float, default=0.2)
    args = parser.parse_args(argv)
    if args.reanalyze:
        rows = json.loads(args.reanalyze.read_text())["rows"]
        print(json.dumps(strategy_summary(rows, args.min_area), indent=2))
        return 0
    if args.images_dir is None or args.manifest is None:
        parser.error("--images-dir and --manifest are required unless --reanalyze is given")
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
