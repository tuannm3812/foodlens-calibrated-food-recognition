"""Running the YOLO detector to propose candidate food regions.

Must not import inference. The public, patchable entry point
(detect_candidate_regions) stays in inference.py as a thin wrapper: it
resolves the weights path via detector_weights_path() -- which can't move,
since it derives the repo root from Path(__file__) and a test patches
inference.__file__ -- and calls run_yolo_detection() here with that value.

The YOLO model is loaded once per weights file and reused across requests.
FastAPI runs sync handlers in a threadpool, and an Ultralytics model is not
safe to call from several threads at once (its predictor is built lazily and
holds per-call state), so each cached model's predict runs under its own lock.
"""

import os
import threading
from pathlib import Path
from typing import Any

from .detector_policy import (
    DETECTOR_CONFIDENCE_THRESHOLD,
    DETECTOR_IOU_THRESHOLD,
    DETECTOR_MAX_DETECTIONS,
    detector_label_filter_config,
    detector_region_role,
    should_export_detection,
)

_cache_lock = threading.Lock()
# Resolved weights path -> (file signature, model, predict lock).
_detector_cache: dict[str, tuple[tuple[int, int] | None, Any, threading.Lock]] = {}


def _weights_signature(resolved_path: str) -> tuple[int, int] | None:
    """mtime and size of the weights file, or None when it does not exist.

    A file replaced in place gets a new signature, so it is reloaded rather
    than served from a stale model. A missing file is passed to YOLO as
    given, which may download it; the signature then changes once it exists.
    """
    try:
        stat = os.stat(resolved_path)
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def cached_detector(weights_path: str, yolo_class: Any) -> tuple[Any, threading.Lock]:
    """Return the shared model for this weights file and the lock guarding it."""
    resolved_path = str(Path(weights_path).resolve())
    signature = _weights_signature(resolved_path)
    with _cache_lock:
        entry = _detector_cache.get(resolved_path)
        if entry is not None and entry[0] == signature:
            return entry[1], entry[2]
        # Construct before storing, so a failed load is not cached.
        detector = yolo_class(weights_path)
        lock = threading.Lock()
        _detector_cache[resolved_path] = (signature, detector, lock)
        return detector, lock


def clear_detector_cache() -> None:
    """Drop every cached model; the next detection loads afresh."""
    with _cache_lock:
        _detector_cache.clear()


def run_yolo_detection(image: Any, weights_path: str) -> list[dict[str, Any]]:
    """Run the YOLO detector against an image and return usable crop rows."""
    detector_filter_mode, configured_labels = detector_label_filter_config()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Multi-food detection requires ultralytics.") from exc

    detector, predict_lock = cached_detector(weights_path, YOLO)
    with predict_lock:
        result = detector.predict(
            source=image,
            conf=DETECTOR_CONFIDENCE_THRESHOLD,
            iou=DETECTOR_IOU_THRESHOLD,
            max_det=DETECTOR_MAX_DETECTIONS,
            verbose=False,
        )[0]

    source_width, source_height = image.size
    source_area = source_width * source_height
    rows: list[dict[str, Any]] = []

    boxes = result.boxes
    if boxes is None:
        return rows

    for detection_index, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        x1 = max(0, min(x1, source_width))
        x2 = max(0, min(x2, source_width))
        y1 = max(0, min(y1, source_height))
        y2 = max(0, min(y2, source_height))
        if x2 <= x1 or y2 <= y1:
            continue

        detector_class_id = int(box.cls[0])
        detector_label = str(result.names.get(detector_class_id, detector_class_id))
        crop_area_ratio = ((x2 - x1) * (y2 - y1)) / source_area
        if not should_export_detection(
            detector_label=detector_label,
            area_ratio=crop_area_ratio,
            filter_mode=detector_filter_mode,
            configured_labels=configured_labels,
        ):
            continue

        rows.append(
            {
                "detection_index": detection_index,
                "detector_label": detector_label,
                "proposal_role": detector_region_role(
                    detector_label=detector_label,
                    filter_mode=detector_filter_mode,
                    configured_labels=configured_labels,
                ),
                "detector_confidence": float(box.conf[0]),
                "crop_area_ratio": crop_area_ratio,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "source_width": source_width,
                "source_height": source_height,
            }
        )

    return rows
