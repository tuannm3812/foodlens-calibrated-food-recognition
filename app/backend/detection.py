"""Running the YOLO detector to propose candidate food regions.

Must not import inference. The public, patchable entry point
(detect_candidate_regions) stays in inference.py as a thin wrapper: it
resolves the weights path via detector_weights_path() -- which can't move,
since it derives the repo root from Path(__file__) and a test patches
inference.__file__ -- and calls run_yolo_detection() here with that value.

The YOLO model is loaded once and reused across requests until the weights
path or the file's contents change.
FastAPI runs sync handlers in a threadpool, and an Ultralytics model is not
safe to call from several threads at once (its predictor is built lazily and
holds per-call state), so the cached model's predict runs under a lock.
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
# One slot: (resolved path, file signature, model, predict lock). The API
# serves one weights file, so loading another (a new path, or the same path
# with new contents) replaces the old model rather than keeping both.
_cached: tuple[str, tuple[int, ...] | None, Any, threading.Lock] | None = None


def _weights_signature(resolved_path: str) -> tuple[int, ...] | None:
    """Identify the weights file's current contents, or None when it is absent.

    Device, inode, size, mtime and ctime together: a file swapped by rename
    gets a new inode, and one rewritten in place gets a new ctime even when
    its size and mtime are made to match, since ctime cannot be set back.
    """
    try:
        stat = os.stat(resolved_path)
    except OSError:
        return None
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def cached_detector(weights_path: str, yolo_class: Any) -> tuple[Any, threading.Lock]:
    """Return the shared model for this weights file and the lock guarding it."""
    global _cached
    resolved_path = str(Path(weights_path).resolve())
    with _cache_lock:
        # Read under the lock, so a caller that waited out another's load
        # compares against what that load stored, not a stale reading.
        before = _weights_signature(resolved_path)
        if _cached is not None and _cached[:2] == (resolved_path, before):
            return _cached[2], _cached[3]
        # Load the resolved file the signature describes, so a symlink moved
        # meanwhile cannot put another file's model under this signature. A
        # missing file is passed as given, for Ultralytics to download.
        load_path = resolved_path if before is not None else weights_path
        # Construct before storing, so a failed load is not cached.
        detector = yolo_class(load_path)
        lock = threading.Lock()
        # Cache only a load whose file was present and unchanged throughout.
        # A file that was missing (downloaded during the load, perhaps then
        # replaced) or that changed while loading is loaded again next time.
        if before is not None and _weights_signature(resolved_path) == before:
            _cached = (resolved_path, before, detector, lock)
        else:
            _cached = None
        return detector, lock


def clear_detector_cache() -> None:
    """Drop the cached model; the next detection loads afresh."""
    global _cached
    with _cache_lock:
        _cached = None


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
