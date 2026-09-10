"""Running the YOLO detector to propose candidate food regions.

Must not import inference. The public, patchable entry point
(detect_candidate_regions) stays in inference.py as a thin wrapper: it
resolves the weights path via detector_weights_path() -- which can't move,
since it derives the repo root from Path(__file__) and a test patches
inference.__file__ -- and calls run_yolo_detection() here with that value.
"""

from typing import Any

from .detector_policy import (
    DETECTOR_CONFIDENCE_THRESHOLD,
    DETECTOR_IOU_THRESHOLD,
    DETECTOR_MAX_DETECTIONS,
    detector_label_filter_config,
    detector_region_role,
    should_export_detection,
)


def run_yolo_detection(image: Any, weights_path: str) -> list[dict[str, Any]]:
    """Run the YOLO detector against an image and return usable crop rows."""
    detector_filter_mode, configured_labels = detector_label_filter_config()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Multi-food detection requires ultralytics.") from exc

    detector = YOLO(weights_path)
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
