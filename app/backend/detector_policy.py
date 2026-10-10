"""Detector label policy: which YOLO detections become classifier regions.

Pure, environment- and constant-driven logic with no artifact or model
dependency. Must not import inference.
"""

import os

DETECTOR_CONFIDENCE_THRESHOLD = 0.25
DETECTOR_IOU_THRESHOLD = 0.50
DETECTOR_MAX_DETECTIONS = 20
MIN_CROP_AREA_RATIO = 0.015
MAX_CROP_AREA_RATIO = 0.80
CANDIDATE_REGION_LABELS = {
    "apple",
    "banana",
    "bowl",
    "broccoli",
    "cake",
    "carrot",
    "donut",
    "hot dog",
    "orange",
    "pizza",
    "sandwich",
}
DIRECT_FOOD_LABELS = CANDIDATE_REGION_LABELS - {"bowl"}


def detector_label_filter_config() -> tuple[str, set[str]]:
    """Resolve detector label filtering from environment overrides."""
    configured_filter = os.getenv("FOODLENS_DETECTOR_LABELS")
    if configured_filter is None:
        return "default", set()

    normalized = configured_filter.strip()
    if not normalized or normalized == "*":
        return "all", set()

    labels = {
        label.strip()
        for label in normalized.replace(";", ",").split(",")
        if label.strip()
    }
    return "configured", labels


def detector_region_role(
    detector_label: str,
    filter_mode: str = "default",
    configured_labels: set[str] | None = None,
) -> str:
    """Map a detector label to its FoodLens proposal role."""
    if detector_label == "bowl":
        return "serving_container"
    if detector_label == "whole_image":
        return "fallback_region"

    if filter_mode == "all":
        return "direct_food"
    if filter_mode == "configured" and configured_labels is not None:
        return "direct_food" if detector_label in configured_labels else "context_object"

    return "direct_food" if detector_label in DIRECT_FOOD_LABELS else "context_object"


def should_export_detection(
    detector_label: str,
    area_ratio: float,
    filter_mode: str = "default",
    configured_labels: set[str] | None = None,
) -> bool:
    """Return whether a detector box is useful as a classifier crop."""
    if not (MIN_CROP_AREA_RATIO <= area_ratio <= MAX_CROP_AREA_RATIO):
        return False

    if filter_mode == "all":
        return True
    if filter_mode == "configured" and configured_labels is not None:
        return detector_label in configured_labels

    return detector_label in CANDIDATE_REGION_LABELS
