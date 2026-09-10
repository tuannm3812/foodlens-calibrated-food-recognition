"""Inference orchestration for the FoodLens API.

Owns artifact-dir/detector-weights path resolution (both derive the repo
root from Path(__file__), and tests patch inference.__file__ to exercise
them), runtime loading, and the public prediction entry points. Delegates
artifact reading, detector policy, detector invocation, image handling,
classifier construction, and demo/fallback responses to focused modules
under app/backend/, re-exporting their public names so `inference.X`
continues to resolve for every caller and test that reaches in this way.
"""

import importlib.util
import os
from io import BytesIO
from pathlib import Path
from typing import Any

from .artifacts import (
    REQUIRED_CLASSIFIER_ARTIFACTS,
    TEMPERATURE,
    artifact_file_status,
    classifier_artifacts_ready,
    read_confusion_pairs,
    read_hard_classes,
    read_json,
    read_policy,
    read_temperature,
)
from .classifier import build_predictions, make_classifier_head
from .decision import DEFAULT_HARD_CLASSES, DEFAULT_POLICY, build_decision
from .demo import (
    MODEL_NAME,
    MULTI_FOOD_POLICY,
    build_classifier_fallback_predictions,
    build_mock_prediction_response,
    build_multi_food_classifier_fallback,
    build_multi_food_mock_response,
)
from .detection import run_yolo_detection
from .detector_policy import (
    CANDIDATE_REGION_LABELS,
    DETECTOR_CONFIDENCE_THRESHOLD,
    DETECTOR_IOU_THRESHOLD,
    DETECTOR_MAX_DETECTIONS,
    DIRECT_FOOD_LABELS,
    MAX_CROP_AREA_RATIO,
    MIN_CROP_AREA_RATIO,
    detector_label_filter_config,
    detector_region_role,
    should_export_detection,
)
from .imaging import build_crop_data_url, build_full_image_region, open_rgb_image
from .schemas import (
    BoundingBox,
    DetectorRegion,
    FoodLensRegionPrediction,
    MultiFoodPrediction,
    MultiFoodPredictionResponse,
    Prediction,
    PredictionResponse,
    RegionArtifacts,
)

__all__ = [
    # Re-exported names below are not referenced elsewhere in this module's
    # own body -- they're kept importable as `inference.X` for callers and
    # tests that reach in this way. Everything else in this file's imports
    # is used directly by the orchestration functions below.
    "REQUIRED_CLASSIFIER_ARTIFACTS",
    "TEMPERATURE",
    "DEFAULT_POLICY",
    "DEFAULT_HARD_CLASSES",
    "DETECTOR_CONFIDENCE_THRESHOLD",
    "DETECTOR_IOU_THRESHOLD",
    "DETECTOR_MAX_DETECTIONS",
    "MIN_CROP_AREA_RATIO",
    "MAX_CROP_AREA_RATIO",
    "CANDIDATE_REGION_LABELS",
    "DIRECT_FOOD_LABELS",
    "detector_region_role",
    "should_export_detection",
    "build_predictions",
    "build_classifier_fallback_predictions",
]

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
IMAGE_SIZE = (224, 224)
DETECTOR_WEIGHTS = "yolo11n.pt"

_RUNTIME: dict[str, Any] | None = None


def artifact_status() -> str:
    """Return whether real model artifacts are currently available."""
    return "ready" if classifier_artifacts_ready(artifact_dir_path()) else "mock"


def artifact_dir_path() -> Path:
    """Resolve classifier artifacts across env overrides, worktrees, and repo roots."""
    configured_artifact_dir = os.getenv("FOODLENS_ARTIFACT_DIR")
    if configured_artifact_dir:
        return Path(configured_artifact_dir)

    if classifier_artifacts_ready(ARTIFACT_DIR):
        return ARTIFACT_DIR

    for parent in Path(__file__).resolve().parents:
        candidate_path = parent / "app" / "artifacts"
        if classifier_artifacts_ready(candidate_path):
            return candidate_path

    return ARTIFACT_DIR


def detector_weights_path() -> str:
    """Resolve detector weights without forcing an Ultralytics download first."""
    configured_weights = os.getenv("FOODLENS_DETECTOR_WEIGHTS")
    if configured_weights:
        return configured_weights

    for parent in Path(__file__).resolve().parents:
        candidate_path = parent / DETECTOR_WEIGHTS
        if candidate_path.exists():
            return str(candidate_path)

    return DETECTOR_WEIGHTS


def runtime_status() -> dict[str, Any]:
    """Return runtime readiness details for backend diagnostics."""
    resolved_artifact_dir = artifact_dir_path()
    checkpoint_path = resolved_artifact_dir / "resnet50_ft_v2_best.pth"
    class_names_path = resolved_artifact_dir / "class_names.json"
    calibration_path = resolved_artifact_dir / "calibration.json"
    decision_policy_path = resolved_artifact_dir / "decision_policy.json"
    hard_classes_path = resolved_artifact_dir / "hard_classes.json"
    confusion_pairs_path = resolved_artifact_dir / "confusion_pairs.json"
    classifier_ready = classifier_artifacts_ready(resolved_artifact_dir)
    weights_path = detector_weights_path()
    weights_found = Path(weights_path).exists()
    detector_dependency_available = importlib.util.find_spec("ultralytics") is not None
    detector_filter_mode, configured_labels = detector_label_filter_config()

    if classifier_ready and detector_dependency_available:
        multi_food_mode = "live_yolo_classifier"
    elif detector_dependency_available:
        multi_food_mode = "detector_only_classifier_fallback"
    else:
        multi_food_mode = "demo_fallback"

    return {
        "classifier": {
            "status": "ready" if classifier_ready else "missing_artifacts",
            "artifact_status": artifact_status(),
            "artifact_dir": str(resolved_artifact_dir),
            "artifacts": {
                "checkpoint": artifact_file_status(checkpoint_path),
                "class_names": artifact_file_status(class_names_path),
                "calibration": artifact_file_status(calibration_path),
                "decision_policy": artifact_file_status(decision_policy_path),
                "hard_classes": artifact_file_status(hard_classes_path),
                "confusion_pairs": artifact_file_status(confusion_pairs_path),
            },
        },
        "detector": {
            "status": "ready" if detector_dependency_available else "missing_dependency",
            "dependency": "ultralytics",
            "dependency_available": detector_dependency_available,
            "weights_path": weights_path,
            "weights_found": weights_found,
            "label_filter": {
                "mode": detector_filter_mode,
                "labels": sorted(configured_labels) if detector_filter_mode == "configured" else [],
            },
            "weights_source": (
                "environment"
                if os.getenv("FOODLENS_DETECTOR_WEIGHTS")
                else "auto_discovered"
                if weights_found
                else "ultralytics_default"
            ),
        },
        "multi_food": {
            "mode": multi_food_mode,
            "detector_status": (
                "live_yolo"
                if classifier_ready and detector_dependency_available
                else "live_yolo_classifier_fallback"
                if detector_dependency_available
                else "fallback_demo"
            ),
        },
    }


def load_runtime() -> dict[str, Any]:
    """Load model and metadata once when real artifacts are present."""
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME

    if artifact_status() != "ready":
        raise FileNotFoundError(
            "Missing real inference artifacts. Expected resnet50_ft_v2_best.pth "
            "and class_names.json under app/artifacts."
        )

    try:
        import torch
        import torch.nn.functional as functional
        from PIL import Image
        from torch import nn
        from torchvision import models, transforms
    except ImportError as exc:
        raise RuntimeError(
            "Real inference requires torch, torchvision, and Pillow."
        ) from exc

    resolved_artifact_dir = artifact_dir_path()
    class_names = read_json(resolved_artifact_dir / "class_names.json", [])
    if len(class_names) != 101:
        raise ValueError("class_names.json must contain 101 ordered Food-101 class names.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = make_classifier_head(nn, model.fc.in_features)
    model.load_state_dict(
        torch.load(resolved_artifact_dir / "resnet50_ft_v2_best.pth", map_location=device)
    )
    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    _RUNTIME = {
        "torch": torch,
        "functional": functional,
        "image_class": Image,
        "device": device,
        "model": model,
        "transform": transform,
        "class_names": class_names,
        "temperature": read_temperature(resolved_artifact_dir),
        "policy": read_policy(resolved_artifact_dir),
        "hard_classes": read_hard_classes(resolved_artifact_dir),
        "confusion_pairs": read_confusion_pairs(resolved_artifact_dir),
    }
    return _RUNTIME


def classify_pil_image(image: Any, runtime: dict[str, Any]) -> list[Prediction]:
    """Classify one PIL image with the loaded FoodLens classifier."""
    torch = runtime["torch"]
    functional = runtime["functional"]
    image_tensor = runtime["transform"](image).unsqueeze(0).to(runtime["device"])

    with torch.no_grad():
        logits = runtime["model"](image_tensor).cpu()
        probabilities = functional.softmax(logits / runtime["temperature"], dim=1)
        top_probabilities, top_indices = probabilities.topk(5, dim=1)

    return [
        Prediction(
            rank=rank + 1,
            class_name=runtime["class_names"][class_index],
            confidence=confidence,
        )
        for rank, (class_index, confidence) in enumerate(
            zip(top_indices[0].tolist(), top_probabilities[0].tolist(), strict=True)
        )
    ]


def predict_mock(
    mode: str = "image",
    fallback_reason: str | None = None,
) -> PredictionResponse:
    """Return a deterministic mock prediction response."""
    return build_mock_prediction_response(mode, fallback_reason, artifact_status())


def build_multi_food_mock(
    fallback_reason: str = "missing_artifacts",
) -> MultiFoodPredictionResponse:
    """Return a deterministic Notebook 8-style multi-food response."""
    return build_multi_food_mock_response(
        fallback_reason,
        artifact_status(),
        read_temperature(artifact_dir_path()),
    )


def detect_candidate_regions(image: Any) -> list[dict[str, Any]]:
    """Detect candidate food regions with YOLO when Ultralytics is available."""
    return run_yolo_detection(image, detector_weights_path())


def build_multi_food_classifier_fallback_response(
    image: Any,
    detection_rows: list[dict[str, Any]],
) -> MultiFoodPredictionResponse:
    """Return real detector crops with explicit classifier-fallback labels."""
    return build_multi_food_classifier_fallback(
        image, detection_rows, read_temperature(artifact_dir_path())
    )


def build_multi_food_response(
    image: Any,
    detection_rows: list[dict[str, Any]],
    runtime: dict[str, Any],
    detector_status: str = "live_yolo",
    fallback_reason: str | None = None,
) -> MultiFoodPredictionResponse:
    """Classify detected regions and return the app-ready multi-food response."""
    predictions: list[MultiFoodPrediction] = []

    for region_index, row in enumerate(detection_rows):
        crop = image.crop((row["x1"], row["y1"], row["x2"], row["y2"]))
        crop_predictions = classify_pil_image(crop, runtime)
        decision = build_decision(
            "image",
            crop_predictions,
            policy=MULTI_FOOD_POLICY,
            hard_classes=runtime["hard_classes"],
            confusion_pairs=runtime["confusion_pairs"],
        )
        crop_name = f"uploaded_image_crop_{region_index:02d}.jpg"

        predictions.append(
            MultiFoodPrediction(
                source_id="uploaded_image",
                detection_index=int(row["detection_index"]),
                bbox=BoundingBox(
                    x1=int(row["x1"]),
                    y1=int(row["y1"]),
                    x2=int(row["x2"]),
                    y2=int(row["y2"]),
                    source_width=int(row["source_width"]),
                    source_height=int(row["source_height"]),
                ),
                detector=DetectorRegion(
                    label=str(row["detector_label"]),
                    proposal_role=str(row["proposal_role"]),
                    confidence=float(row["detector_confidence"]),
                    crop_area_ratio=float(row["crop_area_ratio"]),
                ),
                foodlens=FoodLensRegionPrediction(
                    top_label=crop_predictions[0].class_name,
                    top_confidence=crop_predictions[0].confidence,
                    decision_band=decision.band,
                    top_k_predictions=[
                        (prediction.class_name, prediction.confidence)
                        for prediction in crop_predictions
                    ],
                ),
                artifacts=RegionArtifacts(
                    crop_path=f"runtime/{crop_name}",
                    crop_artifact_path=f"app://runtime/crops/{crop_name}",
                    figure_path="runtime/uploaded_image_detections.jpg",
                    crop_data_url=build_crop_data_url(crop),
                ),
            )
        )

    return MultiFoodPredictionResponse(
        model=MODEL_NAME,
        temperature=runtime["temperature"],
        top_k=5,
        decision_thresholds={
            "auto_accept": MULTI_FOOD_POLICY["auto_confidence"],
            "suggest": MULTI_FOOD_POLICY["suggest_confidence"],
        },
        detector_status=detector_status,
        crop_count=len(predictions),
        predictions=predictions,
        artifact_status="ready",
        fallback_reason=fallback_reason,
    )


def predict_multi_food_image_bytes(image_bytes: bytes) -> MultiFoodPredictionResponse:
    """Return multi-food predictions for an uploaded image.

    Uses live detector proposals and FoodLens crop classification when
    dependencies and artifacts are available. Falls back to a deterministic
    Notebook 8-style response when the detector runtime is unavailable.
    """
    if artifact_status() != "ready":
        try:
            image = open_rgb_image(image_bytes)
        except Exception:
            return build_multi_food_mock(fallback_reason="invalid_image")

        try:
            detections = detect_candidate_regions(image)
        except RuntimeError:
            return build_multi_food_mock(fallback_reason="detector_runtime_unavailable")
        except Exception:
            return build_multi_food_mock(fallback_reason="detector_inference_error")

        if not detections:
            detections = [build_full_image_region(image)]
        return build_multi_food_classifier_fallback_response(image, detections)

    try:
        runtime = load_runtime()
    except Exception:
        return build_multi_food_mock(fallback_reason="classifier_load_error")

    try:
        image = runtime["image_class"].open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return build_multi_food_mock(fallback_reason="invalid_image")

    try:
        detections = detect_candidate_regions(image)
    except RuntimeError:
        return build_multi_food_mock(fallback_reason="detector_runtime_unavailable")
    except Exception:
        return build_multi_food_mock(fallback_reason="detector_inference_error")

    detector_status = "live_yolo"
    fallback_reason = None
    if not detections:
        detections = [build_full_image_region(image)]
        detector_status = "live_yolo_whole_image_fallback"
        fallback_reason = "no_detector_regions"

    try:
        return build_multi_food_response(
            image,
            detections,
            runtime,
            detector_status=detector_status,
            fallback_reason=fallback_reason,
        )
    except Exception:
        return build_multi_food_mock(fallback_reason="classifier_inference_error")


def build_prediction_response(
    image: Any,
    runtime: dict[str, Any],
) -> PredictionResponse:
    """Build a single-image prediction response from an RGB image."""
    predictions = classify_pil_image(image, runtime)
    return PredictionResponse(
        model_name=MODEL_NAME,
        mode="image",
        temperature=runtime["temperature"],
        top_predictions=predictions,
        decision=build_decision(
            "image",
            predictions,
            policy=runtime["policy"],
            hard_classes=runtime["hard_classes"],
            confusion_pairs=runtime["confusion_pairs"],
        ),
        artifact_status="ready",
        fallback_reason=None,
    )


def predict_image_bytes(image_bytes: bytes) -> PredictionResponse:
    """Predict Food-101 classes using real artifacts when available."""
    if artifact_status() != "ready":
        return predict_mock(mode="image", fallback_reason="missing_artifacts")

    try:
        runtime = load_runtime()
    except Exception:
        return predict_mock(mode="image", fallback_reason="classifier_load_error")

    try:
        image = runtime["image_class"].open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return predict_mock(mode="image", fallback_reason="invalid_image")

    try:
        return build_prediction_response(image, runtime)
    except Exception:
        return predict_mock(mode="image", fallback_reason="classifier_inference_error")
