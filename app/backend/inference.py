"""Inference helpers for the FoodLens API."""

import base64
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any, Optional

from .decision import DEFAULT_HARD_CLASSES, DEFAULT_POLICY, build_decision
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


ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
REQUIRED_CLASSIFIER_ARTIFACTS = (
    "resnet50_ft_v2_best.pth",
    "class_names.json",
)
MODEL_NAME = "resnet50_ft_v2"
TEMPERATURE = 0.958111
IMAGE_SIZE = (224, 224)
MULTI_FOOD_POLICY = {
    "auto_confidence": 0.85,
    "suggest_confidence": 0.50,
    "margin_threshold": 0.40,
}
DETECTOR_WEIGHTS = "yolo11n.pt"
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

MOCK_IMAGE_PREDICTIONS: tuple[tuple[str, float], ...] = (
    ("steak", 0.7838),
    ("filet_mignon", 0.1543),
    ("prime_rib", 0.0223),
    ("baby_back_ribs", 0.0102),
    ("pork_chop", 0.0092),
)

MOCK_VIDEO_PREDICTIONS: tuple[tuple[str, float], ...] = (
    ("sushi", 0.6842),
    ("sashimi", 0.2015),
    ("ceviche", 0.0511),
    ("tuna_tartare", 0.0394),
    ("miso_soup", 0.0128),
)

MOCK_MULTI_FOOD_REGIONS: tuple[dict[str, Any], ...] = (
    {
        "source_id": "sample_05_prohibition_table",
        "detection_index": 0,
        "bbox": (410, 132, 662, 382, 960, 733),
        "detector": ("bowl", "serving_container", 0.5368, 0.102),
        "foodlens": ("ravioli", 0.972, "auto_accept"),
        "top_k": (("ravioli", 0.972), ("gnocchi", 0.018), ("lasagna", 0.004)),
    },
    {
        "source_id": "sample_03_food_market",
        "detection_index": 1,
        "bbox": (380, 520, 820, 930, 1366, 1503),
        "detector": ("bowl", "serving_container", 0.3046, 0.088),
        "foodlens": ("lasagna", 0.920, "auto_accept"),
        "top_k": (("lasagna", 0.920), ("ravioli", 0.033), ("pizza", 0.018)),
    },
    {
        "source_id": "sample_01_simplot_table",
        "detection_index": 2,
        "bbox": (455, 374, 701, 595, 960, 733),
        "detector": ("bowl", "serving_container", 0.4445, 0.077),
        "foodlens": ("ramen", 0.768, "suggest"),
        "top_k": (("ramen", 0.768), ("pho", 0.034), ("miso_soup", 0.023)),
    },
    {
        "source_id": "sample_03_food_market",
        "detection_index": 3,
        "bbox": (20, 950, 430, 1290, 1366, 1503),
        "detector": ("bowl", "serving_container", 0.3983, 0.068),
        "foodlens": ("french_fries", 0.752, "suggest"),
        "top_k": (("french_fries", 0.752), ("fish_and_chips", 0.146), ("onion_rings", 0.026)),
    },
    {
        "source_id": "sample_02_party_food",
        "detection_index": 4,
        "bbox": (43, 313, 1363, 1458, 1366, 1503),
        "detector": ("cake", "direct_food", 0.5763, 0.736),
        "foodlens": ("falafel", 0.241, "confirm"),
        "top_k": (("falafel", 0.241), ("donuts", 0.195), ("garlic_bread", 0.112)),
    },
)

_RUNTIME: Optional[dict[str, Any]] = None


def artifact_status() -> str:
    """Return whether real model artifacts are currently available."""
    return "ready" if classifier_artifacts_ready(artifact_dir_path()) else "mock"


def classifier_artifacts_ready(artifact_dir: Path) -> bool:
    """Return whether the required classifier artifacts exist in a directory."""
    return all((artifact_dir / artifact_name).exists() for artifact_name in REQUIRED_CLASSIFIER_ARTIFACTS)


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


def read_json(path: Path, default: Any) -> Any:
    """Read a JSON artifact when available."""
    if not path.exists():
        return default
    return json.loads(path.read_text())


def read_temperature() -> float:
    """Read the calibrated temperature artifact when available."""
    calibration = read_json(artifact_dir_path() / "calibration.json", {})
    return float(calibration.get("temperature", TEMPERATURE))


def read_policy() -> dict[str, float]:
    """Read decision thresholds when available."""
    policy = read_json(artifact_dir_path() / "decision_policy.json", DEFAULT_POLICY)
    return {
        "auto_confidence": float(policy.get("auto_confidence", DEFAULT_POLICY["auto_confidence"])),
        "suggest_confidence": float(
            policy.get("suggest_confidence", DEFAULT_POLICY["suggest_confidence"])
        ),
        "margin_threshold": float(policy.get("margin_threshold", DEFAULT_POLICY["margin_threshold"])),
    }


def read_hard_classes() -> set[str]:
    """Read hard classes when available."""
    hard_classes = read_json(artifact_dir_path() / "hard_classes.json", list(DEFAULT_HARD_CLASSES))
    return set(hard_classes)


def read_confusion_pairs() -> set[tuple[str, str]]:
    """Read known confusion pairs when available."""
    raw_pairs = read_json(artifact_dir_path() / "confusion_pairs.json", [])
    pairs = set()
    for pair in raw_pairs:
        if isinstance(pair, dict) and {"actual", "predicted"}.issubset(pair):
            pairs.add((str(pair["actual"]), str(pair["predicted"])))
        elif isinstance(pair, (list, tuple)) and len(pair) >= 2:
            pairs.add((str(pair[0]), str(pair[1])))
    return pairs


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


def artifact_file_status(path: Path) -> dict[str, Any]:
    """Return status details for one artifact file."""
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


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


def make_classifier_head(torch_nn: Any, in_features: int) -> Any:
    """Create the project-standard Food-101 classifier head."""
    return torch_nn.Sequential(
        torch_nn.Linear(in_features, 512),
        torch_nn.ReLU(),
        torch_nn.Linear(512, 256),
        torch_nn.ReLU(),
        torch_nn.Linear(256, 101),
    )


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
        "temperature": read_temperature(),
        "policy": read_policy(),
        "hard_classes": read_hard_classes(),
        "confusion_pairs": read_confusion_pairs(),
    }
    return _RUNTIME


def build_predictions(raw_predictions: tuple[tuple[str, float], ...]) -> list[Prediction]:
    """Convert raw label-score tuples to API prediction objects."""
    return [
        Prediction(rank=index + 1, class_name=class_name, confidence=confidence)
        for index, (class_name, confidence) in enumerate(raw_predictions)
    ]


def detector_region_role(detector_label: str) -> str:
    """Map a generic detector label to its FoodLens proposal role."""
    if detector_label in DIRECT_FOOD_LABELS:
        return "direct_food"
    if detector_label == "bowl":
        return "serving_container"
    if detector_label == "whole_image":
        return "fallback_region"
    return "context_object"


def should_export_detection(detector_label: str, area_ratio: float) -> bool:
    """Return whether a detector box is useful as a classifier crop."""
    return (
        detector_label in CANDIDATE_REGION_LABELS
        and MIN_CROP_AREA_RATIO <= area_ratio <= MAX_CROP_AREA_RATIO
    )


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
            zip(top_indices[0].tolist(), top_probabilities[0].tolist())
        )
    ]


def predict_mock(
    mode: str = "image",
    fallback_reason: Optional[str] = None,
) -> PredictionResponse:
    """Return a deterministic mock prediction response."""
    raw_predictions = MOCK_VIDEO_PREDICTIONS if mode == "video" else MOCK_IMAGE_PREDICTIONS
    predictions = build_predictions(raw_predictions)
    return PredictionResponse(
        model_name=MODEL_NAME,
        mode=mode,
        temperature=TEMPERATURE,
        top_predictions=predictions,
        decision=build_decision(mode, predictions),
        artifact_status=artifact_status(),
        fallback_reason=fallback_reason,
    )


def build_multi_food_mock(
    fallback_reason: str = "missing_artifacts",
) -> MultiFoodPredictionResponse:
    """Return a deterministic Notebook 8-style multi-food response."""
    predictions: list[MultiFoodPrediction] = []
    for region in MOCK_MULTI_FOOD_REGIONS:
        x1, y1, x2, y2, source_width, source_height = region["bbox"]
        detector_label, proposal_role, detector_confidence, crop_area_ratio = region[
            "detector"
        ]
        top_label, top_confidence, decision_band = region["foodlens"]
        crop_name = f"{region['source_id']}_crop_{region['detection_index']:02d}.jpg"
        predictions.append(
            MultiFoodPrediction(
                source_id=region["source_id"],
                detection_index=region["detection_index"],
                bbox=BoundingBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    source_width=source_width,
                    source_height=source_height,
                ),
                detector=DetectorRegion(
                    label=detector_label,
                    proposal_role=proposal_role,
                    confidence=detector_confidence,
                    crop_area_ratio=crop_area_ratio,
                ),
                foodlens=FoodLensRegionPrediction(
                    top_label=top_label,
                    top_confidence=top_confidence,
                    decision_band=decision_band,
                    top_k_predictions=list(region["top_k"]),
                ),
                artifacts=RegionArtifacts(
                    crop_path=f"crops/{crop_name}",
                    crop_artifact_path=f"app://demo/crops/{crop_name}",
                    figure_path=f"figures/{region['source_id']}_detections.jpg",
                ),
            )
        )

    return MultiFoodPredictionResponse(
        model=MODEL_NAME,
        temperature=read_temperature(),
        top_k=5,
        decision_thresholds={"auto_accept": 0.85, "suggest": 0.50},
        detector_status="fallback_demo",
        crop_count=len(predictions),
        predictions=predictions,
        artifact_status=artifact_status(),
        fallback_reason=fallback_reason,
    )


def detect_candidate_regions(image: Any) -> list[dict[str, Any]]:
    """Detect candidate food regions with YOLO when Ultralytics is available."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Multi-food detection requires ultralytics.") from exc

    detector = YOLO(detector_weights_path())
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
        if not should_export_detection(detector_label, crop_area_ratio):
            continue

        rows.append(
            {
                "detection_index": detection_index,
                "detector_label": detector_label,
                "proposal_role": detector_region_role(detector_label),
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


def build_full_image_region(image: Any) -> dict[str, Any]:
    """Build a fallback region when the detector produces no usable crops."""
    source_width, source_height = image.size
    return {
        "detection_index": 0,
        "detector_label": "whole_image",
        "proposal_role": detector_region_role("whole_image"),
        "detector_confidence": 1.0,
        "crop_area_ratio": 1.0,
        "x1": 0,
        "y1": 0,
        "x2": source_width,
        "y2": source_height,
        "source_width": source_width,
        "source_height": source_height,
    }


def build_crop_data_url(crop: Any) -> str:
    """Encode a crop preview as a browser-ready JPEG data URL."""
    buffer = BytesIO()
    crop.save(buffer, format="JPEG", quality=82)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def open_rgb_image(image_bytes: bytes) -> Any:
    """Open uploaded image bytes without requiring the classifier runtime."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Image decoding requires Pillow.") from exc

    return Image.open(BytesIO(image_bytes)).convert("RGB")


def build_classifier_fallback_predictions(row: dict[str, Any]) -> list[Prediction]:
    """Build honest crop labels when detector works but classifier artifacts are absent."""
    detector_label = str(row["detector_label"])
    if detector_label in DIRECT_FOOD_LABELS:
        top_label = detector_label
    elif detector_label == "bowl":
        top_label = "food_in_container"
    else:
        top_label = "detected_food_region"

    fallback_confidence = min(
        float(row["detector_confidence"]),
        MULTI_FOOD_POLICY["suggest_confidence"] - 0.01,
    )
    return [
        Prediction(rank=1, class_name=top_label, confidence=fallback_confidence),
        Prediction(rank=2, class_name="classifier_unavailable", confidence=0.0),
    ]


def build_multi_food_classifier_fallback_response(
    image: Any,
    detection_rows: list[dict[str, Any]],
) -> MultiFoodPredictionResponse:
    """Return real detector crops with explicit classifier-fallback labels."""
    predictions: list[MultiFoodPrediction] = []

    for region_index, row in enumerate(detection_rows):
        crop = image.crop((row["x1"], row["y1"], row["x2"], row["y2"]))
        crop_predictions = build_classifier_fallback_predictions(row)
        decision = build_decision(
            "image",
            crop_predictions,
            policy=MULTI_FOOD_POLICY,
            hard_classes=set(),
            confusion_pairs=set(),
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
        temperature=read_temperature(),
        top_k=5,
        decision_thresholds={
            "auto_accept": MULTI_FOOD_POLICY["auto_confidence"],
            "suggest": MULTI_FOOD_POLICY["suggest_confidence"],
        },
        detector_status="live_yolo_classifier_fallback",
        crop_count=len(predictions),
        predictions=predictions,
        artifact_status="mock",
        fallback_reason="missing_classifier_artifacts",
    )


def build_multi_food_response(
    image: Any,
    detection_rows: list[dict[str, Any]],
    runtime: dict[str, Any],
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
        detector_status="live_yolo",
        crop_count=len(predictions),
        predictions=predictions,
        artifact_status="ready",
        fallback_reason=None,
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
            return build_multi_food_mock(fallback_reason="missing_artifacts")

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
        image = runtime["image_class"].open(BytesIO(image_bytes)).convert("RGB")
        try:
            detections = detect_candidate_regions(image)
        except RuntimeError:
            return build_multi_food_mock(fallback_reason="detector_runtime_unavailable")
        if not detections:
            detections = [build_full_image_region(image)]
        return build_multi_food_response(image, detections, runtime)
    except Exception:
        return build_multi_food_mock(fallback_reason="inference_error")


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
        image = runtime["image_class"].open(BytesIO(image_bytes)).convert("RGB")
        return build_prediction_response(image, runtime)
    except Exception:
        return predict_mock(mode="image", fallback_reason="inference_error")
