"""Deterministic no-model fallback responses for the FoodLens API.

Used when classifier or detector artifacts/dependencies are unavailable.
Must not import inference. The public, patchable entry points
(predict_mock, build_multi_food_mock) stay in inference.py as thin wrappers:
they need artifact_status() -- and, transitively, artifact_dir_path(), which
resolves the repo root from Path(__file__) and can't move because a test
patches inference.__file__ -- so the wrappers compute that orchestration-level
state and pass it in here explicitly.
"""

from typing import Any

from .artifacts import TEMPERATURE
from .classifier import build_predictions
from .decision import build_decision
from .imaging import build_crop_data_url
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

MODEL_NAME = "resnet50_ft_v2"

MULTI_FOOD_POLICY = {
    "auto_confidence": 0.85,
    "suggest_confidence": 0.50,
    "margin_threshold": 0.40,
}

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


def build_mock_prediction_response(
    mode: str,
    fallback_reason: str | None,
    artifact_status_value: str,
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
        artifact_status=artifact_status_value,
        fallback_reason=fallback_reason,
    )


def build_multi_food_mock_response(
    fallback_reason: str,
    artifact_status_value: str,
    temperature: float,
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
        temperature=temperature,
        top_k=5,
        decision_thresholds={"auto_accept": 0.85, "suggest": 0.50},
        detector_status="fallback_demo",
        crop_count=len(predictions),
        predictions=predictions,
        artifact_status=artifact_status_value,
        fallback_reason=fallback_reason,
    )


def build_multi_food_classifier_fallback(
    image: Any,
    detection_rows: list[dict[str, Any]],
    temperature: float,
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
        temperature=temperature,
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


def build_classifier_fallback_predictions(row: dict[str, Any]) -> list[Prediction]:
    """Build honest crop labels when detector works but classifier artifacts are absent."""
    detector_label = str(row["detector_label"])
    if row["proposal_role"] == "direct_food":
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
