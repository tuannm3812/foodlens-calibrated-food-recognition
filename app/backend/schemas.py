"""API schemas for the FoodLens inference service."""

from typing import Optional

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    """One ranked class prediction."""

    rank: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)


class Decision(BaseModel):
    """Decision-layer output for a prediction request."""

    band: str
    title: str
    recommended_action: str
    top_1_top_2_margin: float


class PredictionResponse(BaseModel):
    """FoodLens prediction response."""

    model_name: str
    mode: str
    temperature: float
    top_predictions: list[Prediction]
    decision: Decision
    artifact_status: str


class BoundingBox(BaseModel):
    """Detected region coordinates in source-image pixels."""

    x1: int
    y1: int
    x2: int
    y2: int
    source_width: int
    source_height: int


class DetectorRegion(BaseModel):
    """Detector proposal metadata for one crop."""

    label: str
    proposal_role: str
    confidence: float = Field(ge=0.0, le=1.0)
    crop_area_ratio: float = Field(ge=0.0, le=1.0)


class FoodLensRegionPrediction(BaseModel):
    """FoodLens classifier output for one detected crop."""

    top_label: str
    top_confidence: float = Field(ge=0.0, le=1.0)
    decision_band: str
    top_k_predictions: list[tuple[str, float]]


class RegionArtifacts(BaseModel):
    """Artifact paths associated with one detected crop."""

    crop_path: str
    crop_artifact_path: str
    figure_path: str
    crop_data_url: Optional[str] = None


class MultiFoodPrediction(BaseModel):
    """One app-ready multi-food region prediction."""

    source_id: str
    detection_index: int
    bbox: BoundingBox
    detector: DetectorRegion
    foodlens: FoodLensRegionPrediction
    artifacts: RegionArtifacts


class MultiFoodPredictionResponse(BaseModel):
    """App-ready multi-food prediction response."""

    model: str
    temperature: float
    top_k: int
    decision_thresholds: dict[str, float]
    detector_status: str
    crop_count: int
    predictions: list[MultiFoodPrediction]
    artifact_status: str
