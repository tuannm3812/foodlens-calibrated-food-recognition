"""API schemas for the FoodLens inference service."""

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
