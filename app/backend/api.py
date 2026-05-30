"""FastAPI entrypoint for the FoodLens inference service."""

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .inference import predict_image_bytes, predict_mock
from .schemas import PredictionResponse


app = FastAPI(
    title="FoodLens API",
    description="Food recognition API with calibrated predictions and decision bands.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health."""
    return {"status": "ok"}


@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)) -> PredictionResponse:
    """Predict a food label from an uploaded image.

    Uses the project ResNet50 FT-V2 artifacts when available. Falls back to a
    deterministic mock when artifacts or runtime dependencies are missing.
    """
    image_bytes = await file.read()
    return predict_image_bytes(image_bytes)


@app.post("/predict/video", response_model=PredictionResponse)
async def predict_video(file: UploadFile = File(...)) -> PredictionResponse:
    """Predict a food label from an uploaded video using mock frame aggregation."""
    _ = await file.read()
    return predict_mock(mode="video")
