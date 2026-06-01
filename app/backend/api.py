"""FastAPI entrypoint for the FoodLens inference service."""

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .inference import (
    predict_image_bytes,
    predict_mock,
    predict_multi_food_image_bytes,
    runtime_status,
)
from .media_download import DownloadError, download_image_url
from .schemas import (
    MultiFoodPredictionResponse,
    PredictionResponse,
    UrlPredictionRequest,
)
from .url_security import UrlValidationError
from .youtube_ingestion import (
    MediaDependencyError,
    MediaIngestionError,
    predict_multi_food_youtube_url,
)


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


@app.get("/runtime/status")
def get_runtime_status() -> dict[str, object]:
    """Return backend runtime readiness for classifier and detector paths."""
    return runtime_status()


@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)) -> PredictionResponse:
    """Predict a food label from an uploaded image.

    Uses the project ResNet50 FT-V2 artifacts when available. Falls back to a
    deterministic mock when artifacts or runtime dependencies are missing.
    """
    image_bytes = await file.read()
    return predict_image_bytes(image_bytes)


@app.post("/predict/multi-food/image", response_model=MultiFoodPredictionResponse)
async def predict_multi_food_image(
    file: UploadFile = File(...),
) -> MultiFoodPredictionResponse:
    """Predict multiple food regions from an uploaded image.

    This endpoint returns the Notebook 8 app contract. The current
    implementation is deterministic while live detector inference is being
    productized.
    """
    image_bytes = await file.read()
    return predict_multi_food_image_bytes(image_bytes)


@app.post("/predict/multi-food/image-url", response_model=MultiFoodPredictionResponse)
async def predict_multi_food_image_url(
    request: UrlPredictionRequest,
) -> MultiFoodPredictionResponse:
    """Predict multiple food regions from a public direct image URL."""
    try:
        image_bytes = download_image_url(request.url)
    except (DownloadError, UrlValidationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return predict_multi_food_image_bytes(image_bytes)


@app.post("/predict/multi-food/youtube-url", response_model=MultiFoodPredictionResponse)
async def predict_multi_food_youtube(
    request: UrlPredictionRequest,
) -> MultiFoodPredictionResponse:
    """Predict multiple food regions from a public YouTube URL."""
    try:
        return predict_multi_food_youtube_url(request.url)
    except UrlValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except MediaDependencyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MediaIngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict/video", response_model=PredictionResponse)
async def predict_video(file: UploadFile = File(...)) -> PredictionResponse:
    """Return deterministic video predictions until live video inference exists."""
    _ = await file.read()
    return predict_mock(mode="video", fallback_reason="video_mock")
