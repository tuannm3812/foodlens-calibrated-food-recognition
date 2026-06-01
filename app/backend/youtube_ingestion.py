"""YouTube URL ingestion for multi-food video analysis."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from .inference import predict_multi_food_image_bytes
from .schemas import MultiFoodPrediction, MultiFoodPredictionResponse
from .url_security import validate_public_http_url


MAX_YOUTUBE_DURATION_SECONDS = 10 * 60
FRAME_POSITIONS = (0.2, 0.5, 0.8)


class MediaDependencyError(RuntimeError):
    """Raised when an optional media ingestion dependency is missing."""


class MediaIngestionError(ValueError):
    """Raised when a user media URL cannot be processed."""


def _yt_dlp_module() -> Any:
    try:
        import yt_dlp
    except ImportError as exc:
        raise MediaDependencyError(
            "YouTube support needs yt-dlp installed on the backend."
        ) from exc
    return yt_dlp


def _ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise MediaDependencyError(
            "YouTube support needs ffmpeg installed on the backend."
        )


def _sample_times(duration: float) -> list[float]:
    if duration <= 0:
        return [0.0]
    return [
        min(max(duration * position, 0.0), max(duration - 0.05, 0.0))
        for position in FRAME_POSITIONS
    ]


def _copy_prediction_for_frame(
    prediction: MultiFoodPrediction,
    frame_index: int,
    global_index: int,
) -> MultiFoodPrediction:
    updates = {
        "source_id": f"youtube_frame_{frame_index + 1}",
        "detection_index": global_index,
    }
    if hasattr(prediction, "model_copy"):
        return prediction.model_copy(update=updates)
    return prediction.copy(update=updates)


def combine_frame_responses(
    responses: list[MultiFoodPredictionResponse],
) -> MultiFoodPredictionResponse:
    """Combine frame-level multi-food responses into one video response."""
    if not responses:
        raise MediaIngestionError("No usable video frames were extracted.")

    first = responses[0]
    predictions: list[MultiFoodPrediction] = []
    for frame_index, response in enumerate(responses):
        for prediction in response.predictions:
            predictions.append(
                _copy_prediction_for_frame(
                    prediction,
                    frame_index=frame_index,
                    global_index=len(predictions),
                )
            )

    fallback_reasons = [
        response.fallback_reason
        for response in responses
        if response.fallback_reason is not None
    ]

    return MultiFoodPredictionResponse(
        model=f"{first.model} · Video review",
        temperature=first.temperature,
        top_k=first.top_k,
        decision_thresholds=first.decision_thresholds,
        detector_status=f"{first.detector_status} · {len(responses)} frames",
        crop_count=len(predictions),
        predictions=predictions,
        artifact_status=first.artifact_status,
        fallback_reason=fallback_reasons[0] if fallback_reasons else None,
    )


def _download_youtube_video(url: str, output_dir: Path) -> tuple[Path, float]:
    yt_dlp = _yt_dlp_module()
    options = {
        "format": "best[ext=mp4][height<=720]/best[height<=720]/best",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
    except Exception as exc:
        raise MediaIngestionError(
            "The YouTube URL could not be downloaded."
        ) from exc

    duration = float(info.get("duration") or 0)
    if duration > MAX_YOUTUBE_DURATION_SECONDS:
        raise MediaIngestionError("The YouTube video is too long for analysis.")

    requested_downloads = info.get("requested_downloads") or []
    if requested_downloads and requested_downloads[0].get("filepath"):
        return Path(requested_downloads[0]["filepath"]), duration

    filepath = info.get("filepath")
    if filepath:
        return Path(filepath), duration

    candidates = sorted(
        output_dir.glob("*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise MediaIngestionError(
            "The YouTube video download did not produce a file."
        )
    return candidates[0], duration


def _extract_frame(video_path: Path, output_path: Path, timestamp: float) -> bytes:
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0 or not output_path.exists():
        raise MediaIngestionError(
            "A frame could not be extracted from the YouTube video."
        )
    return output_path.read_bytes()


def predict_multi_food_youtube_url(url: str) -> MultiFoodPredictionResponse:
    """Download a YouTube URL, sample frames, and return combined predictions."""
    safe_url = validate_public_http_url(url)
    _ensure_ffmpeg()

    with tempfile.TemporaryDirectory(prefix="foodlens-youtube-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        video_path, duration = _download_youtube_video(safe_url, temp_dir)
        frame_responses: list[MultiFoodPredictionResponse] = []
        for index, timestamp in enumerate(_sample_times(duration)):
            frame_path = temp_dir / f"frame-{index + 1}.jpg"
            frame_bytes = _extract_frame(video_path, frame_path, timestamp)
            frame_responses.append(predict_multi_food_image_bytes(frame_bytes))
        return combine_frame_responses(frame_responses)
