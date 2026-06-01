from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
import pytest

import app.backend.api as api
import app.backend.inference as inference
from app.backend.api import app
from app.backend.media_download import DownloadError, download_image_url
from app.backend.url_security import UrlValidationError, validate_public_http_url
from app.backend.youtube_ingestion import (
    MediaDependencyError,
    combine_frame_responses,
)


client = TestClient(app)


def make_jpeg_bytes() -> bytes:
    image = Image.new("RGB", (32, 24), color=(220, 80, 40))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_validate_public_http_url_rejects_private_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.backend.url_security.socket.getaddrinfo",
        lambda host, port, type=0: [
            (None, None, None, None, ("127.0.0.1", port or 443))
        ],
    )

    with pytest.raises(UrlValidationError, match="public media URL"):
        validate_public_http_url("https://example.test/plate.jpg")


def test_validate_public_http_url_allows_public_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.backend.url_security.socket.getaddrinfo",
        lambda host, port, type=0: [
            (None, None, None, None, ("93.184.216.34", port or 443))
        ],
    )

    assert (
        validate_public_http_url("https://example.com/plate.jpg")
        == "https://example.com/plate.jpg"
    )


def test_validate_public_http_url_rejects_non_http_scheme() -> None:
    with pytest.raises(UrlValidationError, match="http or https"):
        validate_public_http_url("file:///tmp/plate.jpg")


def test_download_image_url_rejects_non_image_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        headers = {"Content-Type": "text/html", "Content-Length": "12"}
        status = 200

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _size: int = -1) -> bytes:
            return b"<html></html>"

    monkeypatch.setattr(
        "app.backend.media_download.validate_public_http_url",
        lambda url: url,
    )
    monkeypatch.setattr(
        "app.backend.media_download._open_url",
        lambda request, timeout: Response(),
    )

    with pytest.raises(DownloadError, match="image"):
        download_image_url("https://example.com/page")


def test_download_image_url_reads_bounded_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = make_jpeg_bytes()

    class Response:
        headers = {"Content-Type": "image/jpeg", "Content-Length": str(len(payload))}
        status = 200

        def __init__(self) -> None:
            self.offset = 0

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, size: int = -1) -> bytes:
            if self.offset >= len(payload):
                return b""
            chunk = payload[self.offset : self.offset + size]
            self.offset += len(chunk)
            return chunk

    monkeypatch.setattr(
        "app.backend.media_download.validate_public_http_url",
        lambda url: url,
    )
    monkeypatch.setattr(
        "app.backend.media_download._open_url",
        lambda request, timeout: Response(),
    )

    assert download_image_url("https://example.com/plate.jpg") == payload


def test_image_url_endpoint_runs_multi_food_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_bytes = make_jpeg_bytes()
    monkeypatch.setattr(api, "download_image_url", lambda url: image_bytes)
    monkeypatch.setattr(
        api,
        "predict_multi_food_image_bytes",
        lambda payload: inference.build_multi_food_mock(
            fallback_reason="missing_artifacts"
        ),
    )

    response = client.post(
        "/predict/multi-food/image-url",
        json={"url": "https://example.com/plate.jpg"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["model"] == "resnet50_ft_v2"
    assert body["predictions"]


def test_image_url_endpoint_returns_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api,
        "download_image_url",
        lambda url: (_ for _ in ()).throw(
            UrlValidationError("Enter a public media URL.")
        ),
    )

    response = client.post(
        "/predict/multi-food/image-url",
        json={"url": "http://127.0.0.1/plate.jpg"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Enter a public media URL."


def test_youtube_url_endpoint_returns_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api,
        "predict_multi_food_youtube_url",
        lambda url: (_ for _ in ()).throw(
            MediaDependencyError("YouTube support needs yt-dlp.")
        ),
    )

    response = client.post(
        "/predict/multi-food/youtube-url",
        json={"url": "https://www.youtube.com/watch?v=abc123"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "YouTube support needs yt-dlp."


def test_combine_frame_responses_marks_predictions_as_video_frames() -> None:
    first = inference.build_multi_food_mock(fallback_reason="missing_artifacts")
    second = inference.build_multi_food_mock(fallback_reason="missing_artifacts")

    combined = combine_frame_responses([first, second])

    assert combined.model == "resnet50_ft_v2 · Video review"
    assert combined.detector_status == "fallback_demo · 2 frames"
    assert combined.crop_count == len(first.predictions) + len(second.predictions)
    assert combined.predictions[0].source_id == "youtube_frame_1"
    assert combined.predictions[len(first.predictions)].source_id == "youtube_frame_2"
    assert [prediction.detection_index for prediction in combined.predictions] == list(
        range(combined.crop_count)
    )
