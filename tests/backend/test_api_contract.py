from pathlib import Path

from fastapi.testclient import TestClient
import pytest

import app.backend.inference as inference
from app.backend.api import app


client = TestClient(app)


def test_health_contract() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_single_image_missing_artifacts_returns_mock_with_fallback_reason(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(inference, "ARTIFACT_DIR", tmp_path)

    response = client.post(
        "/predict/image",
        files={"file": ("sample.jpg", b"not-a-real-image", "image/jpeg")},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["artifact_status"] == "mock"
    assert body["fallback_reason"] == "missing_artifacts"
    assert body["top_predictions"][0]["class_name"] == "steak"
    assert body["decision"]["band"] == "suggest"


def test_multi_food_missing_artifacts_returns_demo_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(inference, "ARTIFACT_DIR", tmp_path)

    response = client.post(
        "/predict/multi-food/image",
        files={"file": ("sample.jpg", b"not-a-real-image", "image/jpeg")},
    )

    body = response.json()
    first_prediction = body["predictions"][0]
    assert response.status_code == 200
    assert body["artifact_status"] == "mock"
    assert body["detector_status"] == "fallback_demo"
    assert body["fallback_reason"] == "missing_artifacts"
    assert body["crop_count"] == len(body["predictions"])
    assert first_prediction["bbox"]["source_width"] > 0
    assert first_prediction["detector"]["proposal_role"] in {
        "serving_container",
        "direct_food",
        "fallback_region",
        "context_object",
    }
    assert first_prediction["foodlens"]["top_k_predictions"][0][0]
    assert "crop_path" in first_prediction["artifacts"]


def test_video_mock_reports_explicit_fallback_when_artifacts_are_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(inference, "ARTIFACT_DIR", tmp_path)
    (tmp_path / "resnet50_ft_v2_best.pth").write_bytes(b"placeholder")
    (tmp_path / "class_names.json").write_text("[]")

    response = client.post(
        "/predict/video",
        files={"file": ("sample.mp4", b"not-a-real-video", "video/mp4")},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["mode"] == "video"
    assert body["artifact_status"] == "ready"
    assert body["fallback_reason"] == "video_mock"
