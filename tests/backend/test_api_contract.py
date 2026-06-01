from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
import pytest

import app.backend.inference as inference
from app.backend.api import app


client = TestClient(app)


def make_jpeg_bytes() -> bytes:
    image = Image.new("RGB", (100, 80), color=(220, 80, 40))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health_contract() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_runtime_status_reports_missing_classifier_artifacts_and_detector_readiness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(inference, "ARTIFACT_DIR", tmp_path)
    weights_path = tmp_path / "yolo11n.pt"
    weights_path.write_bytes(b"placeholder")
    monkeypatch.setenv("FOODLENS_DETECTOR_WEIGHTS", str(weights_path))
    monkeypatch.setattr(
        inference.importlib.util,
        "find_spec",
        lambda name: object() if name == "ultralytics" else None,
    )

    response = client.get("/runtime/status")

    body = response.json()
    assert response.status_code == 200
    assert body["classifier"]["status"] == "missing_artifacts"
    assert body["classifier"]["artifact_status"] == "mock"
    assert body["classifier"]["artifacts"]["checkpoint"]["exists"] is False
    assert body["classifier"]["artifacts"]["class_names"]["exists"] is False
    assert body["detector"]["dependency_available"] is True
    assert body["detector"]["weights_found"] is True
    assert body["detector"]["weights_path"] == str(weights_path)
    assert body["multi_food"]["mode"] == "detector_only_classifier_fallback"


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


def test_multi_food_missing_classifier_artifacts_uses_detector_crops_with_fallback_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(inference, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(
        inference,
        "detect_candidate_regions",
        lambda image: [
            {
                "detection_index": 7,
                "detector_label": "pizza",
                "proposal_role": "direct_food",
                "detector_confidence": 0.91,
                "crop_area_ratio": 0.15,
                "x1": 10,
                "y1": 12,
                "x2": 70,
                "y2": 62,
                "source_width": image.size[0],
                "source_height": image.size[1],
            }
        ],
    )

    response = client.post(
        "/predict/multi-food/image",
        files={"file": ("sample.jpg", make_jpeg_bytes(), "image/jpeg")},
    )

    body = response.json()
    first_prediction = body["predictions"][0]
    assert response.status_code == 200
    assert body["artifact_status"] == "mock"
    assert body["detector_status"] == "live_yolo_classifier_fallback"
    assert body["fallback_reason"] == "missing_classifier_artifacts"
    assert body["crop_count"] == 1
    assert first_prediction["source_id"] == "uploaded_image"
    assert first_prediction["detection_index"] == 7
    assert first_prediction["bbox"] == {
        "x1": 10,
        "y1": 12,
        "x2": 70,
        "y2": 62,
        "source_width": 100,
        "source_height": 80,
    }
    assert first_prediction["detector"]["label"] == "pizza"
    assert first_prediction["detector"]["proposal_role"] == "direct_food"
    assert first_prediction["foodlens"]["top_label"] == "pizza"
    assert first_prediction["foodlens"]["decision_band"] == "confirm"
    assert first_prediction["artifacts"]["crop_data_url"].startswith(
        "data:image/jpeg;base64,"
    )


def test_detector_weights_path_finds_nearest_parent_weight_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    backend_dir = repo_root / ".worktrees" / "branch" / "app" / "backend"
    backend_dir.mkdir(parents=True)
    weights_path = repo_root / "yolo11n.pt"
    weights_path.write_bytes(b"placeholder")
    monkeypatch.delenv("FOODLENS_DETECTOR_WEIGHTS", raising=False)
    monkeypatch.setattr(inference, "__file__", str(backend_dir / "inference.py"))

    assert inference.detector_weights_path() == str(weights_path)


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
