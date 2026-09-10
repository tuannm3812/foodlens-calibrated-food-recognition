"""Prove that each of the seven names tests patch on `inference` still bites.

S2 Phase 2 moved most of inference.py's logic into focused modules, keeping
only re-exported names on `inference` so existing monkeypatch.setattr(inference,
"X", fake) calls keep resolving. A resolving attribute is not the same thing
as a patch that actually takes effect: if a call site had moved out of
inference.py, the patch would become a silent no-op -- the attribute would
still exist, but nothing would ever read it. Each test here monkeypatches one
of the seven names with a fake that returns (or lets us detect) a distinctive
sentinel, then drives a real public entry point and asserts the sentinel
surfaces in the observable result -- proving the patched call site is still
inside inference.py.
"""

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

import app.backend.inference as inference


def make_jpeg_bytes() -> bytes:
    image = Image.new("RGB", (64, 48), color=(180, 90, 40))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def write_minimal_classifier_artifacts(artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "resnet50_ft_v2_best.pth").write_bytes(b"placeholder")
    (artifact_dir / "class_names.json").write_text('["apple"]')


def force_missing_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(inference, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setenv("FOODLENS_ARTIFACT_DIR", str(tmp_path))


# ---------------------------------------------------------------------------
# ARTIFACT_DIR
# ---------------------------------------------------------------------------


def test_artifact_dir_patch_changes_artifact_dir_path_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sentinel_dir = tmp_path / "sentinel-artifacts"
    write_minimal_classifier_artifacts(sentinel_dir)
    monkeypatch.delenv("FOODLENS_ARTIFACT_DIR", raising=False)
    monkeypatch.setattr(inference, "ARTIFACT_DIR", sentinel_dir)

    assert inference.artifact_dir_path() == sentinel_dir


# ---------------------------------------------------------------------------
# __file__
# ---------------------------------------------------------------------------


def test_dunder_file_patch_changes_detector_weights_path_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    backend_dir = repo_root / ".worktrees" / "branch" / "app" / "backend"
    backend_dir.mkdir(parents=True)
    sentinel_weights = repo_root / "yolo11n.pt"
    sentinel_weights.write_bytes(b"placeholder")
    monkeypatch.delenv("FOODLENS_DETECTOR_WEIGHTS", raising=False)
    monkeypatch.setattr(inference, "__file__", str(backend_dir / "inference.py"))

    assert inference.detector_weights_path() == str(sentinel_weights)


# ---------------------------------------------------------------------------
# artifact_status
# ---------------------------------------------------------------------------


def test_artifact_status_patch_changes_predict_image_bytes_fallback_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # No real classifier artifacts exist, so the unpatched gate would report
    # "mock" and fail with fallback_reason "missing_artifacts". Forcing the
    # patched artifact_status() to claim "ready" routes predict_image_bytes
    # into the load_runtime() branch instead, which then fails for a
    # different, observable reason -- proof the fake was actually consulted.
    force_missing_artifacts(monkeypatch, tmp_path)
    monkeypatch.setattr(inference, "_RUNTIME", None)
    monkeypatch.setattr(inference, "artifact_status", lambda: "ready")

    response = inference.predict_image_bytes(make_jpeg_bytes())

    assert response.artifact_status == "ready"
    assert response.fallback_reason == "classifier_load_error"


# ---------------------------------------------------------------------------
# artifact_dir_path
# ---------------------------------------------------------------------------


def test_artifact_dir_path_patch_changes_runtime_status_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sentinel_dir = tmp_path / "sentinel-runtime-artifacts"
    monkeypatch.setattr(inference, "artifact_dir_path", lambda: sentinel_dir)

    status = inference.runtime_status()

    assert status["classifier"]["artifact_dir"] == str(sentinel_dir)


# ---------------------------------------------------------------------------
# detect_candidate_regions
# ---------------------------------------------------------------------------


def test_detect_candidate_regions_patch_changes_multi_food_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    force_missing_artifacts(monkeypatch, tmp_path)
    sentinel_row = {
        "detection_index": 0,
        "detector_label": "sentinel_detector_label",
        "proposal_role": "direct_food",
        "detector_confidence": 0.42,
        "crop_area_ratio": 0.5,
        "x1": 0,
        "y1": 0,
        "x2": 10,
        "y2": 10,
        "source_width": 64,
        "source_height": 48,
    }
    monkeypatch.setattr(
        inference, "detect_candidate_regions", lambda image: [sentinel_row]
    )

    response = inference.predict_multi_food_image_bytes(make_jpeg_bytes())

    assert response.predictions[0].detector.label == "sentinel_detector_label"


# ---------------------------------------------------------------------------
# detector_weights_path
# ---------------------------------------------------------------------------


def test_detector_weights_path_patch_changes_runtime_status_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_weights_path = "/sentinel/path/to/weights.pt"
    monkeypatch.setattr(
        inference, "detector_weights_path", lambda: sentinel_weights_path
    )

    status = inference.runtime_status()

    assert status["detector"]["weights_path"] == sentinel_weights_path


# ---------------------------------------------------------------------------
# build_multi_food_mock
# ---------------------------------------------------------------------------


def test_build_multi_food_mock_patch_changes_multi_food_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_response = object()
    monkeypatch.setattr(
        inference, "build_multi_food_mock", lambda fallback_reason=None: sentinel_response
    )

    # Invalid image bytes make predict_multi_food_image_bytes reach for
    # build_multi_food_mock(fallback_reason="invalid_image") before it ever
    # tries the detector, regardless of artifact status.
    result = inference.predict_multi_food_image_bytes(b"not-a-real-image")

    assert result is sentinel_response
