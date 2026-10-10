"""The optional model.json manifest and the /runtime/status ``model`` block.

Without a manifest the backend must behave exactly as before it existed:
ResNet50 from ``resnet50_ft_v2_best.pth``, reported as ``resnet50_ft_v2``. A
present manifest selects the architecture and checkpoint; a bad one is an
artifact failure (``classifier_load_error``), never a silent ResNet50.

The real 100MB checkpoints never load here: the model builder is replaced with a
tiny module whose state dict is saved as the checkpoint, so ``load_runtime()``
otherwise runs unchanged.
"""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn
from torchvision import models

import app.backend.artifacts as artifacts
import app.backend.inference as inference
from app.backend.classifier import build_classifier_model

CLASS_NAMES = [f"class_{index:03d}" for index in range(101)]
CONVNEXT_MANIFEST = {
    "architecture": "convnext_tiny",
    "checkpoint": "convnext_tiny_continued_best.pth",
    "model_name": "a3b_convnext_tiny",
}


def tiny_model(torch_nn) -> nn.Module:
    """A 101-way classifier small enough to save in a test."""
    torch.manual_seed(0)
    return torch_nn.Sequential(
        torch_nn.AdaptiveAvgPool2d(1), torch_nn.Flatten(), torch_nn.Linear(3, 101)
    )


def write_checkpoint(path: Path) -> str:
    torch.save(tiny_model(nn).state_dict(), path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 32), color=(10, 200, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def artifact_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """An artifact dir with class names and no runtime loaded."""
    monkeypatch.setattr(inference, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setenv("FOODLENS_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setattr(inference, "_RUNTIME", None)
    (tmp_path / "class_names.json").write_text(json.dumps(CLASS_NAMES))
    return tmp_path


@pytest.fixture
def builds(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the model builder; record each architecture it is asked for."""
    calls: list[str] = []

    def build(torchvision_models, torch_nn, architecture):
        calls.append(architecture)
        return tiny_model(torch_nn)

    monkeypatch.setattr(inference, "build_classifier_model", build)
    return calls


# --- No manifest: exactly the legacy behaviour --------------------------------


def test_absent_manifest_reads_as_the_legacy_resnet(tmp_path: Path) -> None:
    assert artifacts.read_model_manifest(tmp_path) == {
        "architecture": "resnet50",
        "checkpoint": "resnet50_ft_v2_best.pth",
        "model_name": "resnet50_ft_v2",
    }
    assert artifacts.manifest_source(tmp_path) == "legacy_default"


def test_absent_manifest_requires_the_legacy_checkpoint(artifact_dir: Path) -> None:
    assert not artifacts.classifier_artifacts_ready(artifact_dir)
    (artifact_dir / "convnext_tiny_continued_best.pth").write_bytes(b"x")
    assert not artifacts.classifier_artifacts_ready(artifact_dir)
    (artifact_dir / "resnet50_ft_v2_best.pth").write_bytes(b"x")
    assert artifacts.classifier_artifacts_ready(artifact_dir)


def test_absent_manifest_loads_resnet50_and_reports_its_name(
    artifact_dir: Path, builds: list[str]
) -> None:
    digest = write_checkpoint(artifact_dir / "resnet50_ft_v2_best.pth")
    response = inference.predict_image_bytes(png_bytes())
    assert response.fallback_reason is None
    assert response.artifact_status == "ready"
    assert response.model_name == "resnet50_ft_v2"
    assert builds == ["resnet50"]
    assert inference._RUNTIME["model_identity"] == {
        "manifest": "legacy_default",
        "architecture": "resnet50",
        "checkpoint": "resnet50_ft_v2_best.pth",
        "model_name": "resnet50_ft_v2",
        "checkpoint_sha256": digest,
    }


# --- A valid manifest selects the architecture and checkpoint ------------------


def test_manifest_builds_convnext_from_its_checkpoint(
    artifact_dir: Path, builds: list[str]
) -> None:
    (artifact_dir / "model.json").write_text(json.dumps(CONVNEXT_MANIFEST))
    # A stale ResNet checkpoint left beside it must not be what loads.
    (artifact_dir / "resnet50_ft_v2_best.pth").write_bytes(b"not a checkpoint")
    digest = write_checkpoint(artifact_dir / "convnext_tiny_continued_best.pth")

    response = inference.predict_image_bytes(png_bytes())

    assert response.fallback_reason is None
    assert response.model_name == "a3b_convnext_tiny"
    assert builds == ["convnext_tiny"]
    assert inference._RUNTIME["model_identity"]["checkpoint_sha256"] == digest


def test_manifest_checkpoint_decides_readiness(artifact_dir: Path) -> None:
    (artifact_dir / "model.json").write_text(json.dumps(CONVNEXT_MANIFEST))
    (artifact_dir / "resnet50_ft_v2_best.pth").write_bytes(b"x")
    assert not artifacts.classifier_artifacts_ready(artifact_dir)
    assert inference.predict_image_bytes(png_bytes()).fallback_reason == "missing_artifacts"
    (artifact_dir / "convnext_tiny_continued_best.pth").write_bytes(b"x")
    assert artifacts.classifier_artifacts_ready(artifact_dir)


def test_multi_food_live_response_reports_the_manifest_model_name(
    artifact_dir: Path, builds: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    (artifact_dir / "model.json").write_text(json.dumps(CONVNEXT_MANIFEST))
    write_checkpoint(artifact_dir / "convnext_tiny_continued_best.pth")
    monkeypatch.setattr(inference, "detect_candidate_regions", lambda image: [])
    response = inference.predict_multi_food_image_bytes(png_bytes())
    assert response.artifact_status == "ready"
    assert response.model == "a3b_convnext_tiny"


def test_demo_responses_keep_the_fixed_demo_model_name(artifact_dir: Path) -> None:
    (artifact_dir / "model.json").write_text(json.dumps(CONVNEXT_MANIFEST))
    assert inference.predict_mock().model_name == inference.MODEL_NAME == "resnet50_ft_v2"


# --- A bad manifest is a load error, never a silent ResNet50 --------------------


@pytest.mark.parametrize(
    ("content", "fragment"),
    [
        ('{"architecture": "convnext_tiny"', "unreadable"),
        (json.dumps(["convnext_tiny"]), "must hold a JSON object"),
        (json.dumps({"architecture": "convnext_tiny", "model_name": "m"}), "missing keys"),
        (json.dumps({**CONVNEXT_MANIFEST, "architecture": "vit_b_16"}), "unknown architecture"),
        (json.dumps({**CONVNEXT_MANIFEST, "architecure": "resnet50"}), "unknown keys"),
        (json.dumps({**CONVNEXT_MANIFEST, "checkpoint": "../model.pth"}), "file name"),
        (json.dumps({**CONVNEXT_MANIFEST, "model_name": ""}), "non-empty string"),
    ],
)
def test_bad_manifest_surfaces_as_classifier_load_error(
    artifact_dir: Path, builds: list[str], content: str, fragment: str
) -> None:
    (artifact_dir / "model.json").write_text(content)
    # A loadable legacy checkpoint is present: falling back to it would succeed.
    write_checkpoint(artifact_dir / "resnet50_ft_v2_best.pth")

    with pytest.raises(artifacts.ModelManifestError, match=fragment):
        inference.load_runtime()
    response = inference.predict_image_bytes(png_bytes())
    assert response.artifact_status == "ready"
    assert response.fallback_reason == "classifier_load_error"
    assert builds == []

    status = inference.runtime_status()
    assert status["classifier"]["status"] == "ready"
    assert status["model"]["source"] == "artifact_files"
    assert fragment in status["model"]["error"]
    assert status["classifier"]["artifacts"]["checkpoint"]["path"] is None


def test_bad_manifest_fails_multi_food_into_classifier_load_error(
    artifact_dir: Path, builds: list[str]
) -> None:
    (artifact_dir / "model.json").write_text(json.dumps({"architecture": "resnet50"}))
    write_checkpoint(artifact_dir / "resnet50_ft_v2_best.pth")
    response = inference.predict_multi_food_image_bytes(png_bytes())
    assert response.fallback_reason == "classifier_load_error"
    assert builds == []


# --- The status model block ----------------------------------------------------


@pytest.fixture
def hash_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count checkpoint hashing wherever it could happen."""
    calls: list[int] = []
    real = artifacts.checkpoint_sha256

    def counted(data: bytes) -> str:
        calls.append(len(data))
        return real(data)

    monkeypatch.setattr(artifacts, "checkpoint_sha256", counted)
    monkeypatch.setattr(inference, "checkpoint_sha256", counted)
    return calls


def test_status_before_load_reports_the_manifest_without_hashing(
    artifact_dir: Path, hash_calls: list[int]
) -> None:
    (artifact_dir / "model.json").write_text(json.dumps(CONVNEXT_MANIFEST))
    write_checkpoint(artifact_dir / "convnext_tiny_continued_best.pth")

    block = inference.runtime_status()["model"]

    assert block["source"] == "artifact_files"
    assert block["manifest"] == "model.json"
    assert block["architecture"] == "convnext_tiny"
    assert block["checkpoint"] == "convnext_tiny_continued_best.pth"
    assert block["model_name"] == "a3b_convnext_tiny"
    assert block["checkpoint_sha256"] is None
    assert "not loaded" in block["note"]
    assert hash_calls == []
    assert inference._RUNTIME is None


def test_status_before_load_without_manifest_reports_the_legacy_model(
    artifact_dir: Path,
) -> None:
    block = inference.runtime_status()["model"]
    assert block["manifest"] == "legacy_default"
    assert block["architecture"] == "resnet50"
    assert block["checkpoint"] == "resnet50_ft_v2_best.pth"


def test_status_after_load_reports_the_hash_computed_once(
    artifact_dir: Path, builds: list[str], hash_calls: list[int]
) -> None:
    (artifact_dir / "model.json").write_text(json.dumps(CONVNEXT_MANIFEST))
    digest = write_checkpoint(artifact_dir / "convnext_tiny_continued_best.pth")

    inference.load_runtime()
    blocks = [inference.runtime_status()["model"] for _ in range(3)]

    assert hash_calls == [(artifact_dir / "convnext_tiny_continued_best.pth").stat().st_size]
    for block in blocks:
        assert block == {
            "source": "loaded_runtime",
            "manifest": "model.json",
            **CONVNEXT_MANIFEST,
            "checkpoint_sha256": digest,
            # The effective calibration, captured at load (no calibration.json
            # here, so the backend default).
            "temperature": artifacts.TEMPERATURE,
            "class_names_sha256": artifacts.class_names_sha256(CLASS_NAMES),
        }


def test_status_keeps_the_loaded_identity_after_the_files_change(
    artifact_dir: Path, builds: list[str]
) -> None:
    (artifact_dir / "model.json").write_text(json.dumps(CONVNEXT_MANIFEST))
    digest = write_checkpoint(artifact_dir / "convnext_tiny_continued_best.pth")
    inference.load_runtime()

    # A promotion rewrites the files under a running process.
    (artifact_dir / "model.json").unlink()
    block = inference.runtime_status()["model"]
    assert block["architecture"] == "convnext_tiny"
    assert block["checkpoint_sha256"] == digest


def test_status_reports_the_calibration_cached_at_load(
    artifact_dir: Path, builds: list[str]
) -> None:
    (artifact_dir / "calibration.json").write_text('{"temperature": 0.884}')
    write_checkpoint(artifact_dir / "resnet50_ft_v2_best.pth")

    before = inference.runtime_status()["model"]
    assert before["temperature"] == 0.884
    assert before["class_names_sha256"] is None

    response = inference.predict_image_bytes(png_bytes())
    # A deployment rewrites the files under the running process.
    (artifact_dir / "calibration.json").write_text('{"temperature": 2.0}')
    (artifact_dir / "class_names.json").write_text(json.dumps(CLASS_NAMES[::-1]))
    block = inference.runtime_status()["model"]

    assert response.temperature == block["temperature"] == 0.884
    assert block["class_names_sha256"] == artifacts.class_names_sha256(CLASS_NAMES)


# --- Real architecture construction (no weights, no download) ------------------


def test_convnext_tiny_gets_the_project_head_in_classifier_2() -> None:
    model = build_classifier_model(models, nn, "convnext_tiny")
    head = model.classifier[2]
    assert isinstance(head, nn.Sequential)
    assert [layer.out_features for layer in head if isinstance(layer, nn.Linear)] == [
        512,
        256,
        101,
    ]
    assert head[0].in_features == 768


def test_resnet50_gets_the_project_head_in_fc() -> None:
    model = build_classifier_model(models, nn, "resnet50")
    assert isinstance(model.fc, nn.Sequential)
    assert model.fc[0].in_features == 2048
    assert model.fc[-1].out_features == 101


def test_unknown_architecture_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        build_classifier_model(models, nn, "vit_b_16")
