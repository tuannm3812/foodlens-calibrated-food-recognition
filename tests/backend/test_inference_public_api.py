"""Pin the public surface of `app.backend.inference` across the S2 decomposition.

Before S2, `inference.py` had no `__all__`, so `from app.backend.inference
import *` exported every public top-level name. S2 introduced an `__all__`
listing only the names re-exported from the new modules, which silently dropped
31 of 47 public names from star-imports -- including `predict_image_bytes`,
`runtime_status` and the other entry points. The Codex GitHub reviewer flagged
it on PR #5.

The contract below is the pre-S2 public set (taken from `inference.py` at
`b9caca2`), hard-coded rather than read from git: CI uses a shallow clone, so a
test that consults history fails there.
"""

from __future__ import annotations

import app.backend.inference as inference

PRE_S2_PUBLIC_NAMES = frozenset(
    {
        "ARTIFACT_DIR", "CANDIDATE_REGION_LABELS", "DETECTOR_CONFIDENCE_THRESHOLD",
        "DETECTOR_IOU_THRESHOLD", "DETECTOR_MAX_DETECTIONS", "DETECTOR_WEIGHTS",
        "DIRECT_FOOD_LABELS", "IMAGE_SIZE", "MAX_CROP_AREA_RATIO", "MIN_CROP_AREA_RATIO",
        "MOCK_IMAGE_PREDICTIONS", "MOCK_MULTI_FOOD_REGIONS", "MOCK_VIDEO_PREDICTIONS",
        "MODEL_NAME", "MULTI_FOOD_POLICY", "REQUIRED_CLASSIFIER_ARTIFACTS", "TEMPERATURE",
        "artifact_dir_path", "artifact_file_status", "artifact_status",
        "build_classifier_fallback_predictions", "build_crop_data_url",
        "build_full_image_region", "build_multi_food_classifier_fallback_response",
        "build_multi_food_mock", "build_multi_food_response", "build_prediction_response",
        "build_predictions", "classifier_artifacts_ready", "classify_pil_image",
        "detect_candidate_regions", "detector_label_filter_config",
        "detector_region_role", "detector_weights_path", "load_runtime",
        "make_classifier_head", "open_rgb_image", "predict_image_bytes", "predict_mock",
        "predict_multi_food_image_bytes", "read_confusion_pairs", "read_hard_classes",
        "read_json", "read_policy", "read_temperature", "runtime_status",
        "should_export_detection",
    }
)


def test_contract_has_the_pre_s2_size() -> None:
    assert len(PRE_S2_PUBLIC_NAMES) == 47


def test_star_import_exports_every_pre_s2_public_name() -> None:
    namespace: dict[str, object] = {}
    exec("from app.backend.inference import *", namespace)
    missing = sorted(PRE_S2_PUBLIC_NAMES - namespace.keys())
    assert missing == []


def test_star_import_exports_the_entry_points_api_uses() -> None:
    namespace: dict[str, object] = {}
    exec("from app.backend.inference import *", namespace)
    for name in ("predict_image_bytes", "predict_multi_food_image_bytes", "predict_mock",
                 "runtime_status"):
        assert namespace[name] is getattr(inference, name)


def test_every_all_entry_resolves() -> None:
    unresolved = [name for name in inference.__all__ if not hasattr(inference, name)]
    assert unresolved == []


def test_all_has_no_duplicates() -> None:
    assert len(inference.__all__) == len(set(inference.__all__))
