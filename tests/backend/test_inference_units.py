"""Characterization tests for pure and file-driven logic in inference.py.

S2 Phase 1: these tests pin current behaviour of the code Phase 2 will move
into new modules. They do not assert desired behaviour -- only what the code
does today.
"""

import base64
import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

import app.backend.inference as inference

# ---------------------------------------------------------------------------
# detector_region_role
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filter_mode", ["default", "all", "configured"])
def test_detector_region_role_bowl_is_always_serving_container(filter_mode: str) -> None:
    role = inference.detector_region_role(
        "bowl", filter_mode=filter_mode, configured_labels={"pizza"}
    )

    assert role == "serving_container"


def test_detector_region_role_whole_image_is_fallback_region() -> None:
    role = inference.detector_region_role("whole_image")

    assert role == "fallback_region"


def test_detector_region_role_all_filter_mode_treats_any_label_as_direct_food() -> None:
    role = inference.detector_region_role("car", filter_mode="all")

    assert role == "direct_food"


def test_detector_region_role_configured_mode_matching_label_is_direct_food() -> None:
    role = inference.detector_region_role(
        "car", filter_mode="configured", configured_labels={"car", "pizza"}
    )

    assert role == "direct_food"


def test_detector_region_role_configured_mode_non_matching_label_is_context_object() -> None:
    role = inference.detector_region_role(
        "car", filter_mode="configured", configured_labels={"pizza"}
    )

    assert role == "context_object"


def test_detector_region_role_configured_mode_without_labels_falls_back_to_default() -> None:
    # filter_mode == "configured" but configured_labels is None: the
    # `configured_labels is not None` guard fails, so this falls through to
    # the default-set membership check rather than treating everything as
    # context_object.
    role = inference.detector_region_role("pizza", filter_mode="configured", configured_labels=None)

    assert role == "direct_food"


def test_detector_region_role_default_mode_direct_food_label_is_direct_food() -> None:
    role = inference.detector_region_role("pizza", filter_mode="default")

    assert role == "direct_food"


def test_detector_region_role_default_mode_label_outside_candidate_set_is_context_object() -> None:
    role = inference.detector_region_role("car", filter_mode="default")

    assert role == "context_object"


# ---------------------------------------------------------------------------
# should_export_detection
# ---------------------------------------------------------------------------


def test_should_export_detection_rejects_crop_below_min_area_ratio() -> None:
    exported = inference.should_export_detection(
        "pizza", inference.MIN_CROP_AREA_RATIO - 0.001
    )

    assert exported is False


def test_should_export_detection_accepts_crop_at_min_area_ratio_boundary() -> None:
    exported = inference.should_export_detection("pizza", inference.MIN_CROP_AREA_RATIO)

    assert exported is True


def test_should_export_detection_accepts_crop_at_max_area_ratio_boundary() -> None:
    exported = inference.should_export_detection("pizza", inference.MAX_CROP_AREA_RATIO)

    assert exported is True


def test_should_export_detection_rejects_crop_above_max_area_ratio() -> None:
    exported = inference.should_export_detection(
        "pizza", inference.MAX_CROP_AREA_RATIO + 0.001
    )

    assert exported is False


def test_should_export_detection_default_mode_rejects_label_outside_candidate_set() -> None:
    exported = inference.should_export_detection("car", 0.5, filter_mode="default")

    assert exported is False


def test_should_export_detection_default_mode_accepts_candidate_label() -> None:
    exported = inference.should_export_detection("pizza", 0.5, filter_mode="default")

    assert exported is True


def test_should_export_detection_all_mode_accepts_any_label_within_bounds() -> None:
    exported = inference.should_export_detection("car", 0.5, filter_mode="all")

    assert exported is True


def test_should_export_detection_configured_mode_accepts_only_configured_labels() -> None:
    accepted = inference.should_export_detection(
        "car", 0.5, filter_mode="configured", configured_labels={"car"}
    )
    rejected = inference.should_export_detection(
        "car", 0.5, filter_mode="configured", configured_labels={"pizza"}
    )

    assert accepted is True
    assert rejected is False


# ---------------------------------------------------------------------------
# detector_label_filter_config
# ---------------------------------------------------------------------------


def test_detector_label_filter_config_defaults_when_env_var_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FOODLENS_DETECTOR_LABELS", raising=False)

    mode, labels = inference.detector_label_filter_config()

    assert mode == "default"
    assert labels == set()


def test_detector_label_filter_config_all_mode_when_env_var_is_star(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FOODLENS_DETECTOR_LABELS", "*")

    mode, labels = inference.detector_label_filter_config()

    assert mode == "all"
    assert labels == set()


def test_detector_label_filter_config_all_mode_when_env_var_is_whitespace_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FOODLENS_DETECTOR_LABELS", "   ")

    mode, labels = inference.detector_label_filter_config()

    assert mode == "all"
    assert labels == set()


def test_detector_label_filter_config_configured_mode_parses_comma_separated_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FOODLENS_DETECTOR_LABELS", "pizza, bowl ,cake")

    mode, labels = inference.detector_label_filter_config()

    assert mode == "configured"
    assert labels == {"pizza", "bowl", "cake"}


def test_detector_label_filter_config_configured_mode_parses_semicolons_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FOODLENS_DETECTOR_LABELS", "pizza;bowl, cake")

    mode, labels = inference.detector_label_filter_config()

    assert mode == "configured"
    assert labels == {"pizza", "bowl", "cake"}


# ---------------------------------------------------------------------------
# read_policy / read_hard_classes / read_confusion_pairs
# ---------------------------------------------------------------------------


def _point_artifact_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(inference, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setenv("FOODLENS_ARTIFACT_DIR", str(tmp_path))


def test_read_policy_uses_file_values_when_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_artifact_dir(monkeypatch, tmp_path)
    (tmp_path / "decision_policy.json").write_text(
        json.dumps({"auto_confidence": 0.9, "suggest_confidence": 0.6, "margin_threshold": 0.3})
    )

    policy = inference.read_policy(inference.artifact_dir_path())

    assert policy == {
        "auto_confidence": 0.9,
        "suggest_confidence": 0.6,
        "margin_threshold": 0.3,
    }


def test_read_policy_fills_missing_keys_from_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_artifact_dir(monkeypatch, tmp_path)
    (tmp_path / "decision_policy.json").write_text(json.dumps({"auto_confidence": 0.9}))

    policy = inference.read_policy(inference.artifact_dir_path())

    assert policy["auto_confidence"] == 0.9
    assert policy["suggest_confidence"] == inference.DEFAULT_POLICY["suggest_confidence"]
    assert policy["margin_threshold"] == inference.DEFAULT_POLICY["margin_threshold"]


def test_read_policy_returns_defaults_when_file_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_artifact_dir(monkeypatch, tmp_path)

    policy = inference.read_policy(inference.artifact_dir_path())

    assert policy == {
        "auto_confidence": inference.DEFAULT_POLICY["auto_confidence"],
        "suggest_confidence": inference.DEFAULT_POLICY["suggest_confidence"],
        "margin_threshold": inference.DEFAULT_POLICY["margin_threshold"],
    }


def test_read_policy_raises_on_malformed_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # read_json has no try/except around json.loads: a malformed artifact
    # file crashes the caller rather than degrading to defaults.
    _point_artifact_dir(monkeypatch, tmp_path)
    (tmp_path / "decision_policy.json").write_text("{not valid json")

    with pytest.raises(json.JSONDecodeError):
        inference.read_policy(inference.artifact_dir_path())


def test_read_hard_classes_uses_file_values_when_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_artifact_dir(monkeypatch, tmp_path)
    (tmp_path / "hard_classes.json").write_text(json.dumps(["ramen", "pho"]))

    hard_classes = inference.read_hard_classes(inference.artifact_dir_path())

    assert hard_classes == {"ramen", "pho"}


def test_read_hard_classes_returns_defaults_when_file_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_artifact_dir(monkeypatch, tmp_path)

    hard_classes = inference.read_hard_classes(inference.artifact_dir_path())

    assert hard_classes == set(inference.DEFAULT_HARD_CLASSES)


def test_read_hard_classes_raises_on_malformed_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_artifact_dir(monkeypatch, tmp_path)
    (tmp_path / "hard_classes.json").write_text("[not valid")

    with pytest.raises(json.JSONDecodeError):
        inference.read_hard_classes(inference.artifact_dir_path())


def test_read_confusion_pairs_accepts_dict_and_list_entries_and_skips_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_artifact_dir(monkeypatch, tmp_path)
    (tmp_path / "confusion_pairs.json").write_text(
        json.dumps(
            [
                {"actual": "ramen", "predicted": "pho"},
                ["steak", "prime_rib"],
                "not-a-pair",
                {"actual": "only_one_key"},
            ]
        )
    )

    pairs = inference.read_confusion_pairs(inference.artifact_dir_path())

    assert pairs == {("ramen", "pho"), ("steak", "prime_rib")}


def test_read_confusion_pairs_returns_empty_set_when_file_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_artifact_dir(monkeypatch, tmp_path)

    pairs = inference.read_confusion_pairs(inference.artifact_dir_path())

    assert pairs == set()


def test_read_confusion_pairs_raises_on_malformed_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_artifact_dir(monkeypatch, tmp_path)
    (tmp_path / "confusion_pairs.json").write_text("{bad")

    with pytest.raises(json.JSONDecodeError):
        inference.read_confusion_pairs(inference.artifact_dir_path())


# ---------------------------------------------------------------------------
# build_classifier_fallback_predictions
# ---------------------------------------------------------------------------


def test_build_classifier_fallback_predictions_direct_food_uses_detector_label() -> None:
    row = {
        "detector_label": "pizza",
        "proposal_role": "direct_food",
        "detector_confidence": 0.91,
    }

    predictions = inference.build_classifier_fallback_predictions(row)

    assert predictions[0].class_name == "pizza"
    assert predictions[0].rank == 1
    assert predictions[1] == inference.Prediction(
        rank=2, class_name="classifier_unavailable", confidence=0.0
    )


def test_build_classifier_fallback_predictions_bowl_non_direct_uses_container_label() -> None:
    row = {
        "detector_label": "bowl",
        "proposal_role": "serving_container",
        "detector_confidence": 0.5,
    }

    predictions = inference.build_classifier_fallback_predictions(row)

    assert predictions[0].class_name == "food_in_container"


def test_build_classifier_fallback_predictions_other_non_direct_uses_generic_label() -> None:
    row = {
        "detector_label": "car",
        "proposal_role": "context_object",
        "detector_confidence": 0.5,
    }

    predictions = inference.build_classifier_fallback_predictions(row)

    assert predictions[0].class_name == "detected_food_region"


def test_build_classifier_fallback_predictions_caps_confidence_below_suggest_threshold() -> None:
    row = {
        "detector_label": "pizza",
        "proposal_role": "direct_food",
        "detector_confidence": 0.91,
    }

    predictions = inference.build_classifier_fallback_predictions(row)

    expected = inference.MULTI_FOOD_POLICY["suggest_confidence"] - 0.01
    assert predictions[0].confidence == pytest.approx(expected)


def test_build_classifier_fallback_predictions_keeps_low_confidence_unchanged() -> None:
    row = {
        "detector_label": "pizza",
        "proposal_role": "direct_food",
        "detector_confidence": 0.3,
    }

    predictions = inference.build_classifier_fallback_predictions(row)

    assert predictions[0].confidence == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# build_crop_data_url
# ---------------------------------------------------------------------------


def test_build_crop_data_url_returns_decodable_jpeg_data_url() -> None:
    crop = Image.new("RGB", (12, 8), color=(200, 100, 50))

    data_url = inference.build_crop_data_url(crop)

    assert data_url.startswith("data:image/jpeg;base64,")
    encoded_payload = data_url.removeprefix("data:image/jpeg;base64,")
    decoded_bytes = base64.b64decode(encoded_payload)
    decoded_image = Image.open(BytesIO(decoded_bytes))
    assert decoded_image.format == "JPEG"
    assert decoded_image.size == (12, 8)
