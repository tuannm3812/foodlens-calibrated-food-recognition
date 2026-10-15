"""The pure helpers of scripts/measure_runtime.py; no torch, artifacts or images."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "measure_runtime.py"
_spec = importlib.util.spec_from_file_location("measure_runtime", SCRIPT)
measure = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(measure)

TEST_LIST = [f"{label}/{index}" for label in ("pizza", "apple_pie", "sushi") for index in range(10)]


def test_the_sample_is_fixed_by_its_seed_and_independent_of_input_order() -> None:
    first = measure.select_sample(TEST_LIST, 2, seed=7)
    assert first == measure.select_sample(list(reversed(TEST_LIST)), 2, seed=7)
    assert first != measure.select_sample(TEST_LIST, 2, seed=8)


def test_the_sample_takes_per_class_entries_from_every_class() -> None:
    sample = measure.select_sample([*TEST_LIST, "", "  "], 3, seed=1)
    labels = [entry.split("/")[0] for entry in sample]
    assert sorted(set(labels)) == ["apple_pie", "pizza", "sushi"]
    assert all(labels.count(label) == 3 for label in set(labels))


def test_manifest_entries_keep_the_label_and_file_name_only() -> None:
    rows = [{"path": "/kaggle/input/food41/images/pizza/123.jpg", "label": "pizza"}]
    assert measure.manifest_entries(rows) == ["pizza/123.jpg"]


def test_a_class_with_fewer_entries_than_asked_gives_all_of_them() -> None:
    assert measure.select_sample(["steak/1", "steak/2"], 5, seed=0) == ["steak/1", "steak/2"]


@pytest.mark.parametrize(
    ("q", "expected"), [(0, 1.0), (50, 2.5), (95, 3.85), (100, 4.0)]
)
def test_percentile_interpolates_between_ranks(q, expected) -> None:
    assert measure.percentile([4.0, 1.0, 3.0, 2.0], q) == pytest.approx(expected)


def test_percentile_of_nothing_is_an_error() -> None:
    with pytest.raises(ValueError):
        measure.percentile([], 50)


def test_timing_summary_reports_p50_p95_mean_and_max() -> None:
    assert measure.timing_summary([10.0, 20.0, 30.0]) == {
        "n": 3, "p50_ms": 20.0, "p95_ms": 29.0, "mean_ms": 20.0, "max_ms": 30.0,
    }


def row(label, top1, band, *, top5=True, crop=None, fallback=False, any_crop=None):
    record = {
        "label": label,
        "whole_top1": top1,
        "whole_top1_correct": top1 == label,
        "whole_top5_correct": top5,
        "whole_band": band,
        "crop_count": None,
    }
    if crop is not None:
        record.update(
            crop_count=1 if fallback else 2,
            detector_fallback=fallback,
            best_crop_top1=crop,
            best_crop_top1_correct=crop == label,
            any_crop_top1_correct=crop == label if any_crop is None else any_crop,
            best_crop_band="suggest",
            detector_best_crop_top1_correct=crop == label,
        )
    return record


def test_accuracy_is_reported_overall_and_by_band() -> None:
    rows = [
        row("pizza", "pizza", "auto_accept"),
        row("sushi", "sushi", "auto_accept"),
        row("steak", "filet_mignon", "suggest", top5=True),
        row("steak", "pork_chop", "review", top5=False),
    ]
    summary = measure.accuracy_summary(rows)
    assert summary["whole_image"]["top1_accuracy"] == 0.5
    assert summary["whole_image"]["top5_accuracy"] == 0.75
    assert summary["whole_image"]["by_band"]["auto_accept"] == {
        "n": 2, "share": 0.5, "top1_accuracy": 1.0, "top5_containment": 1.0,
    }
    assert summary["whole_image"]["by_band"]["confirm"]["top1_accuracy"] is None
    assert summary["crops"]["images"] == 0


def test_crop_accuracy_separates_detector_fallbacks_and_disagreements() -> None:
    rows = [
        row("pizza", "pizza", "auto_accept", crop="pizza"),
        row("sushi", "sushi", "auto_accept", crop="sashimi", any_crop=True),
        row("steak", "filet_mignon", "suggest", crop="steak", fallback=True),
    ]
    crops = measure.accuracy_summary(rows)["crops"]
    assert crops["detector_fallback_rate"] == pytest.approx(0.3333)
    assert crops["best_crop_top1_accuracy"] == pytest.approx(0.6667)
    assert crops["any_crop_top1_accuracy"] == 1.0
    assert crops["with_regions"] == {
        "images": 2, "whole_top1_accuracy": 1.0, "best_crop_top1_accuracy": 0.5,
    }
    assert crops["whole_right_best_crop_wrong"] == 1
    assert crops["whole_wrong_best_crop_right"] == 1


def test_confident_false_accepts_count_wrong_auto_accepts_only() -> None:
    rows = [
        row("pizza", "pizza", "auto_accept", crop="pizza"),
        row("steak", "filet_mignon", "auto_accept", crop="steak"),
        row("sushi", "sashimi", "suggest", crop="sashimi"),
    ]
    rows[2]["best_crop_band"] = "auto_accept"
    summary = measure.accuracy_summary(rows)
    assert summary["whole_image"]["confident_false_accepts"] == 1
    assert summary["crops"]["best_crop_confident_false_accepts"] == 1


def test_the_multipart_body_carries_the_file_under_one_boundary() -> None:
    body, content_type = measure.multipart_body("1.jpg", b"\xff\xd8data")
    boundary = content_type.split("boundary=")[1]
    assert content_type.startswith("multipart/form-data; ")
    assert body.startswith(f"--{boundary}\r\n".encode())
    assert body.endswith(f"\r\n--{boundary}--\r\n".encode())
    assert b'name="file"; filename="1.jpg"' in body
    assert b"\r\n\r\n\xff\xd8data\r\n" in body


def test_top_confusions_count_only_whole_image_errors() -> None:
    rows = [
        row("steak", "filet_mignon", "suggest"),
        row("steak", "filet_mignon", "suggest"),
        row("pizza", "pizza", "auto_accept"),
    ]
    assert measure.top_confusions(rows) == [
        {"actual": "steak", "predicted": "filet_mignon", "count": 2}
    ]


# --- measure() end to end, with the classifier and detector faked ----------


@pytest.fixture
def food101(tmp_path: Path) -> Path:
    from PIL import Image

    images = tmp_path / "images"
    lines = ["path,label"]
    for label in ("pizza", "sushi"):
        (images / label).mkdir(parents=True)
        for index in range(3):
            Image.new("RGB", (64, 48), (200, 100, 50)).save(images / label / f"{index}.jpg")
            # The path the training machine recorded; only the file name is used.
            lines.append(f"/kaggle/input/food41/images/{label}/{index}.jpg,{label}")
    manifest = tmp_path / "test_manifest.csv"
    manifest.write_text("\n".join(lines) + "\n")
    return images, manifest


def test_measure_records_timings_accuracy_and_identity(food101, tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    from app.backend import inference
    from app.backend.schemas import Prediction

    runtime = {
        "device": "cpu",
        "temperature": 0.88,
        "class_names_sha256": "c" * 64,
        "model_name": "fake",
        "model_identity": {"architecture": "convnext_tiny", "checkpoint": "a.pth",
                           "model_name": "fake", "checkpoint_sha256": "a" * 64},
        "policy": {"auto_confidence": 0.7, "suggest_confidence": 0.35, "margin_threshold": 0.05},
        "hard_classes": set(),
        "confusion_pairs": set(),
    }

    def classify(image, _runtime):
        # Whole images (64 wide) are "pizza"; crops (32 wide) are "sushi".
        top = "pizza" if image.size[0] == 64 else "sushi"
        names = [top, "steak", "sashimi", "ramen", "pho"]
        return [Prediction(rank=i + 1, class_name=n, confidence=c)
                for i, (n, c) in enumerate(zip(names, (0.9, 0.04, 0.03, 0.02, 0.01), strict=True))]

    def detect(image):
        return [{"detection_index": 0, "detector_label": "bowl", "proposal_role": "food_container",
                 "detector_confidence": 0.8, "crop_area_ratio": 0.5, "x1": 0, "y1": 0,
                 "x2": 32, "y2": 48, "source_width": 64, "source_height": 48}]

    monkeypatch.setattr(inference, "load_runtime", lambda: runtime)
    monkeypatch.setattr(inference, "classify_pil_image", classify)
    monkeypatch.setattr(inference, "detect_candidate_regions", detect)
    monkeypatch.setattr(inference, "decision_layer_status", lambda: {"fingerprint": "f" * 64})

    out = tmp_path / "out"
    images, manifest = food101
    assert measure.main(["--images-dir", str(images), "--manifest", str(manifest),
                         "--per-class", "2", "--output-dir", str(out)]) == 0
    result = json.loads((out / "measurement.json").read_text())
    assert result["sample"]["images"] == 4
    assert result["sample"]["manifest_rows"] == 6
    assert result["rows"][0]["image"] == "pizza/0.jpg"
    assert result["model"]["checkpoint_sha256"] == "a" * 64
    assert result["decision_layer"]["fingerprint"] == "f" * 64
    assert set(result["warm_ms"]) == {
        "decode", "classifier_only", "classifier_with_decision",
        "detector_only", "multi_food_crops", "detector_plus_crops",
    }
    assert set(result["cold_ms"]) == {
        "load_runtime_ms", "first_classification_ms", "first_detection_ms",
    }
    accuracy = result["accuracy"]
    assert accuracy["whole_image"]["top1_accuracy"] == 0.5
    assert accuracy["crops"]["best_crop_top1_accuracy"] == 0.5
    assert accuracy["crops"]["whole_right_best_crop_wrong"] == 2
    assert accuracy["crops"]["detector_fallback_rate"] == 0.0
    assert accuracy["whole_image"]["confident_false_accepts"] == 2
    assert result["api"] is None
    crop = result["rows"][0]["crops"][0]
    assert crop["detector_label"] == "bowl"
    assert crop["box"] == [0, 0, 32, 48]
    assert crop["top1"] == "sushi"
    assert "rows" not in json.loads(capsys.readouterr().out)


def test_a_detector_that_cannot_run_is_reported_and_crops_skipped(food101, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from app.backend import inference
    from app.backend.schemas import Prediction

    def classify(image, _runtime):
        return [Prediction(rank=i + 1, class_name=n, confidence=0.2)
                for i, n in enumerate(["pizza", "sushi", "steak", "ramen", "pho"])]

    def no_detector(image):
        raise RuntimeError("Multi-food detection requires ultralytics.")

    runtime = {"device": "cpu", "temperature": 1.0, "class_names_sha256": "c" * 64,
               "model_identity": {}, "policy": {"auto_confidence": 0.7,
               "suggest_confidence": 0.35, "margin_threshold": 0.05},
               "hard_classes": set(), "confusion_pairs": set()}
    monkeypatch.setattr(inference, "load_runtime", lambda: runtime)
    monkeypatch.setattr(inference, "classify_pil_image", classify)
    monkeypatch.setattr(inference, "detect_candidate_regions", no_detector)
    monkeypatch.setattr(inference, "decision_layer_status", lambda: {})

    out = tmp_path / "out"
    images, manifest = food101
    assert measure.main(["--images-dir", str(images), "--manifest", str(manifest),
                         "--output-dir", str(out)]) == 0
    result = json.loads((out / "measurement.json").read_text())
    assert result["detector"] == {
        "weights": inference.detector_weights_path(),
        "ran": False,
        "error": "RuntimeError: Multi-food detection requires ultralytics.",
    }
    assert "detector_only" not in result["warm_ms"]
    assert result["accuracy"]["crops"]["images"] == 0


API_RUNTIME = {
    "device": "cpu", "temperature": 0.88, "class_names_sha256": "c" * 64,
    "model_name": "a3b",
    "model_identity": {"architecture": "convnext_tiny", "model_name": "a3b",
                       "checkpoint_sha256": "a" * 64},
    "policy": {"auto_confidence": 0.7, "suggest_confidence": 0.35, "margin_threshold": 0.05},
    "hard_classes": set(), "confusion_pairs": set(),
}


def live_status(**model_overrides):
    model = {"source": "loaded_runtime", "architecture": "convnext_tiny", "model_name": "a3b",
             "checkpoint_sha256": "a" * 64, "temperature": 0.88,
             "class_names_sha256": "c" * 64, **model_overrides}
    return {"model": model, "decision_layer": {"fingerprint": "f" * 64},
            "multi_food": {"mode": "live_yolo_classifier"}}


def live_body(endpoint, **overrides):
    body = {"artifact_status": "ready", "fallback_reason": None, "temperature": 0.88,
            "model_name" if endpoint == "image" else "model": "a3b"}
    if endpoint == "multi_food":
        body["detector_status"] = "live_yolo"
    return {**body, **overrides}


@pytest.fixture
def api_run(food101, tmp_path, monkeypatch):
    """Run main() against a fake API; returns (run, posted urls)."""
    pytest.importorskip("torch")
    from app.backend import inference
    from app.backend.schemas import Prediction

    def classify(image, _runtime):
        return [Prediction(rank=i + 1, class_name=n, confidence=0.2)
                for i, n in enumerate(["pizza", "sushi", "steak", "ramen", "pho"])]

    monkeypatch.setattr(inference, "load_runtime", lambda: API_RUNTIME)
    monkeypatch.setattr(inference, "classify_pil_image", classify)
    monkeypatch.setattr(inference, "detect_candidate_regions", lambda image: [])
    monkeypatch.setattr(inference, "decision_layer_status", lambda: {"fingerprint": "f" * 64})
    posted = []

    def run(status=None, respond=None):
        responses = respond or (lambda endpoint, n: live_body(endpoint))

        def post(url, name, data):
            endpoint = "multi_food" if "multi-food" in url else "image"
            posted.append(endpoint)
            return responses(endpoint, posted.count(endpoint))

        monkeypatch.setattr(measure, "post_image", post)
        monkeypatch.setattr(measure, "get_json", lambda url: status or live_status())
        out = tmp_path / "out"
        images, manifest = food101
        code = measure.main(["--images-dir", str(images), "--manifest", str(manifest),
                             "--per-class", "1", "--api-url", "http://api/",
                             "--output-dir", str(out)])
        return code, json.loads((out / "measurement.json").read_text())

    return run, posted


def test_a_matching_api_is_timed_on_both_endpoints_after_one_warmup_each(api_run):
    run, posted = api_run
    code, result = run()
    assert code == 0
    assert result["api"]["invalid_responses"] == []
    assert result["api"]["runtime_status_after_warmup"]["model"]["source"] == "loaded_runtime"
    assert result["warm_ms"]["http_image"]["n"] == 2
    assert result["warm_ms"]["http_multi_food"]["n"] == 2
    # One warm-up per endpoint, then one timed request per image per endpoint.
    assert posted.count("image") == 3
    assert posted.count("multi_food") == 3


@pytest.mark.parametrize(
    "status",
    [
        live_status(checkpoint_sha256="b" * 64),
        live_status(temperature=1.0),
        live_status(class_names_sha256="d" * 64),
        live_status(source="artifact_files"),
        {**live_status(), "decision_layer": {"fingerprint": "e" * 64}},
        {**live_status(), "multi_food": {"mode": "demo_fallback"}},
    ],
    ids=["checkpoint", "temperature", "class-order", "not-loaded", "policy", "demo-mode"],
)
def test_an_api_serving_another_runtime_stops_the_run(api_run, status):
    run, _ = api_run
    with pytest.raises(SystemExit, match="different runtime"):
        run(status=status)


def test_a_warmup_that_falls_back_to_the_demo_stops_the_run(api_run):
    run, _ = api_run
    demo = live_body("multi_food", artifact_status="missing_artifacts",
                     fallback_reason="missing_artifacts", temperature=99)
    with pytest.raises(SystemExit, match="not live A3b: artifact_status"):
        run(respond=lambda endpoint, n: demo if endpoint == "multi_food" else live_body(endpoint))


def test_a_mid_run_fallback_is_left_out_of_the_timings(api_run):
    run, _ = api_run

    def respond(endpoint, n):
        if endpoint == "image" and n == 3:
            return live_body("image", fallback_reason="classifier_load_error")
        return live_body(endpoint)

    code, result = run(respond=respond)
    assert code == 0
    assert result["warm_ms"]["http_image"]["n"] == 1
    assert result["api"]["invalid_responses"] == [
        {"image": "sushi/0.jpg", "endpoint": "image",
         "problem": "fallback_reason 'classifier_load_error'"}
    ]


def test_a_no_regions_fallback_is_still_live_inference() -> None:
    expected = {"model_name": "a3b", "temperature": 0.88}
    body = live_body("multi_food", fallback_reason="no_detector_regions",
                     detector_status="live_yolo_whole_image_fallback")
    assert measure.response_problem("multi_food", body, expected) is None
    assert measure.response_problem("image", live_body("image", model_name="resnet"),
                                    expected) == "model_name 'resnet'"


def crop(top1, confidence, detector, area=0.5):
    return {"top1": top1, "top1_confidence": confidence,
            "detector_confidence": detector, "area_ratio": area}


def test_the_app_picks_by_classifier_confidence_not_detector_confidence() -> None:
    rows = [{
        "label": "pizza", "whole_top1_correct": True,
        # The detector is surer of the bowl; the classifier is surer of the pizza.
        "crops": [crop("ramen", 0.40, 0.90), crop("pizza", 0.80, 0.30)],
    }]
    strategies = measure.strategy_summary(rows)
    assert strategies["ui_ranked"]["correct"] == 1
    assert strategies["detector_ranked"]["correct"] == 0
    assert strategies["ui_and_detector_pick_differ"] == 1


def test_ties_keep_the_response_order_like_the_frontend_sort() -> None:
    first, second = crop("pizza", 0.5, 0.5), crop("ramen", 0.5, 0.5)
    assert measure.ui_best([first, second]) is first
    assert measure.detector_best([second, first]) is second


def test_the_min_area_strategy_falls_back_to_the_whole_image() -> None:
    rows = [
        {"label": "pizza", "whole_top1_correct": True,
         "crops": [crop("ramen", 0.9, 0.9, area=0.05)]},
        {"label": "sushi", "whole_top1_correct": False,
         "crops": [crop("sushi", 0.6, 0.6, area=0.3), crop("pho", 0.9, 0.9, area=0.1)]},
    ]
    strategies = measure.strategy_summary(rows, min_area=0.2)
    assert strategies["ui_ranked"]["correct"] == 0
    assert strategies["ui_ranked_min_area"]["correct"] == 2
    assert strategies["whole_image"]["correct"] == 1


def test_reanalyze_reads_a_saved_run_without_inference(tmp_path, capsys) -> None:
    saved = tmp_path / "measurement.json"
    saved.write_text(json.dumps({"rows": [
        {"label": "pizza", "whole_top1_correct": True,
         "crops": [crop("ramen", 0.40, 0.90), crop("pizza", 0.80, 0.30)]},
    ]}))
    assert measure.main(["--reanalyze", str(saved)]) == 0
    assert json.loads(capsys.readouterr().out)["ui_ranked"]["top1"] == 1.0
