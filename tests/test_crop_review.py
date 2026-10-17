"""scripts/crop_review.py: crop export for hand labelling, and label scoring."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "crop_review.py"
_spec = importlib.util.spec_from_file_location("crop_review", SCRIPT)
review = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(review)


def crop(index, top1, confidence, box=(0, 0, 32, 24), label="bowl", area=0.25):
    return {"detection_index": index, "detector_label": label,
            "proposal_role": "serving_container", "detector_confidence": 0.7,
            "area_ratio": area, "box": list(box), "top1": top1,
            "top1_confidence": confidence, "band": "suggest"}


def measured_row(image, label, crops, fallback=False):
    return {"image": image, "label": label, "detector_fallback": fallback, "crops": crops}


ROWS = [
    measured_row("pizza/1.jpg", "pizza", [crop(0, "pizza", 0.4), crop(1, "ramen", 0.9)]),
    measured_row("pizza/2.jpg", "pizza", [crop(0, "pizza", 0.9, label="whole_image")],
                 fallback=True),
    measured_row("sushi/1.jpg", "sushi", [crop(0, "sushi", 0.8)]),
    measured_row("steak/1.jpg", "steak", [crop(0, "steak", 0.8)]),
]


def test_the_sample_skips_fallbacks_and_is_fixed_by_its_seed() -> None:
    picked = review.select_images(ROWS, 2, seed=3)
    assert all(not row["detector_fallback"] for row in picked)
    assert picked == review.select_images(list(reversed(ROWS)), 2, seed=3)
    assert len(review.select_images(ROWS, 50, seed=3)) == 3


def test_label_rows_mark_the_crop_the_app_shows_and_leave_labels_blank() -> None:
    rows = review.label_rows([ROWS[0]])
    assert [row["app_pick"] for row in rows] == ["no", "yes"]
    assert rows[1]["crop_file"] == "crops/001_1.jpg"
    assert rows[0]["box"] == "0 0 32 24"
    assert all(row[column] == "" for row in rows for column in review.LABEL_COLUMNS)


def labelled(dish, top1, kind, region, pick="yes", detector="bowl"):
    return {"image": f"{dish}/x.jpg", "dish_label": dish, "classifier_top1": top1,
            "region_kind": kind, "region_label": region, "app_pick": pick,
            "detector_label": detector, "labeller": "claude-proposed", "notes": ""}


def test_a_side_item_named_correctly_counts_as_right_for_its_region() -> None:
    rows = [
        # Bibimbap photo; the crop is a side dish of kimchi... named as guacamole.
        labelled("bibimbap", "guacamole", "side_item", ""),
        # Steak photo; the crop is the side of french fries, named correctly.
        labelled("steak", "french_fries", "side_item", "french_fries", pick="no"),
        labelled("pizza", "pizza", "whole_dish", "pizza"),
        labelled("ramen", "pho", "empty_container", "", pick="no"),
    ]
    summary = review.score(rows)
    assert summary["all_crops"]["nameable"] == 2
    assert summary["all_crops"]["region_correct"] == 1.0
    assert summary["all_crops"]["dish_agreement"] == 0.25
    assert summary["app_picks"]["crops"] == 2
    assert summary["app_pick_shows_and_names_dish"] == 0.5
    assert summary["all_crops"]["kinds"]["side_item"] == 2


def test_label_problems_catch_unknown_kinds_classes_and_mismatched_dish_labels() -> None:
    classes = {"pizza", "steak", "french_fries"}
    rows = [
        labelled("pizza", "pizza", "whole_dish", "pizza"),
        labelled("pizza", "pizza", "dessert", ""),
        labelled("steak", "steak", "side_item", "chips"),
        labelled("steak", "steak", "part_of_dish", "french_fries"),
    ]
    problems = review.label_problems(rows, classes)
    assert len(problems) == 3
    assert problems[0].startswith("line 3: region_kind 'dessert'")
    assert "'chips' is not a Food-101 class" in problems[1]
    assert "should carry the dish label 'steak'" in problems[2]


def test_export_writes_crops_labels_and_a_review_page(tmp_path) -> None:
    from PIL import Image

    images = tmp_path / "images"
    for row in ROWS:
        (images / row["image"]).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 48), (200, 100, 50)).save(images / row["image"])
    measurement = tmp_path / "measurement.json"
    measurement.write_text(json.dumps({"measured_at": "t", "model": {}, "sample": {},
                                       "rows": ROWS}))
    out = tmp_path / "out"
    assert review.main(["export", "--measurement", str(measurement), "--images-dir",
                        str(images), "--count", "2", "--output-dir", str(out)]) == 0
    with (out / "labels.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len({row["image"] for row in rows}) == 2
    for row in rows:
        with Image.open(out / row["crop_file"]) as saved:
            assert saved.size == (32, 24)
    page = (out / "review.html").read_text()
    assert "app shows this" in page and "unlabelled" in page
    assert json.loads((out / "export.json").read_text())["images"] == 2

    # Fill one label and rebuild only the page.
    rows[0].update(region_kind="part_of_dish", region_label=rows[0]["dish_label"],
                   labeller="mike")
    review.write_labels(out / "labels.csv", rows)
    assert review.main(["export", "--refresh-html", str(out)]) == 0
    assert "part_of_dish" in (out / "review.html").read_text()


def test_score_refuses_invalid_labels(tmp_path, capsys) -> None:
    path = tmp_path / "labels.csv"
    review.write_labels(path, [{**review.label_rows([ROWS[2]])[0], "region_kind": "?"}])
    classes = tmp_path / "classes.json"
    classes.write_text(json.dumps(["sushi"]))
    assert review.main(["score", str(path), "--class-names", str(classes)]) == 1
    assert "region_kind '?'" in capsys.readouterr().out


@pytest.mark.parametrize("kind", review.REGION_KINDS)
def test_every_region_kind_is_accepted(kind) -> None:
    region = "pizza" if kind in ("whole_dish", "part_of_dish") else ""
    assert review.label_problems([labelled("pizza", "pizza", kind, region)], {"pizza"}) == []
