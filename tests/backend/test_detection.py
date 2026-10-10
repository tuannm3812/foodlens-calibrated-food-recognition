"""Cover run_yolo_detection with a fake YOLO detector.

detection.py imports `ultralytics.YOLO` lazily, inside run_yolo_detection, so
these tests never need the real dependency installed: they install a fake
`ultralytics` module in sys.modules whose YOLO class returns a scripted
result object shaped like the real one (`.boxes` iterable of boxes exposing
`.xyxy`, `.conf`, `.cls`, plus `.names` on the result).
"""

import sys
import types
from typing import Any

import pytest

import app.backend.detector_policy as detector_policy
from app.backend.detection import run_yolo_detection
from app.backend.detector_policy import MAX_CROP_AREA_RATIO, MIN_CROP_AREA_RATIO


class FakeTensor:
    """Minimal stand-in for the torch tensor box.xyxy[0] normally is."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return list(self._values)


class FakeBox:
    def __init__(self, xyxy: list[float], conf: float, cls: int) -> None:
        self.xyxy = [FakeTensor(xyxy)]
        self.conf = [conf]
        self.cls = [cls]


class FakeResult:
    def __init__(self, boxes: list[FakeBox] | None, names: dict[int, str]) -> None:
        self.boxes = boxes
        self.names = names


class FakeImage:
    def __init__(self, width: int, height: int) -> None:
        self.size = (width, height)


def install_fake_ultralytics(
    monkeypatch: pytest.MonkeyPatch,
    boxes: list[FakeBox] | None,
    names: dict[int, str],
    predict_calls: list[dict[str, Any]] | None = None,
) -> None:
    """Register a fake `ultralytics` module so `from ultralytics import YOLO` works
    without the real dependency, and returns a single scripted result.
    """

    class FakeDetector:
        def __init__(self, weights_path: str) -> None:
            self.weights_path = weights_path

        def predict(
            self,
            source: Any,
            conf: float,
            iou: float,
            max_det: int,
            verbose: bool,
        ) -> list[FakeResult]:
            if predict_calls is not None:
                predict_calls.append(
                    {
                        "source": source,
                        "conf": conf,
                        "iou": iou,
                        "max_det": max_det,
                        "verbose": verbose,
                    }
                )
            return [FakeResult(boxes, names)]

    fake_module = types.ModuleType("ultralytics")
    fake_module.YOLO = FakeDetector  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)


@pytest.fixture(autouse=True)
def _default_label_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure detector_label_filter_config() resolves to "default" mode
    # regardless of the ambient environment.
    monkeypatch.delenv("FOODLENS_DETECTOR_LABELS", raising=False)


def test_detection_passing_all_filters_becomes_a_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predict_calls: list[dict[str, Any]] = []
    boxes = [FakeBox(xyxy=[10, 10, 40, 40], conf=0.91, cls=0)]
    install_fake_ultralytics(monkeypatch, boxes, {0: "pizza"}, predict_calls)

    rows = run_yolo_detection(FakeImage(100, 100), "weights.pt")

    assert rows == [
        {
            "detection_index": 0,
            "detector_label": "pizza",
            "proposal_role": "direct_food",
            "detector_confidence": pytest.approx(0.91),
            "crop_area_ratio": pytest.approx(900 / 10000),
            "x1": 10,
            "y1": 10,
            "x2": 40,
            "y2": 40,
            "source_width": 100,
            "source_height": 100,
        }
    ]
    # The weights path and detector knobs flow through to the detector call.
    assert predict_calls[0]["source"] is not None
    assert predict_calls[0]["verbose"] is False


@pytest.mark.parametrize(
    "xyxy",
    [
        pytest.param([30, 10, 30, 40], id="x2_equals_x1"),
        pytest.param([10, 30, 40, 30], id="y2_equals_y1"),
    ],
)
def test_degenerate_bbox_is_rejected_by_the_guard_itself(
    monkeypatch: pytest.MonkeyPatch, xyxy: list[float]
) -> None:
    """The `x2 <= x1 or y2 <= y1` guard must reject a zero-width/zero-height
    box on its own.

    A degenerate box always has area (x2-x1)*(y2-y1) == 0, so the downstream
    MIN_CROP_AREA_RATIO check would reject it anyway -- which would mask a
    `<=` -> `<` mutation of the guard (both leave `rows == []`, since x2<x1
    is False when x2==x1). To isolate the guard, MIN_CROP_AREA_RATIO is
    patched low enough that a zero-area box would otherwise clear the area
    filter, so only the guard itself can be responsible for the rejection.
    """
    monkeypatch.setattr(detector_policy, "MIN_CROP_AREA_RATIO", -1.0)
    boxes = [FakeBox(xyxy=xyxy, conf=0.9, cls=0)]
    install_fake_ultralytics(monkeypatch, boxes, {0: "pizza"})

    rows = run_yolo_detection(FakeImage(100, 100), "weights.pt")

    assert rows == []


def test_detection_below_min_crop_area_ratio_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 3x3 box in a 100x100 image is well under MIN_CROP_AREA_RATIO (0.015).
    boxes = [FakeBox(xyxy=[0, 0, 3, 3], conf=0.9, cls=0)]
    install_fake_ultralytics(monkeypatch, boxes, {0: "pizza"})
    assert (3 * 3) / (100 * 100) < MIN_CROP_AREA_RATIO

    rows = run_yolo_detection(FakeImage(100, 100), "weights.pt")

    assert rows == []


def test_detection_above_max_crop_area_ratio_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 95x95 box in a 100x100 image is well over MAX_CROP_AREA_RATIO (0.80).
    boxes = [FakeBox(xyxy=[0, 0, 95, 95], conf=0.9, cls=0)]
    install_fake_ultralytics(monkeypatch, boxes, {0: "pizza"})
    assert (95 * 95) / (100 * 100) > MAX_CROP_AREA_RATIO

    rows = run_yolo_detection(FakeImage(100, 100), "weights.pt")

    assert rows == []


def test_label_outside_candidate_set_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A well-sized, well-formed box, but "car" is not in CANDIDATE_REGION_LABELS.
    boxes = [FakeBox(xyxy=[10, 10, 40, 40], conf=0.9, cls=0)]
    install_fake_ultralytics(monkeypatch, boxes, {0: "car"})

    rows = run_yolo_detection(FakeImage(100, 100), "weights.pt")

    assert rows == []


def test_bbox_is_clamped_to_image_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    # Detector coordinates can fall outside the image; the region must be
    # clamped to [0, width] / [0, height] rather than kept as-is.
    boxes = [FakeBox(xyxy=[-20, -20, 130, 15], conf=0.9, cls=0)]
    install_fake_ultralytics(monkeypatch, boxes, {0: "pizza"})

    rows = run_yolo_detection(FakeImage(100, 100), "weights.pt")

    assert len(rows) == 1
    row = rows[0]
    assert row["x1"] == 0
    assert row["y1"] == 0
    assert row["x2"] == 100
    assert row["y2"] == 15
    assert row["crop_area_ratio"] == pytest.approx((100 * 15) / (100 * 100))


def test_no_boxes_returns_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_ultralytics(monkeypatch, None, {})

    rows = run_yolo_detection(FakeImage(100, 100), "weights.pt")

    assert rows == []


def test_multiple_detections_get_sequential_detection_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boxes = [
        FakeBox(xyxy=[10, 10, 40, 40], conf=0.9, cls=0),
        FakeBox(xyxy=[50, 50, 80, 80], conf=0.5, cls=1),
    ]
    install_fake_ultralytics(monkeypatch, boxes, {0: "pizza", 1: "broccoli"})

    rows = run_yolo_detection(FakeImage(100, 100), "weights.pt")

    assert [row["detection_index"] for row in rows] == [0, 1]
    assert [row["detector_label"] for row in rows] == ["pizza", "broccoli"]


def test_missing_ultralytics_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ultralytics", None)

    with pytest.raises(RuntimeError, match="Multi-food detection requires ultralytics"):
        run_yolo_detection(FakeImage(100, 100), "weights.pt")
