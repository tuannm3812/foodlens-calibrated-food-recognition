"""Cover run_yolo_detection with a fake YOLO detector.

detection.py imports `ultralytics.YOLO` lazily, inside run_yolo_detection, so
these tests never need the real dependency installed: they install a fake
`ultralytics` module in sys.modules whose YOLO class returns a scripted
result object shaped like the real one (`.boxes` iterable of boxes exposing
`.xyxy`, `.conf`, `.cls`, plus `.names` on the result).
"""

import gc
import os
import sys
import threading
import time
import types
import weakref
from pathlib import Path
from typing import Any

import pytest

import app.backend.detection as detection
import app.backend.detector_policy as detector_policy
from app.backend.detection import clear_detector_cache, run_yolo_detection
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
def _fresh_detector_cache() -> Any:
    # The model cache is process-wide; each test installs its own fake YOLO.
    clear_detector_cache()
    yield
    clear_detector_cache()


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


# ---------------------------------------------------------------------------
# Model cache: one YOLO per weights file, predict serialised per model
# ---------------------------------------------------------------------------


class CountingYOLO:
    """Fake YOLO class recording each construction and tracking overlap."""

    def __init__(self, names: dict[int, str] | None = None) -> None:
        self.constructed: list[str] = []
        self.active = 0
        self.max_active = 0
        self.guard = threading.Lock()
        self.predict_delay = 0.0
        self.construct_delay = 0.0
        self.fail_next_construction = False
        self.on_construct: Any = None
        self.instances: list[weakref.ref[Any]] = []
        self.names = names or {0: "pizza"}

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        counter = self

        class Detector:
            def __init__(self, weights_path: str) -> None:
                time.sleep(counter.construct_delay)
                if counter.fail_next_construction:
                    counter.fail_next_construction = False
                    raise FileNotFoundError(weights_path)
                if counter.on_construct is not None:
                    counter.on_construct(weights_path)
                counter.constructed.append(weights_path)
                counter.instances.append(weakref.ref(self))

            def predict(self, **_: Any) -> list[FakeResult]:
                with counter.guard:
                    counter.active += 1
                    counter.max_active = max(counter.max_active, counter.active)
                time.sleep(counter.predict_delay)
                with counter.guard:
                    counter.active -= 1
                boxes = [FakeBox(xyxy=[10, 10, 40, 40], conf=0.9, cls=0)]
                return [FakeResult(boxes, counter.names)]

        fake_module = types.ModuleType("ultralytics")
        fake_module.YOLO = Detector  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "ultralytics", fake_module)


def test_model_is_loaded_once_and_reused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    yolo = CountingYOLO()
    yolo.install(monkeypatch)
    weights = tmp_path / "yolo.pt"
    weights.write_bytes(b"weights")

    for _ in range(3):
        assert len(run_yolo_detection(FakeImage(100, 100), str(weights))) == 1

    assert yolo.constructed == [str(weights.resolve())]


def test_spellings_of_one_file_share_a_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    yolo = CountingYOLO()
    yolo.install(monkeypatch)
    (tmp_path / "yolo.pt").write_bytes(b"weights")
    monkeypatch.chdir(tmp_path)

    run_yolo_detection(FakeImage(100, 100), "yolo.pt")
    run_yolo_detection(FakeImage(100, 100), str(tmp_path / "yolo.pt"))
    run_yolo_detection(FakeImage(100, 100), "./yolo.pt")

    assert yolo.constructed == [str((tmp_path / "yolo.pt").resolve())]


def test_a_new_weights_path_replaces_the_cached_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One slot: rotating through versioned checkpoints (or a symlink's
    targets) must not keep every earlier model alive."""
    yolo = CountingYOLO()
    yolo.install(monkeypatch)
    paths = [tmp_path / f"v{number}.pt" for number in range(5)]
    for path in paths:
        path.write_bytes(path.name.encode())
        run_yolo_detection(FakeImage(100, 100), str(path))
        run_yolo_detection(FakeImage(100, 100), str(path))
    gc.collect()

    assert yolo.constructed == [str(path.resolve()) for path in paths]
    assert [ref() is not None for ref in yolo.instances] == [False] * 4 + [True]


def test_rotating_a_symlink_reloads_and_keeps_one_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    yolo = CountingYOLO()
    yolo.install(monkeypatch)
    link = tmp_path / "current.pt"
    for number in range(3):
        target = tmp_path / f"v{number}.pt"
        target.write_bytes(b"same size")
        link.unlink(missing_ok=True)
        link.symlink_to(target)
        run_yolo_detection(FakeImage(100, 100), str(link))
    gc.collect()

    assert len(yolo.constructed) == 3
    assert sum(ref() is not None for ref in yolo.instances) == 1


def test_weights_replaced_in_place_are_reloaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    yolo = CountingYOLO()
    yolo.install(monkeypatch)
    weights = tmp_path / "yolo.pt"
    weights.write_bytes(b"old")
    run_yolo_detection(FakeImage(100, 100), str(weights))

    weights.write_bytes(b"new weights")
    stat = weights.stat()
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    run_yolo_detection(FakeImage(100, 100), str(weights))

    assert yolo.constructed == [str(weights), str(weights)]


def test_same_size_rewrite_with_mtime_restored_is_reloaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    yolo = CountingYOLO()
    yolo.install(monkeypatch)
    weights = tmp_path / "yolo.pt"
    weights.write_bytes(b"aaaa")
    original = weights.stat()
    run_yolo_detection(FakeImage(100, 100), str(weights))

    time.sleep(0.01)  # let ctime move on coarse clocks
    weights.write_bytes(b"bbbb")
    os.utime(weights, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert weights.stat().st_size == original.st_size
    assert weights.stat().st_mtime_ns == original.st_mtime_ns
    run_yolo_detection(FakeImage(100, 100), str(weights))

    assert len(yolo.constructed) == 2


def test_same_size_file_swapped_in_by_rename_is_reloaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    yolo = CountingYOLO()
    yolo.install(monkeypatch)
    weights = tmp_path / "yolo.pt"
    weights.write_bytes(b"aaaa")
    original = weights.stat()
    run_yolo_detection(FakeImage(100, 100), str(weights))

    staged = tmp_path / "staged.pt"
    staged.write_bytes(b"bbbb")
    os.utime(staged, ns=(original.st_atime_ns, original.st_mtime_ns))
    os.replace(staged, weights)
    run_yolo_detection(FakeImage(100, 100), str(weights))

    assert len(yolo.constructed) == 2


def test_a_file_downloaded_during_load_is_loaded_once_more_then_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Ultralytics fetches a missing checkpoint while constructing the model.
    # That load is not cached (the file may change before it returns); the
    # next request loads the file now present, and that load is reused.
    yolo = CountingYOLO()
    yolo.on_construct = (
        lambda path: None if Path(path).exists() else Path(path).write_bytes(b"downloaded")
    )
    yolo.install(monkeypatch)
    weights = tmp_path / "yolo.pt"

    for _ in range(3):
        run_yolo_detection(FakeImage(100, 100), str(weights))

    assert yolo.constructed == [str(weights), str(weights.resolve())]


def test_a_download_replaced_before_the_load_returns_is_not_kept(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    weights = tmp_path / "yolo.pt"

    def download_then_replace(path: str) -> None:
        Path(path).write_bytes(b"downloaded A")
        staged = tmp_path / "staged.pt"
        staged.write_bytes(b"replaced B")
        os.replace(staged, weights)

    yolo = CountingYOLO()
    yolo.on_construct = download_then_replace
    yolo.install(monkeypatch)
    run_yolo_detection(FakeImage(100, 100), str(weights))
    yolo.on_construct = None
    run_yolo_detection(FakeImage(100, 100), str(weights))
    run_yolo_detection(FakeImage(100, 100), str(weights))

    assert len(yolo.constructed) == 2


def test_a_missing_file_that_stays_missing_is_never_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # As before caching: each request asks Ultralytics to load it.
    yolo = CountingYOLO()
    yolo.install(monkeypatch)
    weights = tmp_path / "absent.pt"

    for _ in range(3):
        run_yolo_detection(FakeImage(100, 100), str(weights))

    assert yolo.constructed == [str(weights)] * 3


def test_a_symlink_moved_during_load_caches_the_file_actually_loaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first, second = tmp_path / "a.pt", tmp_path / "b.pt"
    first.write_bytes(b"model a")
    second.write_bytes(b"model b")
    link = tmp_path / "current.pt"
    link.symlink_to(first)

    def move_link(path: str) -> None:
        link.unlink()
        link.symlink_to(second)

    yolo = CountingYOLO()
    yolo.on_construct = move_link
    yolo.install(monkeypatch)
    run_yolo_detection(FakeImage(100, 100), str(link))
    yolo.on_construct = None
    run_yolo_detection(FakeImage(100, 100), str(link))
    run_yolo_detection(FakeImage(100, 100), str(link))

    # The first load read a.pt (the resolved file), and is cached as a.pt;
    # the link now names b.pt, which is loaded once and then reused.
    assert yolo.constructed == [str(first.resolve()), str(second.resolve())]


def test_a_file_changed_while_loading_is_not_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    weights = tmp_path / "yolo.pt"
    weights.write_bytes(b"old")
    yolo = CountingYOLO()
    changes = iter([b"replaced mid-load"])
    yolo.on_construct = lambda path: Path(path).write_bytes(next(changes, b"replaced mid-load"))
    yolo.install(monkeypatch)

    run_yolo_detection(FakeImage(100, 100), str(weights))
    yolo.on_construct = None
    run_yolo_detection(FakeImage(100, 100), str(weights))
    run_yolo_detection(FakeImage(100, 100), str(weights))

    # The first model may hold the old contents, so the next call reloads,
    # and that reload (with nothing changing) is then reused.
    assert len(yolo.constructed) == 2
    assert detection._cached is not None


def test_failed_load_is_not_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    yolo = CountingYOLO()
    yolo.install(monkeypatch)
    weights = tmp_path / "yolo.pt"
    weights.write_bytes(b"weights")
    yolo.fail_next_construction = True

    with pytest.raises(FileNotFoundError):
        run_yolo_detection(FakeImage(100, 100), str(weights))
    run_yolo_detection(FakeImage(100, 100), str(weights))

    assert yolo.constructed == [str(weights.resolve())]


def test_concurrent_requests_load_once_and_never_predict_at_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    yolo = CountingYOLO()
    yolo.construct_delay = 0.05
    yolo.predict_delay = 0.02
    yolo.install(monkeypatch)
    weights = tmp_path / "yolo.pt"
    weights.write_bytes(b"weights")
    results: list[int] = []
    errors: list[BaseException] = []
    start = threading.Barrier(8)

    def request() -> None:
        try:
            start.wait()
            results.append(len(run_yolo_detection(FakeImage(100, 100), str(weights))))
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=request) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert results == [1] * 8
    assert yolo.constructed == [str(weights.resolve())]
    assert yolo.max_active == 1
