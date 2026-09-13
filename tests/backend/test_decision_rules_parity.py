"""Differential test: `route_decision` must agree with `build_decision`.

This is the guarantee described in the design doc (section 3.1): production
(`app/backend/decision.py::build_decision`) and offline recalibration
(`scripts/recalibrate_decision_layer.py::assign_decision_band`) both delegate
the band decision to the single `route_decision` function in
`app/backend/decision_rules.py`. If they ever diverged, the offline policy
search would again optimise a decision function production cannot execute.

This test proves they cannot diverge, across a grid covering every branch of
`route_decision` / `build_decision`, and separately proves the structural
property that makes the leak impossible: `route_decision`'s signature has no
actual-label parameter, under any of its common names.
"""

from __future__ import annotations

import inspect
import itertools

from app.backend.decision import build_decision
from app.backend.decision_rules import route_decision
from app.backend.schemas import Prediction

POLICY = {
    "auto_confidence": 0.70,
    "suggest_confidence": 0.35,
    "margin_threshold": 0.40,
}
HARD_CLASSES = {"steak"}
CONFUSION_PAIRS = {("filet_mignon", "steak")}

# Confidences straddling both thresholds, including sitting exactly on them.
TOP_1_CONFIDENCES = [0.10, 0.34, 0.35, 0.36, 0.69, 0.70, 0.71, 0.90]
# Margins straddling margin_threshold, including exactly on it.
MARGINS = [0.0, 0.10, 0.39, 0.40, 0.41, 0.79]
# One hard predicted label, one that isn't; one that's in a confusion pair
# (as the *predicted* member of the pair), one that isn't.
PREDICTED_LABELS = ["steak", "miso_soup", "filet_mignon"]
MODES = ["image", "video"]


def _predictions(top_1_confidence: float, margin: float, predicted_label: str) -> list[Prediction]:
    top_2_confidence = top_1_confidence - margin
    return [
        Prediction(rank=1, class_name=predicted_label, confidence=top_1_confidence),
        Prediction(rank=2, class_name="second_choice", confidence=top_2_confidence),
    ]


def _grid() -> list[tuple[float, float, str, str]]:
    combos = []
    for top_1_confidence, margin, predicted_label, mode in itertools.product(
        TOP_1_CONFIDENCES, MARGINS, PREDICTED_LABELS, MODES
    ):
        # A margin larger than top-1 confidence would require a negative
        # top-2 confidence, which Prediction's schema forbids and which
        # cannot occur with real model outputs anyway.
        if margin > top_1_confidence:
            continue
        combos.append((top_1_confidence, margin, predicted_label, mode))
    return combos


def test_route_decision_matches_build_decision_across_every_branch() -> None:
    combos = _grid()
    assert len(combos) > 100, "grid should exercise a meaningful number of combinations"

    seen_bands = set()
    for top_1_confidence, margin, predicted_label, mode in combos:
        predictions = _predictions(top_1_confidence, margin, predicted_label)

        expected = build_decision(
            mode=mode,
            predictions=predictions,
            policy=POLICY,
            hard_classes=HARD_CLASSES,
            confusion_pairs=CONFUSION_PAIRS,
        ).band

        actual = route_decision(
            top_1_confidence,
            margin,
            predicted_label,
            policy=POLICY,
            hard_classes=HARD_CLASSES,
            confusion_pairs=CONFUSION_PAIRS,
            mode=mode,
        )

        assert actual == expected, (
            f"route_decision != build_decision for "
            f"top_1_confidence={top_1_confidence}, margin={margin}, "
            f"predicted_label={predicted_label!r}, mode={mode!r}: "
            f"route_decision={actual!r}, build_decision={expected!r}"
        )
        seen_bands.add(expected)

    # Every branch (auto_accept, suggest, confirm, review) must actually be
    # exercised by the grid, or the test could pass vacuously.
    assert seen_bands == {"auto_accept", "suggest", "confirm", "review"}

    print(f"\nparity grid covered {len(combos)} combinations across bands {sorted(seen_bands)}")


def test_route_decision_signature_has_no_actual_label_parameter() -> None:
    forbidden_names = {"actual", "actual_label", "true_label", "label", "top_5_contains_actual"}
    signature = inspect.signature(route_decision)
    param_names = set(signature.parameters)

    leaked = param_names & forbidden_names
    assert not leaked, (
        f"route_decision must not accept an actual-label parameter, found: {leaked}"
    )
