import pytest

from app.backend.decision import build_decision
from app.backend.schemas import Prediction


TEST_POLICY = {
    "auto_confidence": 0.70,
    "suggest_confidence": 0.35,
    "margin_threshold": 0.40,
}


def predictions(top_label: str, top_score: float, second_score: float) -> list[Prediction]:
    return [
        Prediction(rank=1, class_name=top_label, confidence=top_score),
        Prediction(rank=2, class_name="second_choice", confidence=second_score),
    ]


def test_auto_accept_when_confident_not_hard_and_margin_is_large() -> None:
    decision = build_decision(
        mode="image",
        predictions=predictions("miso_soup", 0.91, 0.12),
        policy=TEST_POLICY,
        hard_classes={"steak"},
        confusion_pairs=set(),
    )

    assert decision.band == "auto_accept"
    assert decision.title == "Auto-accept"
    assert decision.top_1_top_2_margin == pytest.approx(0.79)


def test_suggest_when_confident_but_margin_is_small() -> None:
    decision = build_decision(
        mode="image",
        predictions=predictions("ramen", 0.72, 0.42),
        policy=TEST_POLICY,
        hard_classes={"steak"},
        confusion_pairs=set(),
    )

    assert decision.band == "suggest"
    assert decision.title == "Show suggestions"


def test_confirm_when_prediction_is_hard_class_below_auto_threshold() -> None:
    decision = build_decision(
        mode="image",
        predictions=predictions("steak", 0.64, 0.12),
        policy=TEST_POLICY,
        hard_classes={"steak"},
        confusion_pairs=set(),
    )

    assert decision.band == "confirm"
    assert decision.title == "Confirm dish"


def test_review_when_prediction_has_confusion_risk_and_small_margin() -> None:
    decision = build_decision(
        mode="image",
        predictions=predictions("filet_mignon", 0.57, 0.31),
        policy=TEST_POLICY,
        hard_classes=set(),
        confusion_pairs={("steak", "filet_mignon")},
    )

    assert decision.band == "review"
    assert decision.title == "Review prediction"


def test_video_predictions_always_require_confirmation() -> None:
    decision = build_decision(
        mode="video",
        predictions=predictions("sushi", 0.88, 0.05),
        policy=TEST_POLICY,
        hard_classes=set(),
        confusion_pairs=set(),
    )

    assert decision.band == "confirm"
    assert decision.title == "Confirm dish"


def test_empty_hard_class_override_is_honored() -> None:
    decision = build_decision(
        mode="image",
        predictions=predictions("steak", 0.91, 0.12),
        policy=TEST_POLICY,
        hard_classes=set(),
        confusion_pairs=set(),
    )

    assert decision.band == "auto_accept"
