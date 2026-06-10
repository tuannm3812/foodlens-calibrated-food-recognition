"""Decision-band helpers for FoodLens predictions."""

from __future__ import annotations

from typing import Optional

from .schemas import Decision, Prediction


DEFAULT_HARD_CLASSES = {
    "chocolate_mousse",
    "steak",
    "pork_chop",
    "bread_pudding",
    "tuna_tartare",
}

DEFAULT_POLICY = {
    "auto_confidence": 0.70,
    "suggest_confidence": 0.35,
    "margin_threshold": 0.40,
}


def build_decision(
    mode: str,
    predictions: list[Prediction],
    policy: Optional[dict[str, float]] = None,
    hard_classes: Optional[set[str]] = None,
    confusion_pairs: Optional[set[tuple[str, str]]] = None,
) -> Decision:
    """Build a FoodLens decision output from ranked predictions."""
    active_policy = DEFAULT_POLICY if policy is None else policy
    active_hard_classes = DEFAULT_HARD_CLASSES if hard_classes is None else hard_classes
    active_confusion_pairs = set() if confusion_pairs is None else confusion_pairs
    top_1 = predictions[0]
    top_2 = predictions[1]
    margin = top_1.confidence - top_2.confidence
    predicted_label = top_1.class_name
    risky_prediction = any(predicted_label in pair for pair in active_confusion_pairs)

    if mode == "video":
        return Decision(
            band="confirm",
            title="Confirm dish",
            recommended_action=(
                "Ask the user to confirm because sampled frames are not fully aligned."
            ),
            top_1_top_2_margin=margin,
        )

    if risky_prediction and margin < active_policy["margin_threshold"]:
        return Decision(
            band="review",
            title="Review prediction",
            recommended_action="Flag for review because this matches a known confusion risk.",
            top_1_top_2_margin=margin,
        )

    if (
        predicted_label in active_hard_classes
        and top_1.confidence < active_policy["auto_confidence"]
    ):
        return Decision(
            band="confirm",
            title="Confirm dish",
            recommended_action="Ask the user to confirm because this is a hard predicted class.",
            top_1_top_2_margin=margin,
        )

    if (
        top_1.confidence >= active_policy["auto_confidence"]
        and margin >= active_policy["margin_threshold"]
        and predicted_label not in active_hard_classes
    ):
        return Decision(
            band="auto_accept",
            title="Auto-accept",
            recommended_action="Accept the top prediction automatically.",
            top_1_top_2_margin=margin,
        )

    if top_1.confidence >= active_policy["suggest_confidence"]:
        return Decision(
            band="suggest",
            title="Show suggestions",
            recommended_action="Show ranked suggestions for user selection.",
            top_1_top_2_margin=margin,
        )

    return Decision(
        band="confirm",
        title="Confirm dish",
        recommended_action="Ask the user to confirm before applying a label.",
        top_1_top_2_margin=margin,
    )
