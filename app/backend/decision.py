"""Decision-band helpers for FoodLens predictions."""

from __future__ import annotations

from .decision_rules import route_decision
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
    policy: dict[str, float] | None = None,
    hard_classes: set[str] | None = None,
    confusion_pairs: set[tuple[str, str]] | None = None,
) -> Decision:
    """Build a FoodLens decision output from ranked predictions."""
    active_policy = DEFAULT_POLICY if policy is None else policy
    active_hard_classes = DEFAULT_HARD_CLASSES if hard_classes is None else hard_classes
    active_confusion_pairs = set() if confusion_pairs is None else confusion_pairs
    top_1 = predictions[0]
    top_2 = predictions[1]
    margin = top_1.confidence - top_2.confidence
    predicted_label = top_1.class_name

    band = route_decision(
        top_1.confidence,
        margin,
        predicted_label,
        policy=active_policy,
        hard_classes=active_hard_classes,
        confusion_pairs=active_confusion_pairs,
        mode=mode,
    )

    if mode == "video":
        return Decision(
            band=band,
            title="Confirm dish",
            recommended_action=(
                "Ask the user to confirm because sampled frames are not fully aligned."
            ),
            top_1_top_2_margin=margin,
        )

    if band == "review":
        return Decision(
            band=band,
            title="Review prediction",
            recommended_action="Flag for review because this matches a known confusion risk.",
            top_1_top_2_margin=margin,
        )

    if (
        band == "confirm"
        and predicted_label in active_hard_classes
        and top_1.confidence < active_policy["auto_confidence"]
    ):
        return Decision(
            band=band,
            title="Confirm dish",
            recommended_action="Ask the user to confirm because this is a hard predicted class.",
            top_1_top_2_margin=margin,
        )

    if band == "auto_accept":
        return Decision(
            band=band,
            title="Auto-accept",
            recommended_action="Accept the top prediction automatically.",
            top_1_top_2_margin=margin,
        )

    if band == "suggest":
        return Decision(
            band=band,
            title="Show suggestions",
            recommended_action="Show ranked suggestions for user selection.",
            top_1_top_2_margin=margin,
        )

    return Decision(
        band=band,
        title="Confirm dish",
        recommended_action="Ask the user to confirm before applying a label.",
        top_1_top_2_margin=margin,
    )
