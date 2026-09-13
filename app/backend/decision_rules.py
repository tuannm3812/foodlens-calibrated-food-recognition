"""The single decision-band routing rule shared by production and offline recalibration.

`route_decision` is the one place the auto_accept / suggest / confirm / review
band is decided. Its signature carries only what an inference-time caller in
production actually has: the top-1 confidence, the top-1/top-2 margin, the
*predicted* label, the active policy thresholds, the hard-class set and the
confusion-pair set. **There is no actual-label parameter, in any form.** That
is deliberate: a decision rule that could consult ground truth would let a
future edit reintroduce a leak by accident. Structurally excluding the
parameter is what keeps that impossible rather than merely discouraged.

Both `app/backend/decision.py::build_decision` (production, called from the
FastAPI backend on live inference requests) and
`scripts/recalibrate_decision_layer.py::assign_decision_band` (offline
recalibration, called while grid-searching policy thresholds against a
finished run's predictions) call this function for the band decision. That is
what keeps the offline policy *executable*: whatever policy the offline grid
search selects is guaranteed to produce the same bands in production, because
both paths route through identical code rather than through two
hand-maintained implementations that can drift apart. Actual labels remain
useful and are still used elsewhere in recalibration, but only *after*
routing, to score how each band performed -- never to decide which band a row
lands in.
"""

from __future__ import annotations


def route_decision(
    top_1_confidence: float,
    margin: float,
    predicted_label: str,
    *,
    policy: dict[str, float],
    hard_classes: set[str],
    confusion_pairs: set[tuple[str, str]],
    mode: str = "image",
) -> str:
    """Route a single prediction to a decision band.

    Reproduces today's production branch order exactly:

    1. ``mode == "video"`` -> ``"confirm"``
    2. predicted label appears in any confusion pair AND
       ``margin < policy["margin_threshold"]`` -> ``"review"``
    3. predicted label in ``hard_classes`` AND
       ``top_1_confidence < policy["auto_confidence"]`` -> ``"confirm"``
    4. ``top_1_confidence >= policy["auto_confidence"]`` AND
       ``margin >= policy["margin_threshold"]`` AND
       predicted label not in ``hard_classes`` -> ``"auto_accept"``
    5. ``top_1_confidence >= policy["suggest_confidence"]`` -> ``"suggest"``
    6. otherwise -> ``"confirm"``
    """
    if mode == "video":
        return "confirm"

    risky_prediction = any(predicted_label in pair for pair in confusion_pairs)
    if risky_prediction and margin < policy["margin_threshold"]:
        return "review"

    if predicted_label in hard_classes and top_1_confidence < policy["auto_confidence"]:
        return "confirm"

    if (
        top_1_confidence >= policy["auto_confidence"]
        and margin >= policy["margin_threshold"]
        and predicted_label not in hard_classes
    ):
        return "auto_accept"

    if top_1_confidence >= policy["suggest_confidence"]:
        return "suggest"

    return "confirm"
