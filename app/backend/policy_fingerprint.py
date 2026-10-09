"""Order-insensitive fingerprint of a decision layer.

The fingerprint is how a deploy step, the artifact files and a running API
process are compared: two decision layers route predictions identically exactly
when their fingerprints match. Kept dependency-free so the deploy script can
use it without loading the model.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

POLICY_KEYS = ("auto_confidence", "suggest_confidence", "margin_threshold")


def _pair(pair: Any) -> list[str]:
    """Normalise an ``{actual, predicted}`` record or a 2-sequence to a list."""
    if isinstance(pair, Mapping):
        return [str(pair["actual"]), str(pair["predicted"])]
    actual, predicted = pair[0], pair[1]
    return [str(actual), str(predicted)]


def decision_layer_fingerprint(
    policy: Mapping[str, Any],
    hard_classes: Iterable[str],
    confusion_pairs: Iterable[Any],
) -> str:
    """Return the SHA-256 of the decision layer's canonical JSON.

    Args:
        policy: Mapping holding the three thresholds; extra keys are ignored.
        hard_classes: Hard-class names, in any order.
        confusion_pairs: ``(actual, predicted)`` tuples or ``{"actual",
            "predicted"}`` records, in any order.

    Returns:
        A hex digest that is independent of class and pair order (and of
        duplicates, which the runtime holds as sets) and changes when any
        threshold changes.
    """
    canonical = {
        "policy": {key: float(policy[key]) for key in POLICY_KEYS},
        "hard_classes": sorted({str(name) for name in hard_classes}),
        "confusion_pairs": sorted({tuple(_pair(pair)) for pair in confusion_pairs}),
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
