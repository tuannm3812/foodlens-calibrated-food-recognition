"""Check whether two decision-layer runs used a compatible derivation.

Each run of `scripts/recalibrate_decision_layer.py` writes
`derivation_provenance.json`. Two runs are **compatible** -- fit for a
controlled comparison -- when every recorded field matches except the
model-specific `path` and `sha256` of their prediction files and derivation
sources. Those differ between models by construction; they are recorded so
each result traces to its inputs, not so the two files compare equal (see
docs/9_agent_log.md, the 2026-09-27 Codex response on methodology closure).

Schema version 3 adds each prediction file's producer `evidence` (see
`scripts/prediction_evidence.py`). What identifies the model -- the sidecar's
own path and hash, and inside its record the predictions, checkpoint,
architecture, class names, temperature, self-check, environment and creation
time -- is model-specific too: two models' runs differ there by construction.
The evidence schema, the `preprocessing` and the `producer` (the rescorer's
path and hash) are compatibility fields: a controlled comparison needs both
models pushed through the same re-scoring code and the same eval transform.
Evidence present on one side only is incompatible.

The code identities are compatibility fields, never expected differences:
`generator.sha256` (the recalibration script) and `routing.module_sha256`
(the routing rule). `generator.sha256` must also be *present* in both
records -- two records that both predate it (schema version 1) cannot show
they came from the same derivation code, so they are reported incompatible
rather than vacuously equal (see docs/9_agent_log.md, the 2026-10-06 Codex
review, finding 3).

Usage:
    python scripts/compare_provenance.py RUN_A/derivation_provenance.json \
        RUN_B/derivation_provenance.json

Exits 0 and prints `compatible`, or exits 1 and prints `incompatible` with
every differing field that is not a model-specific path or hash.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Sections whose `path`/`sha256` name a model-specific input file.
MODEL_SPECIFIC_SECTIONS = (
    ("predictions", "fit"),
    ("predictions", "eval"),
    ("hard_classes", "source"),
    ("confusion_pairs", "source"),
)
MODEL_SPECIFIC_KEYS = ("path", "sha256")

# Under predictions.<fit|eval>.evidence: the sidecar's own path and hash, and the
# top-level record keys that describe the model rather than the procedure.
EVIDENCE_SECTIONS = (("predictions", "fit", "evidence"), ("predictions", "eval", "evidence"))
MODEL_SPECIFIC_EVIDENCE_RECORD_KEYS = (
    "created_at",
    "predictions",
    "checkpoint",
    "architecture",
    "class_names",
    "temperature",
    "self_check",
    "environment",
)

# Code-identity fields that must be recorded on both sides to compare at all.
REQUIRED_FIELDS = (("generator", "sha256"),)


def _flatten(value: object, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], object]:
    if isinstance(value, dict):
        flat: dict[tuple[str, ...], object] = {}
        for key, child in value.items():
            flat.update(_flatten(child, (*prefix, str(key))))
        return flat
    return {prefix: value}


def is_model_specific(field: tuple[str, ...]) -> bool:
    """True for the fields expected to differ between models: input paths and
    hashes, and the model-describing part of the producer evidence."""
    if len(field) == 3 and field[:2] in MODEL_SPECIFIC_SECTIONS:
        return field[2] in MODEL_SPECIFIC_KEYS
    if len(field) >= 4 and field[:3] in EVIDENCE_SECTIONS:
        if len(field) == 4:
            return field[3] in MODEL_SPECIFIC_KEYS
        return field[3] == "record" and field[4] in MODEL_SPECIFIC_EVIDENCE_RECORD_KEYS
    return False


def compare_provenance(
    first: dict[str, object], second: dict[str, object]
) -> tuple[list[str], list[str]]:
    """Compare two provenance records.

    Returns:
        `(incompatible_fields, model_specific_fields)` -- dotted names of the
        fields that differ (or exist on one side only). The runs are
        compatible exactly when `incompatible_fields` is empty;
        `model_specific_fields` lists the expected path/hash differences.
        A `REQUIRED_FIELDS` entry missing from either record is incompatible
        even when it is missing from both.
    """
    flat_first = _flatten(first)
    flat_second = _flatten(second)
    incompatible: list[str] = [
        ".".join(field)
        for field in REQUIRED_FIELDS
        if field not in flat_first and field not in flat_second
    ]
    model_specific: list[str] = []
    for field in sorted(set(flat_first) | set(flat_second)):
        if flat_first.get(field, _MISSING) == flat_second.get(field, _MISSING):
            continue
        name = ".".join(field)
        if is_model_specific(field) and field in flat_first and field in flat_second:
            model_specific.append(name)
        else:
            incompatible.append(name)
    return incompatible, model_specific


_MISSING = object()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("first", type=Path, help="First derivation_provenance.json")
    parser.add_argument("second", type=Path, help="Second derivation_provenance.json")
    args = parser.parse_args(argv)

    records = []
    for path in (args.first, args.second):
        try:
            records.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: cannot read provenance file {path}: {exc}", file=sys.stderr)
            return 2

    incompatible, model_specific = compare_provenance(records[0], records[1])
    if model_specific:
        print("expected model-specific differences: " + ", ".join(model_specific))
    if incompatible:
        print("incompatible: " + ", ".join(incompatible))
        return 1
    print("compatible")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
