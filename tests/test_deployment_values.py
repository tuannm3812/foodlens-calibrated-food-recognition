"""JSON values the deployment code reads but Python cannot convert as-is.

JSON allows integers of any size, and a file may hold bytes that are not UTF-8.
``float()`` and ``math.isfinite()`` raise ``OverflowError`` past the float
range; ``int`` refuses to convert more than 4,300 digits to or from text,
raising ``ValueError``. Each must become a ``DeployError`` naming the field or
file, which the CLI reports as one ``error:`` line, never a traceback.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from app.deployment import identity, paths
from app.deployment.errors import DeployError

POLICY = {"auto_confidence": 0.7, "suggest_confidence": 0.35, "margin_threshold": 0.05}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.884, 0.884),
        (1, 1.0),
        (10**300, 1e300),
        (10**400, None),
        (-(10**400), None),
        (math.inf, None),
        (math.nan, None),
        (True, None),
        ("0.884", None),
        (None, None),
    ],
)
def test_finite_float_returns_none_instead_of_raising(value, expected) -> None:
    assert paths.finite_float(value) == expected


def test_brief_cuts_a_long_value() -> None:
    shown = paths.brief(10**400)
    assert len(shown) == 80 and shown.endswith("...")
    assert paths.brief([]) == "[]"


def test_brief_shows_an_integer_too_long_for_text() -> None:
    # Past 4,300 digits repr() itself raises ValueError.
    assert paths.brief(10**5000) == "<int too large to show>"


def test_read_json_refuses_an_integer_too_long_to_parse(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text('{"temperature": ' + "9" * 5000 + "}\n", encoding="utf-8")
    with pytest.raises(DeployError, match="calibration.json is not valid JSON: Exceeds"):
        paths.read_json(path)


def test_read_json_refuses_bytes_that_are_not_utf8(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_bytes(b'{"temperature": "\xff"}\n')
    with pytest.raises(DeployError, match="calibration.json is not valid JSON"):
        paths.read_json(path)


@pytest.mark.parametrize("value", [10**400, -(10**400)])
def test_a_policy_threshold_past_the_float_range_is_refused(value) -> None:
    with pytest.raises(DeployError, match="auto_confidence must be a finite value in") as info:
        identity.runtime_policy({**POLICY, "auto_confidence": value}, Path("policy.json"))
    assert len(str(info.value)) < 200


@pytest.mark.parametrize("value", [10**400, math.inf, 0, -1])
def test_a_temperature_past_the_float_range_is_refused(value) -> None:
    with pytest.raises(DeployError, match="temperature must be finite and positive") as info:
        identity.validated_temperature({"temperature": value}, Path("calibration.json"))
    assert len(str(info.value)) < 200


def test_an_integer_temperature_in_range_is_read_as_a_float() -> None:
    value = identity.validated_temperature({"temperature": 1}, Path("calibration.json"))
    assert value == 1.0 and isinstance(value, float)
