"""The one exception deployment operations raise for a predictable failure."""

from __future__ import annotations


class DeployError(Exception):
    """A predictable deployment failure, reported without a traceback."""
