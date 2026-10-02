"""Error taxonomy for BalanceForge.

E100  input / data contract violations
E200  imbalance handling / sampling violations
E300  model / training violations
E400  evaluation contract violations
"""

from __future__ import annotations

from collections.abc import Iterable


class BalanceForgeError(Exception):
    """Base class carrying a stable numeric code."""

    code = 0

    def __init__(self, message: str = "") -> None:
        self.message = message
        super().__init__(f"[E{self.code}] {message}" if message else f"[E{self.code}]")


class E100(BalanceForgeError):
    """Input / data contract violation."""

    code = 100


class E200(BalanceForgeError):
    """Imbalance handling / sampling violation."""

    code = 200


class E300(BalanceForgeError):
    """Model / training violation."""

    code = 300


class E400(BalanceForgeError):
    """Evaluation contract violation."""

    code = 400


class E500(BalanceForgeError):
    """Pipeline invariant violation (post-condition failure)."""

    code = 500


def require(condition: bool, exc: type[BalanceForgeError], message: str) -> None:
    """Raise ``exc`` when ``condition`` is false."""
    if not condition:
        raise exc(message)


def require_labels_in(arr: Iterable[int], allowed: Iterable[int], name: str = "y") -> None:
    """Validate that every label in ``arr`` appears in ``allowed`` (E100)."""
    allowed_set = {int(v) for v in allowed}
    seen = sorted({int(v) for v in arr})
    bad = [v for v in seen if v not in allowed_set]
    require(not bad, E100, f"{name} contains labels outside {sorted(allowed_set)}: {bad}")
