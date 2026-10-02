"""Regime routing — Stage 1 of the flagship, expressed as an explicit policy.

The policy maps an :class:`~balanceforge.core.types.ImbalanceProfile` to a
:class:`RegimePlan`: which balancer to run, with which knobs, and whether the
sampler should be *forbidden* outright.

M1 — the negative route
-----------------------
The single most useful thing a router can do is **decline to resample**. In the
``separable`` regime the naive model already has near-zero minority error, so any
resampling injects variance with no signal to recover. FORGE's fourth
experiment (F4) made this precise: for a linearly separable pair,
``b^\\star_{BA} \\equiv 0`` for *every* majority point — **no** bias-variance
term exists, so $B_{\\mathrm{BA}} = B_{\\mathrm{BD}}$ identically. The sampling
variance term is provably zero, not merely small.

That theorem is the routing rule: if the profile says ``separable`` **and** the
validation minority F1 is already at ceiling, the plan is ``no_resampling``.
Otherwise, plan the matching remedy.

Determinism
-----------
Given the same profile the same plan is returned (no randomness anywhere in this
module).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from balanceforge.core.types import ImbalanceProfile

__all__ = ["RegimePlan", "route", "ROUTING_TABLE", "SEPARABLE_CEILING"]

SEPARABLE_CEILING = 0.95
"""Validation minority-F1 above which a ``separable`` profile must not resample."""


@dataclass(frozen=True)
class RegimePlan:
    """What the router decided to do, and why."""

    regime: str
    sampler: str
    target_ratio: float
    allow_resampling: bool
    reason: str
    knobs: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "sampler": self.sampler,
            "target_ratio": self.target_ratio,
            "allow_resampling": self.allow_resampling,
            "reason": self.reason,
            "knobs": dict(self.knobs),
        }


ROUTING_TABLE: dict[str, dict[str, Any]] = {
    "separable": {
        "sampler": "ros",
        "target_ratio": 0.6,
        "knobs": {"synth_budget": 0.25},
        "why": "minority is spatially detached; light expansion only, since the "
        "sampling-variance term is provably ~0 (b*_BA == 0 for separable pairs)",
    },
    "overlap": {
        "sampler": "contamination_gated_smote",
        "target_ratio": 1.0,
        "knobs": {"synth_budget": 0.60},
        "why": "minority sits inside the majority hull; plain SMOTE crosses class "
        "boundaries here, so the contamination gate is the load-bearing part",
    },
    "multimodal": {
        "sampler": "per_mode_expand",
        "target_ratio": 1.0,
        "knobs": {"synth_budget": 0.80},
        "why": "k minority islands; a global SMOTE budget is captured by the "
        "largest mode, so expansion is budgeted per mode",
    },
    "sparse": {
        "sampler": "contamination_gated_smote",
        "target_ratio": 1.0,
        "knobs": {"synth_budget": 1.00, "condense": True},
        "why": "extreme IR with very few minority samples; expand aggressively "
        "and condense the majority so the class ratio can actually be repaired",
    },
}


def route(
    profile: ImbalanceProfile,
    val_minority_f1: float | None = None,
) -> RegimePlan:
    """Pick a :class:`RegimePlan` from a profile (and optional validation F1).

    ``val_minority_f1`` is the *measured* minority F1 on the validation split,
    computed before any resampling. Supplying it enables the M1 negative route;
    omitting it falls back to the geometry-only plan.
    """
    table = ROUTING_TABLE.get(profile.geometric, ROUTING_TABLE["overlap"])

    if (
        profile.geometric == "separable"
        and val_minority_f1 is not None
        and val_minority_f1 >= SEPARABLE_CEILING
    ):
        return RegimePlan(
            regime=profile.geometric,
            sampler="none",
            target_ratio=1.0,
            allow_resampling=False,
            reason=(
                f"separable + val minority F1 {val_minority_f1:.3f} >= "
                f"{SEPARABLE_CEILING} -> M1 negative route: b*_BA == 0 for every "
                "majority point, so resampling can only add variance"
            ),
            knobs={"synth_budget": 0.0},
        )

    return RegimePlan(
        regime=profile.geometric,
        sampler=str(table["sampler"]),
        target_ratio=float(table["target_ratio"]),
        allow_resampling=True,
        reason=str(table["why"]),
        knobs=dict(table["knobs"]),
    )
