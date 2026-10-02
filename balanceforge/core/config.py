"""Configuration with ``BALANCEFORGE_*`` environment overrides.

Any field can be overridden from the environment using the
``BALANCEFORGE_<FIELD>`` convention, e.g. ``BALANCEFORGE_SEED=7``.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import Any

ENV_PREFIX = "BALANCEFORGE_"


@dataclass
class BalanceForgeConfig:
    """Global run configuration."""

    seed: int = 42
    """Master seed; every child seed is derived from it deterministically."""

    n_seeds: int = 3
    """Benchmark repeats; must be >= 3 for the mean+/-std reporting contract."""

    holdout_size: float = 0.3
    """Fraction of samples held out untouched for final evaluation."""

    val_size: float = 0.2
    """Fraction of the *train* split used for routing / HPO validation."""

    base_estimator: str = "rf"
    """Tier-0 classifier: ``rf`` | ``et`` | ``lr`` | ``dt``."""

    n_estimators: int = 120
    max_depth: int | None = None
    max_features: str | float = "sqrt"

    flag_name: str = "polish"
    """HPO objective is aligned to this metric (maximised)."""

    hpo_trials: int = 24
    hpo_timeout_sec: float = 0.0
    """0 means "no wall-clock limit"; only the trial budget bounds the search."""

    n_jobs: int = 1
    """Forced to 1: the sandbox's joblib IPC is unreliable on Windows."""

    ab_seeds: int = 2
    """Ablation repeats (fewer than n_seeds, since ablations are not the headline)."""

    ab_datasets: tuple[str, ...] = ()
    """Datasets used for the ablation sweep (empty => all of them)."""

    random_state_budget: int = 24
    """Trials for the lightweight numpy offline-fallback HPO."""

    def validate(self) -> BalanceForgeConfig:
        if self.n_seeds < 3:
            raise ValueError("n_seeds must be >= 3 for mean+/-std reporting")
        if not 0.05 <= self.holdout_size <= 0.5:
            raise ValueError("holdout_size must be in [0.05, 0.5]")
        if not 0.05 <= self.val_size <= 0.5:
            raise ValueError("val_size must be in [0.05, 0.5]")
        if self.holdout_size + self.val_size >= 0.95:
            raise ValueError("holdout_size + val_size must leave >= 5% for train")
        if self.base_estimator not in {"rf", "et", "lr", "dt"}:
            raise ValueError(f"unknown base_estimator {self.base_estimator!r}")
        if self.n_jobs != 1:
            raise ValueError("n_jobs is pinned to 1 on this platform (joblib IPC limitation)")
        if self.hpo_trials < 1:
            raise ValueError("hpo_trials must be >= 1")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce(raw: str, target: Any) -> Any:
    if isinstance(target, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(target, int):
        return int(raw)
    if isinstance(target, float):
        return float(raw)
    if isinstance(target, tuple):
        return tuple(p.strip() for p in raw.split(",") if p.strip())
    return raw


def load_config(overrides: dict[str, Any] | None = None) -> BalanceForgeConfig:
    """Build a config from defaults + environment + explicit overrides.

    Precedence: explicit ``overrides`` > ``BALANCEFORGE_*`` env > defaults.
    Unknown keys raise ``E100`` rather than being silently ignored.
    """
    from balanceforge.core.errors import E100

    known = {f.name: f for f in fields(BalanceForgeConfig)}
    values: dict[str, Any] = {}

    for name, f in known.items():
        env_key = ENV_PREFIX + name.upper()
        if env_key in os.environ:
            values[name] = _coerce(os.environ[env_key], f.default)

    for key, val in (overrides or {}).items():
        if key not in known:
            raise E100(f"unknown config key {key!r}; valid: {sorted(known)}")
        values[key] = val

    return BalanceForgeConfig(**values).validate()
