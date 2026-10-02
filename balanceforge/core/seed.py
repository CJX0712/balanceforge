"""Global determinism: the single seed entry point for BalanceForge.

Every estimator, sampler, splitter and metric path in BalanceForge must be seeded
from :func:`set_all`. Calling it with the same integer must reproduce bit-identical
benchmark numbers across processes.
"""

from __future__ import annotations

import contextlib
import os
import random
from typing import Any

import numpy as np

_LAST_SEED: int | None = None


def set_all(seed: int) -> int:
    """Seed every global RNG BalanceForge can reach.

    Parameters
    ----------
    seed
        Non-negative integer seed.

    Returns
    -------
    The seed, for call-site convenience.
    """
    global _LAST_SEED
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # scikit-learn / imbalanced-learn expose their own global-free RNGs, but a
    # few legacy helpers read these module-level generators directly.
    for mod_name in ("sklearn", "imblearn"):
        mod = _try_import(mod_name)
        if mod is None:
            continue
        for attr in ("set_random_state",):
            fn = getattr(mod, attr, None)
            if callable(fn):
                # Defensive: these helpers are undocumented and their signature
                # has drifted across releases. A failure here must not prevent
                # the rest of the seeding from happening.
                with contextlib.suppress(Exception):  # pragma: no cover
                    fn(seed)

    _LAST_SEED = seed
    return seed


def last_seed() -> int | None:
    """Return the most recent seed passed to :func:`set_all` (or ``None``)."""
    return _LAST_SEED


def rng(seed: int | None = None) -> np.random.Generator:
    """Return a fresh :class:`numpy.random.Generator` (never the global singleton)."""
    if seed is None:
        seed = _LAST_SEED if _LAST_SEED is not None else 0
    return np.random.default_rng(int(seed))


def derive_seed(base: int, *tags: Any) -> int:
    """Derive a stable child seed from ``base`` and arbitrary tags.

    Deterministic across processes (unlike ``hash()``) so sampler sub-seeds are
    reproducible while remaining decorrelated across call sites.
    """
    h = 1469598103934665603  # FNV-1a offset basis, 64-bit
    for tag in tags:
        for ch in str(tag).encode("utf-8"):
            h ^= ch
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    h ^= int(base) & 0xFFFFFFFFFFFFFFFF
    h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return int(h % (2**31 - 1))


def _try_import(name: str) -> Any:
    try:
        __import__(name)
        import sys

        return sys.modules[name]
    except Exception:  # pragma: no cover - optional backends
        return None
