"""Core package: types, errors, config, interfaces, determinism."""

from __future__ import annotations

from balanceforge.core.config import BalanceForgeConfig, load_config
from balanceforge.core.errors import (
    E100,
    E200,
    E300,
    E400,
    E500,
    BalanceForgeError,
)
from balanceforge.core.seed import set_all
from balanceforge.core.types import (
    Dataset,
    ImbalanceProfile,
    MethodResult,
    SplitData,
)

__all__ = [
    "BalanceForgeConfig",
    "BalanceForgeError",
    "Dataset",
    "E100",
    "E200",
    "E300",
    "E400",
    "E500",
    "ImbalanceProfile",
    "MethodResult",
    "SplitData",
    "load_config",
    "set_all",
]

__author__ = "晨星"
__version__ = "0.1.0"
