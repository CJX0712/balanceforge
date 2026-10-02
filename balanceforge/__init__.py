"""BalanceForge — geometry-routed rebalancing for imbalanced and long-tailed learning.

Author: 晨星 (CJX0712)
"""

from balanceforge.core import (
    BalanceForgeConfig,
    BalanceForgeError,
    Dataset,
    ImbalanceProfile,
    load_config,
    set_all,
)
from balanceforge.pipeline import BalancePipeline, run_benchmark, write_report

__all__ = [
    "BalanceForgeConfig",
    "BalanceForgeError",
    "BalancePipeline",
    "Dataset",
    "ImbalanceProfile",
    "load_config",
    "run_benchmark",
    "set_all",
    "write_report",
]

__author__ = "晨星"
__version__ = "0.1.0"
