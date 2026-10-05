"""Transparent testbed for reproducing HAIL's optimal-execution study."""

from hail_ppo_reproduction.config import ExecutionConfig, SyntheticMarketConfig
from hail_ppo_reproduction.data import TradingDay, make_synthetic_day
from hail_ppo_reproduction.env import ExecutionEnv
from hail_ppo_reproduction.policies import HailBaselinePolicy

__all__ = [
    "ExecutionConfig",
    "ExecutionEnv",
    "HailBaselinePolicy",
    "SyntheticMarketConfig",
    "TradingDay",
    "make_synthetic_day",
]
