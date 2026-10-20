"""Transparent testbed for signal-to-execution reinforcement learning."""

from hail_ppo_reproduction.binance import (
    BinanceKlineSeries,
    download_monthly_trades,
    load_kline_archives,
)
from hail_ppo_reproduction.config import ExecutionConfig, SyntheticMarketConfig
from hail_ppo_reproduction.controlled import (
    ControlledDataConfig,
    ControlledDataset,
    HistoricalPriceDay,
    build_controlled_dataset,
    combine_controlled_datasets,
    make_controlled_trading_day,
)
from hail_ppo_reproduction.data import TradingDay, make_synthetic_day
from hail_ppo_reproduction.env import ExecutionEnv
from hail_ppo_reproduction.historical import (
    HistoricalDataConfig,
    HistoricalDataset,
    build_historical_dataset,
)
from hail_ppo_reproduction.microstructure import (
    FEATURE_NAMES,
    MicrostructureFeatureSeries,
    RAW_TRADE_FEATURE_NAMES,
    RawTradeBucketSeries,
    aggregate_raw_trade_archives,
    build_microstructure_features,
    standardize_raw_trade_features,
)
from hail_ppo_reproduction.policies import HailBaselinePolicy

__all__ = [
    "BinanceKlineSeries",
    "ControlledDataConfig",
    "ControlledDataset",
    "ExecutionConfig",
    "ExecutionEnv",
    "HailBaselinePolicy",
    "HistoricalDataConfig",
    "HistoricalDataset",
    "HistoricalPriceDay",
    "FEATURE_NAMES",
    "MicrostructureFeatureSeries",
    "RAW_TRADE_FEATURE_NAMES",
    "RawTradeBucketSeries",
    "SyntheticMarketConfig",
    "TradingDay",
    "build_historical_dataset",
    "build_microstructure_features",
    "aggregate_raw_trade_archives",
    "download_monthly_trades",
    "build_controlled_dataset",
    "combine_controlled_datasets",
    "load_kline_archives",
    "make_controlled_trading_day",
    "standardize_raw_trade_features",
    "make_synthetic_day",
]
