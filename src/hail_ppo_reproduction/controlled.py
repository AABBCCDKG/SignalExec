"""Controlled-signal episodes on real historical price paths."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable, Iterable

import numpy as np
from numpy.typing import NDArray

from hail_ppo_reproduction.binance import BinanceKlineSeries
from hail_ppo_reproduction.data import TradingDay
from hail_ppo_reproduction.microstructure import MicrostructureFeatureSeries


_DAY_US = 86_400_000_000


@dataclass(frozen=True)
class ControlledDataConfig:
    """Chronological split for the signal-isolation experiment."""

    train_start: date
    validation_start: date
    test_start: date
    test_end: date
    target_quantity: float = 1.0
    validation_end: date | None = None

    def __post_init__(self) -> None:
        validation_end = self.validation_end or self.test_start
        if not (
            self.train_start
            < self.validation_start
            < validation_end
            <= self.test_start
            < self.test_end
        ):
            raise ValueError("split dates must be ordered and non-overlapping")
        if self.target_quantity <= 0:
            raise ValueError("target_quantity must be positive")


@dataclass(frozen=True)
class HistoricalPriceDay:
    """One UTC day of prices without a target or predictive signal."""

    date: date
    prices: NDArray[np.float64]
    realized_returns: NDArray[np.float64]
    return_volatility: float
    extra_features: NDArray[np.float64] | None = None
    symbol: str = "UNKNOWN"

    def __post_init__(self) -> None:
        if len(self.prices) != len(self.realized_returns) + 1:
            raise ValueError("prices must contain one more value than realized returns")
        if not len(self.realized_returns):
            raise ValueError("a price day needs at least one return")
        if np.any(self.prices <= 0) or not np.all(np.isfinite(self.prices)):
            raise ValueError("prices must be finite and positive")
        if not np.all(np.isfinite(self.realized_returns)):
            raise ValueError("returns must be finite")
        if self.return_volatility <= 0:
            raise ValueError("return_volatility must be positive")
        if self.extra_features is not None:
            if self.extra_features.ndim != 2:
                raise ValueError("extra_features must be two-dimensional")
            if len(self.extra_features) != len(self.realized_returns):
                raise ValueError("extra_features must align with decision steps")
            if not np.all(np.isfinite(self.extra_features)):
                raise ValueError("extra_features must be finite")


@dataclass(frozen=True)
class ControlledDataset:
    train: tuple[HistoricalPriceDay, ...]
    validation: tuple[HistoricalPriceDay, ...]
    test: tuple[HistoricalPriceDay, ...]
    bars_per_day: int
    feature_names: tuple[str, ...] = ()

    @property
    def extra_feature_count(self) -> int:
        return len(self.feature_names)

    def split(self, name: str) -> tuple[HistoricalPriceDay, ...]:
        if name not in ("train", "validation", "test"):
            raise ValueError(f"unknown split: {name}")
        return getattr(self, name)


def combine_controlled_datasets(
    datasets: Iterable[ControlledDataset],
) -> ControlledDataset:
    """Pool multiple symbols after symbol-local feature normalization."""

    dataset_tuple = tuple(datasets)
    if not dataset_tuple:
        raise ValueError("at least one controlled dataset is required")
    bars_per_day = dataset_tuple[0].bars_per_day
    feature_names = dataset_tuple[0].feature_names
    if any(dataset.bars_per_day != bars_per_day for dataset in dataset_tuple):
        raise ValueError("all symbols must use the same decision interval")
    if any(dataset.feature_names != feature_names for dataset in dataset_tuple):
        raise ValueError("all symbols must expose the same feature schema")

    def combine(name: str) -> tuple[HistoricalPriceDay, ...]:
        days = [day for dataset in dataset_tuple for day in dataset.split(name)]
        keys = [(day.symbol, day.date) for day in days]
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate symbol-day in {name} split")
        return tuple(sorted(days, key=lambda day: (day.date, day.symbol)))

    return ControlledDataset(
        train=combine("train"),
        validation=combine("validation"),
        test=combine("test"),
        bars_per_day=bars_per_day,
        feature_names=feature_names,
    )


def build_controlled_dataset(
    series: BinanceKlineSeries,
    config: ControlledDataConfig,
    microstructure: MicrostructureFeatureSeries | None = None,
) -> ControlledDataset:
    """Extract complete UTC price days without fitting any forecast model."""

    differences = np.diff(series.open_time_us)
    if not len(differences):
        raise ValueError("at least two klines are required")
    interval_us = int(np.median(differences))
    if interval_us <= 0 or _DAY_US % interval_us:
        raise ValueError("kline interval must divide one UTC day exactly")
    bars_per_day = _DAY_US // interval_us
    boundaries = {
        "train_start": _date_to_us(config.train_start),
        "validation_start": _date_to_us(config.validation_start),
        "validation_end": _date_to_us(config.validation_end or config.test_start),
        "test_start": _date_to_us(config.test_start),
        "test_end": _date_to_us(config.test_end),
    }
    splits: dict[str, list[HistoricalPriceDay]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    midnight_indices = np.flatnonzero(series.open_time_us % _DAY_US == 0)
    for raw_index in midnight_indices:
        index = int(raw_index)
        if index < bars_per_day or index + bars_per_day >= len(series.open):
            continue
        timestamps = series.open_time_us[index : index + bars_per_day + 1]
        if not np.all(np.diff(timestamps) == interval_us):
            continue
        start = int(timestamps[0])
        end = int(timestamps[-1])
        split = _split_name(start, end, boundaries)
        if split is None:
            continue
        prices = series.open[index : index + bars_per_day + 1].copy()
        realized_returns = np.log(prices[1:] / prices[:-1])
        extra_features = (
            None
            if microstructure is None
            else microstructure.features_for(
                series.open_time_us[index : index + bars_per_day]
            )
        )
        past_returns = np.log(
            series.open[index - bars_per_day + 1 : index + 1]
            / series.open[index - bars_per_day : index]
        )
        splits[split].append(
            HistoricalPriceDay(
                date=_us_to_date(start),
                prices=prices,
                realized_returns=realized_returns,
                return_volatility=max(float(np.std(past_returns, ddof=1)), 1e-8),
                extra_features=extra_features,
                symbol=series.symbol,
            )
        )
    if any(not days for days in splits.values()):
        counts = {name: len(days) for name, days in splits.items()}
        raise ValueError(f"each chronological split needs complete days; got {counts}")
    return ControlledDataset(
        train=tuple(splits["train"]),
        validation=tuple(splits["validation"]),
        test=tuple(splits["test"]),
        bars_per_day=int(bars_per_day),
        feature_names=(
            () if microstructure is None else microstructure.feature_names
        ),
    )


def make_controlled_trading_day(
    price_day: HistoricalPriceDay,
    *,
    target_direction: int,
    target_quantity: float,
    signal_accuracy: float | None,
    signal_seed: int,
) -> TradingDay:
    """Attach either no signal or a forecast with controlled return correlation.

    ``signal_accuracy=None`` is the signal-off ablation. Numeric accuracies use
    the realized next return deliberately as a controlled oracle. The result is
    an experiment input, not a point-in-time tradable forecast.
    """

    if target_direction not in (-1, 1):
        raise ValueError("target_direction must be -1 or 1")
    if target_quantity <= 0:
        raise ValueError("target_quantity must be positive")
    if signal_accuracy is None:
        signals = np.zeros_like(price_day.realized_returns)
    else:
        if not -1.0 <= signal_accuracy <= 1.0:
            raise ValueError("signal_accuracy must be between -1 and 1")
        rng = np.random.default_rng(signal_seed)
        noise = rng.normal(
            0.0,
            price_day.return_volatility,
            size=len(price_day.realized_returns),
        )
        signals = (
            signal_accuracy * price_day.realized_returns
            + np.sqrt(max(0.0, 1.0 - signal_accuracy * signal_accuracy)) * noise
        )
    return TradingDay(
        prices=price_day.prices.copy(),
        signals=signals.astype(np.float64),
        target_position=target_direction * target_quantity,
        return_volatility=price_day.return_volatility,
        extra_features=(
            None
            if price_day.extra_features is None
            else price_day.extra_features.copy()
        ),
    )


def make_controlled_day_factory(
    days: Iterable[HistoricalPriceDay],
    *,
    signal_accuracy: float,
    target_quantity: float,
) -> Callable[[np.random.Generator], TradingDay]:
    """Sample a price day, mirrored target direction, and fresh forecast noise."""

    day_tuple = tuple(days)
    if not day_tuple:
        raise ValueError("controlled day factory needs at least one day")

    def factory(rng: np.random.Generator) -> TradingDay:
        price_day = day_tuple[int(rng.integers(0, len(day_tuple)))]
        target_direction = int(rng.choice(np.array([-1, 1])))
        signal_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        return make_controlled_trading_day(
            price_day,
            target_direction=target_direction,
            target_quantity=target_quantity,
            signal_accuracy=signal_accuracy,
            signal_seed=signal_seed,
        )

    return factory


def evaluation_signal_seed(
    *,
    base_seed: int,
    day: date,
    replica: int,
    symbol: str = "",
) -> int:
    """Create one noise seed shared by every policy and every rho value."""

    symbol_code = int.from_bytes(symbol.encode("utf-8"), "little") % (2**32)
    seed_sequence = np.random.SeedSequence(
        [base_seed, day.toordinal(), replica, symbol_code]
    )
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def signal_diagnostics(
    days: Iterable[HistoricalPriceDay],
    *,
    accuracies: tuple[float, ...],
    replicas: int,
    base_seed: int,
) -> list[dict]:
    day_tuple = tuple(days)
    rows = [
        {
            "signal_mode": "off",
            "signal_accuracy": None,
            "replicas": 1,
            "empirical_correlation": 0.0,
            "direction_accuracy": 0.0,
        }
    ]
    for accuracy in accuracies:
        signals = []
        returns = []
        for replica in range(replicas):
            for price_day in day_tuple:
                trading_day = make_controlled_trading_day(
                    price_day,
                    target_direction=1,
                    target_quantity=1.0,
                    signal_accuracy=accuracy,
                    signal_seed=evaluation_signal_seed(
                        base_seed=base_seed,
                        day=price_day.date,
                        replica=replica,
                        symbol=price_day.symbol,
                    ),
                )
                signals.append(trading_day.signals)
                returns.append(price_day.realized_returns)
        signal_values = np.concatenate(signals)
        return_values = np.concatenate(returns)
        rows.append(
            {
                "signal_mode": "controlled",
                "signal_accuracy": accuracy,
                "replicas": replicas,
                "empirical_correlation": float(
                    np.corrcoef(signal_values, return_values)[0, 1]
                ),
                "direction_accuracy": float(
                    np.mean(np.sign(signal_values) == np.sign(return_values))
                ),
            }
        )
    return rows


def _split_name(start: int, end: int, boundaries: dict[str, int]) -> str | None:
    if boundaries["train_start"] <= start and end < boundaries["validation_start"]:
        return "train"
    if boundaries["validation_start"] <= start and end < boundaries["validation_end"]:
        return "validation"
    if boundaries["test_start"] <= start and end < boundaries["test_end"]:
        return "test"
    return None


def _date_to_us(value: date) -> int:
    timestamp = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    return int(timestamp.timestamp() * 1_000_000)


def _us_to_date(value: int) -> date:
    return datetime.fromtimestamp(value / 1_000_000, tz=timezone.utc).date()
