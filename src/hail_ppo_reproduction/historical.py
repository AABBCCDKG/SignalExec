"""Point-in-time historical episodes built from Binance spot klines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable, Iterable

import numpy as np
from numpy.typing import NDArray

from hail_ppo_reproduction.binance import BinanceKlineSeries
from hail_ppo_reproduction.data import TradingDay


_DAY_US = 86_400_000_000


@dataclass(frozen=True)
class HistoricalDataConfig:
    """Chronological split and signal-model assumptions."""

    train_start: date
    validation_start: date
    test_start: date
    test_end: date
    target_quantity: float = 1.0
    ridge_alpha: float = 1.0

    def __post_init__(self) -> None:
        if not self.train_start < self.validation_start < self.test_start < self.test_end:
            raise ValueError("split dates must be strictly increasing")
        if self.target_quantity <= 0:
            raise ValueError("target_quantity must be positive")
        if self.ridge_alpha < 0:
            raise ValueError("ridge_alpha cannot be negative")


@dataclass(frozen=True)
class RidgeModel:
    """Small dependency-free ridge regressor fit only on the training window."""

    feature_mean: NDArray[np.float64]
    feature_scale: NDArray[np.float64]
    target_mean: float
    coefficients: NDArray[np.float64]
    alpha: float

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        matrix = np.asarray(features, dtype=np.float64)
        return self.target_mean + ((matrix - self.feature_mean) / self.feature_scale) @ self.coefficients


@dataclass(frozen=True)
class HistoricalDay:
    """A dated episode plus realized returns kept outside the agent observation."""

    date: date
    trading_day: TradingDay
    realized_returns: NDArray[np.float64]
    target_return: float


@dataclass(frozen=True)
class HistoricalDataset:
    """Chronological train/validation/test episodes and fitted signal models."""

    train: tuple[HistoricalDay, ...]
    validation: tuple[HistoricalDay, ...]
    test: tuple[HistoricalDay, ...]
    short_model: RidgeModel
    long_model: RidgeModel
    bars_per_day: int

    def diagnostics(
        self,
        splits: tuple[str, ...] = ("train", "validation", "test"),
    ) -> dict[str, dict[str, float | int | str]]:
        available = {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }
        unknown = set(splits) - set(available)
        if unknown:
            raise ValueError(f"unknown diagnostic splits: {sorted(unknown)}")
        return {
            name: _split_diagnostics(available[name])
            for name in splits
        }


def fit_ridge(
    features: NDArray[np.float64],
    targets: NDArray[np.float64],
    *,
    alpha: float,
) -> RidgeModel:
    """Fit ridge regression with train-only standardization and intercept."""

    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if matrix.ndim != 2 or target.ndim != 1 or len(matrix) != len(target):
        raise ValueError("ridge features must be 2-D and align with 1-D targets")
    if len(matrix) < 2 or not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(target)):
        raise ValueError("ridge training data must contain at least two finite rows")
    feature_mean = matrix.mean(axis=0)
    feature_scale = matrix.std(axis=0)
    feature_scale = np.where(feature_scale > 1e-12, feature_scale, 1.0)
    target_mean = float(target.mean())
    standardized = (matrix - feature_mean) / feature_scale
    centered_target = target - target_mean
    if alpha == 0:
        coefficients = np.linalg.lstsq(standardized, centered_target, rcond=None)[0]
    else:
        gram = standardized.T @ standardized
        penalty = alpha * np.eye(standardized.shape[1], dtype=np.float64)
        coefficients = np.linalg.solve(
            gram + penalty,
            standardized.T @ centered_target,
        )
    return RidgeModel(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        coefficients=coefficients,
        alpha=alpha,
    )


def build_historical_dataset(
    series: BinanceKlineSeries,
    config: HistoricalDataConfig,
) -> HistoricalDataset:
    """Fit causal signals on train dates and construct strict chronological days.

    At decision time ``t`` the short-horizon features use bars ending before the
    execution bar opens. The action executes at that bar's open and earns the
    next open-to-open return. A separate daily ridge forecast, using only data
    before midnight, sets the target direction for the full UTC day.
    """

    interval_us = _regular_interval(series.open_time_us)
    if _DAY_US % interval_us:
        raise ValueError("kline interval must divide one UTC day exactly")
    bars_per_day = _DAY_US // interval_us
    if bars_per_day < 2:
        raise ValueError("historical episodes require at least two bars per day")

    boundaries = {
        "train_start": _date_to_us(config.train_start),
        "validation_start": _date_to_us(config.validation_start),
        "test_start": _date_to_us(config.test_start),
        "test_end": _date_to_us(config.test_end),
    }
    close_returns = np.empty(len(series.close), dtype=np.float64)
    close_returns[0] = np.nan
    close_returns[1:] = np.log(series.close[1:] / series.close[:-1])
    imbalance = np.divide(
        2.0 * series.taker_buy_volume,
        series.volume,
        out=np.ones_like(series.volume),
        where=series.volume > 0,
    ) - 1.0
    log_volume = np.log1p(series.volume)

    train_short_indices = np.flatnonzero(
        (series.open_time_us >= boundaries["train_start"])
        & (np.roll(series.open_time_us, -1) < boundaries["validation_start"])
    )
    train_short_indices = train_short_indices[
        (train_short_indices >= 12) & (train_short_indices < len(series.open) - 1)
    ]
    short_features = np.stack(
        [
            _short_features(index, close_returns, imbalance, log_volume)
            for index in train_short_indices
        ]
    )
    short_targets = np.log(
        series.open[train_short_indices + 1] / series.open[train_short_indices]
    )
    short_model = fit_ridge(short_features, short_targets, alpha=config.ridge_alpha)

    midnight_indices = np.flatnonzero(series.open_time_us % _DAY_US == 0)
    eligible_midnights = [
        int(index)
        for index in midnight_indices
        if index >= 7 * bars_per_day
        and index + bars_per_day < len(series.open)
        and np.all(
            np.diff(series.open_time_us[index : index + bars_per_day + 1])
            == interval_us
        )
    ]
    train_day_indices = [
        index
        for index in eligible_midnights
        if boundaries["train_start"] <= series.open_time_us[index]
        and series.open_time_us[index + bars_per_day] < boundaries["validation_start"]
    ]
    if len(train_day_indices) < 2:
        raise ValueError("not enough complete training days for the long-horizon model")
    long_features = np.stack(
        [
            _long_features(index, bars_per_day, series, close_returns, imbalance, log_volume)
            for index in train_day_indices
        ]
    )
    long_targets = np.array(
        [
            np.log(series.open[index + bars_per_day] / series.open[index])
            for index in train_day_indices
        ],
        dtype=np.float64,
    )
    long_model = fit_ridge(long_features, long_targets, alpha=config.ridge_alpha)

    splits: dict[str, list[HistoricalDay]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for index in eligible_midnights:
        start = int(series.open_time_us[index])
        end = int(series.open_time_us[index + bars_per_day])
        split = _split_name(start, end, boundaries)
        if split is None:
            continue
        feature_rows = np.stack(
            [
                _short_features(step, close_returns, imbalance, log_volume)
                for step in range(index, index + bars_per_day)
            ]
        )
        signals = short_model.predict(feature_rows)
        target_score = float(
            long_model.predict(
                _long_features(
                    index,
                    bars_per_day,
                    series,
                    close_returns,
                    imbalance,
                    log_volume,
                )[None, :]
            )[0]
        )
        target_direction = 1.0 if target_score >= 0 else -1.0
        prices = series.open[index : index + bars_per_day + 1].copy()
        realized_returns = np.log(prices[1:] / prices[:-1])
        past_returns = np.log(
            series.open[index - bars_per_day + 1 : index + 1]
            / series.open[index - bars_per_day : index]
        )
        volatility = max(float(np.std(past_returns, ddof=1)), 1e-8)
        trading_day = TradingDay(
            prices=prices,
            signals=signals.astype(np.float64),
            target_position=target_direction * config.target_quantity,
            return_volatility=volatility,
        )
        splits[split].append(
            HistoricalDay(
                date=_us_to_date(start),
                trading_day=trading_day,
                realized_returns=realized_returns,
                target_return=float(realized_returns.sum()),
            )
        )

    if any(not values for values in splits.values()):
        counts = {name: len(values) for name, values in splits.items()}
        raise ValueError(f"each chronological split needs complete days; got {counts}")
    return HistoricalDataset(
        train=tuple(splits["train"]),
        validation=tuple(splits["validation"]),
        test=tuple(splits["test"]),
        short_model=short_model,
        long_model=long_model,
        bars_per_day=int(bars_per_day),
    )


def make_historical_day_factory(
    days: Iterable[HistoricalDay],
) -> Callable[[np.random.Generator], TradingDay]:
    day_tuple = tuple(days)
    if not day_tuple:
        raise ValueError("historical day factory needs at least one day")

    def factory(rng: np.random.Generator) -> TradingDay:
        return day_tuple[int(rng.integers(0, len(day_tuple)))].trading_day

    return factory


def _short_features(
    index: int,
    close_returns: NDArray[np.float64],
    imbalance: NDArray[np.float64],
    log_volume: NDArray[np.float64],
) -> NDArray[np.float64]:
    if index < 12:
        raise ValueError("short features require twelve prior bars")
    recent_returns = close_returns[index - 12 : index]
    return np.array(
        [
            close_returns[index - 1],
            recent_returns[-3:].sum(),
            recent_returns.sum(),
            recent_returns.std(ddof=1),
            imbalance[index - 1],
            log_volume[index - 1] - log_volume[index - 12 : index].mean(),
        ],
        dtype=np.float64,
    )


def _long_features(
    index: int,
    bars_per_day: int,
    series: BinanceKlineSeries,
    close_returns: NDArray[np.float64],
    imbalance: NDArray[np.float64],
    log_volume: NDArray[np.float64],
) -> NDArray[np.float64]:
    prior_day = slice(index - bars_per_day, index)
    prior_week = slice(index - 7 * bars_per_day, index)
    return np.array(
        [
            np.log(series.open[index] / series.open[index - bars_per_day]),
            np.log(series.open[index] / series.open[index - 3 * bars_per_day]),
            np.log(series.open[index] / series.open[index - 7 * bars_per_day]),
            np.nanstd(close_returns[prior_day], ddof=1),
            imbalance[prior_day].mean(),
            log_volume[prior_day].mean() - log_volume[prior_week].mean(),
        ],
        dtype=np.float64,
    )


def _regular_interval(timestamps: NDArray[np.int64]) -> int:
    differences = np.diff(timestamps)
    if not len(differences):
        raise ValueError("at least two klines are required")
    interval = int(np.median(differences))
    if interval <= 0:
        raise ValueError("kline interval must be positive")
    return interval


def _split_name(start: int, end: int, boundaries: dict[str, int]) -> str | None:
    if boundaries["train_start"] <= start and end < boundaries["validation_start"]:
        return "train"
    if boundaries["validation_start"] <= start and end < boundaries["test_start"]:
        return "validation"
    if boundaries["test_start"] <= start and end < boundaries["test_end"]:
        return "test"
    return None


def _split_diagnostics(days: tuple[HistoricalDay, ...]) -> dict[str, float | int | str]:
    signals = np.concatenate([day.trading_day.signals for day in days])
    returns = np.concatenate([day.realized_returns for day in days])
    if np.std(signals) > 0 and np.std(returns) > 0:
        correlation = float(np.corrcoef(signals, returns)[0, 1])
    else:
        correlation = 0.0
    direction_accuracy = float(np.mean(np.sign(signals) == np.sign(returns)))
    target_accuracy = float(
        np.mean(
            [
                np.sign(day.trading_day.target_position) == np.sign(day.target_return)
                for day in days
            ]
        )
    )
    return {
        "days": len(days),
        "first_date": days[0].date.isoformat(),
        "last_date": days[-1].date.isoformat(),
        "short_signal_correlation": correlation,
        "short_direction_accuracy": direction_accuracy,
        "target_direction_accuracy": target_accuracy,
    }


def _date_to_us(value: date) -> int:
    return int(datetime(value.year, value.month, value.day, tzinfo=timezone.utc).timestamp() * 1_000_000)


def _us_to_date(value: int) -> date:
    return datetime.fromtimestamp(value / 1_000_000, tz=timezone.utc).date()
