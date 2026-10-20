from datetime import date, datetime, timezone

import numpy as np

from hail_ppo_reproduction.binance import BinanceKlineSeries
from hail_ppo_reproduction.historical import (
    HistoricalDataConfig,
    build_historical_dataset,
    fit_ridge,
    make_historical_day_factory,
)


def make_series() -> BinanceKlineSeries:
    bars_per_day = 4
    count = 44 * bars_per_day
    start = datetime(2024, 12, 20, tzinfo=timezone.utc)
    interval_us = 6 * 60 * 60 * 1_000_000
    timestamps = int(start.timestamp() * 1_000_000) + interval_us * np.arange(count)
    returns = 0.002 * np.sin(np.arange(count) / 5.0) + 0.0002
    opens = 100.0 * np.exp(np.concatenate(([0.0], np.cumsum(returns[:-1]))))
    closes = opens * np.exp(returns)
    volume = 10.0 + 2.0 * np.cos(np.arange(count) / 7.0)
    taker_buy = volume * (0.5 + 0.2 * np.sin(np.arange(count) / 3.0))
    return BinanceKlineSeries(
        symbol="BTCUSDT",
        interval="6h",
        open_time_us=timestamps.astype(np.int64),
        open=opens,
        high=np.maximum(opens, closes) * 1.001,
        low=np.minimum(opens, closes) * 0.999,
        close=closes,
        volume=volume,
        taker_buy_volume=taker_buy,
    )


def config() -> HistoricalDataConfig:
    return HistoricalDataConfig(
        train_start=date(2024, 12, 28),
        validation_start=date(2025, 1, 10),
        test_start=date(2025, 1, 20),
        test_end=date(2025, 2, 1),
    )


def test_ridge_recovers_linear_relationship():
    features = np.arange(20, dtype=np.float64).reshape(10, 2)
    targets = 2.0 * features[:, 0] - features[:, 1] + 3.0

    model = fit_ridge(features, targets, alpha=0.0)

    np.testing.assert_allclose(model.predict(features), targets, atol=1e-10)


def test_historical_dataset_has_strict_splits_and_reproducible_factory():
    dataset = build_historical_dataset(make_series(), config())

    assert dataset.bars_per_day == 4
    assert dataset.train[-1].date < config().validation_start
    assert dataset.validation[0].date >= config().validation_start
    assert dataset.validation[-1].date < config().test_start
    assert dataset.test[0].date >= config().test_start
    assert dataset.test[-1].date < config().test_end
    assert all(len(day.trading_day.signals) == 4 for day in dataset.test)
    factory = make_historical_day_factory(dataset.train)
    assert factory(np.random.default_rng(7)) is factory(np.random.default_rng(7))


def test_future_price_changes_do_not_refit_training_signal_models():
    original = make_series()
    changed_open = original.open.copy()
    changed_close = original.close.copy()
    boundary = int(
        datetime(2025, 1, 10, tzinfo=timezone.utc).timestamp() * 1_000_000
    )
    future = original.open_time_us >= boundary
    changed_open[future] *= 1.2
    changed_close[future] *= 1.2
    changed = BinanceKlineSeries(
        symbol=original.symbol,
        interval=original.interval,
        open_time_us=original.open_time_us,
        open=changed_open,
        high=np.maximum(original.high, changed_open),
        low=np.minimum(original.low, changed_open),
        close=changed_close,
        volume=original.volume,
        taker_buy_volume=original.taker_buy_volume,
    )

    first = build_historical_dataset(original, config())
    second = build_historical_dataset(changed, config())

    np.testing.assert_array_equal(
        first.short_model.coefficients,
        second.short_model.coefficients,
    )
    np.testing.assert_array_equal(
        first.long_model.coefficients,
        second.long_model.coefficients,
    )
