from datetime import date, datetime, timezone
from dataclasses import replace

import numpy as np

from hail_ppo_reproduction.binance import BinanceKlineSeries
from hail_ppo_reproduction.controlled import (
    ControlledDataConfig,
    HistoricalPriceDay,
    build_controlled_dataset,
    combine_controlled_datasets,
    evaluation_signal_seed,
    make_controlled_trading_day,
)
from hail_ppo_reproduction.microstructure import MicrostructureFeatureSeries


def make_series() -> BinanceKlineSeries:
    bars_per_day = 4
    count = 50 * bars_per_day
    start = datetime(2024, 12, 20, tzinfo=timezone.utc)
    interval_us = 6 * 60 * 60 * 1_000_000
    timestamps = int(start.timestamp() * 1_000_000) + interval_us * np.arange(count)
    returns = 0.002 * np.sin(np.arange(count) / 5.0) + 0.0002
    opens = 100.0 * np.exp(np.concatenate(([0.0], np.cumsum(returns[:-1]))))
    closes = opens * np.exp(returns)
    volume = np.full(count, 10.0)
    return BinanceKlineSeries(
        symbol="BTCUSDT",
        interval="6h",
        open_time_us=timestamps.astype(np.int64),
        open=opens,
        high=np.maximum(opens, closes) * 1.001,
        low=np.minimum(opens, closes) * 0.999,
        close=closes,
        volume=volume,
        taker_buy_volume=volume * 0.5,
    )


def test_controlled_dataset_uses_strict_chronological_splits():
    config = ControlledDataConfig(
        train_start=date(2024, 12, 25),
        validation_start=date(2025, 1, 5),
        validation_end=date(2025, 1, 10),
        test_start=date(2025, 1, 15),
        test_end=date(2025, 2, 1),
    )

    dataset = build_controlled_dataset(make_series(), config)

    assert dataset.bars_per_day == 4
    assert dataset.train[-1].date < config.validation_start
    assert dataset.validation[0].date >= config.validation_start
    assert dataset.validation[-1].date < config.validation_end
    assert dataset.test[0].date >= config.test_start
    assert dataset.test[-1].date < config.test_end
    used_dates = {day.date for split in (dataset.train, dataset.validation, dataset.test) for day in split}
    assert date(2025, 1, 12) not in used_dates


def test_controlled_dataset_aligns_extra_features_to_decision_times():
    series = make_series()
    values = np.column_stack(
        (
            np.arange(len(series.open_time_us), dtype=np.float64),
            -np.arange(len(series.open_time_us), dtype=np.float64),
        )
    )
    microstructure = MicrostructureFeatureSeries(
        decision_time_us=series.open_time_us,
        values=values,
        feature_names=("first", "second"),
        training_mean=np.zeros(2),
        training_scale=np.ones(2),
    )
    config = ControlledDataConfig(
        train_start=date(2024, 12, 25),
        validation_start=date(2025, 1, 5),
        test_start=date(2025, 1, 15),
        test_end=date(2025, 2, 1),
    )

    dataset = build_controlled_dataset(series, config, microstructure=microstructure)
    first_day = dataset.train[0]
    first_timestamp = int(
        datetime(
            first_day.date.year,
            first_day.date.month,
            first_day.date.day,
            tzinfo=timezone.utc,
        ).timestamp()
        * 1_000_000
    )
    index = int(np.searchsorted(series.open_time_us, first_timestamp))

    assert dataset.extra_feature_count == 2
    np.testing.assert_array_equal(first_day.extra_features[0], values[index])


def test_multiple_symbols_are_pooled_without_losing_identity():
    btc = make_series()
    eth = replace(btc, symbol="ETHUSDT")
    config = ControlledDataConfig(
        train_start=date(2024, 12, 25),
        validation_start=date(2025, 1, 5),
        test_start=date(2025, 1, 15),
        test_end=date(2025, 2, 1),
    )
    pooled = combine_controlled_datasets(
        (build_controlled_dataset(btc, config), build_controlled_dataset(eth, config))
    )

    assert {day.symbol for day in pooled.train} == {"BTCUSDT", "ETHUSDT"}
    assert len(pooled.train) == 2 * len(build_controlled_dataset(btc, config).train)
    assert evaluation_signal_seed(
        base_seed=5, day=date(2025, 1, 1), replica=0, symbol="BTCUSDT"
    ) != evaluation_signal_seed(
        base_seed=5, day=date(2025, 1, 1), replica=0, symbol="ETHUSDT"
    )


def test_signal_off_is_zero_and_perfect_signal_equals_next_return():
    prices = 100.0 * np.exp(np.array([0.0, 0.01, 0.005, 0.02]))
    returns = np.diff(np.log(prices))
    price_day = HistoricalPriceDay(
        date=date(2025, 1, 1),
        prices=prices,
        realized_returns=returns,
        return_volatility=0.01,
    )

    signal_off = make_controlled_trading_day(
        price_day,
        target_direction=1,
        target_quantity=1.0,
        signal_accuracy=None,
        signal_seed=7,
    )
    perfect = make_controlled_trading_day(
        price_day,
        target_direction=-1,
        target_quantity=1.0,
        signal_accuracy=1.0,
        signal_seed=99,
    )

    np.testing.assert_array_equal(signal_off.signals, np.zeros_like(returns))
    np.testing.assert_allclose(perfect.signals, returns)
    assert perfect.target_position == -1.0


def test_matched_seed_produces_identical_signal_for_mirrored_targets():
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0, 0.01, size=1000)
    prices = 100.0 * np.exp(np.concatenate(([0.0], np.cumsum(returns))))
    price_day = HistoricalPriceDay(
        date=date(2025, 1, 2),
        prices=prices,
        realized_returns=returns,
        return_volatility=0.01,
    )

    buy = make_controlled_trading_day(
        price_day,
        target_direction=1,
        target_quantity=1.0,
        signal_accuracy=0.6,
        signal_seed=123,
    )
    sell = make_controlled_trading_day(
        price_day,
        target_direction=-1,
        target_quantity=1.0,
        signal_accuracy=0.6,
        signal_seed=123,
    )

    np.testing.assert_array_equal(buy.signals, sell.signals)
    assert abs(np.corrcoef(buy.signals, returns)[0, 1] - 0.6) < 0.05
    seed = evaluation_signal_seed(base_seed=5, day=price_day.date, replica=0)
    assert seed == evaluation_signal_seed(base_seed=5, day=price_day.date, replica=0)
    noise_only = make_controlled_trading_day(
        price_day,
        target_direction=1,
        target_quantity=1.0,
        signal_accuracy=0.0,
        signal_seed=seed,
    )
    mixed = make_controlled_trading_day(
        price_day,
        target_direction=1,
        target_quantity=1.0,
        signal_accuracy=0.6,
        signal_seed=seed,
    )
    recovered_noise = (mixed.signals - 0.6 * returns) / np.sqrt(1.0 - 0.6**2)
    np.testing.assert_allclose(recovered_noise, noise_only.signals)
