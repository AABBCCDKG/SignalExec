from datetime import date, datetime, timezone
import zipfile

import numpy as np

from hail_ppo_reproduction.binance import BinanceKlineSeries
from hail_ppo_reproduction.microstructure import (
    FEATURE_NAMES,
    aggregate_raw_trade_archives,
    build_microstructure_features,
    standardize_raw_trade_features,
)


def minute_series(count: int = 3 * 24 * 60) -> BinanceKlineSeries:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    timestamps = int(start.timestamp() * 1_000_000) + 60_000_000 * np.arange(count)
    returns = 0.0001 * np.sin(np.arange(count) / 11.0)
    opens = 100.0 * np.exp(np.concatenate(([0.0], np.cumsum(returns[:-1]))))
    closes = opens * np.exp(returns)
    quote_volume = 1_000.0 + 100.0 * np.cos(np.arange(count) / 7.0)
    trade_count = 100 + (np.arange(count) % 13)
    taker_quote = quote_volume * (0.5 + 0.1 * np.sin(np.arange(count) / 5.0))
    return BinanceKlineSeries(
        symbol="BTCUSDT",
        interval="1m",
        open_time_us=timestamps.astype(np.int64),
        open=opens,
        high=np.maximum(opens, closes) * 1.0002,
        low=np.minimum(opens, closes) * 0.9998,
        close=closes,
        volume=quote_volume / opens,
        taker_buy_volume=taker_quote / opens,
        quote_volume=quote_volume,
        trade_count=trade_count.astype(np.int64),
        taker_buy_quote_volume=taker_quote,
    )


def test_microstructure_features_are_standardized_on_training_rows():
    features = build_microstructure_features(
        minute_series(),
        train_start=date(2025, 1, 1),
        validation_start=date(2025, 1, 2),
    )
    train_end = int(datetime(2025, 1, 2, tzinfo=timezone.utc).timestamp() * 1_000_000)
    train = features.values[features.decision_time_us < train_end]

    assert features.feature_names == FEATURE_NAMES
    assert features.values.shape[1] == len(FEATURE_NAMES)
    np.testing.assert_allclose(train.mean(axis=0), np.zeros(len(FEATURE_NAMES)), atol=1e-12)
    np.testing.assert_allclose(train.std(axis=0), np.ones(len(FEATURE_NAMES)), atol=1e-12)


def test_feature_at_decision_uses_only_previous_completed_minutes():
    original = minute_series()
    changed_quote = original.quote_volume.copy()
    decision_index = 24 * 60 + 5
    changed_quote[decision_index] *= 100.0
    changed = BinanceKlineSeries(
        symbol=original.symbol,
        interval=original.interval,
        open_time_us=original.open_time_us,
        open=original.open,
        high=original.high,
        low=original.low,
        close=original.close,
        volume=original.volume,
        taker_buy_volume=original.taker_buy_volume,
        quote_volume=changed_quote,
        trade_count=original.trade_count,
        taker_buy_quote_volume=original.taker_buy_quote_volume,
    )
    first = build_microstructure_features(
        original,
        train_start=date(2025, 1, 1),
        validation_start=date(2025, 1, 2),
    )
    second = build_microstructure_features(
        changed,
        train_start=date(2025, 1, 1),
        validation_start=date(2025, 1, 2),
    )
    decision_time = original.open_time_us[decision_index]

    np.testing.assert_array_equal(
        first.features_for(np.array([decision_time])),
        second.features_for(np.array([decision_time])),
    )


def test_raw_trades_stream_into_five_minute_buckets_and_cache(tmp_path):
    archive = tmp_path / "BTCUSDT-trades-2025-01.zip"
    start_ms = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    rows = [
        f"1,100,1,100,{start_ms + 1},False,True",
        f"2,101,2,202,{start_ms + 60_000},True,True",
        f"3,102,1,102,{start_ms + 5 * 60_000},False,True",
        f"4,103,1,103,{start_ms + 6 * 60_000},False,True",
    ]
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("BTCUSDT-trades-2025-01.csv", "\n".join(rows) + "\n")
    cache = tmp_path / "buckets.npz"

    buckets = aggregate_raw_trade_archives([archive], cache_path=cache)
    cached = aggregate_raw_trade_archives([archive], cache_path=cache)

    assert len(buckets.decision_time_us) == 2
    np.testing.assert_array_equal(cached.decision_time_us, buckets.decision_time_us)
    np.testing.assert_array_equal(cached.values, buckets.values)
    assert buckets.values[0, 2] < 0  # sell-initiated quote volume dominates
    assert buckets.values[1, 2] == 1.0


def test_raw_trade_bucket_segments_are_reassembled_in_trade_time_order(tmp_path):
    start_ms = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    rows = {
        1: f"1,100,1,100,{start_ms + 1},False,True",
        2: f"2,101,1,101,{start_ms + 2},True,True",
        3: f"3,102,1,102,{start_ms + 3},False,True",
        4: f"4,103,1,103,{start_ms + 3 * 60_000 + 1},False,True",
        5: f"5,104,1,104,{start_ms + 3 * 60_000 + 2},True,True",
    }

    def archive(name, order):
        path = tmp_path / name
        with zipfile.ZipFile(path, "w") as bundle:
            bundle.writestr(name.replace(".zip", ".csv"), "\n".join(rows[i] for i in order))
        return path

    ordered = aggregate_raw_trade_archives(
        [archive("ordered.zip", [1, 2, 3, 4, 5])],
        bucket_minutes=3,
    )
    segmented = aggregate_raw_trade_archives(
        [archive("segmented.zip", [1, 2, 4, 3, 5])],
        bucket_minutes=3,
    )

    np.testing.assert_array_equal(segmented.decision_time_us, ordered.decision_time_us)
    np.testing.assert_allclose(segmented.values, ordered.values)
    assert segmented.values[0, -1] == 1.0


def test_duplicate_trade_ids_are_counted_once(tmp_path):
    start_ms = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    rows = [
        f"1,100,1,100,{start_ms + 1},False,True",
        f"2,101,1,101,{start_ms + 2},True,True",
        f"3,102,1,102,{start_ms + 3 * 60_000 + 1},False,True",
    ]

    def archive(name, archive_rows):
        path = tmp_path / name
        with zipfile.ZipFile(path, "w") as bundle:
            bundle.writestr(name.replace(".zip", ".csv"), "\n".join(archive_rows))
        return path

    expected = aggregate_raw_trade_archives(
        [archive("expected.zip", rows)],
        bucket_minutes=3,
    )
    duplicated = aggregate_raw_trade_archives(
        [archive("duplicated.zip", [*rows, *rows])],
        bucket_minutes=3,
    )

    np.testing.assert_array_equal(duplicated.decision_time_us, expected.decision_time_us)
    np.testing.assert_allclose(duplicated.values, expected.values)


def test_empty_raw_trade_buckets_are_filled_with_zero_features(tmp_path):
    archive = tmp_path / "BTCUSDT-trades-2025-01.zip"
    start_ms = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    rows = [
        f"1,100,1,100,{start_ms + 1},False,True",
        f"2,102,1,102,{start_ms + 2 * 60_000 + 1},True,True",
    ]
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("BTCUSDT-trades-2025-01.csv", "\n".join(rows))

    buckets = aggregate_raw_trade_archives([archive], bucket_minutes=1)

    assert len(buckets.decision_time_us) == 3
    np.testing.assert_array_equal(buckets.values[1], np.zeros(len(FEATURE_NAMES)))


def test_raw_trade_scaler_uses_only_training_buckets(tmp_path):
    archive = tmp_path / "BTCUSDT-trades-2025-01.zip"
    start_ms = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    rows = []
    for bucket in range(4):
        for trade in range(2 + bucket):
            timestamp = start_ms + bucket * 5 * 60_000 + trade * 1_000
            rows.append(
                f"{bucket * 10 + trade},100,{trade + 1},{100 * (trade + 1)},"
                f"{timestamp},{str(trade % 2 == 0)},True"
            )
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("BTCUSDT-trades-2025-01.csv", "\n".join(rows) + "\n")
    buckets = aggregate_raw_trade_archives([archive])

    features = standardize_raw_trade_features(
        buckets,
        train_start=date(2025, 1, 1),
        validation_start=date(2025, 1, 2),
    )

    assert features.values.shape == (4, 8)
    assert np.all(np.isfinite(features.values))
