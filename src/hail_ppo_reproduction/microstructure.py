"""Causal microstructure features from Binance one-minute spot klines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable
import zipfile

import numpy as np
from numpy.typing import NDArray

from hail_ppo_reproduction.binance import BinanceKlineSeries


FEATURE_NAMES = (
    "log_quote_volume",
    "log_trade_count",
    "taker_buy_imbalance",
    "log_mean_trade_quote",
    "realized_volatility_bps",
    "high_low_range_bps",
    "minute_volume_concentration",
    "trade_count_cv",
)

RAW_TRADE_FEATURE_NAMES = (
    "log_trade_quote_volume",
    "log_raw_trade_count",
    "raw_aggressor_imbalance",
    "log_mean_raw_trade_quote",
    "raw_trade_range_bps",
    "raw_trade_return_bps",
    "largest_trade_share",
    "aggressor_side_switch_rate",
)


@dataclass(frozen=True)
class RawTradeBucketSeries:
    decision_time_us: NDArray[np.int64]
    values: NDArray[np.float64]
    feature_names: tuple[str, ...] = RAW_TRADE_FEATURE_NAMES

    def __post_init__(self) -> None:
        if self.decision_time_us.ndim != 1 or self.values.ndim != 2:
            raise ValueError("trade bucket timestamps must be 1-D and values 2-D")
        if len(self.decision_time_us) != len(self.values):
            raise ValueError("trade bucket timestamps and values must align")
        if self.values.shape[1] != len(self.feature_names):
            raise ValueError("trade bucket feature names must match value columns")
        if np.any(np.diff(self.decision_time_us) <= 0):
            raise ValueError("trade bucket timestamps must be strictly increasing")
        if not np.all(np.isfinite(self.values)):
            raise ValueError("trade bucket values must be finite")


def complete_raw_trade_buckets(
    decision_time_us: NDArray[np.int64],
    values: NDArray[np.float64],
    *,
    bucket_minutes: int,
) -> RawTradeBucketSeries:
    """Sort buckets and insert zero features for intervals with no trades."""

    bucket_us = bucket_minutes * 60_000_000
    timestamps = np.asarray(decision_time_us, dtype=np.int64)
    raw_values = np.asarray(values, dtype=np.float64)
    if len(timestamps) == 0:
        raise ValueError("raw trade buckets must not be empty")
    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    raw_values = raw_values[order]
    if len(np.unique(timestamps)) != len(timestamps):
        raise ValueError(
            f"raw trade archives contain duplicate {bucket_minutes}-minute buckets"
        )
    if np.any((timestamps - timestamps[0]) % bucket_us):
        raise ValueError("raw trade buckets are not aligned to one interval grid")
    expected = np.arange(
        timestamps[0], timestamps[-1] + bucket_us, bucket_us, dtype=np.int64
    )
    if len(expected) == len(timestamps):
        return RawTradeBucketSeries(decision_time_us=timestamps, values=raw_values)
    completed_values = np.zeros((len(expected), raw_values.shape[1]), dtype=np.float64)
    indices = ((timestamps - expected[0]) // bucket_us).astype(np.int64)
    completed_values[indices] = raw_values
    return RawTradeBucketSeries(decision_time_us=expected, values=completed_values)


@dataclass(frozen=True)
class _RawTradeSegment:
    first_order: tuple[int, int]
    last_order: tuple[int, int]
    count: int
    total_quote: float
    signed_quote: float
    maximum_quote: float
    first_price: float
    last_price: float
    minimum_price: float
    maximum_price: float
    side_switches: int
    first_side: bytes
    last_side: bytes


@dataclass(frozen=True)
class MicrostructureFeatureSeries:
    """Standardized features available immediately before each decision time."""

    decision_time_us: NDArray[np.int64]
    values: NDArray[np.float64]
    feature_names: tuple[str, ...]
    training_mean: NDArray[np.float64]
    training_scale: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.decision_time_us.ndim != 1 or self.values.ndim != 2:
            raise ValueError("microstructure timestamps must be 1-D and values 2-D")
        if len(self.decision_time_us) != len(self.values):
            raise ValueError("microstructure timestamps and values must align")
        if self.values.shape[1] != len(self.feature_names):
            raise ValueError("feature names must match value columns")
        if np.any(np.diff(self.decision_time_us) <= 0):
            raise ValueError("microstructure decision times must be strictly increasing")
        if not np.all(np.isfinite(self.values)):
            raise ValueError("microstructure values must be finite")

    def features_for(self, decision_times_us: NDArray[np.int64]) -> NDArray[np.float64]:
        indices = np.searchsorted(self.decision_time_us, decision_times_us)
        if np.any(indices >= len(self.decision_time_us)):
            raise KeyError("microstructure features do not cover every decision time")
        if not np.array_equal(self.decision_time_us[indices], decision_times_us):
            raise KeyError("microstructure feature timestamps do not match decision times")
        return self.values[indices].copy()


def build_microstructure_features(
    minute_series: BinanceKlineSeries,
    *,
    train_start: date,
    validation_start: date,
    decision_minutes: int = 5,
) -> MicrostructureFeatureSeries:
    """Aggregate the previous completed one-minute bars for each decision.

    A feature row stamped at time ``t`` uses only minute bars in ``[t-5m, t)``.
    Standardization parameters are fit only on the training timestamps.
    """

    if decision_minutes < 1:
        raise ValueError("decision_minutes must be positive")
    if (
        minute_series.quote_volume is None
        or minute_series.trade_count is None
        or minute_series.taker_buy_quote_volume is None
    ):
        raise ValueError("minute klines need quote volume, trade count and taker-buy quote volume")
    differences = np.diff(minute_series.open_time_us)
    minute_us = 60_000_000
    if not len(differences) or int(np.median(differences)) != minute_us:
        raise ValueError("microstructure input must use one-minute klines")
    decision_us = decision_minutes * minute_us
    timestamps = []
    raw_features = []
    for index in np.flatnonzero(minute_series.open_time_us % decision_us == 0):
        index = int(index)
        if index < decision_minutes:
            continue
        window_times = minute_series.open_time_us[index - decision_minutes : index]
        expected = minute_series.open_time_us[index] - decision_us + minute_us * np.arange(
            decision_minutes
        )
        if not np.array_equal(window_times, expected):
            continue
        window = slice(index - decision_minutes, index)
        quote_volume = minute_series.quote_volume[window]
        trade_count = minute_series.trade_count[window]
        taker_buy_quote = minute_series.taker_buy_quote_volume[window]
        total_quote = float(quote_volume.sum())
        total_trades = int(trade_count.sum())
        imbalance = (
            2.0 * float(taker_buy_quote.sum()) / total_quote - 1.0
            if total_quote > 0
            else 0.0
        )
        mean_trade_quote = total_quote / total_trades if total_trades else 0.0
        minute_returns = np.log(
            minute_series.close[window] / minute_series.open[window]
        )
        realized_volatility_bps = 10_000.0 * float(
            np.sqrt(np.sum(minute_returns * minute_returns))
        )
        high_low_range_bps = 10_000.0 * float(
            np.log(
                np.max(minute_series.high[window])
                / np.min(minute_series.low[window])
            )
        )
        volume_concentration = (
            float(np.max(quote_volume) / total_quote) if total_quote > 0 else 0.0
        )
        mean_count = float(np.mean(trade_count))
        trade_count_cv = (
            float(np.std(trade_count) / mean_count) if mean_count > 0 else 0.0
        )
        timestamps.append(int(minute_series.open_time_us[index]))
        raw_features.append(
            (
                np.log1p(total_quote),
                np.log1p(total_trades),
                imbalance,
                np.log1p(mean_trade_quote),
                realized_volatility_bps,
                high_low_range_bps,
                volume_concentration,
                trade_count_cv,
            )
        )
    decision_times = np.asarray(timestamps, dtype=np.int64)
    raw_values = np.asarray(raw_features, dtype=np.float64)
    train_mask = (
        (decision_times >= _date_to_us(train_start))
        & (decision_times < _date_to_us(validation_start))
    )
    if int(train_mask.sum()) < 2:
        raise ValueError("not enough training rows to standardize microstructure features")
    training_mean = raw_values[train_mask].mean(axis=0)
    training_scale = raw_values[train_mask].std(axis=0)
    training_scale = np.where(training_scale > 1e-12, training_scale, 1.0)
    standardized = np.clip((raw_values - training_mean) / training_scale, -8.0, 8.0)
    return MicrostructureFeatureSeries(
        decision_time_us=decision_times,
        values=standardized,
        feature_names=FEATURE_NAMES,
        training_mean=training_mean,
        training_scale=training_scale,
    )


def aggregate_raw_trade_archives(
    archives: Iterable[Path],
    *,
    cache_path: Path | None = None,
    bucket_minutes: int = 5,
) -> RawTradeBucketSeries:
    """Stream Binance raw-trade zips into compact completed-window features."""

    archive_tuple = tuple(archives)
    if not archive_tuple:
        raise ValueError("at least one raw trade archive is required")
    bucket_us = bucket_minutes * 60_000_000
    signature = np.array(
        [f"{path.name}:{path.stat().st_size}" for path in archive_tuple],
        dtype=np.str_,
    )
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            if np.array_equal(cached["source_signature"], signature):
                cached_times = cached["decision_time_us"].copy()
                cached_values = cached["values"].copy()
                result = complete_raw_trade_buckets(
                    cached_times,
                    cached_values,
                    bucket_minutes=bucket_minutes,
                )
                if len(result.decision_time_us) != len(cached_times):
                    np.savez_compressed(
                        cache_path,
                        decision_time_us=result.decision_time_us,
                        values=result.values,
                        source_signature=signature,
                    )
                return result
    decision_times: list[int] = []
    values: list[tuple[float, ...]] = []
    for archive in archive_tuple:
        with zipfile.ZipFile(archive) as bundle:
            members = [name for name in bundle.namelist() if name.lower().endswith(".csv")]
            if len(members) != 1:
                raise ValueError(f"expected one CSV in {archive}, found {len(members)}")
            segments_by_bucket: dict[int, list[_RawTradeSegment]] = {}
            seen_trade_id_base: int | None = None
            seen_trade_ids = bytearray()
            current_bucket: int | None = None
            first_order: tuple[int, int] | None = None
            last_order: tuple[int, int] | None = None
            count = 0
            total_quote = 0.0
            signed_quote = 0.0
            maximum_quote = 0.0
            first_price = 0.0
            last_price = 0.0
            minimum_price = np.inf
            maximum_price = 0.0
            side_switches = 0
            first_side: bytes | None = None
            previous_side: bytes | None = None

            def finish_segment() -> None:
                if (
                    current_bucket is None
                    or first_order is None
                    or last_order is None
                    or first_side is None
                    or previous_side is None
                    or count == 0
                ):
                    return
                segments_by_bucket.setdefault(current_bucket, []).append(
                    _RawTradeSegment(
                        first_order=first_order,
                        last_order=last_order,
                        count=count,
                        total_quote=total_quote,
                        signed_quote=signed_quote,
                        maximum_quote=maximum_quote,
                        first_price=first_price,
                        last_price=last_price,
                        minimum_price=minimum_price,
                        maximum_price=maximum_price,
                        side_switches=side_switches,
                        first_side=first_side,
                        last_side=previous_side,
                    )
                )

            with bundle.open(members[0]) as raw:
                for line_number, line in enumerate(raw, start=1):
                    fields = line.rstrip().split(b",")
                    if not fields:
                        continue
                    if line_number == 1 and not fields[0].isdigit():
                        continue
                    if len(fields) < 7:
                        raise ValueError(
                            f"{archive}:{line_number} has {len(fields)} columns; expected 7"
                        )
                    try:
                        trade_id = int(fields[0])
                        price = float(fields[1])
                        quote = float(fields[3])
                        timestamp = int(fields[4])
                    except ValueError as error:
                        raise ValueError(
                            f"invalid raw trade row {archive}:{line_number}"
                        ) from error
                    if seen_trade_id_base is None:
                        seen_trade_id_base = trade_id
                    trade_id_offset = trade_id - seen_trade_id_base
                    if trade_id_offset < 0:
                        raise ValueError(
                            f"trade id precedes the archive's first id: "
                            f"{archive}:{line_number}"
                        )
                    byte_index, bit_index = divmod(trade_id_offset, 8)
                    if byte_index >= len(seen_trade_ids):
                        seen_trade_ids.extend(
                            b"\0" * (byte_index + 1 - len(seen_trade_ids))
                        )
                    bit = 1 << bit_index
                    if seen_trade_ids[byte_index] & bit:
                        continue
                    seen_trade_ids[byte_index] |= bit
                    timestamp_us = (
                        timestamp
                        if timestamp >= 100_000_000_000_000
                        else timestamp * 1_000
                    )
                    bucket = timestamp_us // bucket_us * bucket_us
                    order = (timestamp_us, trade_id)
                    buyer_is_maker = fields[5].lower() == b"true"
                    side = b"sell" if buyer_is_maker else b"buy"
                    if current_bucket != bucket:
                        finish_segment()
                        current_bucket = bucket
                        first_order = order
                        last_order = None
                        count = 0
                        total_quote = 0.0
                        signed_quote = 0.0
                        maximum_quote = 0.0
                        first_price = price
                        last_price = price
                        minimum_price = price
                        maximum_price = price
                        side_switches = 0
                        first_side = side
                        previous_side = None
                    if last_order is not None and order <= last_order:
                        raise ValueError(
                            f"raw trades are not ordered within a bucket segment: "
                            f"{archive}:{line_number}"
                        )
                    if previous_side is not None and side != previous_side:
                        side_switches += 1
                    last_order = order
                    previous_side = side
                    count += 1
                    total_quote += quote
                    signed_quote += -quote if buyer_is_maker else quote
                    maximum_quote = max(maximum_quote, quote)
                    last_price = price
                    minimum_price = min(minimum_price, price)
                    maximum_price = max(maximum_price, price)
            finish_segment()

            for bucket, bucket_segments in segments_by_bucket.items():
                bucket_segments.sort(key=lambda segment: segment.first_order)
                for previous, following in zip(bucket_segments, bucket_segments[1:]):
                    if following.first_order <= previous.last_order:
                        raise ValueError(
                            f"raw trade segments overlap in {archive} at bucket {bucket}"
                        )
                bucket_count = sum(segment.count for segment in bucket_segments)
                bucket_total_quote = sum(
                    segment.total_quote for segment in bucket_segments
                )
                bucket_switches = sum(
                    segment.side_switches for segment in bucket_segments
                ) + sum(
                    previous.last_side != following.first_side
                    for previous, following in zip(
                        bucket_segments, bucket_segments[1:]
                    )
                )
                first_segment = bucket_segments[0]
                last_segment = bucket_segments[-1]
                values.append(
                    (
                        np.log1p(bucket_total_quote),
                        np.log1p(bucket_count),
                        (
                            sum(segment.signed_quote for segment in bucket_segments)
                            / bucket_total_quote
                            if bucket_total_quote > 0
                            else 0.0
                        ),
                        np.log1p(bucket_total_quote / bucket_count),
                        10_000.0
                        * np.log(
                            max(segment.maximum_price for segment in bucket_segments)
                            / min(segment.minimum_price for segment in bucket_segments)
                        ),
                        10_000.0
                        * np.log(last_segment.last_price / first_segment.first_price),
                        (
                            max(segment.maximum_quote for segment in bucket_segments)
                            / bucket_total_quote
                            if bucket_total_quote > 0
                            else 0.0
                        ),
                        bucket_switches / (bucket_count - 1)
                        if bucket_count > 1
                        else 0.0,
                    )
                )
                decision_times.append(bucket + bucket_us)

    result = complete_raw_trade_buckets(
        np.asarray(decision_times, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
        bucket_minutes=bucket_minutes,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            decision_time_us=result.decision_time_us,
            values=result.values,
            source_signature=signature,
        )
    return result


def standardize_raw_trade_features(
    buckets: RawTradeBucketSeries,
    *,
    train_start: date,
    validation_start: date,
) -> MicrostructureFeatureSeries:
    train_mask = (
        (buckets.decision_time_us >= _date_to_us(train_start))
        & (buckets.decision_time_us < _date_to_us(validation_start))
    )
    if int(train_mask.sum()) < 2:
        raise ValueError("not enough training buckets to standardize raw trade features")
    training_mean = buckets.values[train_mask].mean(axis=0)
    training_scale = buckets.values[train_mask].std(axis=0)
    training_scale = np.where(training_scale > 1e-12, training_scale, 1.0)
    values = np.clip((buckets.values - training_mean) / training_scale, -8.0, 8.0)
    return MicrostructureFeatureSeries(
        decision_time_us=buckets.decision_time_us,
        values=values,
        feature_names=buckets.feature_names,
        training_mean=training_mean,
        training_scale=training_scale,
    )


def _date_to_us(value: date) -> int:
    timestamp = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    return int(timestamp.timestamp() * 1_000_000)
