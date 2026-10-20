"""Download and parse Binance's official public monthly kline archives."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import re
import shutil
from typing import Iterable, Sequence
from urllib.request import Request, urlopen
import zipfile

import numpy as np
from numpy.typing import NDArray


BINANCE_KLINE_BASE_URL = (
    "https://data.binance.vision/data/spot/monthly/klines"
)
BINANCE_TRADE_BASE_URL = (
    "https://data.binance.vision/data/spot/monthly/trades"
)
_MONTH_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


@dataclass(frozen=True)
class BinanceKlineSeries:
    """Chronologically ordered spot klines with timestamps in microseconds."""

    symbol: str
    interval: str
    open_time_us: NDArray[np.int64]
    open: NDArray[np.float64]
    high: NDArray[np.float64]
    low: NDArray[np.float64]
    close: NDArray[np.float64]
    volume: NDArray[np.float64]
    taker_buy_volume: NDArray[np.float64]
    quote_volume: NDArray[np.float64] | None = None
    trade_count: NDArray[np.int64] | None = None
    taker_buy_quote_volume: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        arrays = (
            self.open_time_us,
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
            self.taker_buy_volume,
        )
        if not arrays[0].ndim == 1 or any(array.ndim != 1 for array in arrays):
            raise ValueError("kline arrays must be one-dimensional")
        if not len(arrays[0]) or any(len(array) != len(arrays[0]) for array in arrays):
            raise ValueError("kline arrays must be nonempty and have equal lengths")
        if np.any(np.diff(self.open_time_us) <= 0):
            raise ValueError("kline timestamps must be strictly increasing")
        prices = np.stack((self.open, self.high, self.low, self.close))
        if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
            raise ValueError("kline prices must be finite and positive")
        if not np.all(np.isfinite(self.volume)) or np.any(self.volume < 0):
            raise ValueError("kline volume must be finite and nonnegative")
        if (
            not np.all(np.isfinite(self.taker_buy_volume))
            or np.any(self.taker_buy_volume < 0)
            or np.any(self.taker_buy_volume > self.volume + 1e-12)
        ):
            raise ValueError("taker-buy volume must lie between zero and total volume")
        optional_arrays = (
            self.quote_volume,
            self.trade_count,
            self.taker_buy_quote_volume,
        )
        if any(array is not None and len(array) != len(self.open_time_us) for array in optional_arrays):
            raise ValueError("optional kline arrays must align with timestamps")
        if self.quote_volume is not None and (
            not np.all(np.isfinite(self.quote_volume)) or np.any(self.quote_volume < 0)
        ):
            raise ValueError("quote volume must be finite and nonnegative")
        if self.trade_count is not None and np.any(self.trade_count < 0):
            raise ValueError("trade count must be nonnegative")
        if self.taker_buy_quote_volume is not None and self.quote_volume is not None and (
            not np.all(np.isfinite(self.taker_buy_quote_volume))
            or np.any(self.taker_buy_quote_volume < 0)
            or np.any(self.taker_buy_quote_volume > self.quote_volume + 1e-9)
        ):
            raise ValueError("taker-buy quote volume must lie within total quote volume")


def download_monthly_klines(
    destination: Path,
    *,
    symbol: str,
    interval: str,
    months: Sequence[str],
    base_url: str = BINANCE_KLINE_BASE_URL,
) -> tuple[Path, ...]:
    """Download archives and verify every file against Binance's checksum."""

    normalized_symbol = symbol.upper()
    destination.mkdir(parents=True, exist_ok=True)
    archives = []
    for month in months:
        if not _MONTH_PATTERN.fullmatch(month):
            raise ValueError(f"invalid month {month!r}; expected YYYY-MM")
        filename = f"{normalized_symbol}-{interval}-{month}.zip"
        url_prefix = f"{base_url}/{normalized_symbol}/{interval}"
        archive = destination / filename
        checksum = destination / f"{filename}.CHECKSUM"
        if not archive.exists():
            _download(f"{url_prefix}/{filename}", archive)
        if not checksum.exists():
            _download(f"{url_prefix}/{filename}.CHECKSUM", checksum)
        try:
            verify_checksum(archive, checksum)
        except ValueError:
            archive.unlink(missing_ok=True)
            _download(f"{url_prefix}/{filename}", archive)
            verify_checksum(archive, checksum)
        archives.append(archive)
    return tuple(archives)


def download_monthly_trades(
    destination: Path,
    *,
    symbol: str,
    months: Sequence[str],
    base_url: str = BINANCE_TRADE_BASE_URL,
) -> tuple[Path, ...]:
    """Download and checksum-verify Binance Spot raw-trade archives."""

    normalized_symbol = symbol.upper()
    destination.mkdir(parents=True, exist_ok=True)
    archives = []
    for month in months:
        if not _MONTH_PATTERN.fullmatch(month):
            raise ValueError(f"invalid month {month!r}; expected YYYY-MM")
        filename = f"{normalized_symbol}-trades-{month}.zip"
        url_prefix = f"{base_url}/{normalized_symbol}"
        archive = destination / filename
        checksum = destination / f"{filename}.CHECKSUM"
        if not archive.exists():
            _download(f"{url_prefix}/{filename}", archive)
        if not checksum.exists():
            _download(f"{url_prefix}/{filename}.CHECKSUM", checksum)
        try:
            verify_checksum(archive, checksum)
        except ValueError:
            archive.unlink(missing_ok=True)
            _download(f"{url_prefix}/{filename}", archive)
            verify_checksum(archive, checksum)
        archives.append(archive)
    return tuple(archives)


def verify_checksum(archive: Path, checksum_file: Path) -> None:
    """Raise when an archive does not match its adjacent official checksum."""

    fields = checksum_file.read_text(encoding="utf-8").strip().split()
    if not fields or not re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]):
        raise ValueError(f"invalid checksum file: {checksum_file}")
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest().lower() != fields[0].lower():
        raise ValueError(f"checksum mismatch for {archive}")


def load_kline_archives(
    archives: Iterable[Path],
    *,
    symbol: str,
    interval: str,
) -> BinanceKlineSeries:
    """Load one or more Binance kline zip files into typed NumPy arrays."""

    rows: list[
        tuple[int, float, float, float, float, float, float, int, float, float]
    ] = []
    for archive in archives:
        rows.extend(_read_archive(archive))
    if not rows:
        raise ValueError("no kline rows were loaded")
    rows.sort(key=lambda row: row[0])
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError("duplicate kline timestamps across archives")
    columns = list(zip(*rows))
    return BinanceKlineSeries(
        symbol=symbol.upper(),
        interval=interval,
        open_time_us=np.asarray(columns[0], dtype=np.int64),
        open=np.asarray(columns[1], dtype=np.float64),
        high=np.asarray(columns[2], dtype=np.float64),
        low=np.asarray(columns[3], dtype=np.float64),
        close=np.asarray(columns[4], dtype=np.float64),
        volume=np.asarray(columns[5], dtype=np.float64),
        quote_volume=np.asarray(columns[6], dtype=np.float64),
        trade_count=np.asarray(columns[7], dtype=np.int64),
        taker_buy_volume=np.asarray(columns[8], dtype=np.float64),
        taker_buy_quote_volume=np.asarray(columns[9], dtype=np.float64),
    )


def _read_archive(
    archive: Path,
) -> list[tuple[int, float, float, float, float, float, float, int, float, float]]:
    with zipfile.ZipFile(archive) as bundle:
        members = [name for name in bundle.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected one CSV in {archive}, found {len(members)}")
        with bundle.open(members[0]) as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
            rows = []
            for line_number, fields in enumerate(reader, start=1):
                if not fields:
                    continue
                if line_number == 1 and not fields[0].strip().isdigit():
                    continue
                if len(fields) < 12:
                    raise ValueError(
                        f"{archive}:{line_number} has {len(fields)} columns; expected 12"
                    )
                try:
                    rows.append(
                        (
                            _timestamp_to_microseconds(fields[0]),
                            float(fields[1]),
                            float(fields[2]),
                            float(fields[3]),
                            float(fields[4]),
                            float(fields[5]),
                            float(fields[7]),
                            int(fields[8]),
                            float(fields[9]),
                            float(fields[10]),
                        )
                    )
                except ValueError as error:
                    raise ValueError(f"invalid kline row {archive}:{line_number}") from error
    return rows


def _timestamp_to_microseconds(value: str) -> int:
    timestamp = int(value)
    # Binance documents that Spot archives use milliseconds before 2025 and
    # microseconds from 2025 onward. Standardize both eras to microseconds.
    return timestamp if timestamp >= 100_000_000_000_000 else timestamp * 1_000


def _download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "signal-to-execution-rl/0.1"})
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
