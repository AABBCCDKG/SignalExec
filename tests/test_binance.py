import hashlib
from pathlib import Path
import zipfile

import numpy as np
import pytest

from hail_ppo_reproduction.binance import load_kline_archives, verify_checksum


def write_archive(path: Path) -> None:
    rows = (
        "open_time,open,high,low,close,volume,close_time,quote_volume,trades,"
        "taker_buy_volume,taker_buy_quote,ignore\n"
        "1735689600000,100,102,99,101,10,0,0,1,4,0,0\n"
        "1735689900000000,101,103,100,102,12,0,0,2,7,0,0\n"
    )
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("BTCUSDT-5m.csv", rows)


def test_load_kline_archives_normalizes_millisecond_and_microsecond_timestamps(tmp_path):
    archive = tmp_path / "BTCUSDT-5m-2025-01.zip"
    write_archive(archive)

    series = load_kline_archives([archive], symbol="BTCUSDT", interval="5m")

    np.testing.assert_array_equal(
        series.open_time_us,
        np.array([1735689600000000, 1735689900000000], dtype=np.int64),
    )
    np.testing.assert_array_equal(series.close, np.array([101.0, 102.0]))
    np.testing.assert_array_equal(series.taker_buy_volume, np.array([4.0, 7.0]))
    np.testing.assert_array_equal(series.quote_volume, np.array([0.0, 0.0]))
    np.testing.assert_array_equal(series.trade_count, np.array([1, 2]))


def test_verify_checksum_rejects_modified_archive(tmp_path):
    archive = tmp_path / "month.zip"
    archive.write_bytes(b"original")
    checksum = tmp_path / "month.zip.CHECKSUM"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum.write_text(f"{digest}  month.zip\n", encoding="utf-8")
    verify_checksum(archive, checksum)

    archive.write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_checksum(archive, checksum)
