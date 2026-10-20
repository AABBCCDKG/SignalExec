"""Download verified Binance Spot trades and build a compact bucket cache."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from hail_ppo_reproduction.binance import download_monthly_trades
from hail_ppo_reproduction.microstructure import (
    aggregate_raw_trade_archives,
    complete_raw_trade_buckets,
)


def aggregate_one(payload: tuple[Path, Path, int]):
    archive, cache, bucket_minutes = payload
    return aggregate_raw_trade_archives(
        [archive],
        cache_path=cache,
        bucket_minutes=bucket_minutes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--months", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--bucket-minutes", type=int, default=3)
    parser.add_argument(
        "--reuse-verified-archives",
        action="store_true",
        help="reuse archives and adjacent checksum files verified in a prior gate",
    )
    args = parser.parse_args()

    if args.reuse_verified_archives:
        archives = tuple(
            args.data_dir / f"{args.symbol.upper()}-trades-{month}.zip"
            for month in args.months
        )
        missing = [
            path
            for archive in archives
            for path in (archive, archive.with_suffix(archive.suffix + ".CHECKSUM"))
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(f"verified archive inputs are missing: {missing[:3]}")
    else:
        archives = download_monthly_trades(
            args.data_dir,
            symbol=args.symbol,
            months=args.months,
        )
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.workers == 1:
        features = aggregate_raw_trade_archives(
            archives,
            cache_path=args.cache,
            bucket_minutes=args.bucket_minutes,
        )
    else:
        part_directory = args.cache.parent / f"{args.cache.stem}-parts"
        part_directory.mkdir(parents=True, exist_ok=True)
        payloads = [
            (archive, part_directory / f"{archive.stem}.npz", args.bucket_minutes)
            for archive in archives
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            parts = list(executor.map(aggregate_one, payloads))
        decision_times = np.concatenate([part.decision_time_us for part in parts])
        values = np.concatenate([part.values for part in parts])
        merged = complete_raw_trade_buckets(
            decision_times,
            values,
            bucket_minutes=args.bucket_minutes,
        )
        signature = np.array(
            [f"{path.name}:{path.stat().st_size}" for path in archives],
            dtype=np.str_,
        )
        np.savez_compressed(
            args.cache,
            decision_time_us=merged.decision_time_us,
            values=merged.values,
            source_signature=signature,
        )
        features = aggregate_raw_trade_archives(
            archives,
            cache_path=args.cache,
            bucket_minutes=args.bucket_minutes,
        )
    print(
        f"complete: {len(archives)} archives -> {len(features.decision_time_us)} "
        f"{args.bucket_minutes}-minute buckets at {args.cache}"
    )


if __name__ == "__main__":
    main()
