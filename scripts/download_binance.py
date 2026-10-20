"""Download checksum-verified Binance spot kline archives."""

import argparse
from pathlib import Path

from hail_ppo_reproduction.binance import download_monthly_klines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--months", nargs="+", required=True)
    args = parser.parse_args()

    archives = download_monthly_klines(
        args.output,
        symbol=args.symbol,
        interval=args.interval,
        months=args.months,
    )
    for archive in archives:
        print(f"verified: {archive}")


if __name__ == "__main__":
    main()
