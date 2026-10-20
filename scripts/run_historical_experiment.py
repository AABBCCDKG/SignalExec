"""Run the HRT-inspired execution comparison on chronological Binance data."""

import argparse
import csv
from dataclasses import asdict
from datetime import date
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys

from hail_ppo_reproduction.binance import download_monthly_klines, load_kline_archives
from hail_ppo_reproduction.config import ExecutionConfig
from hail_ppo_reproduction.experiment import ALGORITHMS, CONTROLS
from hail_ppo_reproduction.historical import HistoricalDataConfig, build_historical_dataset
from hail_ppo_reproduction.historical_experiment import (
    HistoricalExperimentConfig,
    evaluate_historical_policy,
    plot_historical_results,
    summarize_historical,
    train_historical_model,
)


DEFAULT_MONTHS = (
    "2024-12",
    "2025-01",
    "2025-02",
    "2025-03",
    "2025-04",
    "2025-05",
    "2025-06",
    "2025-07",
    "2025-08",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_state() -> dict:
    root = Path(__file__).resolve().parents[1]
    paths = [root / "pyproject.toml", root / "uv.lock"]
    paths += sorted((root / "src").rglob("*.py")) + sorted((root / "scripts").rglob("*.py"))
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }

    def git(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "source_sha256": hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/binance/BTCUSDT-5m"))
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--months", nargs="+", default=list(DEFAULT_MONTHS))
    parser.add_argument("--train-start", type=date.fromisoformat, default=date(2025, 1, 1))
    parser.add_argument("--validation-start", type=date.fromisoformat, default=date(2025, 5, 1))
    parser.add_argument("--test-start", type=date.fromisoformat, default=date(2025, 7, 1))
    parser.add_argument("--test-end", type=date.fromisoformat, default=date(2025, 9, 1))
    parser.add_argument(
        "--evaluation-splits",
        nargs="+",
        choices=("validation", "test"),
        default=["validation"],
        help="development defaults to validation only; add test once after freezing the run",
    )
    parser.add_argument("--timesteps", type=int, default=32768)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--transaction-cost-bps", type=float, default=0.5)
    parser.add_argument("--impact-bps", type=float, default=1.0)
    parser.add_argument("--penalty", type=float, default=0.25)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    archives = download_monthly_klines(
        args.data_dir,
        symbol=args.symbol,
        interval=args.interval,
        months=args.months,
    )
    series = load_kline_archives(archives, symbol=args.symbol, interval=args.interval)
    data_config = HistoricalDataConfig(
        train_start=args.train_start,
        validation_start=args.validation_start,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    dataset = build_historical_dataset(series, data_config)
    experiment = HistoricalExperimentConfig(
        timesteps=args.timesteps,
        train_seeds=tuple(args.seeds),
    )
    execution = ExecutionConfig(
        transaction_cost_bps=args.transaction_cost_bps,
        impact_bps=args.impact_bps,
        terminal_shortfall_penalty=args.penalty,
    )

    import torch

    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    metadata = {
        "status": "running",
        "name": "Signal-to-Execution RL: Binance historical reproduction",
        "symbol": args.symbol.upper(),
        "interval": args.interval,
        "archives": [archive.name for archive in archives],
        "historical_data": asdict(data_config),
        "experiment": asdict(experiment),
        "execution": asdict(execution),
        "evaluation_splits": args.evaluation_splits,
        "diagnostics": dataset.diagnostics(tuple(("train", *args.evaluation_splits))),
        "short_model": {
            "feature_mean": dataset.short_model.feature_mean.tolist(),
            "feature_scale": dataset.short_model.feature_scale.tolist(),
            "target_mean": dataset.short_model.target_mean,
            "coefficients": dataset.short_model.coefficients.tolist(),
            "alpha": dataset.short_model.alpha,
        },
        "long_model": {
            "feature_mean": dataset.long_model.feature_mean.tolist(),
            "feature_scale": dataset.long_model.feature_scale.tolist(),
            "target_mean": dataset.long_model.target_mean,
            "coefficients": dataset.long_model.coefficients.tolist(),
            "alpha": dataset.long_model.alpha,
        },
        "python": sys.version,
        "platform": platform.platform(),
        "torch_threads": 1,
        "device": "cpu",
        "source": source_state(),
        "versions": {
            package: version(package)
            for package in ("numpy", "gymnasium", "torch", "stable-baselines3", "sb3-contrib")
        },
        "training": [],
    }

    def save_metadata() -> None:
        (args.output / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    save_metadata()
    records: list[dict] = []

    def evaluate(name: str, model, seed: int | None) -> None:
        split_days = {
            "validation": dataset.validation,
            "test": dataset.test,
        }
        for split in args.evaluation_splits:
            days = split_days[split]
            for force_completion in (False, True):
                records.extend(
                    evaluate_historical_policy(
                        name,
                        model,
                        seed,
                        split=split,
                        days=days,
                        execution=execution,
                        force_completion=force_completion,
                    )
                )
        write_csv(args.output / "episodes.csv", records)
        write_csv(args.output / "summary.csv", summarize_historical(records))

    try:
        for name in CONTROLS:
            evaluate(name, None, None)
            print(f"evaluated control: {name}", flush=True)
        for seed in experiment.train_seeds:
            for algorithm in ALGORITHMS:
                print(
                    f"training {algorithm}, seed={seed}, steps={experiment.timesteps}",
                    flush=True,
                )
                model, record = train_historical_model(
                    algorithm,
                    seed,
                    dataset.train,
                    experiment,
                    execution,
                    args.output / "models" / f"{algorithm}-{seed}",
                )
                metadata["training"].append(record)
                save_metadata()
                evaluate(algorithm, model, seed)
                print(
                    f"evaluated {algorithm}, seed={seed}, "
                    f"trained in {record['training_seconds']:.1f}s",
                    flush=True,
                )
        summaries = summarize_historical(records)
        (args.output / "summary.json").write_text(
            json.dumps(summaries, indent=2) + "\n",
            encoding="utf-8",
        )
        plot_historical_results(summaries, args.output / "comparison.png")
        metadata["status"] = "complete"
        print(f"complete: {args.output} ({len(records)} evaluated episodes)", flush=True)
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save_metadata()


if __name__ == "__main__":
    main()
