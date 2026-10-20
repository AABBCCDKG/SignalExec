"""Run the paired Signal-Off and controlled-signal execution comparison."""

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

from hail_ppo_reproduction.binance import (
    download_monthly_klines,
    download_monthly_trades,
    load_kline_archives,
)
from hail_ppo_reproduction.config import ExecutionConfig
from hail_ppo_reproduction.controlled import (
    ControlledDataConfig,
    build_controlled_dataset,
    combine_controlled_datasets,
    signal_diagnostics,
)
from hail_ppo_reproduction.controlled_experiment import (
    ControlledExperimentConfig,
    difference_in_differences,
    evaluate_controlled_policy,
    paired_advantages,
    plot_signal_increment,
    summarize_advantages,
    summarize_controlled,
    train_controlled_model,
    validate_matched_records,
)
from hail_ppo_reproduction.experiment import ALGORITHMS, CONTROLS
from hail_ppo_reproduction.microstructure import (
    aggregate_raw_trade_archives,
    build_microstructure_features,
    standardize_raw_trade_features,
)


DEFAULT_FEATURE_MONTHS = tuple(
    [f"2023-{month:02d}" for month in range(5, 13)]
    + [f"2024-{month:02d}" for month in range(1, 13)]
    + [f"2025-{month:02d}" for month in range(1, 13)]
    + [f"2026-{month:02d}" for month in range(1, 9)]
)
DEFAULT_PRICE_MONTHS = (*DEFAULT_FEATURE_MONTHS, "2026-09")


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


def load_model_for_evaluation(algorithm: str, path: Path):
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO, SAC

    model_class = {
        "sac": SAC,
        "ppo": PPO,
        "memory_ppo": RecurrentPPO,
    }[algorithm]
    return model_class.load(path, device="cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/binance"))
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"],
    )
    parser.add_argument("--interval", default="3m")
    parser.add_argument("--months", nargs="+", default=list(DEFAULT_PRICE_MONTHS))
    parser.add_argument("--feature-interval", default="1m")
    parser.add_argument("--feature-months", nargs="+", default=list(DEFAULT_FEATURE_MONTHS))
    parser.add_argument("--decision-minutes", type=int, default=3)
    parser.add_argument(
        "--feature-source",
        choices=("raw-trades", "minute-klines", "none"),
        default="raw-trades",
    )
    parser.add_argument("--train-start", type=date.fromisoformat, default=date(2023, 6, 1))
    parser.add_argument("--validation-start", type=date.fromisoformat, default=date(2025, 1, 1))
    parser.add_argument("--validation-end", type=date.fromisoformat, default=date(2026, 1, 1))
    parser.add_argument("--test-start", type=date.fromisoformat, default=date(2026, 3, 1))
    parser.add_argument("--test-end", type=date.fromisoformat, default=date(2026, 5, 1))
    parser.add_argument(
        "--evaluation-splits",
        nargs="+",
        choices=("validation", "test"),
        default=["validation"],
        help="development defaults to validation only; add test once after freezing",
    )
    parser.add_argument("--timesteps", type=int, default=1572864)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=ALGORITHMS,
        default=list(ALGORITHMS),
        help="train only the selected algorithms; useful for independent parallel runs",
    )
    parser.add_argument(
        "--controls",
        nargs="+",
        choices=CONTROLS,
        default=list(CONTROLS),
        help="evaluate only the selected controls; TWAP is required for paired results",
    )
    parser.add_argument(
        "--load-model",
        type=Path,
        help="load one frozen model for evaluation instead of training it again",
    )
    parser.add_argument("--train-accuracy", type=float, default=0.4)
    parser.add_argument(
        "--evaluation-accuracies",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.4, 0.6, 0.8],
    )
    parser.add_argument("--signal-replicas", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--transaction-cost-bps", type=float, default=0.5)
    parser.add_argument("--impact-bps", type=float, default=1.0)
    parser.add_argument("--penalty", type=float, default=0.25)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    symbols = tuple(symbol.upper() for symbol in args.symbols)
    if len(set(symbols)) != len(symbols):
        raise ValueError("symbols must be unique")
    algorithms = tuple(args.algorithms)
    if len(set(algorithms)) != len(algorithms):
        raise ValueError("algorithms must be unique")
    controls = tuple(args.controls)
    if len(set(controls)) != len(controls):
        raise ValueError("controls must be unique")
    if "twap" not in controls:
        raise ValueError("controls must include twap for paired comparisons")
    if args.load_model is not None:
        if len(algorithms) != 1 or len(args.seeds) != 1:
            raise ValueError("--load-model requires exactly one algorithm and one seed")
        if not args.load_model.is_file():
            raise FileNotFoundError(args.load_model)
    data_config = ControlledDataConfig(
        train_start=args.train_start,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    datasets = []
    archives_by_symbol = {}
    feature_archives_by_symbol = {}
    feature_training_mean = {}
    feature_training_scale = {}
    data_counts_by_symbol = {}
    for symbol in symbols:
        archives = download_monthly_klines(
            args.data_root / f"{symbol}-{args.interval}",
            symbol=symbol,
            interval=args.interval,
            months=args.months,
        )
        series = load_kline_archives(archives, symbol=symbol, interval=args.interval)
        if args.feature_source == "raw-trades":
            feature_archives = download_monthly_trades(
                args.data_root / f"{symbol}-trades",
                symbol=symbol,
                months=args.feature_months,
            )
            raw_buckets = aggregate_raw_trade_archives(
                feature_archives,
                cache_path=(
                    args.data_root
                    / f"{symbol}-trades-{args.decision_minutes}m.npz"
                ),
                bucket_minutes=args.decision_minutes,
            )
            microstructure = standardize_raw_trade_features(
                raw_buckets,
                train_start=data_config.train_start,
                validation_start=data_config.validation_start,
            )
        elif args.feature_source == "minute-klines":
            feature_archives = download_monthly_klines(
                args.data_root / f"{symbol}-{args.feature_interval}",
                symbol=symbol,
                interval=args.feature_interval,
                months=args.feature_months,
            )
            feature_series = load_kline_archives(
                feature_archives,
                symbol=symbol,
                interval=args.feature_interval,
            )
            microstructure = build_microstructure_features(
                feature_series,
                train_start=data_config.train_start,
                validation_start=data_config.validation_start,
                decision_minutes=args.decision_minutes,
            )
        else:
            feature_archives = ()
            microstructure = None
        symbol_dataset = build_controlled_dataset(
            series,
            data_config,
            microstructure=microstructure,
        )
        datasets.append(symbol_dataset)
        archives_by_symbol[symbol] = [archive.name for archive in archives]
        feature_archives_by_symbol[symbol] = [
            archive.name for archive in feature_archives
        ]
        feature_training_mean[symbol] = (
            [] if microstructure is None else microstructure.training_mean.tolist()
        )
        feature_training_scale[symbol] = (
            [] if microstructure is None else microstructure.training_scale.tolist()
        )
        data_counts_by_symbol[symbol] = {
            "train": len(symbol_dataset.train),
            "validation": len(symbol_dataset.validation),
            "test": len(symbol_dataset.test),
        }
    dataset = combine_controlled_datasets(datasets)
    experiment = ControlledExperimentConfig(
        timesteps=args.timesteps,
        train_seeds=tuple(args.seeds),
        train_accuracy=args.train_accuracy,
        evaluation_accuracies=tuple(args.evaluation_accuracies),
        signal_replicas=args.signal_replicas,
        eval_batch_size=args.eval_batch_size,
    )
    execution = ExecutionConfig(
        transaction_cost_bps=args.transaction_cost_bps,
        impact_bps=args.impact_bps,
        terminal_shortfall_penalty=args.penalty,
    )

    import torch

    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    split_days = {
        "validation": dataset.validation,
        "test": dataset.test,
    }
    diagnostics = {}
    for split in args.evaluation_splits:
        days = split_days[split]
        diagnostics[split] = {
            "days": len(days),
            "first_date": days[0].date.isoformat(),
            "last_date": days[-1].date.isoformat(),
            "signals": signal_diagnostics(
                days,
                accuracies=experiment.evaluation_accuracies,
                replicas=experiment.signal_replicas,
                base_seed=experiment.signal_seed,
            ),
        }
    metadata = {
        "status": "running",
        "name": "Signal-to-Execution RL: controlled-signal paired comparison",
        "symbols": list(symbols),
        "interval": args.interval,
        "decision_minutes": args.decision_minutes,
        "archives": archives_by_symbol,
        "feature_archives": feature_archives_by_symbol,
        "feature_source": args.feature_source,
        "feature_names": list(dataset.feature_names),
        "feature_training_mean": feature_training_mean,
        "feature_training_scale": feature_training_scale,
        "historical_data": asdict(data_config),
        "data_counts": {
            "train": len(dataset.train),
            "validation": len(dataset.validation),
            "test": len(dataset.test),
        },
        "data_counts_by_symbol": data_counts_by_symbol,
        "experiment": asdict(experiment),
        "algorithms": list(algorithms),
        "controls": list(controls),
        "loaded_model": None if args.load_model is None else str(args.load_model),
        "execution": asdict(execution),
        "evaluation_splits": args.evaluation_splits,
        "diagnostics": diagnostics,
        "comparison_contract": {
            "targets": [-1, 1],
            "mandatory_final_completion": True,
            "first_difference": "policy minus matched TWAP",
            "second_difference": "rho result minus matched rho=0 result",
            "oracle_warning": "controlled signals use realized next returns and are not tradable forecasts",
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

    records: list[dict] = []

    def save_results() -> None:
        validate_matched_records(records)
        summaries = summarize_controlled(records)
        advantages = paired_advantages(records)
        advantage_summaries = summarize_advantages(advantages)
        did = difference_in_differences(advantages)
        write_csv(args.output / "episodes.csv", records)
        write_csv(args.output / "summary.csv", summaries)
        write_csv(args.output / "paired_advantages.csv", advantages)
        write_csv(args.output / "advantage_summary.csv", advantage_summaries)
        write_csv(args.output / "difference_in_differences.csv", did)
        (args.output / "summary.json").write_text(
            json.dumps(summaries, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output / "difference_in_differences.json").write_text(
            json.dumps(did, indent=2) + "\n",
            encoding="utf-8",
        )

    def evaluate(name: str, model, seed: int | None) -> None:
        for split in args.evaluation_splits:
            records.extend(
                evaluate_controlled_policy(
                    name,
                    model,
                    seed,
                    split=split,
                    days=split_days[split],
                    config=experiment,
                    execution=execution,
                    target_quantity=data_config.target_quantity,
                    extra_feature_count=dataset.extra_feature_count,
                )
            )

    save_metadata()
    try:
        for name in controls:
            evaluate(name, None, None)
            print(f"evaluated control: {name}", flush=True)
        save_results()
        for seed in experiment.train_seeds:
            for algorithm in algorithms:
                print(
                    (
                        f"loading {algorithm}, seed={seed}, model={args.load_model}"
                        if args.load_model is not None
                        else f"training {algorithm}, seed={seed}, steps={experiment.timesteps}"
                    ),
                    flush=True,
                )
                if args.load_model is None:
                    model, record = train_controlled_model(
                        algorithm,
                        seed,
                        dataset.train,
                        experiment,
                        execution,
                        args.output / "models" / f"{algorithm}-{seed}",
                        target_quantity=data_config.target_quantity,
                        extra_feature_count=dataset.extra_feature_count,
                    )
                else:
                    model = load_model_for_evaluation(algorithm, args.load_model)
                    record = {
                        "algorithm": algorithm,
                        "train_seed": seed,
                        "requested_timesteps": experiment.timesteps,
                        "actual_timesteps": model.num_timesteps,
                        "training_days": len(dataset.train),
                        "train_accuracy": experiment.train_accuracy,
                        "extra_feature_count": dataset.extra_feature_count,
                        "training_seconds": 0.0,
                        "execution": asdict(execution),
                        "model": str(args.load_model),
                        "loaded_for_evaluation": True,
                    }
                metadata["training"].append(record)
                save_metadata()
                evaluate(algorithm, model, seed)
                save_results()
                print(
                    f"evaluated {algorithm}, seed={seed}, "
                    + (
                        "loaded frozen model"
                        if args.load_model is not None
                        else f"trained in {record['training_seconds']:.1f}s"
                    ),
                    flush=True,
                )
        did = difference_in_differences(paired_advantages(records))
        plot_signal_increment(did, args.output / "signal_increment.png")
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
