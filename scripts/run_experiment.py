"""Reproducible CPU pilot; saves models, episode-level data and configuration."""

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys

from hail_ppo_reproduction import ExecutionConfig
from hail_ppo_reproduction.experiment import (
    ALGORITHMS, CONTROLS, ExperimentConfig, evaluate_policy, plot_results,
    summarize, train_model,
)
from dataclasses import asdict


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_state() -> dict:
    root = Path(__file__).resolve().parents[1]
    paths = [root / "pyproject.toml", root / "uv.lock"]
    paths += sorted((root / "src").rglob("*.py")) + sorted((root / "scripts").rglob("*.py"))
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], text=True,
                                capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
            "source_sha256": hashes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=32768)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--accuracies", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--steps", type=int, default=78)
    parser.add_argument("--penalty", type=float, default=0.25)
    args = parser.parse_args()
    config = ExperimentConfig(timesteps=args.timesteps, train_seeds=tuple(args.seeds),
                              eval_episodes=args.eval_episodes, accuracies=tuple(args.accuracies),
                              steps=args.steps)
    execution = ExecutionConfig(terminal_shortfall_penalty=args.penalty)
    import torch
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {"status": "running", "experiment": asdict(config), "execution": asdict(execution),
                "python": sys.version, "platform": platform.platform(), "torch_threads": 1,
                "device": "cpu", "source": source_state(),
                "versions": {name: version(name) for name in
                             ("numpy", "gymnasium", "torch", "stable-baselines3", "sb3-contrib")},
                "training": []}
    def save_metadata():
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    save_metadata()
    records = []
    def evaluate(name, model, seed):
        for accuracy in config.accuracies:
            for force_completion in (False, True):
                records.extend(evaluate_policy(name, model, seed, config, execution,
                                                accuracy, force_completion))
        write_csv(args.output / "episodes.csv", records)
        write_csv(args.output / "summary.csv", summarize(records))
    try:
        for name in CONTROLS:
            evaluate(name, None, None)
            print(f"evaluated control: {name}", flush=True)
        for seed in config.train_seeds:
            for algorithm in ALGORITHMS:
                print(f"training {algorithm}, seed={seed}, steps={config.timesteps}", flush=True)
                model, record = train_model(algorithm, seed, config, execution,
                                             args.output / "models" / f"{algorithm}-{seed}")
                metadata["training"].append(record)
                save_metadata()
                evaluate(algorithm, model, seed)
                print(f"evaluated {algorithm}, seed={seed}, trained in {record['training_seconds']:.1f}s", flush=True)
        summaries = summarize(records)
        (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        plot_results(summaries, args.output / "comparison.png")
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
