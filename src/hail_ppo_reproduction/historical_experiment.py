"""Train and evaluate execution policies on chronological historical days."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from hail_ppo_reproduction.config import ExecutionConfig
from hail_ppo_reproduction.env import ExecutionEnv
from hail_ppo_reproduction.experiment import ALGORITHMS, CONTROLS, build_model, control_actions
from hail_ppo_reproduction.historical import HistoricalDay, make_historical_day_factory


@dataclass(frozen=True)
class HistoricalExperimentConfig:
    timesteps: int = 32768
    train_seeds: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self) -> None:
        if self.timesteps < 256 or self.timesteps % 256:
            raise ValueError("timesteps must be a positive multiple of 256")
        if not self.train_seeds or len(set(self.train_seeds)) != len(self.train_seeds):
            raise ValueError("training seeds must be nonempty and unique")


def train_historical_model(
    algorithm: str,
    seed: int,
    days: Iterable[HistoricalDay],
    config: HistoricalExperimentConfig,
    execution: ExecutionConfig,
    directory: Path,
) -> tuple[Any, dict]:
    from stable_baselines3.common.logger import configure

    day_tuple = tuple(days)
    env = ExecutionEnv(make_historical_day_factory(day_tuple), execution)
    model = build_model(algorithm, env, seed, config.timesteps)
    model.get_env().seed(10_000_000 + seed)
    directory.mkdir(parents=True, exist_ok=False)
    model.set_logger(configure(str(directory), ["csv"]))
    started = perf_counter()
    model.learn(total_timesteps=config.timesteps)
    elapsed = perf_counter() - started
    model.save(directory / "model.zip")
    record = {
        "algorithm": algorithm,
        "train_seed": seed,
        "requested_timesteps": config.timesteps,
        "actual_timesteps": model.num_timesteps,
        "training_days": len(day_tuple),
        "training_seconds": elapsed,
        "execution": asdict(execution),
        "model": str(directory / "model.zip"),
    }
    env.close()
    return model, record


def evaluate_historical_policy(
    name: str,
    model: Any,
    train_seed: int | None,
    *,
    split: str,
    days: Iterable[HistoricalDay],
    execution: ExecutionConfig,
    force_completion: bool = False,
) -> list[dict]:
    """Evaluate fixed dated episodes without exposing future prices to policies."""

    records = []
    for historical_day in days:
        env = ExecutionEnv(lambda rng, day=historical_day: day.trading_day, execution)
        observation, _ = env.reset(seed=0)
        state = None
        episode_start = np.ones(1, dtype=bool)
        total_reward = 0.0
        total_cost = 0.0
        total_steps = len(historical_day.trading_day.signals)
        for step in range(total_steps):
            observations = observation[None, :]
            if model is None:
                actions = control_actions(
                    name,
                    observations,
                    total_steps - step,
                    total_steps,
                    historical_day.trading_day.return_volatility,
                )
            else:
                actions, state = model.predict(
                    observations,
                    state=state,
                    episode_start=episode_start,
                    deterministic=True,
                )
            if force_completion and step == total_steps - 1:
                actions = np.ones((1, 1), dtype=np.float32)
            observation, reward, terminated, truncated, info = env.step(actions[0])
            total_reward += reward
            total_cost += info["trading_cost"]
            episode_start = np.array([terminated or truncated])
        notional = (
            abs(historical_day.trading_day.target_position)
            * historical_day.trading_day.prices[0]
        )
        records.append(
            {
                "algorithm": name,
                "train_seed": train_seed,
                "split": split,
                "date": historical_day.date.isoformat(),
                "completion": "forced" if force_completion else "natural",
                "pnl": env.cumulative_pnl,
                "pnl_bps": 10_000.0 * env.cumulative_pnl / notional,
                "fill_fraction": env.fill_fraction,
                "filled": bool(env.fill_fraction >= 1.0 - 1e-6),
                "episode_reward": total_reward,
                "cost_bps": 10_000.0 * total_cost / notional,
                "target_direction": int(np.sign(historical_day.trading_day.target_position)),
                "realized_day_return": historical_day.target_return,
            }
        )
        env.close()
    return records


def summarize_historical(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    keys = ("algorithm", "train_seed", "split", "completion")
    for record in records:
        groups.setdefault(tuple(record[key] for key in keys), []).append(record)
    summaries = []
    for key, rows in groups.items():
        pnls = np.array([row["pnl_bps"] for row in rows])
        summaries.append(
            {
                **dict(zip(keys, key)),
                "episodes": len(rows),
                "mean_pnl_bps": float(pnls.mean()),
                "pnl_std_bps": float(pnls.std(ddof=1)),
                "pnl_se_bps": float(pnls.std(ddof=1) / np.sqrt(len(rows))),
                "mean_fill": float(np.mean([row["fill_fraction"] for row in rows])),
                "completion_rate": float(np.mean([row["filled"] for row in rows])),
                "mean_reward": float(np.mean([row["episode_reward"] for row in rows])),
                "mean_cost_bps": float(np.mean([row["cost_bps"] for row in rows])),
            }
        )
    return summaries


def plot_historical_results(summaries: list[dict], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    policies = (*CONTROLS, *ALGORITHMS)
    splits = [
        split
        for split in ("validation", "test")
        if any(summary["split"] == split for summary in summaries)
    ]
    if not splits:
        raise ValueError("historical plot needs validation or test summaries")
    fig, axes = plt.subplots(len(splits), 2, figsize=(14, 4 * len(splits)), squeeze=False)
    for row, split in enumerate(splits):
        points = [
            summary
            for summary in summaries
            if summary["split"] == split and summary["completion"] == "natural"
        ]
        for column, metric in enumerate(("mean_pnl_bps", "mean_fill")):
            values = []
            errors = []
            for policy in policies:
                per_seed = [point[metric] for point in points if point["algorithm"] == policy]
                values.append(float(np.mean(per_seed)))
                errors.append(float(np.std(per_seed)) if len(per_seed) > 1 else 0.0)
            axes[row, column].bar(np.arange(len(policies)), values, yerr=errors, alpha=0.8)
            axes[row, column].set_xticks(np.arange(len(policies)), policies, rotation=30, ha="right")
            axes[row, column].set_title(
                f"{split}: {'PnL (bps)' if column == 0 else 'fill fraction'}"
            )
            axes[row, column].grid(axis="y", alpha=0.2)
    fig.suptitle("Binance chronological replay — error bars are variation across training seeds")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
