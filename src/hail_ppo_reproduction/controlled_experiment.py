"""Matched controlled-signal comparison on historical price paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from hail_ppo_reproduction.config import ExecutionConfig
from hail_ppo_reproduction.controlled import (
    HistoricalPriceDay,
    evaluation_signal_seed,
    make_controlled_day_factory,
    make_controlled_trading_day,
)
from hail_ppo_reproduction.env import ExecutionEnv
from hail_ppo_reproduction.experiment import ALGORITHMS, CONTROLS, build_model, control_actions
from hail_ppo_reproduction.policies import HailBaselinePolicy


@dataclass(frozen=True)
class ControlledExperimentConfig:
    timesteps: int = 32768
    train_seeds: tuple[int, ...] = (0, 1, 2)
    train_accuracy: float = 0.4
    evaluation_accuracies: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8)
    signal_replicas: int = 1
    signal_seed: int = 4_000_000_000
    eval_batch_size: int = 32

    def __post_init__(self) -> None:
        if self.timesteps < 256 or self.timesteps % 256:
            raise ValueError("timesteps must be a positive multiple of 256")
        if not self.train_seeds or len(set(self.train_seeds)) != len(self.train_seeds):
            raise ValueError("training seeds must be nonempty and unique")
        if self.signal_replicas < 1 or self.eval_batch_size < 1:
            raise ValueError("signal_replicas and eval_batch_size must be positive")
        if not self.evaluation_accuracies:
            raise ValueError("evaluation_accuracies must not be empty")
        if 0.0 not in self.evaluation_accuracies:
            raise ValueError("evaluation_accuracies must include rho=0 for DiD")
        for accuracy in (*self.evaluation_accuracies, self.train_accuracy):
            if not -1.0 <= accuracy <= 1.0:
                raise ValueError("signal accuracies must lie in [-1, 1]")


def train_controlled_model(
    algorithm: str,
    seed: int,
    days: Iterable[HistoricalPriceDay],
    config: ControlledExperimentConfig,
    execution: ExecutionConfig,
    directory: Path,
    *,
    target_quantity: float,
    extra_feature_count: int = 0,
) -> tuple[Any, dict]:
    from stable_baselines3.common.logger import configure

    day_tuple = tuple(days)
    env = ExecutionEnv(
        make_controlled_day_factory(
            day_tuple,
            signal_accuracy=config.train_accuracy,
            target_quantity=target_quantity,
        ),
        execution,
        extra_feature_count=extra_feature_count,
    )
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
        "train_accuracy": config.train_accuracy,
        "extra_feature_count": extra_feature_count,
        "training_seconds": elapsed,
        "execution": asdict(execution),
        "model": str(directory / "model.zip"),
    }
    env.close()
    return model, record


def evaluate_controlled_policy(
    name: str,
    model: Any,
    train_seed: int | None,
    *,
    split: str,
    days: Iterable[HistoricalPriceDay],
    config: ControlledExperimentConfig,
    execution: ExecutionConfig,
    target_quantity: float,
    extra_feature_count: int = 0,
) -> list[dict]:
    """Evaluate matched signal/target episodes with mandatory final completion."""

    day_tuple = tuple(days)
    scenarios: list[tuple[str, float | None, int]] = [("off", None, -1)]
    for accuracy in config.evaluation_accuracies:
        for replica in range(config.signal_replicas):
            scenarios.append(("controlled", accuracy, replica))

    records = []
    for signal_mode, accuracy, replica in scenarios:
        for target_direction in (-1, 1):
            for offset in range(0, len(day_tuple), config.eval_batch_size):
                price_days = day_tuple[offset : offset + config.eval_batch_size]
                trading_days = [
                    make_controlled_trading_day(
                        price_day,
                        target_direction=target_direction,
                        target_quantity=target_quantity,
                        signal_accuracy=accuracy,
                        signal_seed=(
                            0
                            if signal_mode == "off"
                            else evaluation_signal_seed(
                                base_seed=config.signal_seed,
                                day=price_day.date,
                                replica=replica,
                                symbol=price_day.symbol,
                            )
                        ),
                    )
                    for price_day in price_days
                ]
                envs = [
                    ExecutionEnv(
                        lambda rng, day=day: day,
                        execution,
                        extra_feature_count=extra_feature_count,
                    )
                    for day in trading_days
                ]
                observations = np.stack([env.reset(seed=0)[0] for env in envs])
                states = None
                episode_starts = np.ones(len(envs), dtype=bool)
                total_rewards = np.zeros(len(envs), dtype=np.float64)
                total_costs = np.zeros(len(envs), dtype=np.float64)
                executed_quantities = np.zeros(len(envs), dtype=np.float64)
                executed_values = np.zeros(len(envs), dtype=np.float64)
                total_steps = len(trading_days[0].signals)
                for step in range(total_steps):
                    if model is None:
                        if name == "hail_baseline":
                            policy = HailBaselinePolicy()
                            actions = np.array(
                                [
                                    policy.action(
                                        directed_signal=float(observation[0]),
                                        remaining_fraction=float(observation[1]),
                                        steps_remaining=total_steps - step,
                                        total_steps=total_steps,
                                        return_volatility=day.return_volatility,
                                    )
                                    for observation, day in zip(observations, trading_days)
                                ],
                                dtype=np.float32,
                            )
                        else:
                            actions = control_actions(
                                name,
                                observations,
                                total_steps - step,
                                total_steps,
                                trading_days[0].return_volatility,
                            )
                    else:
                        actions, states = model.predict(
                            observations,
                            state=states,
                            episode_start=episode_starts,
                            deterministic=True,
                        )
                    if step == total_steps - 1:
                        actions = np.ones((len(envs), 1), dtype=np.float32)
                    fractions = np.clip(
                        np.asarray(actions).reshape(len(envs), -1)[:, 0],
                        0.0,
                        1.0,
                    )
                    quantities = np.array(
                        [fraction * env.remaining_quantity for env, fraction in zip(envs, fractions)]
                    )
                    prices = np.array([day.prices[step] for day in trading_days])
                    executed_quantities += quantities
                    executed_values += quantities * prices
                    transitions = [
                        env.step(np.array([fraction], dtype=np.float32))
                        for env, fraction in zip(envs, fractions)
                    ]
                    observations = np.stack([transition[0] for transition in transitions])
                    total_rewards += [transition[1] for transition in transitions]
                    total_costs += [transition[4]["trading_cost"] for transition in transitions]
                    episode_starts = np.array(
                        [transition[2] or transition[3] for transition in transitions]
                    )

                for index, (price_day, trading_day, env) in enumerate(
                    zip(price_days, trading_days, envs)
                ):
                    notional = abs(trading_day.target_position) * trading_day.prices[0]
                    average_price = executed_values[index] / executed_quantities[index]
                    cost_bps = 10_000.0 * total_costs[index] / notional
                    shortfall_bps = (
                        10_000.0
                        * target_direction
                        * (average_price - trading_day.prices[0])
                        / trading_day.prices[0]
                        + cost_bps
                    )
                    accuracy_label = "off" if accuracy is None else f"{accuracy:g}"
                    matched_input_id = (
                        f"{split}:{price_day.symbol}:{price_day.date.isoformat()}:"
                        f"{target_direction}:"
                        f"{accuracy_label}:{replica}"
                    )
                    records.append(
                        {
                            "algorithm": name,
                            "train_seed": train_seed,
                            "split": split,
                            "symbol": price_day.symbol,
                            "date": price_day.date.isoformat(),
                            "target_direction": target_direction,
                            "signal_mode": signal_mode,
                            "signal_accuracy": accuracy,
                            "signal_replica": replica,
                            "matched_input_id": matched_input_id,
                            "signal_sha256": hashlib.sha256(
                                trading_day.signals.tobytes()
                            ).hexdigest(),
                            "pnl_bps": 10_000.0 * env.cumulative_pnl / notional,
                            "implementation_shortfall_bps": shortfall_bps,
                            "average_execution_price": average_price,
                            "fill_fraction": env.fill_fraction,
                            "filled": bool(env.fill_fraction >= 1.0 - 1e-6),
                            "episode_reward": float(total_rewards[index]),
                            "cost_bps": cost_bps,
                        }
                    )
                    env.close()
    return records


def validate_matched_records(records: list[dict]) -> None:
    """Verify identical signals and complete targets for every policy comparison."""

    hashes: dict[str, set[str]] = {}
    for record in records:
        if not record["filled"]:
            raise ValueError(
                f"incomplete controlled episode: {record['algorithm']} "
                f"{record['matched_input_id']}"
            )
        hashes.setdefault(record["matched_input_id"], set()).add(record["signal_sha256"])
    mismatched = [key for key, values in hashes.items() if len(values) != 1]
    if mismatched:
        raise ValueError(f"policies received different signals for {mismatched[:3]}")


def summarize_controlled(records: list[dict]) -> list[dict]:
    keys = (
        "algorithm",
        "train_seed",
        "split",
        "signal_mode",
        "signal_accuracy",
    )
    groups: dict[tuple, list[dict]] = {}
    for record in records:
        groups.setdefault(tuple(record[key] for key in keys), []).append(record)
    summaries = []
    for key, rows in groups.items():
        shortfalls = np.array(
            [row["implementation_shortfall_bps"] for row in rows], dtype=np.float64
        )
        pnls = np.array([row["pnl_bps"] for row in rows], dtype=np.float64)
        summaries.append(
            {
                **dict(zip(keys, key)),
                "episodes": len(rows),
                "mean_shortfall_bps": float(shortfalls.mean()),
                "shortfall_se_bps": _standard_error(shortfalls),
                "mean_pnl_bps": float(pnls.mean()),
                "pnl_se_bps": _standard_error(pnls),
                "mean_fill": float(np.mean([row["fill_fraction"] for row in rows])),
                "completion_rate": float(np.mean([row["filled"] for row in rows])),
                "mean_cost_bps": float(np.mean([row["cost_bps"] for row in rows])),
            }
        )
    return summaries


def paired_advantages(records: list[dict]) -> list[dict]:
    """First difference: policy execution outcome minus matched TWAP outcome."""

    twap = {
        record["matched_input_id"]: record
        for record in records
        if record["algorithm"] == "twap"
    }
    rows = []
    for record in records:
        benchmark = twap.get(record["matched_input_id"])
        if benchmark is None:
            raise ValueError(f"missing TWAP match for {record['matched_input_id']}")
        rows.append(
            {
                "algorithm": record["algorithm"],
                "train_seed": record["train_seed"],
                "split": record["split"],
                "symbol": record["symbol"],
                "date": record["date"],
                "target_direction": record["target_direction"],
                "signal_mode": record["signal_mode"],
                "signal_accuracy": record["signal_accuracy"],
                "signal_replica": record["signal_replica"],
                "matched_input_id": record["matched_input_id"],
                "twap_shortfall_advantage_bps": (
                    benchmark["implementation_shortfall_bps"]
                    - record["implementation_shortfall_bps"]
                ),
                "twap_pnl_advantage_bps": (
                    record["pnl_bps"] - benchmark["pnl_bps"]
                ),
            }
        )
    return rows


def summarize_advantages(records: list[dict]) -> list[dict]:
    paired = _pair_mirrored_targets(records)
    keys = (
        "algorithm",
        "train_seed",
        "split",
        "signal_mode",
        "signal_accuracy",
    )
    groups: dict[tuple, list[dict]] = {}
    for record in paired:
        groups.setdefault(tuple(record[key] for key in keys), []).append(record)
    rows = []
    for key, values in groups.items():
        shortfall_values = np.array(
            [value["twap_shortfall_advantage_bps"] for value in values],
            dtype=np.float64,
        )
        pnl_values = np.array(
            [value["twap_pnl_advantage_bps"] for value in values],
            dtype=np.float64,
        )
        rows.append(
            {
                **dict(zip(keys, key)),
                "episodes": len(values),
                "twap_shortfall_advantage_bps": float(shortfall_values.mean()),
                "twap_shortfall_advantage_se_bps": _standard_error(shortfall_values),
                "twap_pnl_advantage_bps": float(pnl_values.mean()),
                "twap_pnl_advantage_se_bps": _standard_error(pnl_values),
            }
        )
    return rows


def difference_in_differences(advantages: list[dict]) -> list[dict]:
    """Second difference: subtract each policy's rho=0 TWAP advantage."""

    paired = _pair_mirrored_targets(advantages)
    identity = (
        "algorithm",
        "train_seed",
        "split",
        "symbol",
        "date",
        "signal_replica",
    )
    rho_zero = {
        tuple(row[key] for key in identity): row
        for row in paired
        if row["signal_mode"] == "controlled" and row["signal_accuracy"] == 0.0
    }
    groups: dict[tuple, list[tuple[float, float]]] = {}
    for row in paired:
        if row["signal_mode"] != "controlled":
            continue
        baseline = rho_zero.get(tuple(row[key] for key in identity))
        if baseline is None:
            raise ValueError(f"missing rho=0 baseline for {tuple(row[key] for key in identity)}")
        group_key = (
            row["algorithm"],
            row["train_seed"],
            row["split"],
            row["signal_accuracy"],
        )
        groups.setdefault(group_key, []).append(
            (
                row["twap_shortfall_advantage_bps"]
                - baseline["twap_shortfall_advantage_bps"],
                row["twap_pnl_advantage_bps"]
                - baseline["twap_pnl_advantage_bps"],
            )
        )
    results = []
    keys = ("algorithm", "train_seed", "split", "signal_accuracy")
    for key, values in groups.items():
        shortfall_values = np.array([value[0] for value in values], dtype=np.float64)
        pnl_values = np.array([value[1] for value in values], dtype=np.float64)
        results.append(
            {
                **dict(zip(keys, key)),
                "episodes": len(values),
                "signal_increment_bps": float(shortfall_values.mean()),
                "signal_increment_se_bps": _standard_error(shortfall_values),
                "signal_pnl_increment_bps": float(pnl_values.mean()),
                "signal_pnl_increment_se_bps": _standard_error(pnl_values),
            }
        )
    return results


def plot_signal_increment(rows: list[dict], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = [
        split
        for split in ("validation", "test")
        if any(row["split"] == split for row in rows)
    ]
    fig, axes = plt.subplots(1, len(splits), figsize=(7 * len(splits), 5), squeeze=False)
    for column, split in enumerate(splits):
        axis = axes[0, column]
        for algorithm in ("hail_baseline", *ALGORITHMS):
            points = [
                row
                for row in rows
                if row["split"] == split and row["algorithm"] == algorithm
            ]
            accuracies = sorted({row["signal_accuracy"] for row in points})
            means = [
                np.mean(
                    [
                        row["signal_increment_bps"]
                        for row in points
                        if row["signal_accuracy"] == accuracy
                    ]
                )
                for accuracy in accuracies
            ]
            axis.plot(accuracies, means, marker="o", label=algorithm)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(f"{split}: signal-use increment vs rho=0")
        axis.set_xlabel("controlled signal correlation")
        axis.set_ylabel("implementation-shortfall improvement, bps")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def _standard_error(values: np.ndarray) -> float:
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def _pair_mirrored_targets(records: list[dict]) -> list[dict]:
    """Average buy/sell outcomes on the same path before estimating uncertainty."""

    keys = (
        "algorithm",
        "train_seed",
        "split",
        "symbol",
        "date",
        "signal_mode",
        "signal_accuracy",
        "signal_replica",
    )
    groups: dict[tuple, list[dict]] = {}
    for record in records:
        groups.setdefault(tuple(record[key] for key in keys), []).append(record)
    paired = []
    numeric_fields = (
        "twap_shortfall_advantage_bps",
        "twap_pnl_advantage_bps",
    )
    for key, rows in groups.items():
        if {row["target_direction"] for row in rows} != {-1, 1} or len(rows) != 2:
            raise ValueError(f"missing mirrored target pair for {key}")
        paired.append(
            {
                **dict(zip(keys, key)),
                **{
                    field: float(np.mean([row[field] for row in rows]))
                    for field in numeric_fields
                },
            }
        )
    return paired
