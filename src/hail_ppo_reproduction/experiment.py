"""Train and evaluate all policies on matched, held-out synthetic trading days."""

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from hail_ppo_reproduction import (
    ExecutionConfig,
    ExecutionEnv,
    HailBaselinePolicy,
    SyntheticMarketConfig,
    make_synthetic_day,
)

ALGORITHMS = ("sac", "ppo", "memory_ppo")
CONTROLS = ("hail_baseline", "twap", "immediate", "wait")


@dataclass(frozen=True)
class ExperimentConfig:
    timesteps: int = 32768
    train_seeds: tuple[int, ...] = (0, 1, 2)
    eval_episodes: int = 200
    eval_seed_start: int = 2_000_000_000
    eval_batch_size: int = 32
    accuracies: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8)
    train_accuracy: float = 0.4
    steps: int = 78

    def __post_init__(self) -> None:
        if self.timesteps < 256 or self.timesteps % 256:
            raise ValueError("timesteps must be a positive multiple of 256")
        if self.eval_episodes < 2 or self.eval_batch_size < 1:
            raise ValueError("need at least two evaluation episodes and a positive batch size")
        if not self.train_seeds or len(set(self.train_seeds)) != len(self.train_seeds):
            raise ValueError("training seeds must be nonempty and unique")
        if any(seed < 0 or seed >= 1_000_000 for seed in self.train_seeds):
            raise ValueError("training seeds must lie in [0, 1000000)")
        if self.eval_seed_start < 1_000_000_000:
            raise ValueError("evaluation seed namespace must be separate from training")
        if not self.accuracies or len(set(self.accuracies)) != len(self.accuracies):
            raise ValueError("evaluation accuracies must be nonempty and unique")
        for accuracy in (*self.accuracies, self.train_accuracy):
            SyntheticMarketConfig(steps=self.steps, signal_accuracy=accuracy)


def create_env(market: SyntheticMarketConfig, execution: ExecutionConfig) -> ExecutionEnv:
    return ExecutionEnv(lambda rng: make_synthetic_day(rng, market), execution)


def build_model(algorithm: str, env: ExecutionEnv, seed: int, timesteps: int) -> Any:
    # Keep the simulator usable without the optional PyTorch dependencies.
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO, SAC

    common = dict(env=env, seed=seed, device="cpu", verbose=0, gamma=1.0,
                  learning_rate=3e-4)
    if algorithm == "sac":
        return SAC("MlpPolicy", **common, buffer_size=min(100_000, timesteps),
                   learning_starts=min(1024, timesteps // 4), batch_size=64,
                   train_freq=4, gradient_steps=1,
                   policy_kwargs={"net_arch": [64, 64]})
    on_policy = dict(n_steps=256, batch_size=64, n_epochs=4)
    if algorithm == "ppo":
        return PPO("MlpPolicy", **common, **on_policy,
                   policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}})
    if algorithm == "memory_ppo":
        return RecurrentPPO("MlpLstmPolicy", **common, **on_policy,
                            policy_kwargs={"lstm_hidden_size": 32,
                                           "net_arch": {"pi": [64], "vf": [64]}})
    raise ValueError(f"unknown algorithm: {algorithm}")


def train_model(algorithm: str, seed: int, config: ExperimentConfig,
                execution: ExecutionConfig, directory: Path) -> tuple[Any, dict]:
    from stable_baselines3.common.logger import configure

    market = SyntheticMarketConfig(steps=config.steps, signal_accuracy=config.train_accuracy)
    env = create_env(market, execution)
    model = build_model(algorithm, env, seed, config.timesteps)
    # Parameter initialization and market simulation use separate RNG streams.
    model.get_env().seed(10_000_000 + seed)
    directory.mkdir(parents=True, exist_ok=False)
    model.set_logger(configure(str(directory), ["csv"]))
    started = perf_counter()
    model.learn(total_timesteps=config.timesteps)
    elapsed = perf_counter() - started
    model.save(directory / "model.zip")
    record = {"algorithm": algorithm, "train_seed": seed,
              "requested_timesteps": config.timesteps, "actual_timesteps": model.num_timesteps,
              "training_seconds": elapsed, "market": asdict(market),
              "execution": asdict(execution), "model": str(directory / "model.zip")}
    env.close()
    return model, record


def control_actions(name: str, observations: np.ndarray, steps_remaining: int,
                    total_steps: int, volatility: float) -> np.ndarray:
    if name == "hail_baseline":
        policy = HailBaselinePolicy()
        return np.array([policy.action(
            directed_signal=float(obs[0]), remaining_fraction=float(obs[1]),
            steps_remaining=steps_remaining, total_steps=total_steps,
            return_volatility=volatility,
        ) for obs in observations], dtype=np.float32)
    if name == "twap":
        return np.full((len(observations), 1), 1.0 / steps_remaining, dtype=np.float32)
    if name == "immediate":
        return np.ones((len(observations), 1), dtype=np.float32)
    if name == "wait":
        return np.zeros((len(observations), 1), dtype=np.float32)
    raise ValueError(f"unknown control: {name}")


def evaluate_policy(name: str, model: Any, train_seed: int | None,
                    config: ExperimentConfig, execution: ExecutionConfig,
                    accuracy: float, force_completion: bool = False) -> list[dict]:
    """Batch independent episodes, carrying LSTM states only within each day.

    Only observations reach the policy; the simulator's future prices are never
    passed to predict(). All methods receive identical seeds, prices and signals.
    """
    market = SyntheticMarketConfig(steps=config.steps, signal_accuracy=accuracy)
    records = []
    for offset in range(0, config.eval_episodes, config.eval_batch_size):
        size = min(config.eval_batch_size, config.eval_episodes - offset)
        seeds = range(config.eval_seed_start + offset, config.eval_seed_start + offset + size)
        envs = [create_env(market, execution) for _ in seeds]
        observations = np.stack([env.reset(seed=seed)[0] for env, seed in zip(envs, seeds)])
        states = None
        episode_starts = np.ones(size, dtype=bool)
        total_rewards = np.zeros(size)
        total_costs = np.zeros(size)
        for step in range(market.steps):
            if model is None:
                actions = control_actions(name, observations, market.steps - step,
                                          market.steps, market.return_volatility)
            else:
                actions, states = model.predict(observations, state=states,
                                                episode_start=episode_starts,
                                                deterministic=True)
            if force_completion and step == market.steps - 1:
                actions = np.ones((size, 1), dtype=np.float32)
            transitions = [env.step(action) for env, action in zip(envs, actions)]
            observations = np.stack([item[0] for item in transitions])
            total_rewards += [item[1] for item in transitions]
            total_costs += [item[4]["trading_cost"] for item in transitions]
            episode_starts = np.array([item[2] or item[3] for item in transitions])
        for index, (env, seed) in enumerate(zip(envs, seeds)):
            day = env.day
            assert day is not None
            notional = abs(day.target_position) * day.prices[0]
            records.append({"algorithm": name, "train_seed": train_seed,
                            "eval_seed": seed, "signal_accuracy": accuracy,
                            "completion": "forced" if force_completion else "natural",
                            "pnl": env.cumulative_pnl,
                            "pnl_bps": 10_000.0 * env.cumulative_pnl / notional,
                            "fill_fraction": env.fill_fraction,
                            "filled": bool(env.fill_fraction >= 1.0 - 1e-6),
                            "episode_reward": float(total_rewards[index]),
                            "cost_bps": 10_000.0 * total_costs[index] / notional})
            env.close()
    return records


def summarize(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    keys = ("algorithm", "train_seed", "signal_accuracy", "completion")
    for record in records:
        groups.setdefault(tuple(record[key] for key in keys), []).append(record)
    summaries = []
    for key, rows in groups.items():
        pnls = np.array([row["pnl_bps"] for row in rows])
        summaries.append({**dict(zip(keys, key)), "episodes": len(rows),
                          "mean_pnl_bps": float(pnls.mean()),
                          "pnl_std_bps": float(pnls.std(ddof=1)),
                          "pnl_se_bps": float(pnls.std(ddof=1) / np.sqrt(len(rows))),
                          "mean_fill": float(np.mean([row["fill_fraction"] for row in rows])),
                          "completion_rate": float(np.mean([row["filled"] for row in rows])),
                          "mean_reward": float(np.mean([row["episode_reward"] for row in rows])),
                          "mean_cost_bps": float(np.mean([row["cost_bps"] for row in rows]))})
    return summaries


def plot_results(summaries: list[dict], destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for row, completion in enumerate(("natural", "forced")):
        for name in (*CONTROLS, *ALGORITHMS):
            points = [s for s in summaries if s["algorithm"] == name and s["completion"] == completion]
            if not points:
                continue
            accuracies = sorted({s["signal_accuracy"] for s in points})
            for col, metric in enumerate(("mean_pnl_bps", "mean_fill")):
                per_seed = [[s[metric] for s in points if s["signal_accuracy"] == a] for a in accuracies]
                line, = axes[row, col].plot(accuracies, [np.mean(v) for v in per_seed], label=name)
                if any(len(v) > 1 for v in per_seed):
                    axes[row, col].fill_between(accuracies, [min(v) for v in per_seed],
                                                [max(v) for v in per_seed], alpha=0.12,
                                                color=line.get_color())
                axes[row, col].set_title(f"{completion}: {'PnL (bps)' if col == 0 else 'fill fraction'}")
                axes[row, col].grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    for axis in axes[1]:
        axis.set_xlabel("Synthetic one-step signal correlation")
    fig.suptitle("Synthetic IID pilot — shading is range across training seeds, not a confidence interval")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
