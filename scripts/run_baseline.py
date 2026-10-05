"""Run the hand-written baseline over seeded synthetic trading days."""

from statistics import mean, pstdev

import numpy as np

from hail_ppo_reproduction import (
    ExecutionEnv,
    HailBaselinePolicy,
    SyntheticMarketConfig,
    make_synthetic_day,
)


def run_episode(env: ExecutionEnv, policy: HailBaselinePolicy, seed: int) -> dict:
    env.reset(seed=seed)
    terminated = False
    info = {}
    while not terminated:
        day = env.day
        assert day is not None
        action = policy.action(
            directed_signal=np.sign(day.target_position) * day.signals[env.step_index],
            remaining_fraction=1.0 - env.fill_fraction,
            steps_remaining=env.steps_remaining,
            total_steps=len(day.signals),
            return_volatility=day.return_volatility,
        )
        _, _, terminated, _, info = env.step(action)
    return info


def main() -> None:
    market = SyntheticMarketConfig(signal_accuracy=0.4)
    env = ExecutionEnv(lambda rng: make_synthetic_day(rng, market))
    policy = HailBaselinePolicy()
    results = [run_episode(env, policy, seed) for seed in range(200)]
    pnls = [result["cumulative_pnl"] for result in results]
    fills = [result["fill_fraction"] for result in results]
    print(f"episodes:  {len(results)}")
    print(f"mean PnL:  {mean(pnls): .6f}")
    print(f"PnL stdev: {pstdev(pnls): .6f}")
    print(f"mean fill: {mean(fills): .4f}")


if __name__ == "__main__":
    main()
