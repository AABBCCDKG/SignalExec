# HAIL PPO Reproduction

An independent reproduction and extension of Hudson River Trading AI Labs' 2026 intern project on reinforcement learning for optimal execution.

Source article: [Intern Spotlight: HRT AI Labs Project](https://www.hudsonrivertrading.com/hrtbeat/intern-spotlight-hrt-ai-labs-project-2026/)

## Goal

Investigate whether the article's qualitative findings hold on a transparent research testbed:

1. compare a hand-written execution rule, SAC, PPO, and memory-augmented PPO;
2. test every policy out of sample across different signal-accuracy levels;
3. report both trading PnL and target-position fill fraction;
4. investigate the under-filling observed in the article.

This repository does **not** claim to reconstruct HRT's private data, simulator, reward, or model architecture. Unknown details are explicit assumptions in [the reproduction specification](docs/reproduction-spec.md).

## Current milestone

The repository includes a synthetic simulator and a complete training/evaluation pilot:

- a one-day execution environment in which actions can only move toward a fixed target position;
- a noisy short-term return signal with configurable predictiveness;
- mark-to-market PnL, transaction costs, and a terminal under-fill penalty;
- the article's hand-written baseline translated into the environment's action space;
- tests for accounting, action constraints, timing incentives, and baseline behavior;
- SAC and PPO from Stable-Baselines3, plus LSTM PPO from SB3-Contrib;
- matched evaluation days, independent training seeds, and a signal-correlation sweep;
- TWAP, immediate-execution, and no-trade controls;
- a paired final-step completion intervention to measure its effect on PnL and fill;
- saved models, training logs, episode-level CSV, configuration, source hashes and plots.

The [first pilot results](docs/pilot-results.md) document unsuccessful execution
policies and the limitations of this synthetic control experiment.

## Run

```bash
uv sync --extra dev --extra rl
uv run --extra dev --extra rl pytest
uv run python scripts/run_baseline.py

# Fast end-to-end check (not evidence of convergence).
uv run --extra rl python scripts/run_experiment.py \
  --output artifacts/smoke --timesteps 512 --seeds 0 \
  --eval-episodes 4 --accuracies 0 0.8 --steps 8

# Pilot: three training seeds per algorithm, five evaluation correlations.
uv run --extra rl python scripts/run_experiment.py --output artifacts/iid-pilot
```

Use Python 3.11 or later. On networks using a system-installed CA, add
`--system-certs` to `uv` if installation reports `UnknownIssuer`.
Each run requires a new output directory and refuses to overwrite previous results.
`artifacts/` is ignored by Git; model files and raw runs stay local.

The current IID market is a pipeline test and an execution control problem. Its
history has no extra information about future returns, so a recurrent policy has
no additional return-prediction advantage by construction. The synthetic signal
is a controlled noisy oracle, not a trained, tradable forecast. Results do not
establish HRT's reported ordering or performance on real market data. The next
research stage needs chronological historical-data splits and an explicit test
of what information memory can recover.
