# HAIL PPO Reproduction

An independent reproduction and extension of Hudson River Trading AI Labs' 2026 intern project on reinforcement learning for optimal execution.

Source article: [Intern Spotlight: HRT AI Labs Project](https://www.hudsonrivertrading.com/hrtbeat/intern-spotlight-hrt-ai-labs-project-2026/)

## Goal

Reproduce the article's qualitative result on a transparent research testbed:

1. compare a hand-written execution rule, SAC, PPO, and memory-augmented PPO;
2. test every policy out of sample across different signal-accuracy levels;
3. report both trading PnL and target-position fill fraction;
4. investigate the under-filling observed in the article.

This repository does **not** claim to reconstruct HRT's private data, simulator, reward, or model architecture. Unknown details are explicit assumptions in [the reproduction specification](docs/reproduction-spec.md).

## Current milestone

Milestone 1 establishes the research contract and a deterministic synthetic testbed. It includes:

- a one-day execution environment in which actions can only move toward a fixed target position;
- a noisy short-term return signal with configurable predictiveness;
- mark-to-market PnL, transaction costs, and a terminal under-fill penalty;
- the article's hand-written baseline translated into the environment's action space;
- tests for accounting, action constraints, timing incentives, and baseline behavior.

## Run

```bash
uv sync --extra dev
uv run pytest
uv run python scripts/run_baseline.py
```

RL dependencies will be added when the simulator contract is stable, so early results cannot be confused with framework defaults.
