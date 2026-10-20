# Raw-trade observation extension

This stage tests whether richer market state helps execution policies use the
same controlled signals. It does not change target directions, forecast quality,
execution costs, completion rules, or model hyperparameters.

## Data and features

- Thirteen checksum-verified Binance Spot BTCUSDT raw-trade archives, December
  2024 through December 2025.
- Approximately 12 GB of compressed source data.
- Streamed directly from zip files into 114,048 five-minute buckets; the source
  CSVs were not extracted to disk.
- Compact cache: `data/binance/BTCUSDT-trades-5m.npz`.
- Training: January 1 through April 29, 2025.
- Validation: May 1 through October 30, 2025.
- New test: November 1 through December 30, 2025.

Each observation receives the previous completed five-minute bucket's:

1. log quote volume;
2. log raw-trade count;
3. aggressive buy/sell quote-volume imbalance;
4. log average trade size;
5. raw-trade price range;
6. first-to-last trade return;
7. largest-trade share;
8. aggressor-side switch rate.

Features are standardized using training dates only and clipped to `[-8, 8]`.
The environment exposes no trade from the current or future decision window.

## Frozen enriched run

```bash
uv run --extra rl python scripts/run_controlled_experiment.py \
  --output artifacts/controlled-rawtrades-paired-20261017-v2 \
  --evaluation-splits validation test
```

- 37,908 evaluated episodes;
- zero signal-hash mismatches;
- zero incomplete targets;
- 34 passing tests;
- two raw-trade formal runs produced byte-identical episode, first-difference
  and DiD CSV files.

## Locked-test result at rho=0.8

Positive TWAP advantage means lower implementation shortfall than matched TWAP.
Signal increment is the improvement relative to the same policy at `rho=0`.
RL rows average three training seeds.

| Policy | TWAP advantage | Signal increment |
|---|---:|---:|
| HRT-inspired rule | +0.347 bps | +4.278 bps |
| SAC | +0.698 bps | +1.273 bps |
| PPO | -2.619 bps | -1.728 bps |
| RecurrentPPO | -1.001 bps | -0.169 bps |

SAC is the first learned policy whose three-seed mean beats TWAP in the current
test period. The mean is not seed-robust: its per-seed TWAP advantages are about
`-0.245`, `-0.380`, and `+2.721 bps`.

## Same-date no-extra-feature ablation

The ablation uses identical November-December dates, signals, targets, costs,
training steps and seeds, changing only the eight raw-trade observation fields.

| Policy | No features | Raw trades | Raw-minus-none |
|---|---:|---:|---:|
| SAC | -0.801 | +0.698 | +1.499 bps |
| PPO | -0.003 | -2.619 | -2.615 bps |
| RecurrentPPO | -0.566 | -1.001 | -0.436 bps |

The raw-minus-none effect on SAC is positive for all three seeds at test
`rho=0.8`: approximately `+0.154`, `+1.007`, and `+3.336 bps`. The same feature
expansion hurts two of three PPO seeds and has mixed effects on RecurrentPPO.
Validation has the same aggregate direction: SAC improves by `+0.663 bps`, PPO
deteriorates by `-2.222 bps`, and RecurrentPPO is nearly unchanged (`-0.010 bps`).

## Interpretation and limits

- Richer state can help, but algorithms do not consume it equally. SAC's replay
  buffer reuses observations, while the PPO-family models receive the same small
  32,768-step budget despite a larger state space.
- The HRT-inspired rule is identical across enriched and ablation runs because
  it does not read the added fields; this is a useful contamination check.
- A positive three-seed SAC mean is not a stable production result when two
  seeds still trail TWAP.
- The simulator still assumes immediate fills and a fixed impact formula. Raw
  trades improve the state but do not reconstruct the historical order book or
  counterfactual fills.
- The post-test no-feature run is an attribution ablation, not a new locked-test
  claim, and no hyperparameter was changed after observing the enriched result.

These results support further investigation of data efficiency and state usage;
they do not establish alpha, profitability, or live execution superiority.
