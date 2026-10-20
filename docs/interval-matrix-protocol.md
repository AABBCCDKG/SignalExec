# Five-interval preregistered comparison

## Question

Does the reliability ordering between HAIL, SAC, PPO, and Memory-PPO change
when the same five symbols and historical period are sampled at `1m`, `3m`,
`5m`, `15m`, or `30m` decisions?

This is a new interval comparison. The previously opened March-April 2026 test
must not be reused as an untouched claim for newly trained intervals.

## Frozen data split

- Symbols: `BTCUSDT`, `ETHUSDT`, `BNBUSDT`, `SOLUSDT`, `XRPUSDT`.
- Lookback begins May 2023; training begins June 1, 2023.
- Training ends before January 1, 2025.
- Formal validation is calendar 2025 only.
- January-April 2026 is observed development history and is not a new test.
- The new locked interval-comparison test is May 1 through August 31, 2026.
- September 2026 price bars supply the right boundary; raw-trade features end
  with August 2026.

All intervals use the same controlled signals, target directions, costs,
forced completion, symbol-local feature normalization, three training seeds,
and algorithm architectures. The old single-BTC five-minute prototype is not
part of this comparison.

## Equal-episode training budgets

The three-minute budget represents about 3,277 sampled episodes. Other budgets
are rounded to a multiple of 256 while preserving approximately the same number
of full trading days:

| Interval | Decisions per day | Steps per model | Approx. episodes |
|---|---:|---:|---:|
| 1m | 1,440 | 4,718,592 | 3,276.8 |
| 3m | 480 | 1,572,864 | 3,276.8 |
| 5m | 288 | 943,616 | 3,276.4 |
| 15m | 96 | 314,624 | 3,277.3 |
| 30m | 48 | 157,184 | 3,274.7 |

The existing frozen three-minute models are reused. The other intervals train
all three algorithms and seeds once. Training timesteps are not changed after
validation begins.

## Reporting contract

- Report every seed; never select the best seed.
- Primary metric: implementation-shortfall advantage versus matched TWAP.
- Signal-use metric: the same advantage minus its `rho=0` value.
- Validation tables are filtered strictly at `date < 2026-01-01`.
- Open May-August 2026 only after all interval caches, training jobs, validation
  gates, and deterministic replay checks complete.
- Dashboard tabs remain result-free until their locked comparison artifact is
  complete and numerically cross-checked.
