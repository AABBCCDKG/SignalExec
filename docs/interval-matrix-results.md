# Five-interval locked comparison

## Contract

The comparison uses five Binance Spot symbols, May 2023 through September 2026
source archives, symbol-local raw-trade normalization, equal sampled-episode
budgets, three training seeds, and the same controlled signals and execution
costs. Training ends before 2025; validation is calendar 2025. The once-opened
locked comparison is May through August 2026.

All 45 learned models were frozen before the new test opened. Each interval's
merged test has 95,160 records, 13 policy instances, and 7,320 matched inputs.
Every target filled, signal hashes matched, and TWAP and `rho=0` identities were
exactly zero. Frozen PPO seed-0 replays reproduced all seven result files
byte-for-byte at every interval.

## Locked-test result

At `rho=0.8`, the table reports mean implementation-shortfall advantage versus
TWAP in basis points. Positive is better.

| Interval | HAIL | SAC mean | PPO mean | Memory-PPO mean |
|---|---:|---:|---:|---:|
| 1m | -2.763 | -0.795 | +3.670 | -0.033 |
| 3m | -0.448 | -1.017 | +1.831 | +0.610 |
| 5m | +2.684 | -2.617 | -0.050 | +3.479 |
| 15m | +7.619 | -2.924 | +8.149 | +0.186 |
| 30m | +13.568 | +6.682 | -0.693 | -0.154 |

The means do not imply reliability across seeds. PPO's 1m mean comes from two
strong seeds (`+6.220`, `-1.401`, `+6.192`); its 15m seeds are all positive but
widely dispersed (`+4.844`, `+18.944`, `+0.659`). Memory-PPO's 5m mean is driven
by one seed (`-0.980`, `+11.681`, `-0.266`). SAC turns positive at 30m but is
also dispersed (`-0.305`, `+9.289`, `+11.063`).

## Interpretation

Decision frequency materially changes the result. The handcrafted HAIL rule is
not universally robust: it trails TWAP at 1m and 3m in this test, then becomes
strong at 5m, 15m, and 30m. No learned architecture dominates every interval.
PPO is strongest on average at 1m and 15m; Memory-PPO leads at 5m; HAIL leads
at 30m. This supports treating interval choice as part of the execution policy,
not as a harmless resampling detail.

The signal remains controlled oracle information rather than a tradable
forecast. These results compare execution responses; they do not establish
alpha, profitability, realistic fills, or market-impact capacity.
