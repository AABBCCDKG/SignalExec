# Five-symbol three-minute result

## Scope

The final study pools `BTCUSDT`, `ETHUSDT`, `BNBUSDT`, `SOLUSDT`, and
`XRPUSDT` with three-minute decisions and causal raw-trade features. Models use
1,572,864 environment steps and seeds 0, 1, and 2. Training ends before January
2025; formal validation is calendar 2025; the once-opened locked test is March
through April 2026.

The signal is controlled oracle information, not a tradable forecast. The
primary outcome is matched implementation-shortfall advantage over TWAP. The
signal increment subtracts the same policy's `rho=0` advantage.

## Integrity gates

- 180 of 180 raw archives and 180 of 180 three-minute cache parts completed.
- Every final cache has 526,080 ordered buckets, eight finite features, and 36
  source signatures.
- The merged validation has 284,700 records, 13 policy instances, and 21,900
  matched inputs after enforcing the calendar-2025 boundary.
- The locked test has 46,800 records, 13 policy instances, and 3,600 matched
  inputs.
- Every target filled; signal hashes matched across policies; TWAP first
  differences and `rho=0` second differences were exactly zero.
- A frozen PPO validation replay reproduced seven result files byte-for-byte;
  its episode SHA256 was
  `0b7bf6c1514fb54f88516cf31ed607993b6488fe118a9c9b6bc17fceb37e5d60`.

## Formal validation

At `rho=0.8`, positive values mean lower shortfall than TWAP:

| Method | TWAP advantage by seed (bps) | Mean | Signal-increment mean |
|---|---:|---:|---:|
| HAIL baseline | `+1.985` | +1.985 | +2.612 |
| SAC | `-2.302 / -1.415 / -2.144` | -1.954 | -1.489 |
| PPO | `+13.447 / -0.986 / -1.808` | +3.551 | +4.492 |
| Memory-PPO | `+1.755 / +8.081 / -0.979` | +2.952 | +3.650 |

PPO and Memory-PPO have positive means but remain seed-sensitive. SAC does not
beat TWAP on any validation seed at this signal quality. No result selects a
favorable seed.

## Locked test

At `rho=0.8`:

| Method | TWAP advantage by seed (bps) | Mean | Signal-increment mean |
|---|---:|---:|---:|
| HAIL baseline | `+0.845` | +0.845 | +1.288 |
| SAC | `-2.206 / +0.477 / -2.599` | -1.443 | -0.928 |
| PPO | `+7.986 / -0.986 / -2.555` | +1.482 | +2.484 |
| Memory-PPO | `+0.615 / +3.936 / -0.965` | +1.195 | +2.203 |

The learned-policy means can exceed the handcrafted baseline, but neither PPO
nor Memory-PPO is reliably better across seeds. PPO has one strong, one
signal-insensitive, and one adverse seed. Memory-PPO has two positive seeds and
one collapsed seed. SAC's mean signal increment is negative at every positive
`rho` in the locked test.

The defensible conclusion is therefore about reliability, not the highest
mean: the HAIL baseline uses better signals consistently and beats TWAP at high
signal quality, while the more complex policies still have material
optimization instability. These execution results do not establish alpha,
profitability, realistic fills, or market-impact capacity.

## Execution record

The nine training jobs ran independently on CPU, one task per algorithm/seed,
with one PyTorch thread each. PPO trained in about six minutes per seed, SAC in
about 19.7 minutes, and Memory-PPO in about 23.7 minutes under concurrent load.
The locked test loaded those exact frozen models and performed no retraining.
