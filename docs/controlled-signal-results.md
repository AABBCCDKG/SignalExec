# Controlled-signal locked results

The corrected paired experiment completed twice with byte-identical core output.
The first formal directory without a version suffix is retained only as an audit
record: it used different forecast-noise draws across `rho` values and estimated
uncertainty before pairing mirrored targets, so it is not valid evidence.

## Frozen run

```bash
uv run --extra rl python scripts/run_controlled_experiment.py \
  --output artifacts/controlled-signal-paired-20261017-v3 \
  --evaluation-splits validation test
```

- 119 training days, 122 validation days, and 60 locked test days.
- Handcrafted rule, TWAP, immediate, wait, SAC, PPO, and RecurrentPPO.
- SAC/PPO/RecurrentPPO: three seeds and 32,768 environment steps each.
- Five controlled accuracies plus signal-off; mirrored buy/sell targets.
- 28,392 evaluated episodes, all fully completed.
- Zero signal-hash mismatches across matched policies.
- `episodes.csv`, `paired_advantages.csv`, and
  `difference_in_differences.csv` match the preceding corrected run exactly.

Test empirical signal correlations were `0.0068, 0.2090, 0.4097, 0.6085,
0.8052` for requested `rho=0, 0.2, 0.4, 0.6, 0.8`.

## First difference: execution quality without signal attribution

The table reports mean implementation-shortfall advantage over matched TWAP.
Positive is better than TWAP; negative is worse. RL entries average three
training seeds.

| Policy | Signal off | rho=0 | rho=0.4 | rho=0.8 |
|---|---:|---:|---:|---:|
| HRT-inspired rule | -0.978 | 0.351 | 0.344 | -2.099 |
| SAC | -0.472 | -0.502 | -0.533 | -0.592 |
| PPO | -0.628 | -0.622 | -0.527 | -0.432 |
| RecurrentPPO | -0.964 | -0.999 | -0.889 | -0.724 |

None of the learned policies beats TWAP on average in this bar-level replay.
The controlled signal can still improve a policy relative to its own `rho=0`
behavior, which is measured by the second difference below.

## Second difference: incremental value from signal quality

Positive values mean that adding return information to the same forecast noise
improved execution relative to the policy's own `rho=0` result.

### Locked test

| Policy | rho=0.2 | rho=0.4 | rho=0.6 | rho=0.8 |
|---|---:|---:|---:|---:|
| HRT-inspired rule | -0.773 | -0.007 | -2.966 | -2.450 |
| SAC | -0.011 | -0.031 | -0.060 | -0.090 |
| PPO | 0.047 | 0.095 | 0.143 | 0.190 |
| RecurrentPPO | 0.055 | 0.110 | 0.174 | 0.275 |

The locked-test mean ordering at `rho=0.8` is therefore:

```text
HRT-inspired rule < SAC < PPO < RecurrentPPO
```

This matches the public qualitative ordering when the outcome is specifically
the incremental use of controlled signal information. It does not establish the
same ordering in absolute execution quality.

## Robustness limits

- Validation puts PPO (`+0.227 bps`) above RecurrentPPO (`+0.092 bps`) at
  `rho=0.8`, so the top-two ordering is not stable across time splits.
- SAC is seed-sensitive: one seed uses the signal positively while another
  deteriorates as `rho` rises.
- PPO is also seed-sensitive: one seed is nearly invariant to the signal while
  another gains about `0.506 bps` at test `rho=0.8`.
- RecurrentPPO's three test increments at `rho=0.8` are approximately `0.714`,
  `0.107`, and `0.006 bps`; all are nonnegative, but their magnitude varies.
- The handcrafted rule's negative means have large day-level uncertainty and do
  not establish that better signals systematically harm the rule.
- Only one deterministic noise replica and one instrument were used.

The result is evidence that PPO-family policies can respond to controlled signal
quality, not evidence of alpha, profitability, live execution advantage, or a
complete reproduction of HRT's private simulator.
