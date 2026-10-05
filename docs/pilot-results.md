# First synthetic pilot

The training and evaluation pipeline runs end to end. This pilot **does not
reproduce HRT's reported algorithm ranking**: SAC trades almost immediately,
while PPO and LSTM PPO often fail to build the target position. A lower loss
achieved by taking little or no position is not evidence of better execution.

## Configuration and evidence

```bash
uv run --extra rl python scripts/run_experiment.py --output artifacts/iid-pilot-20261004
```

- Nine models: SAC, PPO and LSTM PPO, each with training seeds 0, 1 and 2.
- Exactly 32,768 environment steps per model; 294,912 training steps in total.
- 200 held-out synthetic price paths, reused across five signal correlations
  and all policies. Four controls plus nine trained policies, each evaluated
  with and without final-step completion, produce 26,000 episode records.
  These are repeated evaluations, **not 26,000 independent market days**.
- All nine models, training logs, `metadata.json`, `episodes.csv`,
  `summary.csv`, `summary.json` and `comparison.png` are saved in that directory.
  Raw artifacts are local and Git-ignored; the command regenerates them.
- CPU with one PyTorch thread; summed training time was 161.28 seconds,
  excluding evaluation and startup. See the manifest for package versions and
  source SHA-256 hashes; this run used uncommitted phase-two source files.

## Results at signal correlation 0.4

PnL is after costs, in basis points of target notional. Fill is the average
fraction executed. RL entries average the three training seeds; controls run
once on the same 200 days. These means alone do not establish significance.

| Policy | Natural PnL, bps | Mean fill | PnL with forced completion, bps |
|---|---:|---:|---:|
| HAIL rule | -6.2933 | 99.9077% | -6.3082 |
| TWAP | -13.0762 | 100% | -13.0762 |
| Immediate execution | -15.7135 | 100% | -15.7135 |
| Wait | 0 | 0% | -1.1094 |
| SAC | -15.5566 | 100% | -15.5566 |
| PPO | -5.0046 | 33.3333% | -5.7442 |
| LSTM PPO | -4.9165 | 33.6721% | -5.6523 |

PPO's per-seed mean PnL ranges from -15.0138 to 0 bps; LSTM PPO's ranges
from -14.6124 to 0 bps. Both have only one of three seeds reliably completing
the target. The seed variation is substantial. For the HAIL rule, the
day-level standard error is 4.966 bps; it is about 12 bps for immediate
execution and SAC. These errors are conditional on each fixed policy, not
confidence intervals incorporating training randomness.

All forced-completion evaluations finish their targets within 1e-6 by
construction. This verifies an operational completion rule. It does not show
that training learned to satisfy the constraint, and its PnL includes both
additional trading costs and exposure to the last price move.

## Interpretation and next experiment

The toy market has IID returns, random target direction, a synthetic noisy
oracle signal, instantaneous fills and assumed trading costs. It cannot test
the article's real-data claims or the value of recovering hidden predictive
state from history. The low budget also does not establish convergence.

The next questions are why policies collapse to near-immediate execution or
inaction, whether the completion incentive is sufficient, and how much longer
training changes these behaviors. Penalty sweeps, action/observation scaling,
and a training-time hard completion constraint are hypotheses to test; this
pilot has not identified one confirmed cause. Compare each proposed change
against the same controls and inspect execution timing, not just aggregate PnL.

Because these pilot results have now been inspected, use these seeds for
development and reserve a fresh test set before tuning. Historical data,
causal prediction targets and an explicit partial-observability experiment
remain necessary before describing this as a substantive reproduction of the
HAIL study.
