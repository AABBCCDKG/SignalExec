# Signal-to-Execution RL research log

This is the chronological source of truth for hypotheses, experiment changes,
negative results, test-set boundaries, and reproducibility evidence. It is
written so a future paper can distinguish facts observed in this repository
from assumptions needed because HRT's internal implementation is not public.

## Research question

Given an externally supplied target position and a stream of short-horizon
return forecasts, can a learned execution policy use those forecasts better than
a handcrafted rule or TWAP while still completing the same target?

The project studies execution conditional on a target. It does not claim to
build the upstream alpha model or reconstruct HRT's private simulator.

## Public facts and reproduction assumptions

The HRT article publicly describes a long-horizon forecast setting the target,
short-horizon forecasts informing execution timing, a handcrafted baseline,
SAC, PPO, memory-augmented PPO, historical simulation, a signal-accuracy sweep,
PnL, and fill. It does not publish the dataset, state/action definitions, reward,
costs, model architecture, hyperparameters, exact memory mechanism, or seeds.

Every result below therefore belongs to the explicit assumptions in this repo.
Matching HRT's qualitative ordering is an empirical question, not a requirement
that justifies tuning until the desired answer appears.

## Experiment chronology

### Stage 1 — synthetic IID simulator

**Question.** Does the basic environment, accounting, baseline, training and
matched evaluation pipeline run end to end?

**Design.** IID log returns, a noisy one-step oracle signal with configurable
correlation, random target direction, immediate fills and assumed linear costs.

**Observed result.** The public ordering was not reproduced. At signal
correlation `0.4`, SAC filled about 100%, PPO about 33.33%, and LSTM PPO about
33.67%. Low PnL from PPO-family models could not be interpreted as good
execution because most of the target was left unfilled.

**Interpretation.** Completion and raw PnL must be reported together. The IID
process contains no historical information for memory to recover.

**Evidence.** `docs/pilot-results.md` and the reproducible synthetic command in
the README. The 200 inspected evaluation seeds are development data.

### Stage 2 — causal Binance historical forecasts

**Hypothesis.** Real price paths and causal features may provide enough temporal
structure for PPO and recurrent PPO to learn a meaningful policy.

**Design.** Checksum-verified BTCUSDT five-minute klines; train-only Ridge
forecasts; chronological train/validation/test splits; the same execution
environment and three RL algorithms.

**Observed result.** The locked short-horizon signal had correlation `-0.0039`
and direction accuracy `49.08%`. SAC behaved close to immediate execution, PPO
was seed-dependent, and RecurrentPPO almost never traded. The HRT ordering was
not reproduced.

**Interpretation.** This mixed two questions: forecast construction and signal
execution. An execution policy cannot recover information absent from the input
signal. July-August 2025 was opened and cannot support later tuning.

**Evidence.** `docs/historical-reproduction.md`, `docs/binance-results.md`, and
`artifacts/binance-btcusdt-5m-reproduction-20261016-v2`.

### Stage 3 — controlled signal and paired differences

**Hypothesis.** Holding forecast quality fixed and removing market-direction
luck will isolate how each policy uses signal information.

**Design.** Real Binance price paths; mirrored buy/sell targets; common forecast
noise across policies and correlation levels; signal-off plus `rho` values
`0, 0.2, 0.4, 0.6, 0.8`; mandatory final completion. The first difference is
implementation-shortfall advantage over matched TWAP. The second difference
subtracts the same policy's matched `rho=0` advantage.

**Protocol correction.** The first formal output used different noise draws at
different `rho` values and estimated uncertainty before pairing buy/sell
targets. It is retained as an audit record but is not evidence. The corrected
implementation reuses noise and averages mirrored targets before standard
errors.

**Observed result.** On September-October 2025 at `rho=0.8`, signal increments
were approximately `-2.450` bps for the handcrafted rule, `-0.090` for SAC,
`+0.190` for PPO, and `+0.275` for RecurrentPPO. That incremental ordering
matches HRT qualitatively, but every learned policy remained worse than TWAP in
absolute shortfall. Validation placed PPO above RecurrentPPO.

**Interpretation.** PPO-family policies can respond to controlled signal
quality, but the ordering is not stable enough to claim a robust reproduction.

**Evidence.** `docs/controlled-signal-protocol.md`,
`docs/controlled-signal-results.md`, and two byte-identical corrected runs.

### Stage 4 — raw-trade state extension

**Hypothesis.** The five-dimensional state is too poor for learned execution to
identify liquidity and flow conditions that a fixed schedule ignores.

**Design.** Thirteen checksum-verified BTCUSDT Spot raw-trade archives, December
2024 through December 2025, totaling about 12 GB compressed. A streaming parser
aggregates 114,048 five-minute buckets without extracting the source CSVs. Eight
strictly lagged features cover quote volume, trade count, aggressor imbalance,
mean trade size, price range, trade return, largest-trade share and aggressor
side-switch rate. Training-date statistics standardize the state.

**Observed result.** On the new November-December 2025 test at `rho=0.8`, SAC's
three-seed mean TWAP advantage changed from `-0.801` bps without extra features
to `+0.698` with raw-trade features, a `+1.499` bps effect. The raw-minus-none
effect was positive for every SAC seed (`+0.154`, `+1.007`, `+3.336`), although
only one enriched SAC seed beat TWAP absolutely. PPO deteriorated by `-2.615`
bps and RecurrentPPO by `-0.436` under the unchanged 32,768-step budget.

**Interpretation.** Richer state contains useful information, but algorithms do
not consume it equally. Data efficiency or optimization may limit the
PPO-family models. More features alone are not a universal improvement.

**Evidence.** `docs/microstructure-results.md`, the 12 GB local dataset,
`data/binance/BTCUSDT-trades-5m.npz`, same-date no-feature ablation, and two
byte-identical raw-trade formal runs.

### Stage 5 — training-budget sweep

**Hypothesis.** PPO and Memory-PPO deteriorated after the state expansion because
32,768 environment steps are insufficient for the higher-dimensional problem.

**Fixed validation design.** Raw-trade state, May-October 2025 validation only,
three training seeds, unchanged models/costs/signals/targets. Budgets are
32,768, 131,072 and 524,288 steps. November-December is already observed and is
not used for budget selection. January-February 2026 is reserved for a later
one-time test after choosing the budget.

**Observed 32k → 131k result at `rho=0.8`.**

| Algorithm | 32k TWAP advantage | 131k TWAP advantage | Seed SD change |
|---|---:|---:|---:|
| SAC | +0.077 | -0.398 | 0.784 → 0.836 |
| PPO | -2.654 | -0.599 | 3.042 → 0.449 |
| RecurrentPPO | -0.730 | -0.981 | 0.367 → 0.004 |

PPO's signal increment improved from `-1.964` to `+0.216` bps. This supports
the insufficient-samples hypothesis for PPO. RecurrentPPO instead converged
more consistently to a signal-insensitive bad policy, so additional samples
alone had not solved its failure at 131k. SAC did not improve.

**Observed 524k result at `rho=0.8`.**

| Algorithm | TWAP advantage by seed | Mean | Signal increment mean |
|---|---|---:|---:|
| SAC | `+1.778 / -0.057 / +1.874` | +1.198 | +1.970 |
| PPO | `-1.207 / -1.258 / -0.662` | -1.043 | -0.120 |
| RecurrentPPO | `-0.980 / -0.978 / +8.478` | +2.173 | +1.791 |

SAC shows the clearest positive learning trend: every seed uses the signal
positively and two of three beat TWAP. PPO is non-monotonic; 131k is more stable
and better on average than 524k, so insufficient samples were only part of its
problem. RecurrentPPO is bimodal: two seeds remain in the same collapsed policy
while one seed escapes and becomes very strong. Its positive mean is therefore
not a stable result.

**Budget decision before opening a new test.** Use 524,288 environment steps for
all algorithms. This preserves an equal-step comparison and tests the maximum
pre-specified resource level; it does not select a separate favorable budget per
algorithm. January-February 2026 remains untouched at the time of this decision.

**One-time January-February 2026 test.** All gates passed on 58 complete days,
9,048 episodes, zero signal-hash mismatches, and zero incomplete targets. At
`rho=0.8`:

| Algorithm | TWAP advantage by seed | Mean | Signal increment mean |
|---|---|---:|---:|
| Handcrafted rule | `+2.987` | +2.987 | +4.525 |
| SAC | `+0.939 / +0.721 / +3.676` | +1.778 | +2.854 |
| PPO | `-2.531 / -2.476 / -0.697` | -1.901 | -0.986 |
| RecurrentPPO | `-0.973 / -0.979 / +10.556` | +2.868 | +2.680 |

SAC is the first learned method to beat TWAP in all three seeds on a genuinely
untouched period. Its mean signal increment rises monotonically with `rho`:
`0.000, 0.906, 1.652, 2.272, 2.854` bps. This strongly supports the claim that
the enriched state and larger budget let SAC use forecast information.

PPO does not show a simple resource-limited learning curve: it improved sharply
at 131k validation steps, deteriorated at 524k validation, and all three 524k
test seeds trail TWAP. Longer training alone is not the solution.

RecurrentPPO repeats the validation bimodality on test. Two seeds remain in the
same signal-insensitive collapsed solution; one seed escapes and becomes much
stronger than every other learned policy. More training increases the chance of
finding a good recurrent policy for some initializations, but does not make the
method reliable.

### Stage 6 — five-symbol, three-minute, long-history study

**Question.** Are earlier conclusions artifacts of one instrument, five-minute
sampling, and only a few months of training history?

**Pre-registered design.** Pool `BTCUSDT`, `ETHUSDT`, `BNBUSDT`, `SOLUSDT`, and
`XRPUSDT`. Use three-minute price decisions and three-minute raw-trade buckets.
Training begins June 1, 2023, after the May 2023 WHO emergency-status change;
May is lookback only. Validation is calendar 2025. March-April 2026 is reserved
as the next untouched test. Feature normalization is symbol-local and fitted on
training dates; symbol identity remains in every matched evaluation key.

Moving from five-minute to three-minute days increases the episode length from
288 to 480 decisions. Scaling only by `5/3` would preserve total episodes but
would ignore the five-symbol, 19-month expansion. The frozen long-history budget
is therefore 1,572,864 steps (`3 × 524,288`), about 3,277 complete episodes for
roughly 2,895 pooled training symbol-days, or 1.13 sampled episodes per
symbol-day on average.

For reproducibility and future interval ablations, checksum-verified Binance
klines are retained at `1m`, `3m`, `5m`, `15m`, and `30m`. Only `3m` is the
primary experiment. The other intervals have price archives, not completed
long-history experiment inputs or comparable results, and must not display
copied results.

**Current status.** All 180 raw monthly archives and all 180 three-minute cache
parts are complete. Each of the five final caches contains 526,080 ordered
three-minute buckets, eight finite features, and 36 source signatures. Formal
validation, deterministic replay, and the locked test are complete. All
weighted gates are now complete.

**Raw-data incident.** Binance's checksum-valid April 2026 ETH trade archive
contains an exactly duplicated trade-id block and moves backward by 13
three-minute buckets at the block boundary. The original streaming aggregator
correctly rejected the duplicate timestamps but assumed globally ordered rows.
It now deduplicates trade IDs with a compact bitset and reassembles disjoint
bucket segments in `(timestamp, trade id)` order. Tests cover both unique
out-of-order segments and exact duplicate trades; all 37 tests pass.

**Dashboard correction.** The old five-minute result used one symbol and a
shorter period, so it is no longer represented as an available dashboard
experiment. `5m` remains a planned selector and will receive charts only after
a five-symbol long-history rerun. Monthly raw-trade archives are interval
agnostic, so future interval updates reuse verified archives and rebuild the
requested bucket cache before training.

**Validation smoke.** A five-symbol engineering smoke used February 20-27,
2026, which was already observed before this study, with one seed, 256 training
steps, and controlled-signal accuracies 0 and 0.4. All controls and all three
learned policies completed 1,680 matched validation episodes and wrote their
models. The artifact contains zero test rows. This raises weighted progress to
85.0% but provides no algorithm-performance claim; formal calendar-2025
validation still uses the preregistered 1,572,864-step budget and three seeds.

**Formal validation and split correction.** Nine independent CPU jobs completed
all three algorithms, three seeds, five signal accuracies, and 1,572,864 steps
per learned policy. The runner's raw `validation` label extended through
February 2026 because it originally used `test_start` as the only upper bound.
The canonical validation artifact therefore filters the unchanged episode rows
at `date < 2026-01-01`, restoring the preregistered calendar-2025 boundary. The
unfiltered artifact remains available for audit and was not used for the final
validation table.

At `rho=0.8`, HAIL beats TWAP by +1.985 bps. SAC's three advantages are
`-2.302 / -1.415 / -2.144`; PPO's are
`+13.447 / -0.986 / -1.808`; Memory-PPO's are
`+1.755 / +8.081 / -0.979`. Positive learned-policy means are driven by strong
individual seeds rather than stable convergence.

**Deterministic replay and locked test.** Reloading the frozen PPO seed-0 model
reproduced seven validation outputs byte-for-byte. Only then were the frozen
models evaluated once on March-April 2026. At `rho=0.8`, HAIL's TWAP advantage
is +0.845 bps; SAC averages -1.443, PPO +1.482, and Memory-PPO +1.195 bps.
PPO and Memory-PPO again have positive means but large cross-seed disagreement;
SAC's signal increment is negative on average at every positive `rho`.

### Stage 7 — preregistered five-interval comparison

After the original three-minute test, `1m`, `5m`, `15m`, and `30m` were trained
with equal sampled-episode budgets. The previously opened March-April window was
not reused. All 45 learned models, calendar-2025 validations, cross-policy
integrity gates, and byte-identical frozen-model replays completed before
opening May-August 2026 once for all five intervals.

At `rho=0.8`, mean TWAP advantages for HAIL / SAC / PPO / Memory-PPO are:

| Interval | HAIL | SAC | PPO | Memory-PPO |
|---|---:|---:|---:|---:|
| 1m | -2.763 | -0.795 | +3.670 | -0.033 |
| 3m | -0.448 | -1.017 | +1.831 | +0.610 |
| 5m | +2.684 | -2.617 | -0.050 | +3.479 |
| 15m | +7.619 | -2.924 | +8.149 | +0.186 |
| 30m | +13.568 | +6.682 | -0.693 | -0.154 |

No method dominates every interval, and learned means still conceal large
cross-seed differences. The interval is therefore part of the policy design,
not a neutral resampling choice. Full seed-level results and limitations are in
`docs/interval-matrix-results.md`.

## Test-set ledger

| Period / namespace | First use | Current role |
|---|---|---|
| Synthetic seeds 2,000,000,000+ | Synthetic pilot | Development; observed |
| May-June 2025 | First historical smoke | Development; observed |
| July-August 2025 | First historical locked run | Observed; no tuning claims |
| September-October 2025 | Controlled-signal locked run | Observed |
| November-December 2025 | Raw-trade locked run and post-hoc ablation | Observed |
| January-February 2026 | 524k budget test after validation selection | Observed; no further tuning claims |
| March-April 2026 | Five-symbol 3m long-history study | Opened once after freeze; final locked test |

Any parameter or architecture choice made after a period is opened requires a
new later test period for an honest claim.

## Current claim status

**Supported.** The data, simulator and evaluation pipeline are reproducible;
controlled signals isolate policy-signal interaction; HAIL consistently uses
better controlled signals and beats TWAP at high signal quality; individual
PPO and Memory-PPO seeds can outperform HAIL; and the full study exposes
material learned-policy seed sensitivity rather than hiding it in a mean.

**Not supported.** Stable learned-policy superiority over TWAP or HAIL across
seeds; a robust HRT algorithm ordering; alpha; profitability; realistic
order-book execution; counterfactual market impact; or robust recurrent-policy
convergence.

## Future paper path

A future NeurIPS-style paper can be organized around: problem formulation,
public-information boundary, matched controlled-signal design, historical data
and state construction, negative-result chronology, sample-efficiency ablations,
robustness across seeds and periods, and limitations. Drafting should begin only
from the frozen artifacts; the budget sweep and genuinely untouched test are
now complete, so no further test-set tuning is permitted.
