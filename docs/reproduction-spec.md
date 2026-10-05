# Reproduction specification

## What the HAIL article establishes

The public article describes an execution problem with these fixed ingredients:

- a long-term return prediction determines a target position;
- the agent receives short-term return predictions and chooses when to trade;
- during one symbol-day, it may trade only toward the target: buy or wait for a positive target, sell or wait for a negative target;
- a historical-data simulator supplies training episodes;
- the comparison is hand-written baseline, SAC, PPO, then memory-augmented PPO;
- evaluation sweeps the accuracy of the short-term signal and reports PnL and fill fraction;
- the reported qualitative order is baseline < SAC < PPO < memory-PPO, while PPO-family policies sometimes under-fill.

## What the article does not disclose

The article does not publish the dataset, train/test dates, observation vector, action scaling, fill model, costs, reward equation, network sizes, memory architecture, hyperparameters, or random seeds. We test whether the published qualitative behavior appears under explicit assumptions. Matching the reported ordering is a research question, not a pass/fail requirement; negative results must also be reported.

## Our explicit milestone-1 assumptions

| Component | Assumption | Why it is testable |
|---|---|---|
| Market data | Synthetic log returns; the signal is a noisy oracle constructed from the next return | Controls information quality; does not represent a deployable forecast |
| Target | One normalized signed unit, fixed for the day | Buy and sell episodes share the same mechanics |
| Action | A number in `[0, 1]`, interpreted as the fraction of remaining quantity traded now | The environment makes wrong-direction trades and over-filling impossible |
| Fill | Every submitted quantity fills immediately at the current mid price | Removes order-book modeling from the first experiment |
| PnL | Position held after the action earns the next price change | Buying before a rise and selling before a fall are rewarded |
| Costs | Linear transaction cost plus action-dependent impact | Discourages trading everything immediately without evidence |
| Reward | Risk-normalized trading PnL, plus a terminal penalty on the unfilled fraction | Separates economic PnL from the training incentive |
| Memory | Standard PPO sees one observation; memory-PPO will use a recurrent policy over observation history | Matches the article's partial-observability motivation without claiming its private architecture |

## Baseline translation

The article's rule trades a steady slice when directed return is positive, fills the remainder when time pressure dominates the value of waiting, and otherwise waits. Here:

- `directed signal = sign(target) * short-term prediction`;
- a steady slice is `1 / remaining_steps` of the remaining quantity;
- time pressure is `remaining_fraction / remaining_time_fraction`;
- the waiting threshold is `beta * volatility / reference_volatility`.

This preserves the rule's intended monotonic behavior: favorable signals cause steady progress, less time creates urgency, and greater volatility permits more waiting.

## Acceptance criteria

### Milestone 1 — simulator contract

- actions never reverse direction or exceed the target;
- earlier buying earns more than later buying on a rising path, before terminal penalties;
- terminal under-fill is visible separately from raw trading PnL;
- the hand-written baseline reacts correctly to favorable signal and urgency;
- seeded runs are reproducible.

### Milestone 2 — RL comparison on synthetic data

- train SAC, PPO, and recurrent PPO under the same simulator and data budget;
- evaluate unseen seeds and signal-accuracy values;
- report mean, dispersion, fill fraction, and training compute across multiple seeds;
- compare learned decision surfaces for smoothness rather than relying only on aggregate PnL.

### Milestone 3 — under-fill extension

- measure the PnL/fill trade-off while varying the terminal penalty;
- test a constrained or action-masked schedule that guarantees completion;
- state whether improved fill comes from a better policy or simply a stronger penalty.

## Implemented pilot protocol

- Library implementations: Stable-Baselines3 SAC/PPO and SB3-Contrib
  `RecurrentPPO("MlpLstmPolicy")`. This LSTM is our implementation choice;
  HRT's exact memory architecture remains unknown.
- Models run on CPU with one PyTorch thread. `uv.lock` records package versions;
  every output directory records Python/platform/version information and source
  hashes, including whether the working tree was uncommitted.
- Default budget: 32,768 environment steps per model, training seeds 0, 1, 2,
  with signal correlation 0.4. Budgets must be multiples of 256 so PPO's rollout
  rounding cannot silently increase its data allowance.
- PPO: 256-step rollouts, batch size 64, four epochs per update, separate
  two-layer 64-unit actor and critic. LSTM PPO uses hidden size 32 and one
  64-unit layer after the recurrent state. SAC uses two 64-unit layers, batch
  size 64, one gradient update every four environment steps, and at most 1,024
  warm-up steps. All use learning rate 0.0003 and undiscounted episodic rewards
  (`gamma=1`). Architecture and optimizer differences are recorded assumptions.
- Equal environment-step budgets do not imply equal compute or gradient-update
  counts; elapsed training time and actual steps are saved for each model.
- Evaluation: 200 unseen days at correlations 0, 0.2, 0.4, 0.6, 0.8. Each policy
  receives the same day seeds, and each correlation uses the same return path
  with a different mixture of signal and noise. No held-out results select
  hyperparameters or checkpoints. Model initialization and training-market RNG
  streams are separate from the evaluation namespace.
- Recurrent hidden state is carried between decisions and reset between days.
  Policies receive only the observation array. Future prices remain inside the
  simulator. The five features are directed signal/volatility, remaining
  fraction, remaining time fraction, last directed return/volatility, and
  volatility in basis points.
- Evaluation saves raw mark-to-market PnL (after trading costs), PnL in basis
  points of target notional, fill fraction, completion rate (within 1e-6), costs,
  and total training reward. Terminal penalties affect reward, not reported PnL.
- `natural` uses the policy's actions. `forced` reruns the same days and replaces
  the last action by full completion. This is an execution rule applied to a
  fixed trained policy, not a retrained constraint-aware policy. It affects both
  trading costs and exposure to the final price move.
- CSV summaries keep each training seed separate. Plot lines average those seed
  means; shading shows their range, not a confidence interval. Day-level standard
  errors are conditional on a fixed trained policy and do not measure training
  uncertainty. Paired comparisons can be recovered using `eval_seed`.

## Limits and remaining research work

The current returns are IID and the target direction is random rather than
derived from a long-horizon predictor. A history of signals or returns provides
no extra predictive information about the next return. This is an honest
control for the training pipeline, not a reconstruction of the article's
partially observable market. Do not tune this simulator just to force the
published SAC/PPO/memory-PPO ordering.

Remaining work includes a causal long-horizon target, a credible historical
intraday dataset with chronological splits, a partially observable process with
clearly specified latent information, decision-boundary diagnostics conditioned
on histories, and a training-budget/convergence study. Market impact here is a
simple cost assumption; historical replay cannot establish actual counterfactual
fills or impact. The forced-completion intervention is implemented; penalty
sweeps and training a policy with a hard completion constraint remain separate
experiments.

Library reference: [SB3-Contrib recurrent PPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html).
