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

The article does not publish the dataset, train/test dates, observation vector, action scaling, fill model, costs, reward equation, network sizes, memory architecture, hyperparameters, or random seeds. A scientifically honest reproduction therefore aims first at the published qualitative behavior, not identical numbers.

## Our explicit milestone-1 assumptions

| Component | Assumption | Why it is testable |
|---|---|---|
| Market data | Synthetic log returns; the observed signal has configurable correlation with the next return | Signal quality can be swept without leakage |
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

### Milestone 2 — RL comparison

- train SAC, PPO, and recurrent PPO under the same simulator and data budget;
- evaluate unseen seeds and signal-accuracy values;
- report mean, dispersion, fill fraction, and training compute across multiple seeds;
- compare learned decision surfaces for smoothness rather than relying only on aggregate PnL.

### Milestone 3 — under-fill extension

- measure the PnL/fill trade-off while varying the terminal penalty;
- test a constrained or action-masked schedule that guarantees completion;
- state whether improved fill comes from a better policy or simply a stronger penalty.
