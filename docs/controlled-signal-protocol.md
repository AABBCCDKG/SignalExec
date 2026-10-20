# Controlled-signal paired comparison

This protocol isolates execution-policy behavior from upstream forecast quality.
It is a causal diagnostic of signal use, not a backtest of a tradable strategy.

## Data and chronology

- Binance Spot BTCUSDT five-minute klines, checksum verified.
- Training price paths: January 1 through April 29, 2025.
- Development/validation: May 1 through August 30, 2025.
- Locked test: September 1 through October 30, 2025.
- A UTC day crossing a split boundary is excluded.

## Matched targets and signals

Every price day is evaluated twice: once with target `+1` and once with target
`-1`. The pair is averaged before estimating uncertainty, removing the common
effect of the day's market direction.

`signal_off` sets every forecast to zero. Controlled signals use:

```text
signal(rho) = rho * realized_next_return + sqrt(1 - rho^2) * noise
```

The same noise vector is reused across every `rho` and every policy on a matched
day. `rho` takes `0, 0.2, 0.4, 0.6, 0.8`. These signals intentionally contain
future returns so forecast quality is known; they are oracle experiment inputs,
not point-in-time forecasts or trading signals.

All RL policies train at `rho=0.4`, using three initialization seeds and exactly
32,768 environment steps. The handcrafted rule and RL models are evaluated on
the same signal hashes. Every evaluation forces the final action to complete the
target, so avoiding the trade cannot appear superior.

## Outcomes and paired differences

The primary outcome is implementation shortfall including assumed linear fees
and impact. Lower shortfall is better.

The first difference is the policy's matched advantage over TWAP:

```text
TWAP advantage(rho) = shortfall(TWAP, rho) - shortfall(policy, rho)
```

The second difference isolates signal use:

```text
signal increment(rho) = TWAP advantage(rho) - TWAP advantage(rho=0)
```

A positive signal increment means the policy executed better when more return
information was added to the same noise. It does not mean the policy beat TWAP
in absolute execution quality.

## Acceptance gates

- official archives pass SHA-256 verification;
- measured signal correlations track requested `rho` values;
- matched policies receive identical signal hashes;
- every policy fills 100% of the target;
- TWAP first differences and `rho=0` second differences equal zero;
- buy/sell directions are paired before standard errors are calculated;
- the locked test is opened only after validation gates and parameter freeze;
- a deterministic rerun reproduces episode, first-difference, and DiD files.

The replay still assumes immediate fills at bar opens and cannot measure spread,
queue position, counterfactual impact, or real execution capacity.
