# Binance historical reproduction protocol

This experiment is the first real-data stage of **Signal-to-Execution RL**. It
tests the public HRT problem structure on Binance Spot data; it does not claim to
recover HRT's private data, forecasts, simulator, reward, or architecture.

## Data and split

- Source: Binance Public Data monthly Spot klines from
  `https://data.binance.vision/`.
- Instrument: `BTCUSDT`; interval: five minutes.
- Raw months: December 2024 through August 2025. Each archive is checked against
  Binance's adjacent SHA-256 checksum file.
- Training episodes start from January 1, 2025.
- Validation starts May 1, 2025. It includes May and June, which were opened by
  the initial end-to-end development smoke run and are not held-out evidence.
- The replacement locked test starts July 1, 2025 and ends before September 1,
  2025. Development commands evaluate validation only; the test is opened once
  after the implementation and run configuration are frozen.
- A day whose final valuation price touches the next split boundary is excluded,
  so no episode crosses from one split into another.

## Point-in-time contract

Each UTC day is one execution episode. At the open of bar `t`, the short-term
forecast uses only bars that ended before `t`. The action executes at that open
and earns the next open-to-open price change. Features are:

- last-bar, three-bar, and twelve-bar close-return momentum;
- trailing twelve-bar realized volatility;
- previous-bar taker-buy imbalance;
- previous-bar log volume relative to its trailing twelve-bar mean.

A dependency-free ridge model is standardized and fit on training rows only.
A second ridge model sets the day's target direction from prior one-, three-,
and seven-day returns, prior-day volatility and imbalance, and relative volume.
Validation and test prices never refit either forecast.

## Execution and evaluation assumptions

The existing one-direction environment is retained: the policy may only trade
toward its fixed daily target. The historical replay still assumes immediate
fills at bar opens with configurable linear fees and impact. It cannot identify
counterfactual queue position, slippage, or market impact from kline data.

The handcrafted HRT-inspired rule, TWAP, immediate execution, wait, SAC, PPO,
and recurrent PPO receive the same dated episodes. Results report raw PnL after
assumed costs, PnL in basis points of target notional, fill fraction, completion
rate, signal correlation/directional accuracy, and variation across training
seeds. Natural policy behavior and forced final-step completion remain separate.

The chronological test is opened only after the data contract and validation
pipeline pass. The first smoke exposed the earlier May-June test window, so that
window was explicitly reassigned to validation before method tuning. Any later
method development must use training/validation data and reserve a new test
period rather than repeatedly tuning on July-August.
