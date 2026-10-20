# SignalExec

A reproducible benchmark for comparing how execution policies turn the same
trading signal and target position into a sequence of trades.

**Interactive results:** <https://aabbccddkg.github.io/SignalExec/>

SignalExec is an independent research extension inspired by Hudson River
Trading AI Labs' public
[intern project](https://www.hudsonrivertrading.com/hrtbeat/intern-spotlight-hrt-ai-labs-project-2026/).
It does not reproduce HRT's private data, simulator, reward, or architecture.

## Research question

Given the same buy/sell signal and target position, which execution policy
should trade now, how much of the remaining order should it execute, and how
does that answer change with the decision interval?

The benchmark formulates execution as inventory-aware continuous control. The
state combines directed signal strength, remaining inventory, time-to-deadline,
recent returns, volatility, and point-in-time microstructure features. An action
is the fraction of the remaining quantity to execute; rewards include
mark-to-market PnL, fees, linear impact, and a terminal completion penalty.

## Experiment

- **Markets:** BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT.
- **Decision intervals:** 1m, 3m, 5m, 15m, 30m.
- **Policies:** HAIL-inspired rule, TWAP, immediate, wait, SAC, PPO, and
  Recurrent PPO.
- **Features:** quote volume, raw-trade count, aggressor imbalance, average
  trade size, price range and return, largest-trade share, and aggressor-side
  switching.
- **Scale:** 45 learned policies with equal sampled-episode budgets and three
  initialization seeds.

Controlled signals deliberately mix realized next returns with common noise so
their quality is known. They are oracle experiment inputs, not tradable
forecasts. Mirrored buy/sell targets, common signal hashes, mandatory completion,
and matched TWAP comparisons isolate execution-policy behavior from market
direction and signal quality.

## Evaluation protocol

1. Build checksum-verified, point-in-time feature caches from Binance public
   archives.
2. Train before 2025 and select nothing from the final test window.
3. Evaluate on calendar-2025 validation data.
4. Replay frozen PPO models and require byte-identical episode, first-difference,
   and difference-in-differences outputs.
5. Open May-August 2026 once for the five-interval locked comparison.

Every interval passed the fill, signal-hash, TWAP-identity, and `rho=0`
difference-in-differences gates. The complete protocol and result lineage are in
[the research log](docs/research-log.md).

## Locked-test results

At controlled signal quality `rho=0.8`, positive values mean lower
implementation shortfall than matched TWAP, in basis points.

| Interval | HAIL | SAC mean | PPO mean | Recurrent PPO mean |
|---|---:|---:|---:|---:|
| 1m | -2.763 | -0.795 | +3.670 | -0.033 |
| 3m | -0.448 | -1.017 | +1.831 | +0.610 |
| 5m | +2.684 | -2.617 | -0.050 | +3.479 |
| 15m | +7.619 | -2.924 | +8.149 | +0.186 |
| 30m | +13.568 | +6.682 | -0.693 | -0.154 |

The central finding is that decision frequency is a policy variable rather
than a neutral resampling choice. The same handcrafted rule trails TWAP at 1m
and 3m but becomes strong at slower intervals; learned-policy means also remain
sensitive to initialization. See
[the full seed-level results](docs/interval-matrix-results.md).

## Reproduce

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev --extra rl
uv run --extra dev --extra rl pytest -q
```

Binance archives, derived caches, trained models, and experiment artifacts stay
local and are ignored by Git. Downloaders verify official adjacent SHA-256
files before aggregation. The preregistered five-interval design and budgets are
recorded in [the protocol](docs/interval-matrix-protocol.md).

Representative commands:

```bash
# Download checksum-verified price bars.
uv run python scripts/download_binance.py \
  --output data/binance/BTCUSDT-3m --symbol BTCUSDT --interval 3m \
  --months 2026-06 2026-07 2026-08 2026-09

# Stream compressed trades into a point-in-time feature cache.
uv run python scripts/aggregate_binance_trades.py \
  --data-dir data/binance/BTCUSDT-trades \
  --cache data/binance/BTCUSDT-trades-3m.npz \
  --symbol BTCUSDT --months 2026-05 2026-06 2026-07 2026-08 \
  --workers 2 --bucket-minutes 3
```

## Dashboard

The static dashboard is generated from merged locked-test CSVs by
`scripts/export_interval_dashboard.py`. Serve it locally with:

```bash
python3 -m http.server 8000 --directory dashboard
```

Then open <http://localhost:8000/>. GitHub Pages deploys the same `dashboard/`
directory through [the Pages workflow](.github/workflows/pages.yml).

## Limitations

- Controlled signals contain future information by design and do not establish
  alpha or forecast quality.
- The simulator assumes immediate bar-open fills with explicit linear fees and
  impact; it does not model spread, queue position, a historical order book, or
  counterfactual market impact.
- Results compare execution responses on five Binance Spot symbols and do not
  establish live profitability or production capacity.
