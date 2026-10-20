# BTCUSDT historical reproduction results

The first chronological Binance run completed end to end. It reproduces the
public experiment structure, but **does not reproduce HRT's reported qualitative
ordering** of baseline < SAC < PPO < memory-augmented PPO.

## Frozen run

```bash
uv run --extra rl python scripts/run_historical_experiment.py \
  --output artifacts/binance-btcusdt-5m-reproduction-20261016-v2 \
  --evaluation-splits validation test
```

- Data: nine checksum-verified monthly BTCUSDT five-minute archives, December
  2024 through August 2025.
- Chronology: 119 training days, 60 validation days, and 61 locked test days.
- Training: SAC, PPO, and RecurrentPPO; seeds 0, 1, and 2; exactly 32,768
  environment steps per model.
- Evaluation: the HRT-inspired rule, TWAP, immediate execution, wait, and all
  nine trained policies on identical dated episodes, both natural and forced
  final-step completion.
- Costs: 0.5 bps linear transaction cost plus action-dependent impact up to
  1.0 bps. These are explicit assumptions, not measured Binance fills.
- Local evidence: `metadata.json`, model files, training logs, episode-level
  CSV, summaries, and the comparison plot are stored under the output path.

The initial development smoke had exposed the earlier May-June test window.
Those dates were reassigned to validation before this run; July-August was then
opened once as the replacement locked test.

## Signal diagnostics

| Split | Days | Short-signal correlation | Short direction accuracy | Target direction accuracy |
|---|---:|---:|---:|---:|
| Train | 119 | 0.0160 | 51.05% | 52.94% |
| Validation | 60 | 0.0115 | 50.89% | 41.67% |
| Locked test | 61 | -0.0039 | 49.08% | 55.74% |

The causal ridge short-horizon forecast has effectively no out-of-sample edge.
This matters for interpreting the policy comparison: an execution policy cannot
reliably time trades from a signal that carries no observed test predictiveness.

## Natural policy results

PnL is after assumed costs and is expressed in basis points of target notional.
RL entries average three independently trained seeds; ranges show the minimum
and maximum seed mean. These descriptive means are not significance claims.

| Policy | Validation PnL | Test PnL | Test seed range | Test mean fill | Test completion |
|---|---:|---:|---:|---:|---:|
| HRT-inspired rule | -31.907 | -0.433 | - | 100.00% | 100.00% |
| TWAP | -29.327 | 4.595 | - | 100.00% | 100.00% |
| Immediate | -19.664 | 12.114 | - | 100.00% | 100.00% |
| Wait | 0.000 | 0.000 | - | 0.00% | 0.00% |
| SAC | -20.400 | 12.506 | [12.498, 12.511] | 100.00% | 100.00% |
| PPO | -14.086 | 8.347 | [0.015, 12.788] | 66.69% | 66.67% |
| RecurrentPPO | 0.029 | 0.008 | [0.000, 0.024] | 0.07% | 0.00% |

SAC learned behavior close to immediate execution. PPO remained highly
seed-dependent: two seeds completed the target while one barely traded.
RecurrentPPO almost always stayed out of the market, so its near-zero PnL is a
failure to execute rather than superior trading performance.

Forced completion does not change already completed policies. On the locked
test it moves RecurrentPPO's mean PnL to -1.080 bps and the wait control to
-1.088 bps, confirming that their natural near-zero PnL came from under-filling.

## What this establishes

- The public-data reproduction is operational and point-in-time auditable.
- The published HRT ranking is not robust to these transparent assumptions.
- The run does not demonstrate a tradable signal, alpha, or live execution edge.
- Kline replay cannot measure queue position, counterfactual fills, or market
  impact; all actions still fill immediately at observed bar opens.

July-August is now observed and must not be reused to select improvements. The
next research stage should agree on hypotheses using train/validation evidence
and reserve a new chronological test period before changing methods.
