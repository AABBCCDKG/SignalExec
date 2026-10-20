from datetime import date

import numpy as np

from hail_ppo_reproduction import ExecutionConfig, TradingDay
from hail_ppo_reproduction.historical import HistoricalDay
from hail_ppo_reproduction.historical_experiment import (
    evaluate_historical_policy,
    summarize_historical,
)


def historical_day(day: int) -> HistoricalDay:
    trading_day = TradingDay(
        prices=np.array([100.0, 101.0, 102.0]),
        signals=np.array([0.01, 0.01]),
        target_position=1.0,
        return_volatility=0.01,
    )
    returns = np.log(trading_day.prices[1:] / trading_day.prices[:-1])
    return HistoricalDay(
        date=date(2025, 1, day),
        trading_day=trading_day,
        realized_returns=returns,
        target_return=float(returns.sum()),
    )


def test_historical_controls_share_dates_and_complete_targets():
    days = (historical_day(1), historical_day(2))
    execution = ExecutionConfig(
        transaction_cost_bps=0.0,
        impact_bps=0.0,
        terminal_shortfall_penalty=0.0,
    )

    twap = evaluate_historical_policy(
        "twap", None, None, split="test", days=days, execution=execution
    )
    immediate = evaluate_historical_policy(
        "immediate", None, None, split="test", days=days, execution=execution
    )

    assert [row["date"] for row in twap] == [row["date"] for row in immediate]
    assert all(row["filled"] for row in twap + immediate)
    assert all(row["split"] == "test" for row in twap + immediate)


def test_historical_summary_keeps_policy_and_split_separate():
    days = (historical_day(1), historical_day(2))
    records = evaluate_historical_policy(
        "wait",
        None,
        None,
        split="validation",
        days=days,
        execution=ExecutionConfig(),
    )

    summary = summarize_historical(records)

    assert len(summary) == 1
    assert summary[0]["algorithm"] == "wait"
    assert summary[0]["split"] == "validation"
    assert summary[0]["episodes"] == 2
