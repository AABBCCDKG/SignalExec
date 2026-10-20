from datetime import date

import numpy as np

from hail_ppo_reproduction import ExecutionConfig
from hail_ppo_reproduction.controlled import HistoricalPriceDay
from hail_ppo_reproduction.controlled_experiment import (
    ControlledExperimentConfig,
    difference_in_differences,
    evaluate_controlled_policy,
    paired_advantages,
    summarize_advantages,
    validate_matched_records,
)


def price_day(day: int, rising: bool) -> HistoricalPriceDay:
    prices = np.array([100.0, 101.0, 102.0, 103.0])
    if not rising:
        prices = prices[::-1].copy()
    returns = np.diff(np.log(prices))
    return HistoricalPriceDay(
        date=date(2025, 1, day),
        prices=prices,
        realized_returns=returns,
        return_volatility=0.01,
    )


def config() -> ControlledExperimentConfig:
    return ControlledExperimentConfig(
        timesteps=256,
        train_seeds=(0,),
        evaluation_accuracies=(0.0, 0.8),
        signal_replicas=1,
        eval_batch_size=2,
    )


def test_matched_policies_fill_and_receive_identical_signals():
    days = (price_day(1, True), price_day(2, False))
    execution = ExecutionConfig()
    twap = evaluate_controlled_policy(
        "twap",
        None,
        None,
        split="validation",
        days=days,
        config=config(),
        execution=execution,
        target_quantity=1.0,
    )
    immediate = evaluate_controlled_policy(
        "immediate",
        None,
        None,
        split="validation",
        days=days,
        config=config(),
        execution=execution,
        target_quantity=1.0,
    )
    records = twap + immediate

    validate_matched_records(records)

    assert len(twap) == 12
    assert all(row["filled"] and row["fill_fraction"] == 1.0 for row in records)
    hashes = {}
    for row in records:
        hashes.setdefault(row["matched_input_id"], set()).add(row["signal_sha256"])
    assert all(len(values) == 1 for values in hashes.values())


def test_first_and_second_differences_are_paired_to_twap_and_rho_zero():
    days = (price_day(1, True), price_day(2, False))
    execution = ExecutionConfig()
    records = []
    for name in ("twap", "immediate"):
        records.extend(
            evaluate_controlled_policy(
                name,
                None,
                None,
                split="validation",
                days=days,
                config=config(),
                execution=execution,
                target_quantity=1.0,
            )
        )

    advantages = paired_advantages(records)
    summaries = summarize_advantages(advantages)
    did = difference_in_differences(advantages)

    assert all(
        row["twap_shortfall_advantage_bps"] == 0.0
        for row in advantages
        if row["algorithm"] == "twap"
    )
    assert any(row["algorithm"] == "immediate" for row in summaries)
    immediate_off = next(
        row
        for row in summaries
        if row["algorithm"] == "immediate" and row["signal_mode"] == "off"
    )
    assert immediate_off["episodes"] == 2  # two dates after buy/sell pairing
    assert {row["signal_accuracy"] for row in did} == {0.0, 0.8}
    assert all(row["episodes"] == 2 for row in did)
    assert all(row["signal_increment_bps"] == 0.0 for row in did if row["signal_accuracy"] == 0.0)


def test_mirrored_targets_cancel_common_price_direction_for_immediate_execution():
    records = evaluate_controlled_policy(
        "immediate",
        None,
        None,
        split="validation",
        days=(price_day(1, True),),
        config=config(),
        execution=ExecutionConfig(),
        target_quantity=1.0,
    )
    off = [row for row in records if row["signal_mode"] == "off"]

    assert len(off) == 2
    assert np.isclose(np.mean([row["pnl_bps"] for row in off]), -1.5)
    assert all(np.isclose(row["implementation_shortfall_bps"], 1.5) for row in off)
