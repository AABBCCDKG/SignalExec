import numpy as np

from hail_ppo_reproduction import HailBaselinePolicy


def test_favorable_signal_trades_one_equal_schedule_slice():
    policy = HailBaselinePolicy()

    action = policy.action(
        directed_signal=0.01,
        remaining_fraction=1.0,
        steps_remaining=10,
        total_steps=10,
        return_volatility=0.002,
    )

    assert np.isclose(action[0], 0.1)


def test_unfavorable_signal_waits_when_there_is_time():
    policy = HailBaselinePolicy(beta=4.0)

    action = policy.action(
        directed_signal=-0.01,
        remaining_fraction=1.0,
        steps_remaining=10,
        total_steps=10,
        return_volatility=0.002,
    )

    assert action[0] == 0.0


def test_unfavorable_signal_fills_when_time_pressure_is_high():
    policy = HailBaselinePolicy(beta=4.0)

    action = policy.action(
        directed_signal=-0.01,
        remaining_fraction=1.0,
        steps_remaining=1,
        total_steps=10,
        return_volatility=0.002,
    )

    assert action[0] == 1.0
