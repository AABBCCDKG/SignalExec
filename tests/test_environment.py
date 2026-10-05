import numpy as np

from hail_ppo_reproduction import (
    ExecutionConfig,
    ExecutionEnv,
    SyntheticMarketConfig,
    TradingDay,
    make_synthetic_day,
)


def fixed_day(prices, signals, target=1.0, volatility=0.01):
    day = TradingDay(
        prices=np.asarray(prices, dtype=np.float64),
        signals=np.asarray(signals, dtype=np.float64),
        target_position=target,
        return_volatility=volatility,
    )
    return lambda rng: day


def test_action_cannot_reverse_or_overfill_target():
    env = ExecutionEnv(fixed_day([100.0, 99.0, 98.0], [0.0, 0.0], target=-2.0))
    env.reset(seed=1)

    env.step(np.array([0.75], dtype=np.float32))
    _, _, terminated, _, info = env.step(np.array([1.0], dtype=np.float32))

    assert terminated
    assert info["position"] == -2.0
    assert info["fill_fraction"] == 1.0


def test_buying_earlier_earns_more_on_a_rising_path():
    config = ExecutionConfig(
        transaction_cost_bps=0.0,
        impact_bps=0.0,
        terminal_shortfall_penalty=0.0,
    )
    factory = fixed_day([100.0, 101.0, 102.0], [0.0, 0.0])

    early = ExecutionEnv(factory, config)
    early.reset(seed=1)
    early.step(np.array([1.0], dtype=np.float32))
    _, _, _, _, early_info = early.step(np.array([0.0], dtype=np.float32))

    late = ExecutionEnv(factory, config)
    late.reset(seed=1)
    late.step(np.array([0.0], dtype=np.float32))
    _, _, _, _, late_info = late.step(np.array([1.0], dtype=np.float32))

    assert early_info["cumulative_pnl"] == 2.0
    assert late_info["cumulative_pnl"] == 1.0


def test_terminal_shortfall_is_separate_from_trading_pnl():
    env = ExecutionEnv(
        fixed_day([100.0, 100.0], [0.0]),
        ExecutionConfig(
            transaction_cost_bps=0.0,
            impact_bps=0.0,
            terminal_shortfall_penalty=0.5,
        ),
    )
    env.reset(seed=1)

    _, reward, terminated, _, info = env.step(np.array([0.0], dtype=np.float32))

    assert terminated
    assert reward == -0.5
    assert info["cumulative_pnl"] == 0.0
    assert info["shortfall_penalty"] == 0.5


def test_seeded_market_resets_are_reproducible():
    market = SyntheticMarketConfig()
    env = ExecutionEnv(lambda rng: make_synthetic_day(rng, market))

    first_observation, _ = env.reset(seed=123)
    first_prices = env.day.prices.copy()
    first_signals = env.day.signals.copy()
    first_target = env.day.target_position

    second_observation, _ = env.reset(seed=123)

    np.testing.assert_array_equal(second_observation, first_observation)
    np.testing.assert_array_equal(env.day.prices, first_prices)
    np.testing.assert_array_equal(env.day.signals, first_signals)
    assert env.day.target_position == first_target
