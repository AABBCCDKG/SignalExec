"""Synthetic data used to validate the execution research contract."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from hail_ppo_reproduction.config import SyntheticMarketConfig


@dataclass(frozen=True)
class TradingDay:
    """Exogenous inputs for one symbol-day execution episode."""

    prices: NDArray[np.float64]
    signals: NDArray[np.float64]
    target_position: float
    return_volatility: float

    def __post_init__(self) -> None:
        if self.prices.ndim != 1 or self.signals.ndim != 1:
            raise ValueError("prices and signals must be one-dimensional")
        if len(self.prices) != len(self.signals) + 1:
            raise ValueError("prices must contain one more value than signals")
        if len(self.signals) < 1:
            raise ValueError("a trading day needs at least one decision step")
        if np.any(self.prices <= 0) or not np.all(np.isfinite(self.prices)):
            raise ValueError("prices must be finite and positive")
        if not np.all(np.isfinite(self.signals)):
            raise ValueError("signals must be finite")
        if self.target_position == 0:
            raise ValueError("target_position cannot be zero")
        if self.return_volatility <= 0:
            raise ValueError("return_volatility must be positive")


def make_synthetic_day(
    rng: np.random.Generator,
    config: SyntheticMarketConfig,
    target_direction: Optional[int] = None,
) -> TradingDay:
    """Generate returns and a noisy one-step forecast with known correlation.

    The signal at step ``t`` is correlated with the return from price ``t`` to
    price ``t + 1``. It is observable before that price move in the simulator.
    """

    if target_direction is None:
        direction = int(rng.choice(np.array([-1, 1])))
    elif target_direction in (-1, 1):
        direction = target_direction
    else:
        raise ValueError("target_direction must be -1, 1, or None")

    returns = rng.normal(0.0, config.return_volatility, size=config.steps)
    noise = rng.normal(0.0, config.return_volatility, size=config.steps)
    rho = config.signal_accuracy
    signals = rho * returns + np.sqrt(max(0.0, 1.0 - rho * rho)) * noise
    prices = config.start_price * np.exp(
        np.concatenate((np.array([0.0]), np.cumsum(returns)))
    )

    return TradingDay(
        prices=prices.astype(np.float64),
        signals=signals.astype(np.float64),
        target_position=direction * config.target_quantity,
        return_volatility=config.return_volatility,
    )
