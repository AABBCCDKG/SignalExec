"""Gymnasium environment for one-direction intraday execution."""

from typing import Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from hail_ppo_reproduction.config import ExecutionConfig
from hail_ppo_reproduction.data import TradingDay

DayFactory = Callable[[np.random.Generator], TradingDay]


class ExecutionEnv(gym.Env):
    """Execute toward a fixed target while timing trades from noisy signals.

    An action is the fraction of the remaining quantity to trade at the current
    step. This construction enforces the article's buy-or-wait / sell-or-wait
    constraint by design.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        day_factory: DayFactory,
        config: Optional[ExecutionConfig] = None,
    ) -> None:
        super().__init__()
        self.day_factory = day_factory
        self.config = config or ExecutionConfig()
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-8.0, 0.0, 0.0, -8.0, 0.0], dtype=np.float32),
            high=np.array([8.0, 1.0, 1.0, 8.0, 1000.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.day: Optional[TradingDay] = None
        self.step_index = 0
        self.position = 0.0
        self.cumulative_pnl = 0.0

    @property
    def steps_remaining(self) -> int:
        self._require_day()
        return len(self.day.signals) - self.step_index  # type: ignore[union-attr]

    @property
    def remaining_quantity(self) -> float:
        self._require_day()
        target = abs(self.day.target_position)  # type: ignore[union-attr]
        return max(0.0, target - abs(self.position))

    @property
    def fill_fraction(self) -> float:
        self._require_day()
        return abs(self.position) / abs(self.day.target_position)  # type: ignore[union-attr]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[NDArray[np.float32], Dict[str, float]]:
        super().reset(seed=seed)
        self.day = self.day_factory(self.np_random)
        self.step_index = 0
        self.position = 0.0
        self.cumulative_pnl = 0.0
        return self._observation(), self._info(0.0, 0.0, 0.0)

    def step(
        self, action: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, float]]:
        self._require_day()
        if self.step_index >= len(self.day.signals):  # type: ignore[union-attr]
            raise RuntimeError("step called after the episode ended")

        fraction = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
        quantity = fraction * self.remaining_quantity
        direction = float(np.sign(self.day.target_position))  # type: ignore[union-attr]
        signed_quantity = direction * quantity

        price_now = float(self.day.prices[self.step_index])  # type: ignore[union-attr]
        price_next = float(self.day.prices[self.step_index + 1])  # type: ignore[union-attr]
        cost_bps = self.config.transaction_cost_bps + self.config.impact_bps * fraction
        trading_cost = quantity * price_now * cost_bps / 10_000.0

        self.position += signed_quantity
        step_pnl = self.position * (price_next - price_now) - trading_cost
        self.cumulative_pnl += step_pnl
        self.step_index += 1

        terminated = self.step_index == len(self.day.signals)  # type: ignore[union-attr]
        risk_scale = (
            abs(self.day.target_position)  # type: ignore[union-attr]
            * float(self.day.prices[0])  # type: ignore[union-attr]
            * self.day.return_volatility  # type: ignore[union-attr]
        )
        reward = step_pnl / risk_scale
        shortfall_penalty = 0.0
        if terminated:
            shortfall_penalty = (
                self.config.terminal_shortfall_penalty * (1.0 - self.fill_fraction)
            )
            reward -= shortfall_penalty

        observation = self._observation(terminal=terminated)
        info = self._info(step_pnl, trading_cost, shortfall_penalty)
        return observation, float(reward), terminated, False, info

    def _observation(self, terminal: bool = False) -> NDArray[np.float32]:
        self._require_day()
        if terminal:
            directed_signal_z = 0.0
        else:
            directed_signal_z = (
                np.sign(self.day.target_position)  # type: ignore[union-attr]
                * self.day.signals[self.step_index]  # type: ignore[union-attr]
                / self.day.return_volatility  # type: ignore[union-attr]
            )

        if self.step_index == 0:
            last_directed_return_z = 0.0
        else:
            price_before = self.day.prices[self.step_index - 1]  # type: ignore[union-attr]
            price_now = self.day.prices[self.step_index]  # type: ignore[union-attr]
            last_directed_return_z = (
                np.sign(self.day.target_position)  # type: ignore[union-attr]
                * np.log(price_now / price_before)
                / self.day.return_volatility  # type: ignore[union-attr]
            )

        total_steps = len(self.day.signals)  # type: ignore[union-attr]
        return np.array(
            [
                np.clip(directed_signal_z, -8.0, 8.0),
                1.0 - self.fill_fraction,
                self.steps_remaining / total_steps,
                np.clip(last_directed_return_z, -8.0, 8.0),
                self.day.return_volatility * 10_000.0,  # type: ignore[union-attr]
            ],
            dtype=np.float32,
        )

    def _info(
        self,
        step_pnl: float,
        trading_cost: float,
        shortfall_penalty: float,
    ) -> Dict[str, float]:
        return {
            "position": self.position,
            "remaining_quantity": self.remaining_quantity,
            "fill_fraction": self.fill_fraction,
            "step_pnl": step_pnl,
            "cumulative_pnl": self.cumulative_pnl,
            "trading_cost": trading_cost,
            "shortfall_penalty": shortfall_penalty,
        }

    def _require_day(self) -> None:
        if self.day is None:
            raise RuntimeError("reset must be called before reading or stepping the environment")
