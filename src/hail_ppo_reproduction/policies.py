"""Non-learning policies used as experiment controls."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class HailBaselinePolicy:
    """Translation of the three-branch rule described in the HAIL article."""

    beta: float = 4.0
    reference_volatility: float = 0.002

    def action(
        self,
        *,
        directed_signal: float,
        remaining_fraction: float,
        steps_remaining: int,
        total_steps: int,
        return_volatility: float,
    ) -> NDArray[np.float32]:
        if steps_remaining <= 0 or remaining_fraction <= 0:
            return np.array([0.0], dtype=np.float32)

        if directed_signal > 0:
            fraction = 1.0 / steps_remaining
        else:
            remaining_time_fraction = steps_remaining / total_steps
            required_rate = remaining_fraction / remaining_time_fraction
            threshold = self.beta * return_volatility / self.reference_volatility
            fraction = 1.0 if required_rate > threshold else 0.0

        return np.array([fraction], dtype=np.float32)
