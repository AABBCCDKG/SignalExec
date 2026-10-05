"""Configuration objects for the first reproduction milestone."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticMarketConfig:
    """Parameters for a synthetic one-day market episode."""

    steps: int = 78
    start_price: float = 100.0
    target_quantity: float = 1.0
    return_volatility: float = 0.002
    signal_accuracy: float = 0.4

    def __post_init__(self) -> None:
        if self.steps < 2:
            raise ValueError("steps must be at least 2")
        if self.start_price <= 0:
            raise ValueError("start_price must be positive")
        if self.target_quantity <= 0:
            raise ValueError("target_quantity must be positive")
        if self.return_volatility <= 0:
            raise ValueError("return_volatility must be positive")
        if not -1.0 <= self.signal_accuracy <= 1.0:
            raise ValueError("signal_accuracy must be between -1 and 1")


@dataclass(frozen=True)
class ExecutionConfig:
    """Accounting and reward assumptions for the execution environment."""

    transaction_cost_bps: float = 0.5
    impact_bps: float = 1.0
    terminal_shortfall_penalty: float = 0.25

    def __post_init__(self) -> None:
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps cannot be negative")
        if self.impact_bps < 0:
            raise ValueError("impact_bps cannot be negative")
        if self.terminal_shortfall_penalty < 0:
            raise ValueError("terminal_shortfall_penalty cannot be negative")
