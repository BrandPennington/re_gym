"""Mortgage hedging environment with interest rate derivatives."""

from typing import Any

import numpy as np
from gymnasium import spaces

from realestate_options_gym.envs.base import BaseRealEstateEnv


class MortgageHedgingEnv(BaseRealEstateEnv):
    """Environment for hedging mortgage portfolio interest rate risk.

    The agent manages a portfolio of mortgages and must use interest rate
    derivatives (swaps, caps, floors) to hedge duration and convexity risk.

    Actions:
        Continuous: [swap_notional, cap_notional, floor_notional]
        Each normalized to [-1, 1] representing sell/buy direction and size.

    Observation Space:
        - Mortgage portfolio value and duration
        - Interest rate levels and curve shape
        - Hedge positions and Greeks
        - P&L metrics
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        render_mode: str | None = None,
    ):
        """Initialize mortgage hedging environment."""
        default_config = {
            "mortgage_notional": 10_000_000,
            "mortgage_rate": 0.05,
            "mortgage_term_years": 30,
            "hedge_cost_bps": 5,
            "max_hedge_ratio": 2.0,
        }
        if config:
            default_config.update(config)

        super().__init__(config=default_config, render_mode=render_mode)

        # Mortgage and hedge state
        self.mortgage_value = 0.0
        self.mortgage_duration = 0.0
        self.swap_position = 0.0
        self.cap_position = 0.0
        self.floor_position = 0.0
        self.cumulative_pnl = 0.0

    def _create_observation_space(self) -> spaces.Box:
        """Create observation space."""
        # [mortgage_value, duration, convexity, short_rate, curve_slope,
        #  swap_pos, cap_pos, floor_pos, hedge_pnl, cumulative_pnl]
        return spaces.Box(
            low=np.array([-np.inf] * 10, dtype=np.float32),
            high=np.array([np.inf] * 10, dtype=np.float32),
        )

    def _create_action_space(self) -> spaces.Box:
        """Create action space for hedge adjustments."""
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),  # swap, cap, floor
            dtype=np.float32,
        )

    def _reset_state(self) -> None:
        """Reset mortgage and hedge state."""
        self.mortgage_value = self.config.mortgage_notional
        self.mortgage_duration = self._calculate_duration()
        self.swap_position = 0.0
        self.cap_position = 0.0
        self.floor_position = 0.0
        self.cumulative_pnl = 0.0
        self.prev_portfolio_value = self.mortgage_value

    def _calculate_duration(self) -> float:
        """Calculate modified duration of mortgage."""
        # Simplified duration calculation
        term = self.config.mortgage_term_years
        rate = self.short_rate
        # Approximate duration for amortizing mortgage
        return (1 - (1 + rate) ** (-term)) / rate

    def _calculate_convexity(self) -> float:
        """Calculate convexity of mortgage."""
        duration = self.mortgage_duration
        # Approximate convexity
        return duration * (duration + 1) / (1 + self.short_rate) ** 2

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        norm = self.config.mortgage_notional

        # Curve slope (10y - 2y spread, simplified)
        curve_slope = 0.01 + 0.005 * np.sin(self.current_step / 12)

        obs = np.array(
            [
                self.mortgage_value / norm,
                self.mortgage_duration / 10,  # Normalize by typical duration
                self._calculate_convexity() / 100,
                self.short_rate,
                curve_slope,
                self.swap_position / norm,
                self.cap_position / norm,
                self.floor_position / norm,
                (self.mortgage_value - self.prev_portfolio_value) / norm,
                self.cumulative_pnl / norm,
            ],
            dtype=np.float32,
        )
        return obs

    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward from hedging actions."""
        max_hedge = self.config.mortgage_notional * self.config.max_hedge_ratio

        # Interpret actions as hedge adjustments
        swap_delta = action[0] * max_hedge * 0.1
        cap_delta = action[1] * max_hedge * 0.1
        floor_delta = action[2] * max_hedge * 0.1

        # Transaction costs
        hedge_cost = (
            (abs(swap_delta) + abs(cap_delta) + abs(floor_delta))
            * self.config.hedge_cost_bps
            / 10000
        )

        # Update positions
        self.swap_position += swap_delta
        self.cap_position += cap_delta
        self.floor_position += floor_delta

        # Clip to limits
        self.swap_position = np.clip(self.swap_position, -max_hedge, max_hedge)
        self.cap_position = np.clip(self.cap_position, 0, max_hedge)
        self.floor_position = np.clip(self.floor_position, 0, max_hedge)

        # Calculate P&L from rate move
        rate_change = self.short_rate - self.rate_history[-2] if len(self.rate_history) > 1 else 0

        # Mortgage P&L (rates up = value down)
        mortgage_pnl = -self.mortgage_duration * rate_change * self.mortgage_value

        # Swap P&L (receive fixed = rates up is good)
        swap_pnl = self.swap_position * self.mortgage_duration * rate_change

        # Cap P&L (nonlinear, simplified)
        if self.short_rate > self.config.mortgage_rate:
            cap_pnl = self.cap_position * (self.short_rate - self.config.mortgage_rate)
        else:
            cap_pnl = 0

        # Floor P&L
        if self.short_rate < self.config.mortgage_rate - 0.02:
            floor_pnl = self.floor_position * (
                self.config.mortgage_rate - 0.02 - self.short_rate
            )
        else:
            floor_pnl = 0

        total_pnl = mortgage_pnl + swap_pnl + cap_pnl + floor_pnl - hedge_cost

        # Update mortgage value
        self.prev_portfolio_value = self.mortgage_value
        self.mortgage_value += mortgage_pnl
        self.mortgage_duration = self._calculate_duration()
        self.cumulative_pnl += total_pnl

        # Reward: Sharpe-like ratio (P&L / volatility penalty)
        vol_penalty = abs(mortgage_pnl + swap_pnl) / self.config.mortgage_notional
        reward = total_pnl / self.config.mortgage_notional - 0.1 * vol_penalty

        return float(reward)

    def _check_terminated(self) -> bool:
        """Check termination conditions."""
        # Terminate if mortgage value drops too much
        return self.mortgage_value < 0.5 * self.config.mortgage_notional

    def _get_info(self) -> dict[str, Any]:
        """Get hedge info."""
        info = super()._get_info()
        info.update(
            {
                "mortgage_value": self.mortgage_value,
                "mortgage_duration": self.mortgage_duration,
                "swap_position": self.swap_position,
                "cap_position": self.cap_position,
                "floor_position": self.floor_position,
                "cumulative_pnl": self.cumulative_pnl,
            }
        )
        return info

    def _render_ansi(self) -> str:
        """Render hedge state."""
        lines = [
            "=" * 50,
            "MORTGAGE HEDGING STATUS",
            "=" * 50,
            f"Step: {self.current_step}",
            f"Short Rate: {self.short_rate:.2%}",
            f"Mortgage Value: ${self.mortgage_value:,.0f}",
            f"Duration: {self.mortgage_duration:.2f}",
            "-" * 50,
            "Hedge Positions:",
            f"  Swap: ${self.swap_position:,.0f}",
            f"  Cap: ${self.cap_position:,.0f}",
            f"  Floor: ${self.floor_position:,.0f}",
            f"Cumulative P&L: ${self.cumulative_pnl:,.0f}",
            "=" * 50,
        ]
        return "\n".join(lines)
