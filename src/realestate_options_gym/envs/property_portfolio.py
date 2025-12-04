"""Multi-property portfolio management environment."""

from typing import Any

import numpy as np
from gymnasium import spaces

from realestate_options_gym.envs.base import BaseRealEstateEnv


class PropertyPortfolioEnv(BaseRealEstateEnv):
    """Environment for managing a portfolio of real estate properties.

    The agent allocates capital across multiple properties with different
    risk/return characteristics. Properties can be acquired, held, or sold.

    Actions:
        Continuous action space representing target allocation weights
        for each property in the universe.

    Observation Space:
        - Property values for each asset
        - Property returns and volatilities
        - Interest rate environment
        - Portfolio composition
        - Market conditions
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        render_mode: str | None = None,
    ):
        """Initialize the portfolio environment."""
        default_config = {
            "n_properties": 5,
            "initial_capital": 1_000_000,
            "transaction_cost_pct": 0.02,
            "holding_cost_monthly_pct": 0.005,
            "leverage_limit": 0.7,  # Max LTV
            "property_correlations": None,  # If None, use default
        }
        if config:
            default_config.update(config)

        super().__init__(config=default_config, render_mode=render_mode)

        # Portfolio state
        self.n_properties = self.config.n_properties
        self.capital = 0.0
        self.property_values = np.zeros(self.n_properties)
        self.property_holdings = np.zeros(self.n_properties)  # Fraction owned
        self.property_volatilities = np.zeros(self.n_properties)
        self.portfolio_value = 0.0

    def _create_observation_space(self) -> spaces.Box:
        """Create observation space for portfolio environment."""
        n = self.config.n_properties
        # Per property: value, 12m return, volatility, holding
        # Plus: cash, short rate, portfolio value, leverage
        obs_dim = n * 4 + 4

        low = np.full(obs_dim, -np.inf, dtype=np.float32)
        high = np.full(obs_dim, np.inf, dtype=np.float32)

        # Bound holdings to [0, 1]
        low[3 * n : 4 * n] = 0
        high[3 * n : 4 * n] = 1

        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _create_action_space(self) -> spaces.Box:
        """Create continuous action space for allocations."""
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.config.n_properties,),
            dtype=np.float32,
        )

    def _reset_state(self) -> None:
        """Reset portfolio state."""
        n = self.n_properties

        # Initialize property values with some variation
        base_value = self.config.initial_property_value
        self.property_values = base_value * (
            1 + 0.3 * self._np_random.standard_normal(n)
        )
        self.property_values = np.maximum(self.property_values, base_value * 0.5)

        # Initialize volatilities (different for each property type)
        self.property_volatilities = 0.10 + 0.08 * self._np_random.random(n)

        # Start with cash only
        self.capital = self.config.initial_capital
        self.property_holdings = np.zeros(n)
        self.portfolio_value = self.capital

        # Track history for each property
        self.property_histories = [[v] for v in self.property_values]

    def _step_dynamics(self) -> None:
        """Simulate property and rate dynamics."""
        super()._step_dynamics()

        dt = self.config.dt
        n = self.n_properties

        # Update each property value independently
        for i in range(n):
            drift = 0.03 * dt  # 3% annual appreciation
            diffusion = self.property_volatilities[i] * np.sqrt(dt)
            shock = self._np_random.standard_normal()

            # Log-normal dynamics
            self.property_values[i] *= np.exp(
                (drift - 0.5 * self.property_volatilities[i] ** 2) * dt
                + diffusion * shock
            )
            self.property_histories[i].append(self.property_values[i])

    def _get_observation(self) -> np.ndarray:
        """Get current portfolio observation."""
        n = self.n_properties
        norm = self.config.initial_capital

        # Property values (normalized)
        values = self.property_values / norm

        # 12-month returns
        returns = np.zeros(n)
        for i in range(n):
            if len(self.property_histories[i]) > 12:
                returns[i] = (
                    self.property_histories[i][-1] / self.property_histories[i][-12] - 1
                )

        # Volatilities
        vols = self.property_volatilities

        # Holdings
        holdings = self.property_holdings

        # Portfolio metrics
        cash_norm = self.capital / norm
        portfolio_norm = self.portfolio_value / norm
        leverage = 1 - (self.capital / max(self.portfolio_value, 1))

        obs = np.concatenate(
            [
                values,
                returns,
                vols,
                holdings,
                [cash_norm, self.short_rate, portfolio_norm, leverage],
            ]
        ).astype(np.float32)

        return obs

    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward from portfolio rebalancing."""
        # Interpret action as target weights
        target_weights = np.clip(action, 0, 1)
        target_weights = target_weights / (target_weights.sum() + 1e-8)

        # Current portfolio value
        current_holdings_value = np.sum(self.property_holdings * self.property_values)
        old_portfolio = self.capital + current_holdings_value

        # Execute trades to reach target
        target_values = target_weights * old_portfolio
        current_values = self.property_holdings * self.property_values

        trades = target_values - current_values
        transaction_costs = np.sum(np.abs(trades)) * self.config.transaction_cost_pct

        # Update holdings
        self.property_holdings = target_values / np.maximum(self.property_values, 1)
        self.capital = old_portfolio - np.sum(target_values) - transaction_costs

        # Apply holding costs
        holding_cost = (
            np.sum(self.property_holdings * self.property_values)
            * self.config.holding_cost_monthly_pct
        )
        self.capital -= holding_cost

        # New portfolio value
        new_holdings_value = np.sum(self.property_holdings * self.property_values)
        self.portfolio_value = self.capital + new_holdings_value

        # Reward: risk-adjusted return
        pnl = self.portfolio_value - old_portfolio
        reward = pnl / old_portfolio

        return float(reward)

    def _check_terminated(self) -> bool:
        """Check if portfolio is bankrupt."""
        return self.portfolio_value < 0.1 * self.config.initial_capital

    def _get_info(self) -> dict[str, Any]:
        """Get portfolio info."""
        info = super()._get_info()
        info.update(
            {
                "portfolio_value": self.portfolio_value,
                "capital": self.capital,
                "holdings": self.property_holdings.tolist(),
                "property_values": self.property_values.tolist(),
                "leverage": 1 - (self.capital / max(self.portfolio_value, 1)),
            }
        )
        return info

    def _render_ansi(self) -> str:
        """Render portfolio state."""
        lines = [
            "=" * 60,
            "PROPERTY PORTFOLIO STATUS",
            "=" * 60,
            f"Step: {self.current_step}",
            f"Portfolio Value: ${self.portfolio_value:,.0f}",
            f"Cash: ${self.capital:,.0f}",
            f"Short Rate: {self.short_rate:.2%}",
            "-" * 60,
            "Holdings:",
        ]

        for i in range(self.n_properties):
            value = self.property_holdings[i] * self.property_values[i]
            pct = value / max(self.portfolio_value, 1) * 100
            lines.append(
                f"  Property {i + 1}: ${value:,.0f} ({pct:.1f}%) "
                f"[Vol: {self.property_volatilities[i]:.1%}]"
            )

        lines.append("=" * 60)
        return "\n".join(lines)
