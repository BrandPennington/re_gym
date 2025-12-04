"""REIT trading environment with property-level simulation."""

from typing import Any

import numpy as np
from gymnasium import spaces

from realestate_options_gym.envs.base import BaseRealEstateEnv


class REITTradingEnv(BaseRealEstateEnv):
    """Environment for trading REITs with underlying property simulation.

    Unlike standard equity environments, this models the underlying property
    portfolio that drives REIT valuations, including NAV premiums/discounts.

    Actions:
        Discrete: 0=Hold, 1=Buy, 2=Sell
        Or continuous position sizing.

    Observation Space:
        - REIT price and NAV
        - Premium/discount to NAV
        - Underlying property metrics
        - Interest rate sensitivity
        - Dividend yield
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        render_mode: str | None = None,
    ):
        """Initialize REIT trading environment."""
        default_config = {
            "initial_nav": 100.0,
            "initial_premium": 0.0,  # Premium to NAV
            "dividend_yield": 0.04,
            "nav_volatility": 0.15,
            "premium_mean_reversion": 0.5,
            "premium_volatility": 0.10,
            "initial_cash": 100_000,
            "shares_per_trade": 100,
            "transaction_cost_per_share": 0.01,
        }
        if config:
            default_config.update(config)

        super().__init__(config=default_config, render_mode=render_mode)

        # REIT state
        self.nav = 0.0
        self.premium = 0.0
        self.price = 0.0
        self.shares_held = 0
        self.cash = 0.0
        self.dividends_received = 0.0

    def _create_observation_space(self) -> spaces.Box:
        """Create observation space."""
        # [nav, price, premium, dividend_yield, short_rate,
        #  rate_sensitivity, shares_held_norm, cash_norm, portfolio_return]
        return spaces.Box(
            low=np.array([0, 0, -1, 0, -0.1, -10, 0, 0, -1], dtype=np.float32),
            high=np.array([np.inf, np.inf, 1, 0.2, 0.5, 10, 1, np.inf, 1], dtype=np.float32),
        )

    def _create_action_space(self) -> spaces.Discrete:
        """Create discrete action space."""
        return spaces.Discrete(3)  # Hold, Buy, Sell

    def _reset_state(self) -> None:
        """Reset REIT state."""
        self.nav = self.config.initial_nav
        self.premium = self.config.initial_premium
        self.price = self.nav * (1 + self.premium)
        self.shares_held = 0
        self.cash = self.config.initial_cash
        self.dividends_received = 0.0
        self.initial_portfolio_value = self.cash

    def _step_dynamics(self) -> None:
        """Simulate REIT dynamics."""
        super()._step_dynamics()

        dt = self.config.dt

        # NAV follows property dynamics (correlated with property_value)
        nav_return = np.log(self.property_value / self.property_value_history[-2]) if len(self.property_value_history) > 1 else 0
        self.nav *= np.exp(nav_return * 0.8)  # 80% correlation

        # Premium mean-reverts
        premium_drift = -self.config.premium_mean_reversion * self.premium * dt
        premium_diffusion = self.config.premium_volatility * np.sqrt(dt)
        self.premium += premium_drift + premium_diffusion * self._np_random.standard_normal()
        self.premium = np.clip(self.premium, -0.5, 0.5)

        # Update price
        self.price = self.nav * (1 + self.premium)

        # Quarterly dividend
        if self.current_step % 3 == 0 and self.shares_held > 0:
            quarterly_div = self.nav * self.config.dividend_yield / 4
            dividend = quarterly_div * self.shares_held
            self.cash += dividend
            self.dividends_received += dividend

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Rate sensitivity (duration-like measure)
        rate_sensitivity = -5.0 * (1 + self.premium)  # REITs are rate sensitive

        # Normalize holdings
        max_shares = self.config.initial_cash / self.config.initial_nav
        shares_norm = self.shares_held / max_shares

        # Cash normalized
        cash_norm = self.cash / self.config.initial_cash

        # Portfolio return
        portfolio_value = self.cash + self.shares_held * self.price
        portfolio_return = portfolio_value / self.initial_portfolio_value - 1

        obs = np.array(
            [
                self.nav,
                self.price,
                self.premium,
                self.config.dividend_yield,
                self.short_rate,
                rate_sensitivity,
                shares_norm,
                cash_norm,
                portfolio_return,
            ],
            dtype=np.float32,
        )
        return obs

    def _calculate_reward(self, action: int) -> float:
        """Calculate reward from trading action."""
        shares_to_trade = self.config.shares_per_trade
        cost_per_share = self.config.transaction_cost_per_share

        old_portfolio = self.cash + self.shares_held * self.price
        transaction_cost = 0.0

        if action == 1:  # Buy
            cost = shares_to_trade * self.price + shares_to_trade * cost_per_share
            if self.cash >= cost:
                self.shares_held += shares_to_trade
                self.cash -= cost
                transaction_cost = shares_to_trade * cost_per_share

        elif action == 2:  # Sell
            if self.shares_held >= shares_to_trade:
                self.shares_held -= shares_to_trade
                proceeds = shares_to_trade * self.price - shares_to_trade * cost_per_share
                self.cash += proceeds
                transaction_cost = shares_to_trade * cost_per_share

        # New portfolio value
        new_portfolio = self.cash + self.shares_held * self.price

        # Reward: portfolio return minus transaction costs
        reward = (new_portfolio - old_portfolio) / old_portfolio

        return float(reward)

    def _check_terminated(self) -> bool:
        """Check if trading should stop."""
        portfolio_value = self.cash + self.shares_held * self.price
        # Terminate if lost 50% or more
        return portfolio_value < 0.5 * self.initial_portfolio_value

    def _get_info(self) -> dict[str, Any]:
        """Get trading info."""
        info = super()._get_info()
        portfolio_value = self.cash + self.shares_held * self.price
        info.update(
            {
                "nav": self.nav,
                "price": self.price,
                "premium_to_nav": self.premium,
                "shares_held": self.shares_held,
                "cash": self.cash,
                "portfolio_value": portfolio_value,
                "dividends_received": self.dividends_received,
                "total_return": portfolio_value / self.initial_portfolio_value - 1,
            }
        )
        return info

    def _render_ansi(self) -> str:
        """Render REIT trading state."""
        portfolio_value = self.cash + self.shares_held * self.price
        total_return = (portfolio_value / self.initial_portfolio_value - 1) * 100

        lines = [
            "=" * 50,
            "REIT TRADING STATUS",
            "=" * 50,
            f"Step: {self.current_step}",
            f"NAV: ${self.nav:.2f}",
            f"Price: ${self.price:.2f}",
            f"Premium/Discount: {self.premium:+.1%}",
            f"Short Rate: {self.short_rate:.2%}",
            "-" * 50,
            f"Shares Held: {self.shares_held}",
            f"Cash: ${self.cash:,.2f}",
            f"Holdings Value: ${self.shares_held * self.price:,.2f}",
            f"Portfolio Value: ${portfolio_value:,.2f}",
            f"Total Return: {total_return:+.2f}%",
            f"Dividends Received: ${self.dividends_received:,.2f}",
            "=" * 50,
        ]
        return "\n".join(lines)
