"""Lease option trading environment."""

from typing import Any

import numpy as np
from gymnasium import spaces

from realestate_options_gym.envs.base import BaseRealEstateEnv


class LeaseOptionEnv(BaseRealEstateEnv):
    """Environment for lease-option trading on a single property.

    A lease-option gives the tenant the right (but not obligation) to purchase
    the property at a predetermined strike price during or at the end of the
    lease term. The tenant pays monthly rent, a portion of which may be
    credited toward the purchase price.

    Actions:
        0: Hold - continue lease, pay rent
        1: Exercise - purchase property at strike price
        2: Abandon - terminate lease, forfeit accumulated credits
        3: Renegotiate - attempt to modify terms (may fail)

    Observation Space:
        - property_value: Current estimated property value
        - property_value_ma: Moving averages (3m, 6m, 12m)
        - short_rate: Current short-term interest rate
        - rate_curve: Simulated term structure (1y, 2y, 5y, 10y, 30y)
        - time_to_expiry: Remaining option term (normalized 0-1)
        - accumulated_rent_credit: Credits toward purchase
        - strike_price: Option strike price
        - volatility_regime: Estimated vol state (0, 1, 2)
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        render_mode: str | None = None,
    ):
        """Initialize the lease option environment."""
        # Set defaults specific to lease options
        default_config = {
            "lease_term_months": 36,
            "monthly_rent": 2500,
            "option_premium_pct": 0.03,
            "rent_credit_pct": 0.25,
            "strike_premium_pct": 0.05,
            "renegotiation_success_prob": 0.3,
            "renegotiation_cost": 1000,
            "transaction_cost_pct": 0.02,
        }
        if config:
            default_config.update(config)

        super().__init__(config=default_config, render_mode=render_mode)

        # Lease-specific state
        self.strike_price = 0.0
        self.accumulated_rent_credit = 0.0
        self.option_premium_paid = 0.0
        self.total_rent_paid = 0.0
        self.remaining_term = 0
        self.is_exercised = False
        self.is_abandoned = False

    def _create_observation_space(self) -> spaces.Box:
        """Create the observation space for lease option environment."""
        # 13-dimensional observation
        # [property_value, ma_3m, ma_6m, ma_12m, short_rate,
        #  rate_1y, rate_2y, rate_5y, rate_10y, rate_30y,
        #  time_to_expiry, accumulated_credit_ratio, volatility_regime]
        low = np.array(
            [0, 0, 0, 0, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 0, 0, 0],
            dtype=np.float32,
        )
        high = np.array(
            [np.inf, np.inf, np.inf, np.inf, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 2],
            dtype=np.float32,
        )
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _create_action_space(self) -> spaces.Discrete:
        """Create discrete action space."""
        return spaces.Discrete(4)

    def _reset_state(self) -> None:
        """Reset lease-specific state."""
        self.strike_price = self.config.initial_property_value * (
            1 + self.config.strike_premium_pct
        )
        self.accumulated_rent_credit = 0.0
        self.option_premium_paid = (
            self.config.initial_property_value * self.config.option_premium_pct
        )
        self.total_rent_paid = 0.0
        self.remaining_term = self.config.lease_term_months
        self.is_exercised = False
        self.is_abandoned = False

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Normalize property value by initial value for stability
        norm_factor = self.config.initial_property_value

        # Moving averages
        mas = self._calculate_moving_averages([3, 6, 12]) / norm_factor

        # Simulated term structure (simplified parallel shift from short rate)
        rate_curve = self._simulate_rate_curve()

        # Time to expiry (normalized)
        time_to_expiry = self.remaining_term / self.config.lease_term_months

        # Accumulated credit ratio (relative to strike)
        credit_ratio = self.accumulated_rent_credit / self.strike_price

        # Volatility regime
        vol_regime = float(self._estimate_volatility_regime())

        observation = np.array(
            [
                self.property_value / norm_factor,
                mas[0],
                mas[1],
                mas[2],
                self.short_rate,
                rate_curve[0],
                rate_curve[1],
                rate_curve[2],
                rate_curve[3],
                rate_curve[4],
                time_to_expiry,
                credit_ratio,
                vol_regime,
            ],
            dtype=np.float32,
        )

        return observation

    def _simulate_rate_curve(self) -> np.ndarray:
        """Simulate term structure from short rate."""
        # Simplified: assume upward sloping curve with mean reversion
        tenors = np.array([1, 2, 5, 10, 30])
        long_term_rate = 0.04  # Assume 4% long-term rate
        rates = self.short_rate + (long_term_rate - self.short_rate) * (
            1 - np.exp(-0.1 * tenors)
        )
        return rates.astype(np.float32)

    def _calculate_reward(self, action: int) -> float:
        """Calculate reward based on action taken."""
        if self.is_exercised or self.is_abandoned:
            return 0.0

        reward = 0.0
        monthly_rent = self.config.monthly_rent

        if action == 0:  # Hold
            # Pay rent, accumulate credit
            rent_credit = monthly_rent * self.config.rent_credit_pct
            self.accumulated_rent_credit += rent_credit
            self.total_rent_paid += monthly_rent
            self.remaining_term -= 1

            # Small negative reward for rent payment (opportunity cost)
            reward = -monthly_rent / self.config.initial_property_value

        elif action == 1:  # Exercise
            if self.remaining_term > 0:
                # Can exercise early
                effective_strike = self.strike_price - self.accumulated_rent_credit
                transaction_cost = self.property_value * self.config.transaction_cost_pct

                # Net value: property - cost - transaction fees - premium already paid
                net_value = (
                    self.property_value
                    - effective_strike
                    - transaction_cost
                    - self.option_premium_paid
                    - self.total_rent_paid
                )

                reward = net_value / self.config.initial_property_value
                self.is_exercised = True

        elif action == 2:  # Abandon
            # Forfeit all credits and premium
            reward = -(self.option_premium_paid + self.total_rent_paid) / self.config.initial_property_value
            self.is_abandoned = True

        elif action == 3:  # Renegotiate
            # Attempt to renegotiate terms
            if self._np_random.random() < self.config.renegotiation_success_prob:
                # Success: extend term by 6 months, reduce strike by 2%
                self.remaining_term += 6
                self.strike_price *= 0.98
                reward = 0.01  # Small positive reward
            else:
                # Failed: pay cost, no benefit
                reward = -self.config.renegotiation_cost / self.config.initial_property_value

        return float(reward)

    def _check_terminated(self) -> bool:
        """Check if episode should terminate."""
        if self.is_exercised or self.is_abandoned:
            return True

        if self.remaining_term <= 0:
            # Option expired - must exercise or abandon
            effective_strike = self.strike_price - self.accumulated_rent_credit

            if self.property_value > effective_strike:
                # In the money - auto exercise
                self.is_exercised = True
            else:
                # Out of money - auto abandon
                self.is_abandoned = True

            return True

        return False

    def _get_info(self) -> dict[str, Any]:
        """Get additional info about current state."""
        info = super()._get_info()
        info.update(
            {
                "strike_price": self.strike_price,
                "accumulated_rent_credit": self.accumulated_rent_credit,
                "effective_strike": self.strike_price - self.accumulated_rent_credit,
                "remaining_term": self.remaining_term,
                "option_premium_paid": self.option_premium_paid,
                "total_rent_paid": self.total_rent_paid,
                "is_exercised": self.is_exercised,
                "is_abandoned": self.is_abandoned,
                "intrinsic_value": max(
                    0, self.property_value - (self.strike_price - self.accumulated_rent_credit)
                ),
            }
        )
        return info

    def _render_ansi(self) -> str:
        """Render lease option state as ASCII."""
        effective_strike = self.strike_price - self.accumulated_rent_credit
        intrinsic = max(0, self.property_value - effective_strike)
        moneyness = "ITM" if intrinsic > 0 else "OTM"

        lines = [
            "=" * 50,
            "LEASE OPTION STATUS",
            "=" * 50,
            f"Step: {self.current_step} | Remaining: {self.remaining_term} months",
            f"Property Value: ${self.property_value:,.0f}",
            f"Strike Price: ${self.strike_price:,.0f}",
            f"Rent Credits: ${self.accumulated_rent_credit:,.0f}",
            f"Effective Strike: ${effective_strike:,.0f}",
            f"Intrinsic Value: ${intrinsic:,.0f} ({moneyness})",
            f"Short Rate: {self.short_rate:.2%}",
            f"Status: {'Exercised' if self.is_exercised else 'Abandoned' if self.is_abandoned else 'Active'}",
            "=" * 50,
        ]
        return "\n".join(lines)
