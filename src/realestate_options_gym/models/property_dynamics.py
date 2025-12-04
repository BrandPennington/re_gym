"""Property value dynamics model.

Implements a mean-reverting jump-diffusion process for property values.
"""

import numpy as np


class PropertyDynamicsModel:
    """Mean-reverting jump-diffusion model for property values.

    dP = kappa * (theta - P) * dt + sigma * P * dW + P * dJ

    where:
        kappa: Mean reversion speed
        theta: Long-term property value (can be linked to market index)
        sigma: Base volatility (can be regime-dependent)
        dJ: Compound Poisson jump process

    The model captures:
        - Mean reversion toward local market trends
        - Stochastic volatility via regime switching
        - Jump risk for market shocks (e.g., 2008 crisis)
    """

    def __init__(
        self,
        initial_value: float = 500_000,
        volatility: float = 0.12,
        mean_reversion_speed: float = 0.5,
        long_term_value: float | None = None,
        jump_intensity: float = 0.1,
        jump_mean: float = -0.05,
        jump_std: float = 0.10,
        regime_probs: tuple[float, float, float] | None = None,
        regime_vol_multipliers: tuple[float, float, float] | None = None,
        seed: int | None = None,
    ):
        """Initialize property dynamics model.

        Args:
            initial_value: Initial property value.
            volatility: Base annual volatility.
            mean_reversion_speed: Speed of mean reversion (kappa).
            long_term_value: Long-term target value (theta). If None, uses initial.
            jump_intensity: Annual jump frequency (lambda).
            jump_mean: Average jump size (negative = crash).
            jump_std: Jump size standard deviation.
            regime_probs: Probability of (low, medium, high) vol regimes.
            regime_vol_multipliers: Vol multipliers for each regime.
            seed: Random seed.
        """
        self.initial_value = initial_value
        self.value = initial_value
        self.base_volatility = volatility
        self.kappa = mean_reversion_speed
        self.theta = long_term_value if long_term_value is not None else initial_value
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std

        # Regime switching
        self.regime_probs = regime_probs or (0.6, 0.3, 0.1)
        self.regime_vol_multipliers = regime_vol_multipliers or (0.7, 1.0, 1.5)
        self.current_regime = 1  # Start in medium vol

        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        """Reset model to initial state.

        Args:
            seed: Optional new random seed.
        """
        self.value = self.initial_value
        self.current_regime = 1
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def step(self, dt: float) -> float:
        """Simulate one time step.

        Args:
            dt: Time step size in years.

        Returns:
            New property value.
        """
        # Regime switching (Markov chain)
        self._update_regime()

        # Current volatility based on regime
        vol = self.base_volatility * self.regime_vol_multipliers[self.current_regime]

        # Mean-reverting drift
        drift = self.kappa * (self.theta - self.value) * dt

        # Diffusion
        diffusion = vol * self.value * np.sqrt(dt) * self._rng.standard_normal()

        # Jump component
        jump = self._simulate_jump(dt)

        # Update value (ensure non-negative)
        self.value = max(self.value + drift + diffusion + jump, 0.01 * self.initial_value)

        return self.value

    def _update_regime(self) -> None:
        """Update volatility regime using simplified Markov transition."""
        # Transition probabilities (simplified)
        # High persistence within regime, small probability of change
        stay_prob = 0.95
        u = self._rng.random()

        if u > stay_prob:
            # Transition to adjacent regime
            if self.current_regime == 0:
                self.current_regime = 1
            elif self.current_regime == 2:
                self.current_regime = 1
            else:
                # From medium, can go to low or high
                self.current_regime = 0 if self._rng.random() < 0.5 else 2

    def _simulate_jump(self, dt: float) -> float:
        """Simulate jump component.

        Args:
            dt: Time step size.

        Returns:
            Jump size (can be zero).
        """
        # Poisson arrival
        n_jumps = self._rng.poisson(self.jump_intensity * dt)

        if n_jumps == 0:
            return 0.0

        # Sum of jump sizes (log-normal)
        total_jump = 0.0
        for _ in range(n_jumps):
            jump_size = np.exp(
                self.jump_mean + self.jump_std * self._rng.standard_normal()
            ) - 1
            total_jump += self.value * jump_size

        return total_jump

    def get_regime(self) -> int:
        """Get current volatility regime.

        Returns:
            Regime index (0=low, 1=medium, 2=high).
        """
        return self.current_regime

    def get_current_volatility(self) -> float:
        """Get current effective volatility.

        Returns:
            Annual volatility.
        """
        return self.base_volatility * self.regime_vol_multipliers[self.current_regime]

    def simulate_path(
        self,
        T: float,
        dt: float,
        n_paths: int = 1,
    ) -> np.ndarray:
        """Simulate multiple property value paths.

        Args:
            T: Total time horizon in years.
            dt: Time step size.
            n_paths: Number of paths to simulate.

        Returns:
            Array of shape (n_paths, n_steps + 1) with simulated values.
        """
        n_steps = int(T / dt)
        paths = np.zeros((n_paths, n_steps + 1))

        for i in range(n_paths):
            self.reset()
            paths[i, 0] = self.value

            for j in range(n_steps):
                paths[i, j + 1] = self.step(dt)

        return paths

    def estimate_var(
        self,
        horizon_years: float = 1.0,
        confidence: float = 0.95,
        n_simulations: int = 10000,
    ) -> float:
        """Estimate Value-at-Risk via Monte Carlo.

        Args:
            horizon_years: VaR time horizon.
            confidence: Confidence level (e.g., 0.95 for 95% VaR).
            n_simulations: Number of Monte Carlo paths.

        Returns:
            VaR as a positive loss amount.
        """
        dt = 1 / 12  # Monthly steps
        paths = self.simulate_path(horizon_years, dt, n_simulations)
        final_values = paths[:, -1]

        # Calculate returns
        returns = (final_values - self.initial_value) / self.initial_value

        # VaR is the negative of the quantile
        var = -np.percentile(returns, (1 - confidence) * 100)

        return var * self.initial_value
