"""Short-rate interest rate models.

Implements Hull-White, Vasicek, and CIR models for simulating
the evolution of short-term interest rates.
"""

from abc import ABC, abstractmethod

import numpy as np


class InterestRateModel(ABC):
    """Abstract base class for interest rate models."""

    def __init__(
        self,
        initial_rate: float = 0.05,
        seed: int | None = None,
    ):
        """Initialize the interest rate model.

        Args:
            initial_rate: Initial short rate.
            seed: Random seed for reproducibility.
        """
        self.initial_rate = initial_rate
        self.rate = initial_rate
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        """Reset the model to initial state.

        Args:
            seed: Optional new random seed.
        """
        self.rate = self.initial_rate
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    @abstractmethod
    def step(self, dt: float) -> float:
        """Simulate one time step.

        Args:
            dt: Time step size in years.

        Returns:
            New short rate.
        """
        raise NotImplementedError

    @abstractmethod
    def zero_coupon_bond_price(self, T: float) -> float:
        """Calculate zero-coupon bond price.

        Args:
            T: Time to maturity in years.

        Returns:
            Bond price.
        """
        raise NotImplementedError

    def simulate_path(self, T: float, dt: float) -> np.ndarray:
        """Simulate a rate path.

        Args:
            T: Total time horizon in years.
            dt: Time step size.

        Returns:
            Array of rates at each time step.
        """
        n_steps = int(T / dt)
        path = np.zeros(n_steps + 1)
        path[0] = self.rate

        for i in range(n_steps):
            path[i + 1] = self.step(dt)

        return path


class VasicekModel(InterestRateModel):
    """Vasicek interest rate model.

    dr = a(b - r)dt + sigma * dW

    where:
        a: Mean reversion speed
        b: Long-term mean rate
        sigma: Volatility

    The Vasicek model allows negative rates and has closed-form
    solutions for bond prices.
    """

    def __init__(
        self,
        initial_rate: float = 0.05,
        mean_reversion: float = 0.1,
        long_term_mean: float = 0.05,
        volatility: float = 0.01,
        seed: int | None = None,
    ):
        """Initialize Vasicek model.

        Args:
            initial_rate: Initial short rate.
            mean_reversion: Speed of mean reversion (a).
            long_term_mean: Long-term mean rate (b).
            volatility: Rate volatility (sigma).
            seed: Random seed.
        """
        super().__init__(initial_rate, seed)
        self.a = mean_reversion
        self.b = long_term_mean
        self.sigma = volatility

    def step(self, dt: float) -> float:
        """Simulate one time step using exact discretization.

        Args:
            dt: Time step size in years.

        Returns:
            New short rate.
        """
        # Exact simulation for Vasicek
        exp_a = np.exp(-self.a * dt)
        mean = self.rate * exp_a + self.b * (1 - exp_a)
        variance = (self.sigma**2 / (2 * self.a)) * (1 - np.exp(-2 * self.a * dt))
        std = np.sqrt(variance)

        self.rate = mean + std * self._rng.standard_normal()
        return self.rate

    def zero_coupon_bond_price(self, T: float) -> float:
        """Calculate zero-coupon bond price (closed-form).

        Args:
            T: Time to maturity.

        Returns:
            Bond price.
        """
        B = (1 - np.exp(-self.a * T)) / self.a
        A = np.exp(
            (self.b - self.sigma**2 / (2 * self.a**2)) * (B - T)
            - self.sigma**2 * B**2 / (4 * self.a)
        )
        return A * np.exp(-B * self.rate)


class HullWhiteModel(InterestRateModel):
    """Hull-White (extended Vasicek) interest rate model.

    dr = (theta(t) - a*r)dt + sigma * dW

    where theta(t) is calibrated to match the initial term structure.
    For simplicity, we use a constant theta here.
    """

    def __init__(
        self,
        initial_rate: float = 0.05,
        mean_reversion: float = 0.1,
        volatility: float = 0.01,
        theta: float | None = None,
        seed: int | None = None,
    ):
        """Initialize Hull-White model.

        Args:
            initial_rate: Initial short rate.
            mean_reversion: Speed of mean reversion (a).
            volatility: Rate volatility (sigma).
            theta: Mean level (if None, set to a * initial_rate).
            seed: Random seed.
        """
        super().__init__(initial_rate, seed)
        self.a = mean_reversion
        self.sigma = volatility
        self.theta = theta if theta is not None else self.a * initial_rate

    def step(self, dt: float) -> float:
        """Simulate one time step.

        Args:
            dt: Time step size.

        Returns:
            New short rate.
        """
        # Euler-Maruyama discretization
        drift = (self.theta - self.a * self.rate) * dt
        diffusion = self.sigma * np.sqrt(dt) * self._rng.standard_normal()
        self.rate += drift + diffusion
        return self.rate

    def zero_coupon_bond_price(self, T: float) -> float:
        """Calculate approximate zero-coupon bond price.

        Uses the Vasicek formula as approximation.

        Args:
            T: Time to maturity.

        Returns:
            Bond price.
        """
        B = (1 - np.exp(-self.a * T)) / self.a
        # Simplified A calculation
        long_term = self.theta / self.a
        A = np.exp(
            (long_term - self.sigma**2 / (2 * self.a**2)) * (B - T)
            - self.sigma**2 * B**2 / (4 * self.a)
        )
        return A * np.exp(-B * self.rate)


class CIRModel(InterestRateModel):
    """Cox-Ingersoll-Ross interest rate model.

    dr = a(b - r)dt + sigma * sqrt(r) * dW

    The CIR model ensures positive rates when 2ab > sigma^2 (Feller condition).
    """

    def __init__(
        self,
        initial_rate: float = 0.05,
        mean_reversion: float = 0.1,
        long_term_mean: float = 0.05,
        volatility: float = 0.02,
        seed: int | None = None,
    ):
        """Initialize CIR model.

        Args:
            initial_rate: Initial short rate.
            mean_reversion: Speed of mean reversion (a).
            long_term_mean: Long-term mean rate (b).
            volatility: Rate volatility (sigma).
            seed: Random seed.
        """
        super().__init__(initial_rate, seed)
        self.a = mean_reversion
        self.b = long_term_mean
        self.sigma = volatility

        # Check Feller condition
        if 2 * self.a * self.b <= self.sigma**2:
            import warnings

            warnings.warn(
                f"Feller condition not satisfied: 2ab={2*self.a*self.b:.4f} <= "
                f"sigma^2={self.sigma**2:.4f}. Rates may hit zero."
            )

    def step(self, dt: float) -> float:
        """Simulate one time step using full truncation scheme.

        Args:
            dt: Time step size.

        Returns:
            New short rate (always non-negative).
        """
        # Full truncation scheme for CIR
        r_plus = max(self.rate, 0)
        drift = self.a * (self.b - r_plus) * dt
        diffusion = self.sigma * np.sqrt(r_plus * dt) * self._rng.standard_normal()

        self.rate = max(r_plus + drift + diffusion, 0)
        return self.rate

    def zero_coupon_bond_price(self, T: float) -> float:
        """Calculate zero-coupon bond price (closed-form).

        Args:
            T: Time to maturity.

        Returns:
            Bond price.
        """
        h = np.sqrt(self.a**2 + 2 * self.sigma**2)

        numerator = 2 * h * np.exp((self.a + h) * T / 2)
        denominator = 2 * h + (self.a + h) * (np.exp(h * T) - 1)

        A = (numerator / denominator) ** (2 * self.a * self.b / self.sigma**2)

        B = 2 * (np.exp(h * T) - 1) / denominator

        return A * np.exp(-B * self.rate)
