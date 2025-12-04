"""Base environment class for real estate option trading."""

from abc import abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from realestate_options_gym.models.interest_rates import (
    CIRModel,
    HullWhiteModel,
    InterestRateModel,
    VasicekModel,
)
from realestate_options_gym.models.property_dynamics import PropertyDynamicsModel
from realestate_options_gym.utils.config import EnvConfig


class BaseRealEstateEnv(gym.Env):
    """Base class for real estate option trading environments.

    This class provides common functionality for simulating property values,
    interest rates, and market dynamics.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        render_mode: str | None = None,
    ):
        """Initialize the environment.

        Args:
            config: Environment configuration dictionary.
            render_mode: Rendering mode ('human', 'ansi', or None).
        """
        super().__init__()
        self.render_mode = render_mode
        self.config = EnvConfig(**(config or {}))

        # Initialize models
        self.rate_model = self._create_rate_model()
        self.property_model = PropertyDynamicsModel(
            initial_value=self.config.initial_property_value,
            volatility=self.config.property_volatility,
            mean_reversion_speed=self.config.mean_reversion_speed,
            jump_intensity=self.config.jump_intensity,
            jump_mean=self.config.jump_mean,
            jump_std=self.config.jump_std,
            seed=self.config.seed,
        )

        # State variables
        self.current_step = 0
        self.property_value = self.config.initial_property_value
        self.short_rate = self.config.initial_short_rate
        self.property_value_history: list[float] = []
        self.rate_history: list[float] = []

        # Define spaces (to be overridden by subclasses)
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()

        # Random generator
        self._np_random: np.random.Generator | None = None

    def _create_rate_model(self) -> InterestRateModel:
        """Create the interest rate model based on configuration."""
        model_type = self.config.interest_rate_model.lower()

        if model_type == "hull-white":
            return HullWhiteModel(
                initial_rate=self.config.initial_short_rate,
                mean_reversion=self.config.hw_mean_reversion,
                volatility=self.config.hw_volatility,
                seed=self.config.seed,
            )
        elif model_type == "vasicek":
            return VasicekModel(
                initial_rate=self.config.initial_short_rate,
                mean_reversion=self.config.vasicek_mean_reversion,
                long_term_mean=self.config.vasicek_long_term_mean,
                volatility=self.config.vasicek_volatility,
                seed=self.config.seed,
            )
        elif model_type == "cir":
            return CIRModel(
                initial_rate=self.config.initial_short_rate,
                mean_reversion=self.config.cir_mean_reversion,
                long_term_mean=self.config.cir_long_term_mean,
                volatility=self.config.cir_volatility,
                seed=self.config.seed,
            )
        else:
            raise ValueError(f"Unknown interest rate model: {model_type}")

    @abstractmethod
    def _create_observation_space(self) -> spaces.Space:
        """Create the observation space. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def _create_action_space(self) -> spaces.Space:
        """Create the action space. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def _get_observation(self) -> np.ndarray:
        """Get the current observation. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def _calculate_reward(self, action: int | np.ndarray) -> float:
        """Calculate reward for the action. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def _check_terminated(self) -> bool:
        """Check if episode is terminated. Must be implemented by subclasses."""
        raise NotImplementedError

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            Tuple of (observation, info dictionary).
        """
        super().reset(seed=seed)

        if seed is not None:
            self._np_random = np.random.default_rng(seed)
            self.rate_model.reset(seed=seed)
            self.property_model.reset(seed=seed)
        else:
            self._np_random = np.random.default_rng(self.config.seed)

        # Reset state
        self.current_step = 0
        self.property_value = self.config.initial_property_value
        self.short_rate = self.config.initial_short_rate
        self.property_value_history = [self.property_value]
        self.rate_history = [self.short_rate]

        # Subclass-specific reset
        self._reset_state()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def _reset_state(self) -> None:
        """Reset subclass-specific state. Override in subclasses."""
        pass

    def step(
        self, action: int | np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: The action to take.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        self.current_step += 1

        # Simulate market dynamics
        self._step_dynamics()

        # Calculate reward based on action
        reward = self._calculate_reward(action)

        # Check termination conditions
        terminated = self._check_terminated()
        truncated = False

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _step_dynamics(self) -> None:
        """Simulate one time step of market dynamics."""
        dt = self.config.dt

        # Update interest rate
        self.short_rate = self.rate_model.step(dt)
        self.rate_history.append(self.short_rate)

        # Update property value
        self.property_value = self.property_model.step(dt)
        self.property_value_history.append(self.property_value)

    def _get_info(self) -> dict[str, Any]:
        """Get additional information about the current state."""
        return {
            "step": self.current_step,
            "property_value": self.property_value,
            "short_rate": self.short_rate,
            "time_elapsed_months": self.current_step,
        }

    def _calculate_moving_averages(self, window_months: list[int]) -> np.ndarray:
        """Calculate moving averages of property value.

        Args:
            window_months: List of window sizes in months.

        Returns:
            Array of moving averages.
        """
        mas = []
        for window in window_months:
            if len(self.property_value_history) >= window:
                ma = np.mean(self.property_value_history[-window:])
            else:
                ma = np.mean(self.property_value_history)
            mas.append(ma)
        return np.array(mas, dtype=np.float32)

    def _estimate_volatility_regime(self) -> int:
        """Estimate current volatility regime (0=low, 1=medium, 2=high).

        Returns:
            Integer regime indicator.
        """
        if len(self.property_value_history) < 12:
            return 1  # Default to medium

        returns = np.diff(np.log(self.property_value_history[-12:]))
        realized_vol = np.std(returns) * np.sqrt(12)

        if realized_vol < 0.08:
            return 0  # Low
        elif realized_vol < 0.15:
            return 1  # Medium
        else:
            return 2  # High

    def render(self) -> str | None:
        """Render the environment state."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            print(self._render_ansi())
            return None
        return None

    def _render_ansi(self) -> str:
        """Render environment state as ASCII string."""
        lines = [
            f"Step: {self.current_step}",
            f"Property Value: ${self.property_value:,.0f}",
            f"Short Rate: {self.short_rate:.4f}",
            f"Volatility Regime: {['Low', 'Medium', 'High'][self._estimate_volatility_regime()]}",
        ]
        return "\n".join(lines)

    def close(self) -> None:
        """Clean up environment resources."""
        pass
