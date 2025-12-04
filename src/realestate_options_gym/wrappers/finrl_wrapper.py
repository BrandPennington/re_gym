"""Wrapper for FinRL compatibility.

Adapts RealEstateOptionsGym environments to work with FinRL's
DRLAgent and backtesting infrastructure.
"""

from typing import Any

import gymnasium as gym
import numpy as np


class FinRLWrapper(gym.Wrapper):
    """Wrapper to make environments compatible with FinRL.

    FinRL expects specific attributes and methods that this wrapper provides:
    - state: Current observation as a flat array
    - action_dim: Dimension of action space
    - stock_dim: Number of assets (mapped to n_properties)
    - reward_memory: List of rewards for analysis
    - asset_memory: Portfolio value history
    """

    def __init__(self, env: gym.Env):
        """Initialize FinRL wrapper.

        Args:
            env: A RealEstateOptionsGym environment.
        """
        super().__init__(env)

        # FinRL expected attributes
        self.reward_memory: list[float] = []
        self.asset_memory: list[float] = []
        self.actions_memory: list[Any] = []
        self.date_memory: list[int] = []

        # Map to FinRL conventions
        self._setup_finrl_attributes()

    def _setup_finrl_attributes(self) -> None:
        """Set up attributes expected by FinRL."""
        # Action dimension
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, gym.spaces.Box):
            self.action_dim = self.action_space.shape[0]
        else:
            self.action_dim = 1

        # Stock dimension (number of assets)
        # For single-property envs, this is 1
        # For portfolio envs, get from config
        self.stock_dim = getattr(self.env.config, "n_properties", 1)

        # State dimension
        self.state_dim = self.observation_space.shape[0]

        # Placeholder for state
        self.state: np.ndarray | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment and tracking.

        Args:
            seed: Random seed.
            options: Additional options.

        Returns:
            Tuple of (observation, info).
        """
        # Clear memory
        self.reward_memory = []
        self.asset_memory = []
        self.actions_memory = []
        self.date_memory = []

        # Reset underlying env
        obs, info = self.env.reset(seed=seed, options=options)

        # Track initial state
        self.state = obs.flatten()
        self.asset_memory.append(self._get_portfolio_value(info))
        self.date_memory.append(0)

        return obs, info

    def step(
        self, action: int | np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Take a step and track for FinRL.

        Args:
            action: Action to take.

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Update state
        self.state = obs.flatten()

        # Track for analysis
        self.reward_memory.append(reward)
        self.actions_memory.append(action)
        self.asset_memory.append(self._get_portfolio_value(info))
        self.date_memory.append(info.get("step", len(self.date_memory)))

        return obs, reward, terminated, truncated, info

    def _get_portfolio_value(self, info: dict[str, Any]) -> float:
        """Extract portfolio value from info dict.

        Args:
            info: Step info dictionary.

        Returns:
            Portfolio value.
        """
        # Try common keys
        for key in ["portfolio_value", "property_value", "mortgage_value", "capital"]:
            if key in info:
                return float(info[key])

        # Default: use property value from base env
        return float(getattr(self.env, "property_value", 0))

    def get_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from episode.

        Args:
            risk_free_rate: Annual risk-free rate.

        Returns:
            Sharpe ratio (annualized).
        """
        if len(self.reward_memory) < 2:
            return 0.0

        returns = np.array(self.reward_memory)
        excess_returns = returns - risk_free_rate / 252  # Assume daily

        if np.std(excess_returns) == 0:
            return 0.0

        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return float(sharpe)

    def get_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio from episode.

        Args:
            risk_free_rate: Annual risk-free rate.

        Returns:
            Sortino ratio (annualized).
        """
        if len(self.reward_memory) < 2:
            return 0.0

        returns = np.array(self.reward_memory)
        excess_returns = returns - risk_free_rate / 252

        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return float("inf") if np.mean(excess_returns) > 0 else 0.0

        sortino = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
        return float(sortino)

    def get_max_drawdown(self) -> float:
        """Calculate maximum drawdown from episode.

        Returns:
            Maximum drawdown as a positive fraction.
        """
        if len(self.asset_memory) < 2:
            return 0.0

        values = np.array(self.asset_memory)
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak

        return float(np.max(drawdown))

    def get_episode_stats(self) -> dict[str, float]:
        """Get comprehensive episode statistics.

        Returns:
            Dictionary with performance metrics.
        """
        if len(self.asset_memory) < 2:
            return {}

        values = np.array(self.asset_memory)
        returns = np.diff(values) / values[:-1]

        return {
            "total_return": (values[-1] / values[0]) - 1,
            "sharpe_ratio": self.get_sharpe_ratio(),
            "sortino_ratio": self.get_sortino_ratio(),
            "max_drawdown": self.get_max_drawdown(),
            "volatility": float(np.std(returns) * np.sqrt(252)),
            "num_steps": len(self.reward_memory),
            "final_value": values[-1],
            "cumulative_reward": sum(self.reward_memory),
        }

    def save_episode_data(self, filepath: str) -> None:
        """Save episode data to CSV for analysis.

        Args:
            filepath: Path to save CSV file.
        """
        import pandas as pd

        data = {
            "step": self.date_memory[: len(self.reward_memory)],
            "reward": self.reward_memory,
            "portfolio_value": self.asset_memory[1:],  # Skip initial
        }

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
