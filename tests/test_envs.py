"""Tests for environment implementations."""

import gymnasium as gym
import numpy as np
import pytest

import realestate_options_gym
from realestate_options_gym.envs import (
    LeaseOptionEnv,
    MortgageHedgingEnv,
    PropertyPortfolioEnv,
    REITTradingEnv,
)


class TestLeaseOptionEnv:
    """Tests for LeaseOptionEnv."""

    def test_creation(self):
        """Test environment can be created."""
        env = LeaseOptionEnv()
        assert env is not None
        assert env.observation_space is not None
        assert env.action_space is not None

    def test_gym_make(self):
        """Test creation via gym.make."""
        env = gym.make("LeaseOption-v1")
        assert env is not None

    def test_reset(self):
        """Test environment reset."""
        env = LeaseOptionEnv()
        obs, info = env.reset(seed=42)

        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)
        assert "property_value" in info

    def test_step_hold(self):
        """Test hold action."""
        env = LeaseOptionEnv()
        env.reset(seed=42)

        obs, reward, terminated, truncated, info = env.step(0)  # Hold

        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert not terminated  # Should not terminate on first step
        assert info["remaining_term"] == env.config.lease_term_months - 1

    def test_step_exercise(self):
        """Test exercise action terminates episode."""
        env = LeaseOptionEnv()
        env.reset(seed=42)

        obs, reward, terminated, truncated, info = env.step(1)  # Exercise

        assert terminated
        assert info["is_exercised"]

    def test_step_abandon(self):
        """Test abandon action terminates episode."""
        env = LeaseOptionEnv()
        env.reset(seed=42)

        obs, reward, terminated, truncated, info = env.step(2)  # Abandon

        assert terminated
        assert info["is_abandoned"]

    def test_full_episode(self):
        """Test running a full episode with holds."""
        env = LeaseOptionEnv(config={"lease_term_months": 12})
        obs, _ = env.reset(seed=42)

        total_reward = 0
        for _ in range(12):
            obs, reward, terminated, truncated, info = env.step(0)  # Hold
            total_reward += reward
            if terminated:
                break

        assert terminated  # Should terminate when lease expires

    def test_rent_credit_accumulation(self):
        """Test that rent credits accumulate correctly."""
        env = LeaseOptionEnv()
        env.reset(seed=42)

        initial_credit = env.accumulated_rent_credit
        env.step(0)  # Hold

        expected_credit = env.config.monthly_rent * env.config.rent_credit_pct
        assert env.accumulated_rent_credit == initial_credit + expected_credit


class TestPropertyPortfolioEnv:
    """Tests for PropertyPortfolioEnv."""

    def test_creation(self):
        """Test environment creation."""
        env = PropertyPortfolioEnv()
        assert env is not None

    def test_gym_make(self):
        """Test creation via gym.make."""
        env = gym.make("PropertyPortfolio-v1")
        assert env is not None

    def test_action_space(self):
        """Test action space is continuous."""
        env = PropertyPortfolioEnv(config={"n_properties": 5})
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (5,)

    def test_reset(self):
        """Test reset initializes portfolio."""
        env = PropertyPortfolioEnv()
        obs, info = env.reset(seed=42)

        assert info["capital"] == env.config.initial_capital
        assert info["portfolio_value"] == info["capital"]  # Start in cash

    def test_step(self):
        """Test step with allocation."""
        env = PropertyPortfolioEnv(config={"n_properties": 3})
        env.reset(seed=42)

        # Allocate 50% to first property
        action = np.array([0.5, 0.25, 0.25])
        obs, reward, terminated, truncated, info = env.step(action)

        assert not terminated
        assert info["portfolio_value"] > 0


class TestMortgageHedgingEnv:
    """Tests for MortgageHedgingEnv."""

    def test_creation(self):
        """Test environment creation."""
        env = MortgageHedgingEnv()
        assert env is not None

    def test_gym_make(self):
        """Test creation via gym.make."""
        env = gym.make("MortgageHedging-v1")
        assert env is not None

    def test_action_space(self):
        """Test action space for hedging."""
        env = MortgageHedgingEnv()
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (3,)  # swap, cap, floor

    def test_hedge_positions(self):
        """Test that hedge positions update."""
        env = MortgageHedgingEnv()
        env.reset(seed=42)

        # Add swap position
        action = np.array([1.0, 0.0, 0.0])  # Max swap
        env.step(action)

        assert env.swap_position != 0


class TestREITTradingEnv:
    """Tests for REITTradingEnv."""

    def test_creation(self):
        """Test environment creation."""
        env = REITTradingEnv()
        assert env is not None

    def test_gym_make(self):
        """Test creation via gym.make."""
        env = gym.make("REITTrading-v1")
        assert env is not None

    def test_buy_action(self):
        """Test buying shares."""
        env = REITTradingEnv()
        env.reset(seed=42)

        initial_shares = env.shares_held
        initial_cash = env.cash

        env.step(1)  # Buy

        assert env.shares_held > initial_shares
        assert env.cash < initial_cash

    def test_sell_without_shares(self):
        """Test selling without shares does nothing."""
        env = REITTradingEnv()
        env.reset(seed=42)

        initial_shares = env.shares_held
        initial_cash = env.cash

        env.step(2)  # Sell

        assert env.shares_held == initial_shares  # No change
        assert env.cash == initial_cash

    def test_dividend_payment(self):
        """Test dividend is paid quarterly."""
        env = REITTradingEnv()
        env.reset(seed=42)

        # Buy shares
        env.step(1)
        initial_dividends = env.dividends_received

        # Step through a quarter
        for _ in range(3):
            env.step(0)  # Hold

        assert env.dividends_received > initial_dividends


class TestEnvironmentSeeding:
    """Test reproducibility with seeds."""

    @pytest.mark.parametrize(
        "env_id",
        ["LeaseOption-v1", "PropertyPortfolio-v1", "MortgageHedging-v1", "REITTrading-v1"],
    )
    def test_seed_reproducibility(self, env_id):
        """Test that same seed gives same results."""
        env1 = gym.make(env_id)
        env2 = gym.make(env_id)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        np.testing.assert_array_equal(obs1, obs2)

        # Take same action
        if isinstance(env1.action_space, gym.spaces.Discrete):
            action = 0
        else:
            action = env1.action_space.sample()
            env2.action_space.seed(42)
            action = action  # Use same action

        obs1, r1, _, _, _ = env1.step(action)
        obs2, r2, _, _, _ = env2.step(action)

        np.testing.assert_array_almost_equal(obs1, obs2)
        assert r1 == r2


class TestRenderModes:
    """Test rendering functionality."""

    def test_ansi_render(self):
        """Test ANSI rendering."""
        env = LeaseOptionEnv(render_mode="ansi")
        env.reset(seed=42)

        output = env.render()

        assert isinstance(output, str)
        assert "LEASE OPTION STATUS" in output

    def test_human_render(self, capsys):
        """Test human rendering prints to stdout."""
        env = LeaseOptionEnv(render_mode="human")
        env.reset(seed=42)

        env.render()

        captured = capsys.readouterr()
        assert "LEASE OPTION STATUS" in captured.out
