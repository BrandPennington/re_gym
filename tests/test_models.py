"""Tests for stochastic models."""

import numpy as np
import pytest

from realestate_options_gym.models.interest_rates import (
    CIRModel,
    HullWhiteModel,
    VasicekModel,
)
from realestate_options_gym.models.property_dynamics import PropertyDynamicsModel


class TestVasicekModel:
    """Tests for Vasicek interest rate model."""

    def test_creation(self):
        """Test model creation."""
        model = VasicekModel()
        assert model.rate == model.initial_rate

    def test_step(self):
        """Test single step simulation."""
        model = VasicekModel(initial_rate=0.05, seed=42)
        new_rate = model.step(1 / 12)

        assert new_rate != 0.05  # Should have changed
        assert model.rate == new_rate

    def test_mean_reversion(self):
        """Test that rates mean revert over time."""
        model = VasicekModel(
            initial_rate=0.10,  # Start high
            mean_reversion=0.5,
            long_term_mean=0.05,
            seed=42,
        )

        # Simulate for 10 years
        for _ in range(120):
            model.step(1 / 12)

        # Should be closer to long-term mean
        assert abs(model.rate - 0.05) < abs(0.10 - 0.05)

    def test_bond_price_positive(self):
        """Test that bond prices are positive."""
        model = VasicekModel(initial_rate=0.05)

        for T in [0.5, 1.0, 5.0, 10.0]:
            price = model.zero_coupon_bond_price(T)
            assert price > 0
            assert price < 1  # Should be less than 1 for positive rates

    def test_bond_price_decreasing_maturity(self):
        """Test that bond prices decrease with maturity."""
        model = VasicekModel(initial_rate=0.05)

        prices = [model.zero_coupon_bond_price(T) for T in [1, 5, 10, 30]]

        # Each price should be less than the previous
        for i in range(1, len(prices)):
            assert prices[i] < prices[i - 1]

    def test_path_simulation(self):
        """Test path simulation."""
        model = VasicekModel(seed=42)
        path = model.simulate_path(T=1.0, dt=1 / 12)

        assert len(path) == 13  # 12 months + initial
        assert path[0] == model.initial_rate


class TestHullWhiteModel:
    """Tests for Hull-White model."""

    def test_creation(self):
        """Test model creation."""
        model = HullWhiteModel()
        assert model.rate == model.initial_rate

    def test_step(self):
        """Test single step."""
        model = HullWhiteModel(seed=42)
        new_rate = model.step(1 / 12)

        assert isinstance(new_rate, float)

    def test_bond_price(self):
        """Test bond pricing."""
        model = HullWhiteModel(initial_rate=0.05)
        price = model.zero_coupon_bond_price(1.0)

        assert 0 < price < 1


class TestCIRModel:
    """Tests for CIR model."""

    def test_creation(self):
        """Test model creation."""
        model = CIRModel()
        assert model.rate == model.initial_rate

    def test_rates_stay_positive(self):
        """Test that rates remain non-negative."""
        model = CIRModel(
            initial_rate=0.01,  # Start low
            volatility=0.05,  # High vol
            seed=42,
        )

        for _ in range(1000):
            model.step(1 / 12)
            assert model.rate >= 0

    def test_feller_condition_warning(self):
        """Test warning when Feller condition violated."""
        with pytest.warns(UserWarning, match="Feller condition"):
            CIRModel(
                mean_reversion=0.1,
                long_term_mean=0.01,
                volatility=0.10,  # Violates 2ab > sigma^2
            )

    def test_bond_price(self):
        """Test CIR bond pricing."""
        model = CIRModel(initial_rate=0.05)
        price = model.zero_coupon_bond_price(1.0)

        assert 0 < price < 1


class TestPropertyDynamicsModel:
    """Tests for property dynamics model."""

    def test_creation(self):
        """Test model creation."""
        model = PropertyDynamicsModel()
        assert model.value == model.initial_value

    def test_step(self):
        """Test single step."""
        model = PropertyDynamicsModel(seed=42)
        new_value = model.step(1 / 12)

        assert new_value > 0
        assert model.value == new_value

    def test_value_stays_positive(self):
        """Test values remain positive."""
        model = PropertyDynamicsModel(
            initial_value=100_000,
            volatility=0.30,  # High vol
            jump_intensity=0.5,  # Frequent jumps
            seed=42,
        )

        for _ in range(100):
            model.step(1 / 12)
            assert model.value > 0

    def test_regime_switching(self):
        """Test volatility regime changes."""
        model = PropertyDynamicsModel(seed=42)

        regimes_seen = set()
        for _ in range(1000):
            model.step(1 / 12)
            regimes_seen.add(model.get_regime())

        # Should see multiple regimes over many steps
        assert len(regimes_seen) >= 2

    def test_current_volatility(self):
        """Test current volatility reflects regime."""
        model = PropertyDynamicsModel()

        for regime in [0, 1, 2]:
            model.current_regime = regime
            vol = model.get_current_volatility()
            expected = model.base_volatility * model.regime_vol_multipliers[regime]
            assert vol == expected

    def test_path_simulation(self):
        """Test multi-path simulation."""
        model = PropertyDynamicsModel(seed=42)
        paths = model.simulate_path(T=1.0, dt=1 / 12, n_paths=100)

        assert paths.shape == (100, 13)
        assert np.all(paths > 0)  # All values positive

    def test_var_estimation(self):
        """Test Value-at-Risk calculation."""
        model = PropertyDynamicsModel(seed=42)
        var = model.estimate_var(
            horizon_years=1.0,
            confidence=0.95,
            n_simulations=1000,
        )

        assert var > 0  # VaR should be positive loss
        assert var < model.initial_value  # Should be less than total value


class TestModelReset:
    """Test model reset functionality."""

    def test_vasicek_reset(self):
        """Test Vasicek reset."""
        model = VasicekModel(initial_rate=0.05, seed=42)

        # Change state
        for _ in range(10):
            model.step(1 / 12)

        # Reset
        model.reset()

        assert model.rate == model.initial_rate

    def test_property_reset(self):
        """Test property model reset."""
        model = PropertyDynamicsModel(seed=42)

        # Change state
        for _ in range(10):
            model.step(1 / 12)

        # Reset
        model.reset()

        assert model.value == model.initial_value
        assert model.current_regime == 1  # Medium

    def test_reset_with_new_seed(self):
        """Test reset with new seed gives different results."""
        model = VasicekModel(seed=42)
        model.step(1 / 12)
        rate1 = model.rate

        model.reset(seed=123)
        model.step(1 / 12)
        rate2 = model.rate

        assert rate1 != rate2
