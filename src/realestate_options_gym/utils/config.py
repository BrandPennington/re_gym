"""Environment configuration management."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnvConfig:
    """Configuration for real estate option environments.

    This dataclass holds all configurable parameters for the simulation
    environments. Parameters are organized by category.
    """

    # Property dynamics
    initial_property_value: float = 500_000
    property_volatility: float = 0.12
    mean_reversion_speed: float = 0.5
    jump_intensity: float = 0.1
    jump_mean: float = -0.05
    jump_std: float = 0.10

    # Interest rate model selection
    interest_rate_model: str = "hull-white"  # "hull-white", "vasicek", "cir"
    initial_short_rate: float = 0.05

    # Hull-White parameters
    hw_mean_reversion: float = 0.1
    hw_volatility: float = 0.01

    # Vasicek parameters
    vasicek_mean_reversion: float = 0.1
    vasicek_long_term_mean: float = 0.05
    vasicek_volatility: float = 0.01

    # CIR parameters
    cir_mean_reversion: float = 0.1
    cir_long_term_mean: float = 0.05
    cir_volatility: float = 0.02

    # Lease option parameters
    lease_term_months: int = 36
    monthly_rent: float = 2_500
    option_premium_pct: float = 0.03
    rent_credit_pct: float = 0.25
    strike_premium_pct: float = 0.05
    renegotiation_success_prob: float = 0.3
    renegotiation_cost: float = 1_000

    # Portfolio parameters
    n_properties: int = 5
    initial_capital: float = 1_000_000
    leverage_limit: float = 0.7

    # Transaction costs
    transaction_cost_pct: float = 0.02
    holding_cost_monthly_pct: float = 0.005
    hedge_cost_bps: float = 5

    # Mortgage hedging
    mortgage_notional: float = 10_000_000
    mortgage_rate: float = 0.05
    mortgage_term_years: int = 30
    max_hedge_ratio: float = 2.0

    # REIT trading
    initial_nav: float = 100.0
    initial_premium: float = 0.0
    dividend_yield: float = 0.04
    nav_volatility: float = 0.15
    premium_mean_reversion: float = 0.5
    premium_volatility: float = 0.10
    initial_cash: float = 100_000
    shares_per_trade: int = 100
    transaction_cost_per_share: float = 0.01

    # Simulation
    dt: float = 1 / 12  # Monthly steps (1/12 year)
    seed: int | None = 42

    # Reward configuration
    reward_type: str = "pnl"  # "pnl", "sharpe", "sortino", "calmar"
    transaction_penalty: float = 0.001

    # Additional custom parameters
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration values."""
        if self.initial_property_value <= 0:
            raise ValueError("initial_property_value must be positive")

        if self.property_volatility < 0:
            raise ValueError("property_volatility must be non-negative")

        if self.initial_short_rate < -0.1:
            raise ValueError("initial_short_rate seems unreasonably negative")

        if self.dt <= 0 or self.dt > 1:
            raise ValueError("dt must be in (0, 1]")

        if self.interest_rate_model not in ("hull-white", "vasicek", "cir"):
            raise ValueError(
                f"Unknown interest_rate_model: {self.interest_rate_model}. "
                "Choose from: hull-white, vasicek, cir"
            )

        if self.reward_type not in ("pnl", "sharpe", "sortino", "calmar"):
            raise ValueError(
                f"Unknown reward_type: {self.reward_type}. "
                "Choose from: pnl, sharpe, sortino, calmar"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary of all configuration parameters.
        """
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EnvConfig":
        """Create config from dictionary.

        Args:
            d: Dictionary of configuration parameters.

        Returns:
            EnvConfig instance.
        """
        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}

        # Put unknown fields in extra
        extra = {k: v for k, v in d.items() if k not in known_fields}
        if extra:
            filtered["extra"] = extra

        return cls(**filtered)

    def update(self, **kwargs: Any) -> "EnvConfig":
        """Create new config with updated values.

        Args:
            **kwargs: Parameters to update.

        Returns:
            New EnvConfig with updates applied.
        """
        current = self.to_dict()
        current.update(kwargs)
        return EnvConfig.from_dict(current)


# Preset configurations
PRESETS = {
    "default": EnvConfig(),
    "high_volatility": EnvConfig(
        property_volatility=0.20,
        jump_intensity=0.2,
        hw_volatility=0.02,
    ),
    "low_rates": EnvConfig(
        initial_short_rate=0.02,
        hw_mean_reversion=0.05,
        vasicek_long_term_mean=0.02,
    ),
    "uk_residential": EnvConfig(
        initial_property_value=350_000,
        property_volatility=0.10,
        monthly_rent=1_500,
        transaction_cost_pct=0.04,  # Higher stamp duty
    ),
    "us_commercial": EnvConfig(
        initial_property_value=2_000_000,
        property_volatility=0.15,
        monthly_rent=15_000,
        lease_term_months=60,  # 5-year commercial lease
    ),
}


def get_preset(name: str) -> EnvConfig:
    """Get a preset configuration.

    Args:
        name: Preset name.

    Returns:
        EnvConfig for the preset.

    Raises:
        ValueError: If preset name is unknown.
    """
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]
