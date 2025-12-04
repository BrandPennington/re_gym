"""RealEstateOptionsGym: RL environments for real estate option trading."""

from gymnasium.envs.registration import register

from realestate_options_gym.envs import (
    LeaseOptionEnv,
    MortgageHedgingEnv,
    PropertyPortfolioEnv,
    REITTradingEnv,
)

__version__ = "0.1.0"
__all__ = [
    "LeaseOptionEnv",
    "PropertyPortfolioEnv",
    "MortgageHedgingEnv",
    "REITTradingEnv",
    "register_envs",
]


def register_envs():
    """Register all environments with Gymnasium."""
    register(
        id="LeaseOption-v1",
        entry_point="realestate_options_gym.envs:LeaseOptionEnv",
        max_episode_steps=360,  # 30 years monthly
    )
    register(
        id="PropertyPortfolio-v1",
        entry_point="realestate_options_gym.envs:PropertyPortfolioEnv",
        max_episode_steps=240,  # 20 years monthly
    )
    register(
        id="MortgageHedging-v1",
        entry_point="realestate_options_gym.envs:MortgageHedgingEnv",
        max_episode_steps=360,
    )
    register(
        id="REITTrading-v1",
        entry_point="realestate_options_gym.envs:REITTradingEnv",
        max_episode_steps=252,  # 1 year daily
    )


# Auto-register on import
register_envs()
