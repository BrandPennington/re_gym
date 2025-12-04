# RealEstateOptionsGym Documentation

Welcome to RealEstateOptionsGym, a Gymnasium-compatible reinforcement learning
environment for real estate option trading with stochastic interest rates and
property value dynamics.

## Quick Start

```python
import gymnasium as gym
import realestate_options_gym

env = gym.make("LeaseOption-v1")
obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## Available Environments

| Environment | Description |
|-------------|-------------|
| `LeaseOption-v1` | Single property lease-option with exercise decision |
| `PropertyPortfolio-v1` | Multi-property portfolio management |
| `MortgageHedging-v1` | Interest rate hedging for mortgages |
| `REITTrading-v1` | REIT trading with NAV modeling |

## Installation

```bash
pip install realestate-options-gym
```

For development:
```bash
pip install realestate-options-gym[dev]
```

For RL training:
```bash
pip install realestate-options-gym[rl]
```

## Documentation Sections

- [Environments](environments.md) - Detailed environment documentation
- [Models](models.md) - Stochastic process models
- [Configuration](configuration.md) - Environment configuration options
- [Examples](examples.md) - Usage examples and tutorials

## Citation

```bibtex
@software{realestate_options_gym,
  author = {Pennington, Brandon},
  title = {RealEstateOptionsGym: RL Environments for Property Derivatives},
  year = {2025},
  url = {https://github.com/BrandPennington/realestate-options-gym}
}
```
