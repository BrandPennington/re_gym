# RealEstateOptionsGym
A Gymnasium-compatible reinforcement learning environment for real estate option trading with stochastic interest rates and property value dynamics.

## Overview
RealEstateOptionsGym provides realistic simulation environments for training RL agents on lease-option and property derivative strategies. Unlike equity-focused RL frameworks, this library models the unique characteristics of real estate markets:

- **Illiquidity and transaction costs** - Realistic bid-ask spreads and holding periods
- **Stochastic interest rates** - Hull-White and Vasicek short-rate models affecting mortgage rates and discount factors
- **Property value dynamics** - Mean-reverting processes with regime-switching volatility
- **Lease option mechanics** - Rent credits, option premiums, exercise windows

## Installation
```bash
pip install realestate-options-gym
```

From source:
```bash
git clone https://github.com/BrandPennington/realestate-options-gym.git
cd realestate-options-gym
pip install -e .
```

## Quick Start
```python
import gymnasium as gym
import realestate_options_gym

# Create a lease option trading environment
env = gym.make("LeaseOption-v1", config={
    "initial_property_value": 500_000,
    "lease_term_months": 36,
    "option_premium_pct": 0.03,
    "rent_credit_pct": 0.25,
    "interest_rate_model": "hull-white",
    "hw_mean_reversion": 0.1,
    "hw_volatility": 0.01,
})

obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # Replace with trained agent
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()
```

## Environments
| Environment | Description |
|-------------|-------------|
| `LeaseOption-v1` | Single property lease-option with exercise decision |
| `PropertyPortfolio-v1` | Multi-property portfolio with acquisition/disposition |
| `MortgageHedging-v1` | Interest rate hedging for property-backed loans |
| `REITTrading-v1` | REIT allocation with property-level simulation |

## Observation Space
The default observation includes:

| Feature | Shape | Description |
|---------|-------|-------------|
| `property_value` | (1,) | Current estimated property value |
| `property_value_ma` | (3,) | Moving averages (3m, 6m, 12m) |
| `short_rate` | (1,) | Current short-term interest rate |
| `rate_curve` | (5,) | Term structure (1y, 2y, 5y, 10y, 30y) |
| `time_to_expiry` | (1,) | Remaining option term (normalized) |
| `accumulated_rent_credit` | (1,) | Rent credits accumulated toward purchase |
| `volatility_regime` | (1,) | Estimated volatility state (low/medium/high) |

## Action Space

### LeaseOption-v1
| Action | Description |
|--------|-------------|
| 0 | Hold - continue lease, pay rent |
| 1 | Exercise - purchase property at strike |
| 2 | Abandon - terminate lease, forfeit credits |
| 3 | Renegotiate - attempt to extend/modify terms |

### PropertyPortfolio-v1
Continuous action space `Box(low=-1, high=1, shape=(n_properties,))` representing target allocation weights.

## Integration with RL Libraries

### Stable-Baselines3
```python
from stable_baselines3 import PPO
import gymnasium as gym
import realestate_options_gym

env = gym.make("LeaseOption-v1")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
```

### FinRL Integration
```python
from finrl.agents.stablebaselines3 import DRLAgent
from realestate_options_gym.wrappers import FinRLWrapper

env = FinRLWrapper(gym.make("PropertyPortfolio-v1"))
agent = DRLAgent(env=env)
model = agent.get_model("ppo")
trained_model = agent.train_model(model, total_timesteps=50_000)
```

## Interest Rate Models

### Hull-White (default)
```
dr = (theta(t) - a*r) * dt + sigma * dW
```

Calibrated to initial term structure with configurable mean reversion `a` and volatility `sigma`.

### Vasicek
```
dr = a * (b - r) * dt + sigma * dW
```

Constant parameters with closed-form bond pricing.

### CIR (Cox-Ingersoll-Ross)
```
dr = a * (b - r) * dt + sigma * sqrt(r) * dW
```

Ensures positive rates with volatility proportional to rate level.

## Property Value Dynamics
Property values follow a mean-reverting jump-diffusion process:

```
dP = kappa * (theta - P) * dt + sigma * P * dW + P * dJ
```

Where:
- `kappa`: Mean reversion speed
- `theta`: Long-term property value (linked to local index)
- `sigma`: Base volatility (regime-dependent)
- `dJ`: Poisson jump process for market shocks

## Data Sources
The library includes adapters for real market data:

```python
from realestate_options_gym.data import UKLandRegistry, FREDRates

# UK Land Registry price paid data
uk_data = UKLandRegistry(postcode_prefix="SW1")
historical_prices = uk_data.get_price_series(start="2015-01-01")

# Interest rate curves from FRED
rates = FREDRates()
curve = rates.get_treasury_curve()
```

## Contributing
Contributions welcome. Priority areas:

- [ ] Commercial real estate environments (office, retail, industrial)
- [ ] Multi-agent market simulation
- [ ] Additional interest rate models (LMM, SABR)
- [ ] Integration with PropTech data APIs
- [ ] GPU-accelerated simulation via JAX

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License
MIT License. See [LICENSE](LICENSE) for details.

## Citation
```bibtex
@software{realestate_options_gym,
  author = {Pennington, Brandon},
  title = {RealEstateOptionsGym: RL Environments for Property Derivatives},
  year = {2025},
  url = {https://github.com/BrandPennington/realestate-options-gym}
}
```

## References
1. Titman, S. (1985). "Urban Land Prices Under Uncertainty." *American Economic Review*
2. Williams, J.T. (1991). "Real Estate Development as an Option." *Journal of Real Estate Finance and Economics*
3. Grenadier, S.R. (1996). "Leasing and Credit Risk." *Journal of Financial Economics*
4. Hull, J. & White, A. (1990). "Pricing Interest-Rate-Derivative Securities." *Review of Financial Studies*
