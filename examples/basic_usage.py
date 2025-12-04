#!/usr/bin/env python
"""Basic usage example for RealEstateOptionsGym.

This script demonstrates how to:
1. Create environments using gym.make
2. Run episodes with random actions
3. Access environment info and render states
"""

import gymnasium as gym

# Import to register environments
import realestate_options_gym  # noqa: F401


def run_lease_option_example():
    """Run a simple lease option episode."""
    print("=" * 60)
    print("Lease Option Environment Example")
    print("=" * 60)

    # Create environment
    env = gym.make(
        "LeaseOption-v1",
        config={
            "initial_property_value": 400_000,
            "lease_term_months": 24,
            "option_premium_pct": 0.02,
            "rent_credit_pct": 0.30,
        },
        render_mode="ansi",
    )

    # Reset
    obs, info = env.reset(seed=42)
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Strike price: ${info['strike_price']:,.0f}")
    print(f"Option premium paid: ${info['option_premium_paid']:,.0f}")

    # Run episode
    total_reward = 0
    step = 0

    while True:
        # Simple strategy: hold until option is deep in the money
        intrinsic = info.get("intrinsic_value", 0)

        if intrinsic > 50_000:  # Exercise if $50k+ in the money
            action = 1  # Exercise
        elif info["remaining_term"] <= 1:
            action = 1 if intrinsic > 0 else 2  # Exercise or abandon at expiry
        else:
            action = 0  # Hold

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        if step % 6 == 0:  # Print every 6 months
            print(f"\nMonth {step}:")
            print(env.render())

        if terminated or truncated:
            break

    print(f"\n{'='*60}")
    print(f"Episode finished after {step} steps")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Final property value: ${info['property_value']:,.0f}")
    print(f"Exercised: {info['is_exercised']}, Abandoned: {info['is_abandoned']}")

    env.close()


def run_portfolio_example():
    """Run a portfolio management episode."""
    print("\n" + "=" * 60)
    print("Property Portfolio Environment Example")
    print("=" * 60)

    env = gym.make(
        "PropertyPortfolio-v1",
        config={
            "n_properties": 3,
            "initial_capital": 500_000,
            "initial_property_value": 200_000,
        },
    )

    obs, info = env.reset(seed=42)
    print(f"\nInitial capital: ${info['capital']:,.0f}")
    print(f"Number of properties: {len(info['property_values'])}")

    # Run for 2 years (24 months)
    for month in range(24):
        # Simple equal-weight strategy
        action = [0.33, 0.33, 0.34]

        obs, reward, terminated, truncated, info = env.step(action)

        if month % 6 == 5:
            print(f"\nMonth {month + 1}:")
            print(f"  Portfolio value: ${info['portfolio_value']:,.0f}")
            print(f"  Cash: ${info['capital']:,.0f}")
            print(f"  Leverage: {info['leverage']:.1%}")

        if terminated or truncated:
            break

    final_return = (info["portfolio_value"] / 500_000 - 1) * 100
    print(f"\nFinal portfolio value: ${info['portfolio_value']:,.0f}")
    print(f"Total return: {final_return:+.2f}%")

    env.close()


def run_reit_trading_example():
    """Run a REIT trading episode."""
    print("\n" + "=" * 60)
    print("REIT Trading Environment Example")
    print("=" * 60)

    env = gym.make(
        "REITTrading-v1",
        config={
            "initial_nav": 50.0,
            "initial_premium": -0.05,  # Trading at 5% discount
            "dividend_yield": 0.05,
        },
    )

    obs, info = env.reset(seed=42)
    print(f"\nInitial NAV: ${info['nav']:.2f}")
    print(f"Initial price: ${info['price']:.2f}")
    print(f"Premium to NAV: {info['premium_to_nav']:.1%}")
    print(f"Starting cash: ${info['cash']:,.0f}")

    # Simple strategy: buy when discount > 5%, sell when premium > 5%
    for day in range(252):  # 1 year of trading days
        premium = info["premium_to_nav"]

        if premium < -0.05 and info["cash"] > info["price"] * 100:
            action = 1  # Buy
        elif premium > 0.05 and info["shares_held"] > 0:
            action = 2  # Sell
        else:
            action = 0  # Hold

        obs, reward, terminated, truncated, info = env.step(action)

        if (day + 1) % 63 == 0:  # Quarterly update
            print(f"\nQuarter {(day + 1) // 63}:")
            print(f"  NAV: ${info['nav']:.2f}, Price: ${info['price']:.2f}")
            print(f"  Shares: {info['shares_held']}, Cash: ${info['cash']:,.2f}")
            print(f"  Dividends received: ${info['dividends_received']:,.2f}")

        if terminated or truncated:
            print("\nPortfolio depleted!")
            break

    print(f"\nFinal Results:")
    print(f"  Portfolio value: ${info['portfolio_value']:,.2f}")
    print(f"  Total return: {info['total_return']:.2%}")
    print(f"  Total dividends: ${info['dividends_received']:,.2f}")

    env.close()


if __name__ == "__main__":
    run_lease_option_example()
    run_portfolio_example()
    run_reit_trading_example()
