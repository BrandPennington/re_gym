#!/usr/bin/env python
"""Train a PPO agent on the LeaseOption environment.

This script demonstrates:
1. Integration with Stable-Baselines3
2. Training an RL agent on real estate options
3. Evaluating and analyzing performance

Requires: pip install stable-baselines3
"""

import gymnasium as gym
import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor

    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("stable-baselines3 not installed. Install with:")
    print("  pip install stable-baselines3")

# Import to register environments
import realestate_options_gym  # noqa: F401


def make_env(seed: int = 0):
    """Create a monitored environment instance."""
    env = gym.make(
        "LeaseOption-v1",
        config={
            "lease_term_months": 36,
            "initial_property_value": 500_000,
            "property_volatility": 0.12,
            "interest_rate_model": "hull-white",
        },
    )
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def train_agent(total_timesteps: int = 50_000):
    """Train PPO agent on lease option environment."""
    print("Creating training environment...")
    env = make_env(seed=42)

    print("Initializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=42,
    )

    print(f"\nTraining for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    return model, env


def evaluate_agent(model, n_episodes: int = 100):
    """Evaluate trained agent."""
    print(f"\nEvaluating over {n_episodes} episodes...")

    eval_env = make_env(seed=123)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=n_episodes
    )

    print(f"Mean reward: {mean_reward:.4f} +/- {std_reward:.4f}")

    # Detailed episode analysis
    exercise_count = 0
    abandon_count = 0
    episode_lengths = []
    final_intrinsics = []

    for ep in range(n_episodes):
        obs, info = eval_env.reset()
        done = False
        ep_len = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            ep_len += 1

        episode_lengths.append(ep_len)
        final_intrinsics.append(info.get("intrinsic_value", 0))

        if info.get("is_exercised"):
            exercise_count += 1
        elif info.get("is_abandoned"):
            abandon_count += 1

    print(f"\nEpisode Statistics:")
    print(f"  Exercise rate: {exercise_count / n_episodes:.1%}")
    print(f"  Abandon rate: {abandon_count / n_episodes:.1%}")
    print(f"  Avg episode length: {np.mean(episode_lengths):.1f} steps")
    print(f"  Avg final intrinsic value: ${np.mean(final_intrinsics):,.0f}")

    return mean_reward


def compare_with_baseline(model, n_episodes: int = 100):
    """Compare trained agent with simple baselines."""
    print("\n" + "=" * 60)
    print("Comparison with Baselines")
    print("=" * 60)

    eval_env = make_env(seed=456)

    # Baseline 1: Always hold until expiry
    hold_rewards = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0

        while not done:
            obs, reward, terminated, truncated, _ = eval_env.step(0)  # Hold
            total_reward += reward
            done = terminated or truncated

        hold_rewards.append(total_reward)

    print(f"\nHold-until-expiry baseline: {np.mean(hold_rewards):.4f} +/- {np.std(hold_rewards):.4f}")

    # Baseline 2: Random actions
    random_rewards = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0

        while not done:
            action = eval_env.action_space.sample()
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated

        random_rewards.append(total_reward)

    print(f"Random baseline: {np.mean(random_rewards):.4f} +/- {np.std(random_rewards):.4f}")

    # Trained agent
    agent_rewards = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated

        agent_rewards.append(total_reward)

    print(f"Trained PPO agent: {np.mean(agent_rewards):.4f} +/- {np.std(agent_rewards):.4f}")

    # Improvement
    improvement_vs_hold = (np.mean(agent_rewards) - np.mean(hold_rewards)) / abs(np.mean(hold_rewards)) * 100
    improvement_vs_random = (np.mean(agent_rewards) - np.mean(random_rewards)) / abs(np.mean(random_rewards)) * 100

    print(f"\nImprovement vs hold: {improvement_vs_hold:+.1f}%")
    print(f"Improvement vs random: {improvement_vs_random:+.1f}%")


def main():
    """Main training loop."""
    if not HAS_SB3:
        print("Please install stable-baselines3 to run this example.")
        return

    print("=" * 60)
    print("Training PPO on LeaseOption-v1")
    print("=" * 60)

    # Train
    model, env = train_agent(total_timesteps=50_000)

    # Evaluate
    evaluate_agent(model, n_episodes=100)

    # Compare with baselines
    compare_with_baseline(model, n_episodes=100)

    # Save model
    model.save("lease_option_ppo")
    print("\nModel saved to lease_option_ppo.zip")

    env.close()


if __name__ == "__main__":
    main()
