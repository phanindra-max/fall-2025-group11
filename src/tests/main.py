# -*- coding: utf-8 -*-
"""
Author: Satya Phanindra Kumar Kalaga
Date: 2025-09-10
Version: 1.0
"""


import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def heuristic_policy(observation):
    """
    Simple heuristic policy for CartPole based on pole angle and angular velocity.
    
    Args:
        observation: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    
    Returns:
        action: 0 (left) or 1 (right)
    """
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = observation
    
    # If pole is tilting right (positive angle) and/or has positive angular velocity,
    # move cart right (action 1) to get under the pole
    # If pole is tilting left (negative angle) and/or has negative angular velocity,
    # move cart left (action 0)
    
    if pole_angle > 0 or pole_angular_velocity > 0:
        return 1  # Move right
    else:
        return 0  # Move left

def q_learning_policy(observation, q_table, epsilon=0.1):
    """
    Epsilon-greedy Q-learning policy.
    
    Args:
        observation: Current state observation
        q_table: Q-table for action values
        epsilon: Exploration rate
    
    Returns:
        action: Selected action
    """
    if np.random.random() < epsilon:
        return np.random.choice([0, 1])  # Explore
    else:
        # Discretize the observation for Q-table lookup
        state = discretize_observation(observation)
        return np.argmax(q_table[state])

def discretize_observation(observation, bins=(6, 6, 12, 12)):
    """
    Discretize continuous observation space for Q-learning.
    
    Args:
        observation: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
        bins: Number of bins for each dimension
    
    Returns:
        state: Discretized state index
    """
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = observation
    
    # Define bounds for discretization
    cart_pos_bins = np.linspace(-2.4, 2.4, bins[0])
    cart_vel_bins = np.linspace(-2.0, 2.0, bins[1])
    pole_angle_bins = np.linspace(-0.2095, 0.2095, bins[2])
    pole_vel_bins = np.linspace(-2.0, 2.0, bins[3])
    
    # Discretize each dimension
    cart_pos_idx = np.digitize(cart_position, cart_pos_bins) - 1
    cart_vel_idx = np.digitize(cart_velocity, cart_vel_bins) - 1
    pole_angle_idx = np.digitize(pole_angle, pole_angle_bins) - 1
    pole_vel_idx = np.digitize(pole_angular_velocity, pole_vel_bins) - 1
    
    # Clip to valid range
    cart_pos_idx = np.clip(cart_pos_idx, 0, bins[0] - 1)
    cart_vel_idx = np.clip(cart_vel_idx, 0, bins[1] - 1)
    pole_angle_idx = np.clip(pole_angle_idx, 0, bins[2] - 1)
    pole_vel_idx = np.clip(pole_vel_idx, 0, bins[3] - 1)
    
    return (cart_pos_idx, cart_vel_idx, pole_angle_idx, pole_vel_idx)

def run_episodes(env, policy_func, num_episodes=10, policy_name="Policy", **kwargs):
    """
    Run episodes with a given policy and return results.
    
    Args:
        env: Gymnasium environment
        policy_func: Function that takes observation and returns action
        num_episodes: Number of episodes to run
        policy_name: Name for display purposes
        **kwargs: Additional arguments for policy function
    
    Returns:
        episode_rewards: List of total rewards per episode
    """
    episode_rewards = []
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        total_reward = 0
        
        for step in range(500):  # Increased max steps
            # Get action from policy
            if 'q_table' in kwargs:
                action = policy_func(observation, **kwargs)
            else:
                action = policy_func(observation)
            
            # Take action and get results
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Episode ends if terminated or truncated
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        print(f"{policy_name} Episode {episode + 1}: Total reward = {total_reward}")
    
    return episode_rewards

def main():
    """
    Main function to compare different reinforcement learning policies in the CartPole environment.
    """

    # Create the CartPole environment
    env = gym.make("CartPole-v1", render_mode='rgb_array')

    # Reset the environment to get initial observation
    observation, info = env.reset(seed=42)

    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)
    print("Initial observation:", observation)
    print("=" * 60)

    # 1. Random Policy (baseline)
    print("Running Random Policy (Baseline)...")
    random_rewards = run_episodes(env, lambda obs: env.action_space.sample(), 
                                num_episodes=10, policy_name="Random")
    
    print(f"Random Policy - Average reward: {np.mean(random_rewards):.2f}")
    print("=" * 60)

    # 2. Heuristic Policy
    print("Running Heuristic Policy...")
    heuristic_rewards = run_episodes(env, heuristic_policy, 
                                   num_episodes=10, policy_name="Heuristic")
    
    print(f"Heuristic Policy - Average reward: {np.mean(heuristic_rewards):.2f}")
    print("=" * 60)

    # 3. Q-Learning Policy (with training)
    print("Training Q-Learning Policy...")
    
    # Initialize Q-table
    bins = (6, 6, 12, 12)
    q_table = np.zeros(bins + (2,))  # 2 actions: left and right
    
    # Training parameters
    learning_rate = 0.1
    discount_factor = 0.95
    epsilon = 0.1
    training_episodes = 1000
    
    # Training loop
    for episode in range(training_episodes):
        observation, info = env.reset()
        
        for step in range(500):
            # Get current state
            state = discretize_observation(observation)
            
            # Choose action using epsilon-greedy
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            
            # Take action
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = discretize_observation(next_observation)
            
            # Q-learning update
            current_q = q_table[state + (action,)]
            next_max_q = np.max(q_table[next_state])
            new_q = current_q + learning_rate * (reward + discount_factor * next_max_q - current_q)
            q_table[state + (action,)] = new_q
            
            observation = next_observation
            

            if terminated or truncated:
                break
        
        # Print progress
        if (episode + 1) % 200 == 0:
            print(f"Training episode {episode + 1}/{training_episodes}")
    
    print("Testing trained Q-Learning Policy...")
    qlearning_rewards = run_episodes(env, q_learning_policy, 
                                   num_episodes=10, policy_name="Q-Learning",
                                   q_table=q_table, epsilon=0.0)  # No exploration during testing
    
    print(f"Q-Learning Policy - Average reward: {np.mean(qlearning_rewards):.2f}")
    print("=" * 60)

    env.close()

    # Performance comparison
    print("PERFORMANCE COMPARISON:")
    print(f"Random Policy:    {np.mean(random_rewards):.2f} ± {np.std(random_rewards):.2f}")
    print(f"Heuristic Policy: {np.mean(heuristic_rewards):.2f} ± {np.std(heuristic_rewards):.2f}")
    print(f"Q-Learning Policy: {np.mean(qlearning_rewards):.2f} ± {np.std(qlearning_rewards):.2f}")

    # Plotting the rewards comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Individual episode rewards
    episodes = range(1, 11)
    ax1.plot(episodes, random_rewards, 'o-', label='Random', alpha=0.7)
    ax1.plot(episodes, heuristic_rewards, 's-', label='Heuristic', alpha=0.7)
    ax1.plot(episodes, qlearning_rewards, '^-', label='Q-Learning', alpha=0.7)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards by Policy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Average performance comparison
    policies = ['Random', 'Heuristic', 'Q-Learning']
    means = [np.mean(random_rewards), np.mean(heuristic_rewards), np.mean(qlearning_rewards)]
    stds = [np.std(random_rewards), np.std(heuristic_rewards), np.std(qlearning_rewards)]
    
    bars = ax2.bar(policies, means, yerr=stds, capsize=5, alpha=0.7)
    ax2.set_ylabel('Average Total Reward')
    ax2.set_title('Average Performance Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{mean:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
