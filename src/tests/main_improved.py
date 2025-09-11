# -*- coding: utf-8 -*-
"""
Author: Satya Phanindra Kumar Kalaga
Date: 2025-09-10
Version: 1.0

Improved main script using modular policy classes
"""

import sys
import os
# Add the component directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'component'))

import gymnasium as gym
import numpy as np
from policies import RandomPolicy, HeuristicPolicy, QLearningPolicy, ImprovedHeuristicPolicy
from trainer import PolicyTrainer


def main():
    """
    Main function to compare different reinforcement learning policies.
    """
    
    # Create the CartPole environment
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    
    # Initialize trainer
    trainer = PolicyTrainer(env)
    
    # Reset environment to get information
    observation, info = env.reset(seed=42)
    
    print("CARTPOLE ENVIRONMENT INFO")
    print("=" * 60)
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)
    print("Initial observation:", observation)
    print("Observation details:")
    print("  [0] Cart Position:      ", observation[0])
    print("  [1] Cart Velocity:      ", observation[1])
    print("  [2] Pole Angle:         ", observation[2])
    print("  [3] Pole Angular Velocity:", observation[3])
    print()
    
    # Initialize policies
    policies = [
        RandomPolicy(env.action_space),
        HeuristicPolicy(),
        ImprovedHeuristicPolicy(),
    ]
    
    # Train Q-Learning policy
    print("TRAINING Q-LEARNING POLICY")
    print("=" * 60)
    qlearning_policy = QLearningPolicy(
        bins=(6, 6, 12, 12),
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.1
    )
    
    # Train the Q-learning policy
    training_rewards = trainer.train_qlearning(
        qlearning_policy, 
        num_episodes=1000, 
        verbose=True, 
        progress_interval=200
    )
    
    # Add trained policy to comparison
    policies.append(qlearning_policy)
    
    print("\nTRAINING COMPLETED")
    print("=" * 60)
    
    # Compare all policies
    results = trainer.compare_policies(
        policies, 
        num_episodes=10, 
        max_steps=500, 
        plot_results=True
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Detailed analysis
    print("\nDETAILED ANALYSIS")
    print("=" * 60)
    
    best_policy = max(results.keys(), key=lambda k: results[k]['mean'])
    worst_policy = min(results.keys(), key=lambda k: results[k]['mean'])
    
    print(f"Best performing policy: {best_policy}")
    print(f"  Average reward: {results[best_policy]['mean']:.2f}")
    print(f"  Standard deviation: {results[best_policy]['std']:.2f}")
    print(f"  Success rate (>400): {sum(1 for r in results[best_policy]['rewards'] if r > 400)}/10")
    
    print(f"\nWorst performing policy: {worst_policy}")
    print(f"  Average reward: {results[worst_policy]['mean']:.2f}")
    print(f"  Standard deviation: {results[worst_policy]['std']:.2f}")
    
    # Calculate improvement
    improvement = (results[best_policy]['mean'] - results[worst_policy]['mean']) / results[worst_policy]['mean'] * 100
    print(f"\nImprovement: {improvement:.1f}% better than worst policy")
    
    print("\nPOLICY EXPLANATIONS")
    print("=" * 60)
    print("Random Policy: Selects actions uniformly at random")
    print("Heuristic Policy: Moves cart in direction of pole tilt")
    print("Improved Heuristic: Balances pole and keeps cart centered")
    print("Q-Learning Policy: Learns optimal actions through experience")
    
    env.close()
    
    return results


def demo_single_policy(policy_class, policy_name, episodes=5):
    """
    Demonstrate a single policy in detail.
    
    Args:
        policy_class: Policy class to instantiate
        policy_name: Name for display
        episodes: Number of episodes to run
    """
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    
    if policy_class == RandomPolicy:
        policy = policy_class(env.action_space)
    else:
        policy = policy_class()
    
    print(f"\nDEMONSTRATING {policy_name.upper()}")
    print("=" * 50)
    
    total_rewards = []
    
    for episode in range(episodes):
        observation, info = env.reset()
        total_reward = 0
        step_count = 0
        
        print(f"\nEpisode {episode + 1}:")
        
        for step in range(500):
            action = policy.get_action(observation)
            
            # Print first few steps for insight
            if step < 3:
                print(f"  Step {step + 1}: obs={observation}, action={'right' if action == 1 else 'left'}")
            
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
        
        total_rewards.append(total_reward)
        print(f"  Episode {episode + 1} completed: {step_count} steps, reward = {total_reward}")
    
    print(f"\n{policy_name} Summary:")
    print(f"  Average reward: {np.mean(total_rewards):.2f}")
    print(f"  Best episode: {max(total_rewards)}")
    print(f"  Worst episode: {min(total_rewards)}")
    
    env.close()


if __name__ == "__main__":
    # Run main comparison
    results = main()
    
    # Optional: Run detailed demos
    print("\n" + "="*80)
    print("DETAILED POLICY DEMONSTRATIONS")
    print("="*80)
    
    # Demonstrate each policy individually for better understanding
    demo_single_policy(RandomPolicy, "Random Policy", episodes=3)
    demo_single_policy(HeuristicPolicy, "Simple Heuristic Policy", episodes=3)
    demo_single_policy(ImprovedHeuristicPolicy, "Improved Heuristic Policy", episodes=3)
