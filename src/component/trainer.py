# -*- coding: utf-8 -*-
"""
Author: Satya Phanindra Kumar Kalaga
Date: 2025-09-10
Version: 1.0

Policy Trainer and Evaluator for Reinforcement Learning
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any


class PolicyTrainer:
    """Class to train and evaluate reinforcement learning policies."""
    
    def __init__(self, env):
        """
        Initialize policy trainer.
        
        Args:
            env: Gymnasium environment
        """
        self.env = env
        self.training_history = {}
    
    def train_qlearning(self, policy, num_episodes=1000, max_steps=500, 
                       verbose=True, progress_interval=200):
        """
        Train a Q-learning policy.
        
        Args:
            policy: QLearningPolicy instance
            num_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            verbose: Whether to print progress
            progress_interval: Episodes between progress prints
        
        Returns:
            training_rewards: List of rewards per training episode
        """
        training_rewards = []
        policy.set_training_mode(True)
        
        if verbose:
            print(f"Training {policy.get_name()} for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            observation, info = self.env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                # Get current state and action
                state = policy.discretize_observation(observation)
                action = policy.get_action(observation)
                
                # Take action
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                next_state = policy.discretize_observation(next_observation)
                
                # Update Q-value
                policy.update_q_value(state, action, reward, next_state)
                
                episode_reward += reward
                observation = next_observation
                
                if terminated or truncated:
                    break
            
            training_rewards.append(episode_reward)
            
            # Print progress
            if verbose and (episode + 1) % progress_interval == 0:
                avg_reward = np.mean(training_rewards[-progress_interval:])
                print(f"Episode {episode + 1}/{num_episodes}, "
                      f"Average reward (last {progress_interval}): {avg_reward:.2f}")
        
        # Store training history
        self.training_history[policy.get_name()] = training_rewards
        
        return training_rewards
    
    def evaluate_policy(self, policy, num_episodes=10, max_steps=500, 
                       verbose=True, test_mode=True):
        """
        Evaluate a policy's performance.
        
        Args:
            policy: Policy to evaluate
            num_episodes: Number of evaluation episodes
            max_steps: Maximum steps per episode
            verbose: Whether to print results
            test_mode: Set policy to test mode (no exploration)
        
        Returns:
            episode_rewards: List of rewards per episode
        """
        episode_rewards = []
        
        # Set to test mode if supported
        if hasattr(policy, 'set_training_mode') and test_mode:
            policy.set_training_mode(False)
        
        if verbose:
            print(f"Evaluating {policy.get_name()}...")
        
        for episode in range(num_episodes):
            observation, info = self.env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = policy.get_action(observation)
                observation, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(total_reward)
            
            if verbose:
                print(f"{policy.get_name()} Episode {episode + 1}: "
                      f"Total reward = {total_reward}")
        
        if verbose:
            print(f"{policy.get_name()} - Average reward: {np.mean(episode_rewards):.2f}")
            print("=" * 60)
        
        return episode_rewards
    
    def compare_policies(self, policies: List[Any], num_episodes=10, 
                        max_steps=500, plot_results=True):
        """
        Compare multiple policies.
        
        Args:
            policies: List of policy objects
            num_episodes: Number of episodes per policy
            max_steps: Maximum steps per episode
            plot_results: Whether to plot comparison
        
        Returns:
            results: Dictionary with policy names as keys and results as values
        """
        results = {}
        
        print("POLICY COMPARISON")
        print("=" * 60)
        
        for policy in policies:
            episode_rewards = self.evaluate_policy(
                policy, num_episodes, max_steps, verbose=True, test_mode=True
            )
            results[policy.get_name()] = {
                'rewards': episode_rewards,
                'mean': np.mean(episode_rewards),
                'std': np.std(episode_rewards),
                'max': np.max(episode_rewards),
                'min': np.min(episode_rewards)
            }
        
        # Print summary
        print("PERFORMANCE SUMMARY:")
        print("-" * 60)
        for name, stats in results.items():
            print(f"{name:20s}: {stats['mean']:6.2f} ± {stats['std']:5.2f} "
                  f"(min: {stats['min']:3.0f}, max: {stats['max']:3.0f})")
        
        if plot_results:
            self._plot_comparison(results)
        
        return results
    
    def _plot_comparison(self, results: Dict[str, Dict[str, Any]]):
        """Plot policy comparison results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Individual episode rewards
        episodes = range(1, len(list(results.values())[0]['rewards']) + 1)
        
        for i, (name, stats) in enumerate(results.items()):
            marker = ['o', 's', '^', 'D', 'v'][i % 5]
            ax1.plot(episodes, stats['rewards'], marker + '-', 
                    label=name, alpha=0.7, markersize=4)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Episode Rewards by Policy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Average performance comparison
        names = list(results.keys())
        means = [results[name]['mean'] for name in names]
        stds = [results[name]['std'] for name in names]
        
        bars = ax2.bar(names, means, yerr=stds, capsize=5, alpha=0.7)
        ax2.set_ylabel('Average Total Reward')
        ax2.set_title('Average Performance Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Rotate x-axis labels if needed
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                    f'{mean:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self, policy_names=None):
        """
        Plot training history for trained policies.
        
        Args:
            policy_names: List of policy names to plot (default: all)
        """
        if not self.training_history:
            print("No training history available.")
            return
        
        if policy_names is None:
            policy_names = list(self.training_history.keys())
        
        plt.figure(figsize=(12, 6))
        
        for name in policy_names:
            if name in self.training_history:
                rewards = self.training_history[name]
                # Smooth the curve using moving average
                window_size = min(50, len(rewards) // 10)
                if window_size > 1:
                    smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                    plt.plot(range(window_size-1, len(rewards)), smoothed, label=f'{name} (smoothed)')
                plt.plot(rewards, alpha=0.3, label=f'{name} (raw)')
        
        plt.xlabel('Training Episode')
        plt.ylabel('Episode Reward')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
