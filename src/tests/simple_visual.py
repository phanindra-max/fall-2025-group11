# -*- coding: utf-8 -*-
"""
Author: Satya Phanindra Kumar Kalaga
Date: 2025-09-10
Version: 1.0

Simple visual demonstration using matplotlib only
"""

import sys
import os
# Add the component directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'component'))

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from policies import RandomPolicy, HeuristicPolicy, ImprovedHeuristicPolicy


def capture_episode_frames(policy, policy_name, max_steps=200):
    """
    Capture frames from a single episode run.
    
    Args:
        policy: Policy to run
        policy_name: Name for display
        max_steps: Maximum steps to capture
    
    Returns:
        frames, rewards, actions, observations
    """
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    
    frames = []
    rewards = []
    actions = []
    observations = []
    action_names = []
    
    print(f"🎬 Recording {policy_name} episode...")
    
    observation, info = env.reset()
    total_reward = 0
    
    for step in range(max_steps):
        # Capture frame
        frame = env.render()
        frames.append(frame)
        
        # Store current state
        observations.append(observation.copy())
        
        # Get action
        action = policy.get_action(observation)
        actions.append(action)
        action_names.append("Left" if action == 0 else "Right")
        
        # Take action
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        rewards.append(total_reward)
        
        if terminated or truncated:
            print(f"Episode completed: {step + 1} steps, total reward = {total_reward}")
            break
    
    env.close()
    return frames, rewards, actions, observations, action_names


def show_episode_summary(frames, rewards, actions, observations, action_names, policy_name):
    """
    Show a comprehensive summary of the episode with multiple views.
    """
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'{policy_name} - Episode Analysis', fontsize=18, fontweight='bold')
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, height_ratios=[2, 1, 1], width_ratios=[1, 1, 1, 1])
    
    # Show key frames from the episode
    key_frames_ax = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 3])
    ]
    
    # Select 4 key frames: start, middle, near end, final
    frame_indices = [
        0,  # Start
        len(frames) // 3,  # Early
        2 * len(frames) // 3,  # Middle-late
        len(frames) - 1  # End
    ]
    
    for i, (ax, frame_idx) in enumerate(zip(key_frames_ax, frame_indices)):
        if frame_idx < len(frames):
            ax.imshow(frames[frame_idx])
            ax.set_title(f'Step {frame_idx + 1}\nAction: {action_names[frame_idx]}\nReward: {rewards[frame_idx]:.0f}')
            ax.axis('off')
    
    # Reward progression
    ax_reward = fig.add_subplot(gs[1, :2])
    steps = range(1, len(rewards) + 1)
    ax_reward.plot(steps, rewards, 'b-', linewidth=2, marker='o', markersize=3)
    ax_reward.set_title('Cumulative Reward Over Time')
    ax_reward.set_xlabel('Step')
    ax_reward.set_ylabel('Total Reward')
    ax_reward.grid(True, alpha=0.3)
    
    # Action sequence
    ax_actions = fig.add_subplot(gs[1, 2:])
    action_values = [0 if action == 0 else 1 for action in actions]
    colors = ['red' if a == 0 else 'blue' for a in actions]
    ax_actions.scatter(steps, action_values, c=colors, alpha=0.7, s=20)
    ax_actions.set_title('Action Sequence (Red=Left, Blue=Right)')
    ax_actions.set_xlabel('Step')
    ax_actions.set_ylabel('Action')
    ax_actions.set_yticks([0, 1])
    ax_actions.set_yticklabels(['Left', 'Right'])
    ax_actions.grid(True, alpha=0.3)
    
    # State variables over time
    ax_states = fig.add_subplot(gs[2, :])
    
    if observations:
        cart_positions = [obs[0] for obs in observations]
        cart_velocities = [obs[1] for obs in observations]
        pole_angles = [obs[2] for obs in observations]
        pole_velocities = [obs[3] for obs in observations]
        
        ax_states.plot(steps, cart_positions, 'r-', label='Cart Position', linewidth=2)
        ax_states.plot(steps, pole_angles, 'g-', label='Pole Angle', linewidth=2)
        ax_states.plot(steps, np.array(cart_velocities) * 0.1, 'r--', alpha=0.7, label='Cart Velocity (×0.1)')
        ax_states.plot(steps, np.array(pole_velocities) * 0.1, 'g--', alpha=0.7, label='Pole Velocity (×0.1)')
        
        ax_states.set_title('State Variables Over Time')
        ax_states.set_xlabel('Step')
        ax_states.set_ylabel('Value')
        ax_states.legend()
        ax_states.grid(True, alpha=0.3)
        
        # Add danger zones
        ax_states.axhline(y=0.2, color='orange', linestyle=':', alpha=0.5, label='Danger Zone')
        ax_states.axhline(y=-0.2, color='orange', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.show()


def compare_policies_side_by_side():
    """
    Compare multiple policies in a side-by-side view.
    """
    # Create policies
    env_temp = gym.make("CartPole-v1")
    policies = [
        (RandomPolicy(env_temp.action_space), "Random Policy"),
        (HeuristicPolicy(), "Heuristic Policy"),
        (ImprovedHeuristicPolicy(), "Improved Heuristic")
    ]
    env_temp.close()
    
    print("🎯 COMPARING POLICIES SIDE BY SIDE")
    print("=" * 60)
    
    # Collect data for all policies
    all_data = []
    for policy, name in policies:
        frames, rewards, actions, observations, action_names = capture_episode_frames(policy, name)
        all_data.append((frames, rewards, actions, observations, action_names, name))
    
    # Create comparison plot
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Policy Comparison - Key Moments', fontsize=16, fontweight='bold')
    
    for col, (frames, rewards, actions, observations, action_names, name) in enumerate(all_data):
        # Show start, middle, and end frames
        frame_indices = [0, len(frames) // 2, len(frames) - 1]
        labels = ['Start', 'Middle', 'End']
        
        for row, (frame_idx, label) in enumerate(zip(frame_indices, labels)):
            ax = axes[row, col]
            if frame_idx < len(frames):
                ax.imshow(frames[frame_idx])
                title = f'{name}\n{label} - Step {frame_idx + 1}'
                if frame_idx < len(rewards):
                    title += f'\nReward: {rewards[frame_idx]:.0f}'
                ax.set_title(title, fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Show performance comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for frames, rewards, actions, observations, action_names, name in all_data:
        steps = range(1, len(rewards) + 1)
        ax1.plot(steps, rewards, linewidth=2, label=name, marker='o', markersize=2)
    
    ax1.set_title('Reward Progression Comparison')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Cumulative Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Final performance bar chart
    final_rewards = [data[1][-1] if data[1] else 0 for data in all_data]
    episode_lengths = [len(data[1]) for data in all_data]
    policy_names = [data[5] for data in all_data]
    
    x = range(len(policy_names))
    bars = ax2.bar(x, final_rewards, alpha=0.7, color=['red', 'green', 'blue'])
    ax2.set_title('Final Episode Performance')
    ax2.set_xlabel('Policy')
    ax2.set_ylabel('Total Reward')
    ax2.set_xticks(x)
    ax2.set_xticklabels(policy_names, rotation=45)
    
    # Add value labels on bars
    for bar, reward, length in zip(bars, final_rewards, episode_lengths):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{reward:.0f}\n({length} steps)', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return all_data


def show_policy_behavior_analysis():
    """
    Detailed analysis of how different policies behave.
    """
    print("🧠 POLICY BEHAVIOR ANALYSIS")
    print("=" * 60)
    
    # Run heuristic policy for detailed analysis
    env_temp = gym.make("CartPole-v1")
    heuristic_policy = HeuristicPolicy()
    env_temp.close()
    
    frames, rewards, actions, observations, action_names = capture_episode_frames(
        heuristic_policy, "Heuristic Policy (Detailed Analysis)", max_steps=300
    )
    
    show_episode_summary(frames, rewards, actions, observations, action_names, "Heuristic Policy")
    
    # Show decision-making process
    if observations:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Heuristic Policy - Decision Making Process', fontsize=14)
        
        steps = range(len(observations))
        
        # Pole angle vs actions
        pole_angles = [obs[2] for obs in observations]
        action_colors = ['red' if a == 0 else 'blue' for a in actions]
        
        ax1.scatter(steps, pole_angles, c=action_colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_title('Pole Angle vs Actions\n(Red=Left, Blue=Right)')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Pole Angle (radians)')
        ax1.grid(True, alpha=0.3)
        
        # Angular velocity vs actions
        pole_velocities = [obs[3] for obs in observations]
        ax2.scatter(steps, pole_velocities, c=action_colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('Pole Angular Velocity vs Actions')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Angular Velocity (rad/s)')
        ax2.grid(True, alpha=0.3)
        
        # Decision boundaries
        angle_action_pairs = [(obs[2], actions[i]) for i, obs in enumerate(observations)]
        velocity_action_pairs = [(obs[3], actions[i]) for i, obs in enumerate(observations)]
        
        angles_left = [pair[0] for pair in angle_action_pairs if pair[1] == 0]
        angles_right = [pair[0] for pair in angle_action_pairs if pair[1] == 1]
        
        ax3.hist([angles_left, angles_right], bins=20, alpha=0.7, 
                label=['Left Actions', 'Right Actions'], color=['red', 'blue'])
        ax3.set_title('Action Distribution by Pole Angle')
        ax3.set_xlabel('Pole Angle (radians)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Performance correlation
        window_size = 10
        if len(rewards) > window_size:
            windowed_rewards = [np.mean(rewards[max(0, i-window_size):i+1]) 
                              for i in range(len(rewards))]
            ax4.plot(steps, windowed_rewards, 'g-', linewidth=2, label='Smoothed Reward')
            ax4.plot(steps, rewards, 'b-', alpha=0.3, label='Actual Reward')
            ax4.set_title('Performance Over Time')
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Reward')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("🎮 CARTPOLE VISUAL ANALYSIS")
    print("=" * 60)
    
    choice = input("""
Choose visualization mode:
1. Single policy detailed analysis
2. Compare all policies side-by-side  
3. Policy behavior analysis
4. All visualizations

Enter your choice (1/2/3/4): """).strip()
    
    if choice in ['1', '4']:
        print("\n1️⃣ SINGLE POLICY ANALYSIS")
        env_temp = gym.make("CartPole-v1")
        policy = HeuristicPolicy()
        env_temp.close()
        
        frames, rewards, actions, observations, action_names = capture_episode_frames(
            policy, "Heuristic Policy"
        )
        show_episode_summary(frames, rewards, actions, observations, action_names, "Heuristic Policy")
    
    if choice in ['2', '4']:
        print("\n2️⃣ POLICY COMPARISON")
        all_data = compare_policies_side_by_side()
    
    if choice in ['3', '4']:
        print("\n3️⃣ BEHAVIOR ANALYSIS")
        show_policy_behavior_analysis()
    
    print("\n✅ Visual analysis completed!")
