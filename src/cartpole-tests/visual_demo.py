# -*- coding: utf-8 -*-
"""
Author: Satya Phanindra Kumar Kalaga
Date: 2025-09-10
Version: 1.0

Visual demonstration of CartPole policies
"""

import sys
import os
import time
# Add the component directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'component'))

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from policies import RandomPolicy, HeuristicPolicy, QLearningPolicy, ImprovedHeuristicPolicy
from trainer import PolicyTrainer


def run_policy_with_visual(policy, policy_name, episodes=3, render_mode='human'):
    """
    Run a policy with visual rendering.
    
    Args:
        policy: Policy to run
        policy_name: Name for display
        episodes: Number of episodes to run
        render_mode: 'human' for window, 'rgb_array' for matplotlib
    """
    print(f"\n🎮 Running {policy_name} with visual rendering...")
    print("=" * 60)
    
    if render_mode == 'human':
        env = gym.make("CartPole-v1", render_mode='human')
        
        for episode in range(episodes):
            observation, info = env.reset()
            total_reward = 0
            step_count = 0
            
            print(f"\nEpisode {episode + 1}: Press Ctrl+C to stop early")
            
            for step in range(500):
                # Render the environment
                env.render()
                
                # Get action from policy
                action = policy.get_action(observation)
                
                # Take action
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step_count += 1
                
                # Add small delay to make it watchable
                time.sleep(0.02)  # 50 FPS
                
                if terminated or truncated:
                    print(f"Episode {episode + 1} finished: {step_count} steps, reward = {total_reward}")
                    time.sleep(1)  # Pause between episodes
                    break
        
        env.close()
    
    elif render_mode == 'rgb_array':
        run_policy_with_matplotlib(policy, policy_name, episodes)


def run_policy_with_matplotlib(policy, policy_name, episodes=1):
    """
    Run a policy and display using matplotlib animation.
    """
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    
    # Collect frames from one episode
    frames = []
    rewards = []
    actions = []
    observations = []
    
    print(f"🎬 Recording {policy_name} episode for matplotlib animation...")
    
    observation, info = env.reset()
    total_reward = 0
    
    for step in range(500):
        # Capture frame
        frame = env.render()
        frames.append(frame)
        
        # Get action
        action = policy.get_action(observation)
        actions.append("Left" if action == 0 else "Right")
        
        # Store current state
        observations.append(observation.copy())
        
        # Take action
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        rewards.append(total_reward)
        
        if terminated or truncated:
            print(f"Episode completed: {step + 1} steps, total reward = {total_reward}")
            break
    
    env.close()
    
    # Create animated plot
    create_animated_plot(frames, rewards, actions, observations, policy_name)


def create_animated_plot(frames, rewards, actions, observations, policy_name):
    """
    Create an animated matplotlib plot showing the CartPole environment and metrics.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{policy_name} - CartPole Performance', fontsize=16)
    
    # Environment view
    im = ax1.imshow(frames[0])
    ax1.set_title('CartPole Environment')
    ax1.axis('off')
    
    # Reward over time
    ax2.set_title('Cumulative Reward')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Total Reward')
    reward_line, = ax2.plot([], [], 'b-', linewidth=2)
    ax2.grid(True, alpha=0.3)
    
    # State variables over time
    ax3.set_title('Cart Position & Pole Angle')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Value')
    pos_line, = ax3.plot([], [], 'r-', label='Cart Position', linewidth=2)
    angle_line, = ax3.plot([], [], 'g-', label='Pole Angle', linewidth=2)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Action history
    ax4.set_title('Action History (Last 20 steps)')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Action')
    ax4.set_ylim(-0.5, 1.5)
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Left', 'Right'])
    action_line, = ax4.plot([], [], 'mo-', markersize=6)
    ax4.grid(True, alpha=0.3)
    
    # Text displays
    step_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def animate(frame_idx):
        if frame_idx >= len(frames):
            return im, reward_line, pos_line, angle_line, action_line, step_text
        
        # Update environment image
        im.set_array(frames[frame_idx])
        
        # Update reward plot
        steps = list(range(frame_idx + 1))
        reward_line.set_data(steps, rewards[:frame_idx + 1])
        ax2.set_xlim(0, max(10, frame_idx))
        ax2.set_ylim(0, max(10, max(rewards[:frame_idx + 1]) * 1.1))
        
        # Update state plots
        cart_positions = [obs[0] for obs in observations[:frame_idx + 1]]
        pole_angles = [obs[2] for obs in observations[:frame_idx + 1]]
        
        pos_line.set_data(steps, cart_positions)
        angle_line.set_data(steps, pole_angles)
        
        # Set axis limits for state plot
        if cart_positions and pole_angles:
            ax3.set_xlim(0, max(10, frame_idx))
            y_min = min(min(cart_positions), min(pole_angles)) * 1.1
            y_max = max(max(cart_positions), max(pole_angles)) * 1.1
            ax3.set_ylim(y_min, y_max)
        
        # Update action history (last 20 steps)
        action_window = min(20, frame_idx + 1)
        start_idx = max(0, frame_idx + 1 - action_window)
        action_steps = list(range(start_idx, frame_idx + 1))
        action_values = [0 if actions[i] == "Left" else 1 for i in range(start_idx, frame_idx + 1)]
        
        action_line.set_data(action_steps, action_values)
        ax4.set_xlim(max(0, frame_idx - 19), frame_idx + 1)
        
        # Update text
        step_text.set_text(f'Step: {frame_idx + 1}\nAction: {actions[frame_idx]}\nReward: {rewards[frame_idx]:.1f}')
        
        return im, reward_line, pos_line, angle_line, action_line, step_text
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(frames), interval=100, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return anim


def compare_policies_visually():
    """
    Compare different policies with visual rendering.
    """
    print("🎯 VISUAL POLICY COMPARISON")
    print("=" * 80)
    
    # Create policies
    env_temp = gym.make("CartPole-v1")
    policies = [
        (RandomPolicy(env_temp.action_space), "Random Policy"),
        (HeuristicPolicy(), "Heuristic Policy"),
        (ImprovedHeuristicPolicy(), "Improved Heuristic Policy")
    ]
    env_temp.close()
    
    choice = input("""
Choose visualization mode:
1. Live window (human rendering) - Shows real-time CartPole animation
2. Matplotlib animation - Shows detailed metrics with environment
3. Both modes

Enter your choice (1/2/3): """).strip()
    
    if choice in ['1', '3']:
        print("\n🎮 LIVE WINDOW MODE")
        print("Close the window to proceed to the next policy")
        print("=" * 60)
        
        for policy, name in policies:
            input(f"Press Enter to start {name}...")
            try:
                run_policy_with_visual(policy, name, episodes=2, render_mode='human')
            except KeyboardInterrupt:
                print(f"\n{name} demonstration stopped by user.")
                continue
    
    if choice in ['2', '3']:
        print("\n🎬 MATPLOTLIB ANIMATION MODE")
        print("=" * 60)
        
        for policy, name in policies:
            input(f"Press Enter to record and animate {name}...")
            run_policy_with_matplotlib(policy, name, episodes=1)


def train_and_visualize_qlearning():
    """
    Train Q-learning policy and show before/after visualization.
    """
    print("🧠 Q-LEARNING TRAINING VISUALIZATION")
    print("=" * 60)
    
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    trainer = PolicyTrainer(env)
    
    # Create untrained Q-learning policy
    qlearning_policy = QLearningPolicy(epsilon=0.1)
    
    print("Showing UNTRAINED Q-learning policy...")
    run_policy_with_matplotlib(qlearning_policy, "Untrained Q-Learning", episodes=1)
    
    input("Press Enter to start training...")
    
    # Train the policy
    trainer.train_qlearning(qlearning_policy, num_episodes=1000, verbose=True, progress_interval=200)
    
    print("Training completed! Showing TRAINED Q-learning policy...")
    qlearning_policy.set_training_mode(False)  # Turn off exploration
    run_policy_with_matplotlib(qlearning_policy, "Trained Q-Learning", episodes=1)
    
    env.close()


if __name__ == "__main__":
    print("🎮 CARTPOLE VISUAL DEMONSTRATION")
    print("=" * 80)
    
    menu_choice = input("""
Choose what you'd like to see:
1. Compare all policies visually
2. Train Q-learning and see before/after
3. Quick heuristic policy demo
4. All of the above

Enter your choice (1/2/3/4): """).strip()
    
    if menu_choice in ['1', '4']:
        compare_policies_visually()
    
    if menu_choice in ['2', '4']:
        train_and_visualize_qlearning()
    
    if menu_choice in ['3', '4']:
        print("\n🚀 QUICK HEURISTIC DEMO")
        env_temp = gym.make("CartPole-v1")
        heuristic = HeuristicPolicy()
        env_temp.close()
        
        run_policy_with_visual(heuristic, "Quick Heuristic Demo", episodes=1, render_mode='human')
    
    print("\n✅ Visual demonstration completed!")
