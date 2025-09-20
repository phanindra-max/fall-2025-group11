# -*- coding: utf-8 -*-
"""
Author: Satya Phanindra Kumar Kalaga
Date: 2025-09-10
Version: 1.0

Automatic visual demonstration - no user input required
"""

import sys
import os
# Add the component directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'component'))

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from policies import RandomPolicy, HeuristicPolicy, ImprovedHeuristicPolicy


def capture_and_compare_all_policies():
    """
    Automatically capture and compare all policies.
    """
    print("🎯 AUTOMATIC POLICY COMPARISON WITH VISUALS")
    print("=" * 70)
    
    # Create policies
    env_temp = gym.make("CartPole-v1")
    policies = [
        (RandomPolicy(env_temp.action_space), "Random Policy", 'red'),
        (HeuristicPolicy(), "Heuristic Policy", 'green'),
        (ImprovedHeuristicPolicy(), "Improved Heuristic", 'blue')
    ]
    env_temp.close()
    
    # Collect data for all policies
    all_results = []
    
    for policy, name, color in policies:
        print(f"🎬 Recording {name}...")
        
        env = gym.make("CartPole-v1", render_mode='rgb_array')
        
        frames = []
        rewards = []
        actions = []
        observations = []
        
        observation, info = env.reset()
        total_reward = 0
        
        for step in range(300):  # Max steps
            # Capture frame
            frame = env.render()
            frames.append(frame)
            
            # Store current state
            observations.append(observation.copy())
            
            # Get action
            action = policy.get_action(observation)
            actions.append(action)
            
            # Take action
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            rewards.append(total_reward)
            
            if terminated or truncated:
                print(f"  Episode completed: {step + 1} steps, total reward = {total_reward}")
                break
        
        env.close()
        all_results.append((frames, rewards, actions, observations, name, color))
    
    # Create comprehensive comparison visualization
    create_comprehensive_comparison(all_results)
    
    return all_results


def create_comprehensive_comparison(all_results):
    """
    Create a comprehensive comparison visualization.
    """
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('CartPole Policy Comparison - Visual Analysis', fontsize=20, fontweight='bold')
    
    # Create grid layout
    gs = fig.add_gridspec(4, 4, height_ratios=[2, 1, 1, 1])
    
    # 1. Show key frames for each policy (top row)
    print("📊 Creating visual comparison...")
    
    frame_positions = [0, 1, 2]  # Start, middle, end positions
    
    for col, (frames, rewards, actions, observations, name, color) in enumerate(all_results):
        for row in range(3):  # Show 3 key frames per policy
            ax = fig.add_subplot(gs[row, col])
            
            if row == 0:  # Start frame
                frame_idx = 0
                label = "Start"
            elif row == 1:  # Middle frame
                frame_idx = min(len(frames) // 2, len(frames) - 1)
                label = "Middle"
            else:  # End frame
                frame_idx = len(frames) - 1
                label = "End"
            
            if frame_idx < len(frames):
                ax.imshow(frames[frame_idx])
                title = f'{name}\n{label} - Step {frame_idx + 1}'
                if frame_idx < len(rewards):
                    title += f'\nReward: {rewards[frame_idx]:.0f}'
                ax.set_title(title, fontsize=10, color=color, fontweight='bold')
            ax.axis('off')
    
    # 2. Performance comparison (bottom row)
    ax_performance = fig.add_subplot(gs[3, :])
    
    for frames, rewards, actions, observations, name, color in all_results:
        steps = range(1, len(rewards) + 1)
        ax_performance.plot(steps, rewards, linewidth=3, label=name, 
                          color=color, marker='o', markersize=2, alpha=0.8)
    
    ax_performance.set_title('Reward Progression Comparison', fontsize=14, fontweight='bold')
    ax_performance.set_xlabel('Step', fontsize=12)
    ax_performance.set_ylabel('Cumulative Reward', fontsize=12)
    ax_performance.legend(fontsize=12)
    ax_performance.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Create separate detailed analysis plots
    create_detailed_analysis(all_results)


def create_detailed_analysis(all_results):
    """
    Create detailed analysis plots.
    """
    # Performance summary
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed Policy Analysis', fontsize=16, fontweight='bold')
    
    # Extract summary data
    policy_names = [result[4] for result in all_results]
    final_rewards = [result[1][-1] if result[1] else 0 for result in all_results]
    episode_lengths = [len(result[1]) for result in all_results]
    colors = [result[5] for result in all_results]
    
    # 1. Final performance bar chart
    bars = ax1.bar(policy_names, final_rewards, color=colors, alpha=0.7)
    ax1.set_title('Final Episode Performance', fontweight='bold')
    ax1.set_ylabel('Total Reward')
    ax1.set_xticklabels(policy_names, rotation=45, ha='right')
    
    # Add value labels
    for bar, reward, length in zip(bars, final_rewards, episode_lengths):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{reward:.0f}\n({length} steps)', ha='center', va='bottom', fontweight='bold')
    
    # 2. Episode length comparison
    bars2 = ax2.bar(policy_names, episode_lengths, color=colors, alpha=0.7)
    ax2.set_title('Episode Length (Survival Time)', fontweight='bold')
    ax2.set_ylabel('Steps')
    ax2.set_xticklabels(policy_names, rotation=45, ha='right')
    
    # Add value labels
    for bar, length in zip(bars2, episode_lengths):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{length}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Action distribution analysis
    for i, (frames, rewards, actions, observations, name, color) in enumerate(all_results):
        if actions:
            left_actions = sum(1 for a in actions if a == 0)
            right_actions = sum(1 for a in actions if a == 1)
            
            action_data = [left_actions, right_actions]
            action_labels = ['Left', 'Right']
            
            if i == 0:  # Only show for first policy to avoid clutter
                ax3.pie(action_data, labels=action_labels, autopct='%1.1f%%', 
                       colors=['lightcoral', 'skyblue'], startangle=90)
                ax3.set_title(f'{name} - Action Distribution', fontweight='bold')
    
    # 4. Performance efficiency (reward per step)
    efficiency = [final_rewards[i] / episode_lengths[i] if episode_lengths[i] > 0 else 0 
                 for i in range(len(policy_names))]
    
    bars3 = ax4.bar(policy_names, efficiency, color=colors, alpha=0.7)
    ax4.set_title('Performance Efficiency (Reward/Step)', fontweight='bold')
    ax4.set_ylabel('Reward per Step')
    ax4.set_xticklabels(policy_names, rotation=45, ha='right')
    
    # Add value labels
    for bar, eff in zip(bars3, efficiency):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{eff:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Show policy behavior insights
    show_policy_insights(all_results)


def show_policy_insights(all_results):
    """
    Show insights about policy behavior.
    """
    print("\n🧠 POLICY BEHAVIOR INSIGHTS")
    print("=" * 60)
    
    for frames, rewards, actions, observations, name, color in all_results:
        print(f"\n📋 {name}:")
        print(f"  • Final Reward: {rewards[-1] if rewards else 0:.0f}")
        print(f"  • Episode Length: {len(rewards)} steps")
        
        if actions:
            left_pct = (sum(1 for a in actions if a == 0) / len(actions)) * 100
            right_pct = 100 - left_pct
            print(f"  • Action Balance: {left_pct:.1f}% Left, {right_pct:.1f}% Right")
        
        if observations:
            max_angle = max(abs(obs[2]) for obs in observations)
            avg_angle = np.mean([abs(obs[2]) for obs in observations])
            print(f"  • Pole Control: Max angle {max_angle:.3f}, Avg angle {avg_angle:.3f}")
            
            cart_range = max(obs[0] for obs in observations) - min(obs[0] for obs in observations)
            print(f"  • Cart Movement Range: {cart_range:.3f}")
    
    # Performance comparison
    final_rewards = [result[1][-1] if result[1] else 0 for result in all_results]
    best_idx = np.argmax(final_rewards)
    worst_idx = np.argmin(final_rewards)
    
    print(f"\n🏆 PERFORMANCE SUMMARY:")
    print(f"  • Best Policy: {all_results[best_idx][4]} ({final_rewards[best_idx]:.0f} points)")
    print(f"  • Worst Policy: {all_results[worst_idx][4]} ({final_rewards[worst_idx]:.0f} points)")
    
    if final_rewards[worst_idx] > 0:
        improvement = ((final_rewards[best_idx] - final_rewards[worst_idx]) / final_rewards[worst_idx]) * 100
        print(f"  • Improvement: {improvement:.1f}% better than worst")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"  • Random policy performs poorly due to lack of strategy")
    print(f"  • Heuristic policies leverage physics understanding")
    print(f"  • Simple rules can dramatically outperform random actions")
    print(f"  • The CartPole problem rewards consistent, physics-based decisions")


if __name__ == "__main__":
    print("🎮 CARTPOLE AUTOMATIC VISUAL DEMONSTRATION")
    print("=" * 80)
    print("This will automatically show visual comparisons of all policies...")
    print("Close the matplotlib windows to continue through the demo.\n")
    
    # Run automatic comparison
    results = capture_and_compare_all_policies()
    
    print("\n✅ Visual demonstration completed!")
    print("You should have seen:")
    print("  📸 CartPole environment frames for each policy")
    print("  📊 Performance comparison charts")
    print("  📈 Detailed analysis of policy behavior")
    print("  🧠 Insights about what makes policies effective")
