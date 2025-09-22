# -*- coding: utf-8 -*-
"""
Author: Prudhvi Chekuri
Date: 2025-09-21
Version: 1.0
Description: A Q-learning agent to solve the CartPole-v1 environment from OpenAI Gymnasium.
"""


import time
import random
import collections
import numpy as np
import gymnasium as gym



# Record the start time for performance measurement
start_time = time.time()

# --- 1. Hyperparameters and Setup ---

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Environment
train_env = gym.make("CartPole-v1")

# Learning parameters
LEARNING_RATE = 0.1         # Alpha: How much we update Q-values based on new info
DISCOUNT = 0.95             # Gamma: How much we value future rewards
EPISODES = 50000            # Max episodes to run if the agent doesn't solve it early

# Exploration parameters (Epsilon-greedy strategy)
epsilon = 1.0               # Initial exploration rate
MIN_EPSILON = 0.01          # Minimum exploration rate
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = (epsilon - MIN_EPSILON) / (END_EPSILON_DECAYING - START_EPSILON_DECAYING) # Linear decay


# --- 2. State Discretization ---

# Define how many "buckets" to split each of the 4 continuous state values into
NUM_BUCKETS = (10, 10, 12, 12) # (pos, vel, ang, ang_vel)

# Get the state space bounds directly from the environment
# This makes the code more adaptable to other environments
STATE_BOUNDS = list(zip(train_env.observation_space.low, train_env.observation_space.high))
# Some bounds are very large, so we manually override them for better bucket distribution
STATE_BOUNDS[1] = (-4, 4)
STATE_BOUNDS[3] = (-4, 4)

# Create the bins that will be used to discretize the continuous state space
bucket_bins = [
    np.linspace(STATE_BOUNDS[i][0], STATE_BOUNDS[i][1], NUM_BUCKETS[i] - 1)
    for i in range(len(STATE_BOUNDS))
]

def discretize_state(state):
    """Converts a continuous state from the environment into a discrete tuple."""
    discrete_state = tuple(
        np.digitize(s_val, bucket_bins[i])
        for i, s_val in enumerate(state)
    )
    return discrete_state


# --- 3. Q-Table Initialization ---

# The Q-table size will be (num_pos, num_vel, num_ang, num_ang_vel, num_actions)
q_table_shape = NUM_BUCKETS + (train_env.action_space.n,)
q_table = np.random.uniform(low=-2, high=0, size=q_table_shape)


# --- 4. Training Loop ---

rewards_queue = collections.deque(maxlen=100)
print("🚀 Starting training...")

for episode in range(EPISODES):
    # Reset the environment for a new episode
    initial_state, info = train_env.reset(seed=SEED)
    discrete_state = discretize_state(initial_state)
    
    terminated = truncated = False
    episode_reward = 0
    
    while not terminated and not truncated:
        # Epsilon-greedy action selection
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state]) # Exploit
        else:
            action = train_env.action_space.sample()    # Explore
            
        # Take the action and get the outcome
        new_state, reward, terminated, truncated, _ = train_env.step(action)
        new_discrete_state = discretize_state(new_state)
        episode_reward += reward
        
        # Update Q-table using the Bellman equation
        if not terminated:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        
        # If the action led to failure, penalize it by setting Q-value to 0
        elif terminated:
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

    # Decay epsilon to reduce exploration over time, with a floor at MIN_EPSILON
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon = max(MIN_EPSILON, epsilon - epsilon_decay_value)
        
    # Check for solved condition
    rewards_queue.append(episode_reward)
    if len(rewards_queue) == 100:
        average_reward = sum(rewards_queue) / len(rewards_queue)
        # Progress update every 2000 episodes
        if (episode + 1) % 2000 == 0:
            print(f"Episode: {episode + 1} | Avg. Reward (last 100): {average_reward:.2f} | Epsilon: {epsilon:.4f}")
        
        # If solved, stop training
        if average_reward >= 495.0: # CartPole-v1 is considered solved at 475, but we use 495 for a stricter criterion
            print(f"\n✅ Solved! Achieved an average reward of {average_reward:.2f} over the last 100 episodes.")
            print(f"Training finished after {episode + 1} episodes.")
            break

train_env.close()


# --- 5. Fast, Non-Visual Evaluation Phase ---

print("\n⚙️  Running fast evaluation (no visuals)...")
eval_env = gym.make("CartPole-v1")
total_rewards = []
num_test_episodes = 100

for episode in range(num_test_episodes):
    state, info = eval_env.reset(seed=SEED + episode) # Different seed for robust evaluation
    discrete_state = discretize_state(state)
    
    terminated = truncated = False
    episode_reward = 0
    
    while not terminated and not truncated:
        # Always choose the best action (greedy policy)
        action = np.argmax(q_table[discrete_state])
        new_state, reward, terminated, truncated, _ = eval_env.step(action)
        discrete_state = discretize_state(new_state)
        episode_reward += reward
        
    total_rewards.append(episode_reward)

eval_env.close()

# Final performance report
final_average_reward = sum(total_rewards) / num_test_episodes
print(f"\n--- Evaluation Report ---")
print(f"Final average score over {num_test_episodes} episodes: {final_average_reward:.2f}")

if final_average_reward >= 475:
    print("🏆 Performance confirmed: The agent has successfully solved CartPole-v1!")
else:
    print("📉 Performance did not meet the 'solved' threshold in the final evaluation.")


# --- 6. Final Visual Demonstration ---
print("\n🎬 Showing one visual demonstration of the trained agent...")

demo_env = gym.make("CartPole-v1", render_mode="human")
state, info = demo_env.reset(seed=SEED)
discrete_state = discretize_state(state)
terminated = truncated = False
demo_reward = 0

while not terminated and not truncated:
    action = np.argmax(q_table[discrete_state]) # Use the learned policy
    new_state, reward, terminated, truncated, _ = demo_env.step(action)
    discrete_state = discretize_state(new_state)
    demo_reward += reward
    time.sleep(0.01) # Slow down rendering slightly to make it easier to watch

print(f"Score for the demonstration episode: {demo_reward}")
demo_env.close()

end_time = time.time()
print(f"\n⏱️ Total execution time: {end_time - start_time:.2f} seconds")