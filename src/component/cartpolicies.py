# -*- coding: utf-8 -*-
"""
Author: Satya Phanindra Kumar Kalaga
Date: 2025-09-10
Version: 1.0

Reinforcement Learning Policies for CartPole Environment
"""

import numpy as np
from abc import ABC, abstractmethod


class BasePolicy(ABC):
    """Abstract base class for all policies."""
    
    @abstractmethod
    def get_action(self, observation):
        """Return action given observation."""
        pass
    
    @abstractmethod
    def get_name(self):
        """Return policy name."""
        pass


class RandomPolicy(BasePolicy):
    """Random policy that selects actions uniformly at random."""
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def get_action(self, observation):
        """Return random action."""
        return self.action_space.sample()
    
    def get_name(self):
        return "Random Policy"


class HeuristicPolicy(BasePolicy):
    """
    Simple heuristic policy for CartPole based on pole angle and angular velocity.
    
    Strategy:
    - If pole is tilting right (positive angle) or has positive angular velocity,
      move cart right to get under the pole
    - If pole is tilting left (negative angle) or has negative angular velocity,
      move cart left
    """
    
    def get_action(self, observation):
        """
        Return action based on pole angle and angular velocity.
        
        Args:
            observation: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
        
        Returns:
            action: 0 (left) or 1 (right)
        """
        cart_position, cart_velocity, pole_angle, pole_angular_velocity = observation
        
        # If pole is tilting right and/or has positive angular velocity, move right
        # If pole is tilting left and/or has negative angular velocity, move left
        if pole_angle > 0 or pole_angular_velocity > 0:
            return 1  # Move right
        else:
            return 0  # Move left
    
    def get_name(self):
        return "Heuristic Policy"


class QLearningPolicy(BasePolicy):
    """
    Q-Learning policy with epsilon-greedy action selection.
    """
    
    def __init__(self, bins=(6, 6, 12, 12), learning_rate=0.1, 
                 discount_factor=0.95, epsilon=0.1):
        """
        Initialize Q-Learning policy.
        
        Args:
            bins: Number of bins for discretizing each state dimension
            learning_rate: Learning rate for Q-value updates
            discount_factor: Discount factor for future rewards
            epsilon: Exploration rate for epsilon-greedy
        """
        self.bins = bins
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros(bins + (2,))  # 2 actions: left and right
        self.training_mode = True
    
    def discretize_observation(self, observation):
        """
        Discretize continuous observation space for Q-table lookup.
        
        Args:
            observation: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
        
        Returns:
            state: Discretized state index tuple
        """
        cart_position, cart_velocity, pole_angle, pole_angular_velocity = observation
        
        # Define bounds for discretization based on CartPole environment limits
        cart_pos_bins = np.linspace(-2.4, 2.4, self.bins[0])
        cart_vel_bins = np.linspace(-2.0, 2.0, self.bins[1])
        pole_angle_bins = np.linspace(-0.2095, 0.2095, self.bins[2])
        pole_vel_bins = np.linspace(-2.0, 2.0, self.bins[3])
        
        # Discretize each dimension
        cart_pos_idx = np.digitize(cart_position, cart_pos_bins) - 1
        cart_vel_idx = np.digitize(cart_velocity, cart_vel_bins) - 1
        pole_angle_idx = np.digitize(pole_angle, pole_angle_bins) - 1
        pole_vel_idx = np.digitize(pole_angular_velocity, pole_vel_bins) - 1
        
        # Clip to valid range
        cart_pos_idx = np.clip(cart_pos_idx, 0, self.bins[0] - 1)
        cart_vel_idx = np.clip(cart_vel_idx, 0, self.bins[1] - 1)
        pole_angle_idx = np.clip(pole_angle_idx, 0, self.bins[2] - 1)
        pole_vel_idx = np.clip(pole_vel_idx, 0, self.bins[3] - 1)
        
        return (cart_pos_idx, cart_vel_idx, pole_angle_idx, pole_vel_idx)
    
    def get_action(self, observation):
        """
        Return action using epsilon-greedy policy.
        
        Args:
            observation: Current state observation
        
        Returns:
            action: Selected action (0 or 1)
        """
        state = self.discretize_observation(observation)
        
        if self.training_mode and np.random.random() < self.epsilon:
            return np.random.choice([0, 1])  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
        """
        current_q = self.q_table[state + (action,)]
        next_max_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        self.q_table[state + (action,)] = new_q
    
    def set_training_mode(self, training=True):
        """Set training mode (affects exploration)."""
        self.training_mode = training
    
    def set_epsilon(self, epsilon):
        """Set exploration rate."""
        self.epsilon = epsilon
    
    def get_name(self):
        return "Q-Learning Policy"


class ImprovedHeuristicPolicy(BasePolicy):
    """
    Improved heuristic policy that considers both angle and position.
    This policy tries to balance the pole while keeping the cart centered.
    """
    
    def __init__(self, angle_threshold=0.05, position_threshold=1.0):
        """
        Initialize improved heuristic policy.
        
        Args:
            angle_threshold: Threshold for considering pole balanced
            position_threshold: Threshold for cart position from center
        """
        self.angle_threshold = angle_threshold
        self.position_threshold = position_threshold
    
    def get_action(self, observation):
        """
        Return action based on pole angle, angular velocity, and cart position.
        
        Args:
            observation: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
        
        Returns:
            action: 0 (left) or 1 (right)
        """
        cart_position, cart_velocity, pole_angle, pole_angular_velocity = observation
        
        # Primary concern: balance the pole
        if abs(pole_angle) > self.angle_threshold or abs(pole_angular_velocity) > 0.1:
            # Pole needs balancing - move in direction to stabilize
            if pole_angle > 0 or pole_angular_velocity > 0:
                return 1  # Move right
            else:
                return 0  # Move left
        
        # Secondary concern: keep cart centered when pole is balanced
        if abs(cart_position) > self.position_threshold:
            if cart_position > 0:
                return 0  # Move left to center
            else:
                return 1  # Move right to center
        
        # If pole is balanced and cart is centered, use subtle corrections
        # Consider cart velocity to prevent oscillation
        if cart_velocity > 0.1:
            return 0  # Slow down rightward movement
        elif cart_velocity < -0.1:
            return 1  # Slow down leftward movement
        
        # Default: maintain current direction based on slight pole tilt
        return 1 if pole_angle >= 0 else 0
    
    def get_name(self):
        return "Improved Heuristic Policy"
