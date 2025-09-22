# Q-Learning Agent for CartPole-v1

This experiment provides an implementation of a Q-learning agent designed to solve the CartPole-v1 environment from the Gymnasium library. The purpose of this code is to understand the RL training loop and how it is implemented using the Gymnasium library and also understand the core concepts of Q-Learning Algorithm.

## The CartPole Problem

The goal in the CartPole environment is to balance a pole on top of a movable cart. The agent can only perform two actions: push the cart to the left or push it to the right. The episode ends if the pole tilts more than 15 degrees from the vertical or if the cart moves more than 2.4 units from the center. The agent receives a reward of +1 for every timestep it keeps the pole upright. The environment is considered "solved" if the agent achieves an average reward of 475 over 100 consecutive episodes.

## Core Concepts Implemented

### The Reinforcement Learning Loop

This project is built around the fundamental agent-environment interaction loop:

1. The agent observes the current **State** of the environment.
2. Based on the state, the agent chooses an **Action**.
3. The environment transitions to a **New State** and provides a **Reward** as feedback.
4. The agent uses this feedback to update its knowledge and improve its decision-making.

### Q-Learning

Q-learning is a model-free reinforcement learning algorithm used to find the optimal action-selection policy. It works by learning a Q-function, which estimates the value (or quality) of taking a certain action in a given state. This code uses a Q-table - a large matrix where rows represent all possible states and columns represent all possible actions. Each cell in the table, Q(s, a), stores the expected future reward for taking action a in state s.

The table is updated using the Bellman equation:
```
Q(s, a) ← (1 - α) * Q(s, a) + α * [R + γ * max_a' Q(s', a')]
```

### State Discretization

The CartPole environment has a continuous state space (the cart position, pole angle, etc., are floating-point numbers). A Q-table, however, requires discrete states to be used as table indices. To solve this, we use discretization. We divide the continuous range of each state variable into a finite number of "buckets" or "bins". This converts a continuous state like [-0.23, 0.5, 0.01, -0.1] into a discrete, indexable tuple like (4, 7, 6, 5).

### Epsilon-Greedy Strategy

To ensure the agent both finds new strategies and perfects existing ones, it must balance exploration (trying random actions) and exploitation (taking the best-known action). This is achieved with an epsilon-greedy strategy. The agent acts greedily with a probability of 1 - ε and explores randomly with a probability of ε. The value of ε (epsilon) starts at 1.0 (100% exploration) and gradually decays to a small minimum value, allowing the agent to rely more on its learned knowledge as training progresses.

## How to Run the Code

### Setup with a Virtual Environment (venv)

It is highly recommended to run this project in a dedicated virtual environment to avoid conflicts with other Python packages.

#### 1. Create the Virtual Environment:
Navigate to the project directory in your terminal and run:

```bash
# For macOS/Linux
python3 -m venv venv

# For Windows
python -m venv venv
```

This will create a new folder named `venv` in your project directory.

#### 2. Activate the Virtual Environment:
You must activate the environment before installing dependencies.

```bash
# For macOS/Linux
source venv/bin/activate

# For Windows
.\venv\Scripts\activate
```

Your terminal prompt should now be prefixed with `(venv)`.

#### 3. Install Prerequisites:
With the virtual environment active, install the required libraries:

```bash
pip install gymnasium numpy
```

### Execution

Once the setup is complete, run the Python script from your terminal:

```bash
python cartpole.py
```

After you are finished, you can deactivate the virtual environment by simply typing `deactivate` in the terminal.

## Expected Outcome

When you run the script, it will:

1. **Start Training**: Display a 🚀 Starting training... message and begin the learning process without any visuals for maximum speed. It will print its average performance every 2000 episodes.

2. **Stop Automatically**: Training will automatically conclude once the agent achieves an average reward of 495 or more over 100 consecutive episodes.

3. **Run Evaluation**: After training, it will run a fast, non-visual evaluation over 100 episodes and print a final performance report.

4. **Show Demonstration**: Finally, a window will pop up showing the fully trained agent playing one episode of CartPole, allowing you to visually confirm its performance.