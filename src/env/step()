import gymnasium as gym
import numpy as np

class PseudoLabelEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.data = np.random.rand(100, 10).astype(np.float32)
        self.labels = np.random.randint(0, 2, size=(100,))
        self.total_steps = len(self.data)
        self.current_index = 0
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_index = 0
        state = self.data[self.current_index]
        info = {}
        return state, info

    def step(self, action):
        correct_label = self.labels[self.current_index]
        reward = 1.0 if action == correct_label else 0.0
        self.current_index += 1
        terminated = self.current_index >= self.total_steps
        truncated = False
        if not terminated:
            next_state = self.data[self.current_index]
        else:
            next_state = np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {}
        return next_state, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
