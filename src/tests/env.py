import gymnasium as gym
import numpy as np

class PseudoLabelEnv(gym.Env):
    def __init__(self):
        super().__init__()

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)
        self.current_index = 0
        state = np.zeros(self.observation_space.shape, dtype = np.float32)
        info = {}
        return state, info

    def step(self, action):
        pass

    def render(self):
        pass

    def close(self):
        pass