"""Optional Gymnasium interface. Install requirements-rl.txt to use this module."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .world import World
from .config import Config


class USVEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None, scenarios=None, domain_randomization=True):
        self.config = config or Config()
        self.scenarios = scenarios or ("calm", "storm", "intermittent", "sensor_fault", "moving_front")
        self.domain_randomization = domain_randomization
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(-np.inf, np.inf, (13,), np.float32)
        self.world = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        world_seed = options.get("world_seed", int(self.np_random.integers(100000, 2000000000)))
        scenario = options.get("scenario", self.scenarios[int(self.np_random.integers(len(self.scenarios)))])
        data = self.config.dict()
        if self.domain_randomization:
            data["radar_r50_calm_m"] *= float(self.np_random.uniform(.85, 1.15))
            data["radio_r50_calm_m"] *= float(self.np_random.uniform(.85, 1.15))
            data["wave_memory_s"] *= float(self.np_random.uniform(.7, 1.4))
        self.world = World(Config(**data), scenario, world_seed, "active")
        return self.world.features(), {}

    def step(self, action):
        if self.world is None:
            raise RuntimeError("Call reset() first")
        obs, reward, done = self.world.step_decision(int(action))
        info = {"episode_metrics": self.world.result()} if done else {}
        # The duration defines a finite mission and is observed through time/duration.
        return obs, reward, done, False, info

