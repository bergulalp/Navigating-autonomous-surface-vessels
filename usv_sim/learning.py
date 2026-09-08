"""Dependency-light Double Q-learning of high-level formation options.

This is tabular reinforcement learning in a partially observed, discretized state;
it is not deep RL, multi-agent RL, or a convergence claim for the physical system.
"""
import json
from pathlib import Path
import numpy as np
from .world import World
from .config import Config, source_hash


class DoubleQ:
    shape = (4, 3, 3, 3, 8, 8)

    def __init__(self, seed=0):
        self.q1 = np.zeros(self.shape)
        self.q2 = np.zeros(self.shape)
        self.visits = np.zeros(self.shape, dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        self.episodes = 0

    @staticmethod
    def state(features):
        f = features
        return (int(np.digitize(f[0], [.3, .55, .75])),
                int(np.digitize(f[1], [.80, 1.02])),
                int(np.digitize(f[2], [.18, .28])),
                int(np.digitize(f[6], [.05, .3])),
                int(np.clip(round(float(f[10]) * 7), 0, 7)))

    def action(self, features, epsilon=0.0):
        s = self.state(features)
        if self.rng.uniform() < epsilon:
            return int(self.rng.integers(8))
        # Deterministic tie breaking is part of the checkpoint's specification.
        return int(np.argmax(self.q1[s] + self.q2[s]))

    def update(self, features, action, reward, next_features, terminal, gamma=.97):
        s, sn = self.state(features), self.state(next_features)
        index = s + (action,)
        self.visits[index] += 1
        alpha = max(0.04, 0.35 / (1 + self.visits[index] / 30) ** 0.6)
        qa, qb = (self.q1, self.q2) if self.rng.uniform() < .5 else (self.q2, self.q1)
        greedy = int(np.argmax(qa[sn]))
        target = reward + (0 if terminal else gamma * qb[sn + (greedy,)])
        qa[index] += alpha * (target - qa[index])

    def save(self, path, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(temp, q1=self.q1, q2=self.q2, visits=self.visits,
                            episodes=self.episodes, rng_state=json.dumps(self.rng.bit_generator.state),
                            metadata=json.dumps(metadata))
        temp.replace(path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            obj = cls()
            obj.q1, obj.q2, obj.visits = data["q1"].copy(), data["q2"].copy(), data["visits"].copy()
            obj.episodes = int(data["episodes"])
            obj.rng.bit_generator.state = json.loads(str(data["rng_state"]))
            obj.metadata = json.loads(str(data["metadata"]))
        return obj


def train(config, output, episodes=200, train_seed=0, resume=False, randomize=True):
    path = Path(output)
    agent = DoubleQ.load(path) if resume and path.exists() else DoubleQ(train_seed)
    training_scenarios = ("calm", "storm", "intermittent", "sensor_fault", "moving_front")
    meta = dict(algorithm="tabular_double_q", training_seed=train_seed, config=config.dict(),
                source_sha256=source_hash(), scenarios=training_scenarios, randomize=randomize,
                reward="mean delivered freshness - 0.08 normalized interval energy - 0.1 stale command fraction",
                observation="13 coordinator-visible features; DoubleQ discretizes five",
                training_episode_seed_rule="100000 + training_seed*10000 + episode_index",
                trained_episodes=agent.episodes)
    if resume and hasattr(agent, "metadata"):
        for key in ("config", "source_sha256", "training_seed", "randomize"):
            if agent.metadata.get(key) != meta[key]:
                raise ValueError(f"Cannot resume with changed {key}; create a new checkpoint")
    history_path = path.with_suffix(".history.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a" if resume else "w") as history:
        for episode in range(agent.episodes, agent.episodes + episodes):
            seed = 100000 + train_seed * 10000 + episode
            rng = np.random.default_rng(seed + 810000)
            scenario = training_scenarios[episode % len(training_scenarios)]
            data = config.dict()
            if randomize:
                data["radio_r50_calm_m"] *= float(rng.uniform(.85, 1.15))
                data["radar_r50_calm_m"] *= float(rng.uniform(.85, 1.15))
                data["wave_memory_s"] *= float(rng.uniform(.7, 1.4))
            world = World(Config(**data), scenario, seed, "active")
            features = world.features()
            total_reward = 0.0
            epsilon = max(.12, .8 * np.exp(-episode / 150))
            done = False
            while not done:
                action = agent.action(features, epsilon)
                following, reward, done = world.step_decision(action)
                agent.update(features, action, reward, following, done)
                features = following
                total_reward += reward
            agent.episodes = episode + 1
            result = world.result()
            row = dict(episode=episode, seed=seed, scenario=scenario, epsilon=epsilon,
                       total_reward=total_reward, delivered=result["delivered"], energy_wh=result["energy_wh"])
            history.write(json.dumps(row) + "\n")
            history.flush()
            meta["trained_episodes"] = agent.episodes
            if agent.episodes % 10 == 0:
                agent.save(path, meta)
                print(json.dumps(row), flush=True)
        agent.save(path, meta)
    return path

