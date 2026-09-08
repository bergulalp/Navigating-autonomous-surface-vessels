"""Dependency-light mission adapter for a bounded high-level residual learner."""
import hashlib
from pathlib import Path
import numpy as np
from usv_sim.config import Config, source_hash
from .experiment import ResearchWorld, extension_hash
from .control import LocalPolicy, ControlConfig

STEPS=(60.,120.,180.)
PRICES=(.0125,.025,.05)


def rl_source_hash():
    h=hashlib.sha256((source_hash()+extension_hash()).encode())
    for name in ('residual.py','ppo.py','native_pg.py'):
        h.update(name.encode());h.update(Path(__file__).with_name(name).read_bytes())
    return h.hexdigest()


class ResidualMission:
    def __init__(self, config=None, domain_randomization=True):
        self.config=config or Config()
        self.domain_randomization=domain_randomization
        self.rng=np.random.default_rng(0)
        self.world=None
        self.previous_action=4

    @property
    def observation_size(self):return 13*self.config.n+5

    def reset(self, seed=None, options=None):
        if seed is not None:self.rng=np.random.default_rng(seed)
        options=options or {}
        scenarios=('calm','storm','intermittent','sensor_fault','moving_front')
        scenario=options.get('scenario',scenarios[int(self.rng.integers(len(scenarios)))])
        world_seed=int(options.get('world_seed',self.rng.integers(100000,2000000000)))
        data=self.config.dict()
        if self.domain_randomization:
            data['radio_r50_calm_m']*=float(self.rng.uniform(.7,1.1))
            data['radar_r50_calm_m']*=float(self.rng.uniform(.9,1.1))
        cfg=Config(**data)
        self.world=ResearchWorld(cfg,scenario,world_seed,'economic')
        self.world.policy=LocalPolicy('economic',cfg)
        self.previous_action=4
        return self.observation()

    def observation(self):
        w=self.world;cfg=w.cfg
        p,m,ages,s=w.observation()
        # A logged observation count is bounded by log compression. All inputs
        # are cached coordinator state or previously issued goals, never true health.
        mm=m.copy();mm[:,6]=np.log1p(mm[:,6])/8
        energy=w.telemetry[-1,:,9]/cfg.battery_wh
        per=np.c_[p/cfg.mission_radius_m,mm,np.minimum(ages/cfg.command_timeout_s,3),energy,
                  (w.policy.goals-p)/cfg.mission_radius_m]
        global_obs=[s,np.sin(w.wind),np.cos(w.wind),w.t/cfg.duration_s,self.previous_action/8]
        return np.clip(np.r_[per.ravel(),global_obs],-20,20).astype(np.float32)

    def step(self, action):
        if self.world is None:raise RuntimeError('Reset the mission first')
        if int(action)!=action or not 0<=int(action)<9:raise ValueError('Action must be an integer in 0..8')
        action=int(action);w=self.world;cfg=w.cfg
        w.policy.settings=ControlConfig(step_m=STEPS[action//3],energy_price=PRICES[action%3])
        before_e=float(w.fleet.energy.sum());before_i=len(w.logs)
        _,_,done=w.step_decision()
        rows=w.logs[before_i:]
        reward=float(np.mean([r['delivered'] for r in rows]))
        ref=cfg.n*100*len(rows)*cfg.dt_s/3600
        reward-=.025*(w.fleet.energy.sum()-before_e)/ref
        self.previous_action=action
        info={'episode_metrics':w.result()} if done else {}
        return self.observation(),float(reward),done,info
