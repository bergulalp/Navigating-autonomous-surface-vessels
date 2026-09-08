"""Optional PPO training/evaluation around the v3 economic controller.

Resuming restores policy/optimizer/timestep state. Environment/RNG trajectories
restart; it is not a bitwise continuation of the entire experiment.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from usv_sim.config import Config
from .residual import ResidualMission, rl_source_hash


def env_factory(config, stack=4, randomize=True, seed=0):
    def make():
        import gymnasium as gym
        from gymnasium import spaces
        from gymnasium.wrappers import FrameStackObservation, FlattenObservation
        class ResidualEnv(gym.Env):
            metadata={'render_modes':[]}
            def __init__(self):
                self.mission=ResidualMission(config,randomize)
                self.action_space=spaces.Discrete(9)
                self.observation_space=spaces.Box(-20,20,(self.mission.observation_size,),np.float32)
            def reset(self,*,seed=None,options=None):
                super().reset(seed=seed)
                return self.mission.reset(seed,options),{}
            def step(self,action):
                o,r,d,i=self.mission.step(action)
                return o,r,d,False,i
        env=ResidualEnv()
        env.reset(seed=seed)
        if stack>1:env=FlattenObservation(FrameStackObservation(env,stack_size=stack))
        return env
    return make


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for command in ('train','evaluate','check'):
        q=sub.add_parser(command);q.add_argument('--config',default='configs/benchmark.json')
        q.add_argument('--out',required=True);q.add_argument('--stack',type=int,default=4)
        if command=='train':
            q.add_argument('--steps',type=int,default=20000);q.add_argument('--seed',type=int,default=0)
            q.add_argument('--workers',type=int,default=2);q.add_argument('--resume')
        if command=='evaluate':
            q.add_argument('--checkpoint',required=True);q.add_argument('--seeds',type=int,default=20)
            q.add_argument('--seed-start',type=int,default=12000)
            q.add_argument('--scenarios',nargs='+',default=['storm','sensor_fault','moving_front'])
    args=p.parse_args()
    try:
        import torch, gymnasium, stable_baselines3
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        from stable_baselines3.common.callbacks import CheckpointCallback
    except ImportError as e:
        raise SystemExit('Install the optional RL dependencies using docs/RL_EXPERIMENT.md') from e
    torch.set_num_threads(1)
    cfg=Config.read(args.config);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    versions=dict(torch=torch.__version__,gymnasium=gymnasium.__version__,stable_baselines3=stable_baselines3.__version__)
    signature=dict(config=cfg.dict(),stack=args.stack,rl_source_sha256=rl_source_hash())
    if args.command=='check':
        env=env_factory(cfg,args.stack,False)()
        check_env(env,warn=True)
        obs,_=env.reset(seed=55,options={'world_seed':300,'scenario':'storm'})
        done=False;steps=0;start=time.perf_counter()
        while not done:
            obs,r,done,_,info=env.step(4);steps+=1
        report=dict(status='passed',steps=steps,wall_s=time.perf_counter()-start,
                    observation_shape=list(obs.shape),versions=versions,signature=signature,
                    metrics=info['episode_metrics'])
        (out/'check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));env.close();return
    if args.command=='train':
        if args.steps<1 or args.workers<1:p.error('Use positive steps and workers')
        signature['training_seed']=args.seed
        manifest=out/'manifest.json'
        if args.resume:
            old=json.loads(Path(args.resume).parent.joinpath('manifest.json').read_text())
            if old['signature']!=signature:raise ValueError('Resume source/configuration/seed/stack mismatch')
        elif manifest.exists():raise ValueError('Use --resume or a new output directory')
        manifest.write_text(json.dumps(dict(signature=signature,versions=versions,arguments=vars(args)),indent=2))
        fs=[env_factory(cfg,args.stack,True,500000+args.seed*1000+i) for i in range(args.workers)]
        vec=SubprocVecEnv(fs) if args.workers>1 else DummyVecEnv(fs)
        model=PPO.load(args.resume,env=vec,device='cpu') if args.resume else PPO(
            'MlpPolicy',vec,seed=args.seed,device='cpu',n_steps=128,batch_size=64,gamma=.99,
            learning_rate=3e-4,policy_kwargs={'net_arch':[128,128]},verbose=0)
        start=time.perf_counter();before=model.num_timesteps
        callback=CheckpointCallback(save_freq=max(1,10000//args.workers),save_path=str(out),name_prefix='ppo')
        model.learn(args.steps,callback=callback,reset_num_timesteps=not bool(args.resume))
        model.save(out/'final');elapsed=time.perf_counter()-start
        log=dict(requested_steps=args.steps,actual_added_steps=model.num_timesteps-before,
                 total_steps=model.num_timesteps,wall_s=elapsed,steps_per_second=(model.num_timesteps-before)/elapsed,
                 checkpoint_sha256=hashlib.sha256((out/'final.zip').read_bytes()).hexdigest())
        with (out/'training_timing.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
        print(json.dumps(log,indent=2));vec.close();return
    meta=json.loads(Path(args.checkpoint).parent.joinpath('manifest.json').read_text())
    if meta['signature']['rl_source_sha256']!=rl_source_hash():raise ValueError('Checkpoint source differs')
    if meta['signature']['stack']!=args.stack or meta['signature']['config']['n']!=cfg.n:
        raise ValueError('Checkpoint observation size/stack differs')
    model=PPO.load(args.checkpoint,device='cpu')
    checkpoint_hash=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    env=env_factory(cfg,args.stack,False)();records=[]
    for scenario in args.scenarios:
        for seed in range(args.seed_start,args.seed_start+args.seeds):
            obs,_=env.reset(seed=seed,options={'world_seed':seed,'scenario':scenario})
            done=False;actions=np.zeros(9,int)
            while not done:
                action,_=model.predict(obs,deterministic=True);action=int(action);actions[action]+=1
                obs,_,done,_,info=env.step(action)
            row=info['episode_metrics'];row.update(policy='ppo_residual',residual_actions=actions.tolist(),
                                                  checkpoint_sha256=checkpoint_hash,rl_source_sha256=rl_source_hash())
            (out/f'{scenario}_{seed}.json').write_text(json.dumps(row,indent=2));records.append(row)
    (out/'manifest.json').write_text(json.dumps(dict(config=cfg.dict(),checkpoint_sha256=checkpoint_hash,
                                                   versions=versions,arguments=vars(args)),indent=2))
    print(f'Evaluated {len(records)} episodes');env.close()


if __name__=='__main__':main()
