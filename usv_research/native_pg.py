"""NumPy REINFORCE residual policy with observation history and resumable Adam.

This small linear-softmax policy is an executable learning baseline. It is not
PPO, SAC, a recurrent network, or a claim that learning beats economic control.
"""
import argparse
from collections import deque
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import softmax
from usv_sim.config import Config
from .residual import ResidualMission, rl_source_hash


class Agent:
    def __init__(self, size, decisions, stack=4, seed=0):
        self.stack=stack;self.rng=np.random.default_rng(seed)
        self.w=np.zeros((size*stack+1,9));self.adam_m=np.zeros_like(self.w);self.adam_v=np.zeros_like(self.w)
        self.baseline=np.zeros(decisions);self.episodes=0;self.history=deque(maxlen=stack)
    def reset(self,obs):
        self.history.clear()
        for _ in range(self.stack):self.history.append(obs.copy())
    def features(self,obs,append=True):
        if append:self.history.append(obs.copy())
        return np.r_[np.concatenate(self.history)/2,1.0]
    def act(self,obs,deterministic=False):
        x=self.features(obs);prob=softmax(x@self.w)
        action=int(np.argmax(prob)) if deterministic else int(self.rng.choice(9,p=prob))
        return action,x,prob
    def update(self,trajectory,gamma=.99,lr=.003):
        rewards=np.array([r[3] for r in trajectory]);returns=np.zeros(len(rewards));future=0.
        for k in range(len(rewards)-1,-1,-1):
            future=rewards[k]+gamma*future;returns[k]=future
        gradient=np.zeros_like(self.w)
        for t,(x,prob,action,_) in enumerate(trajectory):
            advantage=returns[t]-self.baseline[t]
            score=-prob.copy();score[action]+=1
            H=-np.sum(prob*np.log(np.maximum(prob,1e-15)))
            entropy_grad=-prob*(np.log(np.maximum(prob,1e-15))+H)
            gradient+=np.outer(x,gamma**t*advantage*score+.001*entropy_grad)
        gradient/=len(trajectory)
        gradient/=max(1.,np.linalg.norm(gradient))
        self.episodes+=1
        self.adam_m=.9*self.adam_m+.1*gradient;self.adam_v=.999*self.adam_v+.001*gradient**2
        self.w+=lr*(self.adam_m/(1-.9**self.episodes))/(np.sqrt(self.adam_v/(1-.999**self.episodes))+1e-8)
        # The baseline used above depends on previous episodes, not this action's outcome.
        self.baseline[:len(returns)]=.9*self.baseline[:len(returns)]+.1*returns
        return float(returns[0])
    def save(self,path,metadata):
        path=Path(path);temp=path.with_suffix('.tmp')
        with temp.open('wb') as f:
            np.savez_compressed(f,w=self.w,adam_m=self.adam_m,adam_v=self.adam_v,baseline=self.baseline,
                                episodes=self.episodes,stack=self.stack,rng=json.dumps(self.rng.bit_generator.state),
                                metadata=json.dumps(metadata))
        temp.replace(path)
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            stack=int(z['stack']);a=cls((z['w'].shape[0]-1)//stack,len(z['baseline']),stack)
            for name in ('w','adam_m','adam_v','baseline'):setattr(a,name,z[name].copy())
            a.episodes=int(z['episodes']);a.rng.bit_generator.state=json.loads(str(z['rng']))
            meta=json.loads(str(z['metadata']))
        return a,meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for command in ('train','evaluate'):
        q=sub.add_parser(command);q.add_argument('--config',default='configs/train.json');q.add_argument('--out',required=True)
        if command=='train':
            q.add_argument('--episodes',type=int,default=100);q.add_argument('--seed',type=int,default=0)
            q.add_argument('--stack',type=int,default=4);q.add_argument('--resume',action='store_true')
        else:
            q.add_argument('--checkpoint',required=True);q.add_argument('--seeds',type=int,default=10)
            q.add_argument('--seed-start',type=int,default=12000)
            q.add_argument('--scenarios',nargs='+',default=['storm','sensor_fault','moving_front'])
            q.add_argument('--constant-action',type=int)
    args=p.parse_args();cfg=Config.read(args.config);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.command=='train':
        meta=dict(config=cfg.dict(),seed=args.seed,stack=args.stack,rl_source_sha256=rl_source_hash(),algorithm='linear-softmax REINFORCE')
        checkpoint=out/'checkpoint.npz';history=out/'episodes.jsonl'
        if args.resume:
            a,old=Agent.load(checkpoint)
            if old!=meta:raise ValueError('Resume config, source, stack or seed differs')
            if history.exists():
                rows=[json.loads(x) for x in history.read_text().splitlines()]
                history.write_text(''.join(json.dumps(x)+'\n' for x in rows if x['episode']<=a.episodes))
        else:
            if checkpoint.exists() or history.exists():raise ValueError('Use --resume or a new output directory')
            a=Agent(13*cfg.n+5,int(cfg.duration_s/cfg.decision_s),args.stack,args.seed)
        (out/'manifest.json').write_text(json.dumps(meta,indent=2))
        start=time.perf_counter();before=a.episodes
        for _ in range(args.episodes):
            env=ResidualMission(cfg,True)
            obs=env.reset(seed=int(a.rng.integers(100000,2000000000)));a.reset(obs)
            done=False;trajectory=[];actions=np.zeros(9,int)
            while not done:
                action,x,prob=a.act(obs);actions[action]+=1
                obs,r,done,info=env.step(action);trajectory.append((x,prob,action,r))
            G=a.update(trajectory);row=info['episode_metrics']
            row.update(episode=a.episodes,discounted_return=G,actions=actions.tolist(),rl_source_sha256=rl_source_hash())
            # Checkpoint commits complete episodes; history recovery trims any extra line.
            with history.open('a') as f:f.write(json.dumps(row)+'\n')
            a.save(checkpoint,meta)
            if a.episodes%10==0:print(f'{a.episodes} episodes; C={row["delivered"]:.3f}, E={row["energy_wh"]:.1f} Wh',flush=True)
        timing=dict(added_episodes=a.episodes-before,total_episodes=a.episodes,wall_s=time.perf_counter()-start)
        with (out/'timing.jsonl').open('a') as f:f.write(json.dumps(timing)+'\n')
        print(json.dumps(timing));return
    a,meta=Agent.load(args.checkpoint)
    if meta['rl_source_sha256']!=rl_source_hash():raise ValueError('Checkpoint source differs')
    if a.w.shape[0]!=(13*cfg.n+5)*a.stack+1:raise ValueError('Checkpoint vessel count differs')
    if args.constant_action is not None and not 0<=args.constant_action<9:raise ValueError('Constant action must be in 0..8')
    checkpoint_hash=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    for scenario in args.scenarios:
        for seed in range(args.seed_start,args.seed_start+args.seeds):
            env=ResidualMission(cfg,False);obs=env.reset(seed=seed,options={'world_seed':seed,'scenario':scenario});a.reset(obs)
            done=False;actions=np.zeros(9,int)
            while not done:
                action=a.act(obs,True)[0] if args.constant_action is None else args.constant_action
                actions[action]+=1;obs,_,done,info=env.step(action)
            row=info['episode_metrics'];row.update(policy='native_residual' if args.constant_action is None else f'constant_{args.constant_action}',
                checkpoint_sha256=checkpoint_hash,rl_source_sha256=rl_source_hash(),residual_actions=actions.tolist())
            (out/f'{scenario}_{seed}.json').write_text(json.dumps(row,indent=2))
    (out/'manifest.json').write_text(json.dumps(dict(arguments=vars(args),config=cfg.dict(),checkpoint_sha256=checkpoint_hash),indent=2))
    print(f'Evaluated {len(args.scenarios)*args.seeds} missions')


if __name__=='__main__':main()
