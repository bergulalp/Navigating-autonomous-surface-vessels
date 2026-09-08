"""Paired evaluation of PPO and all nine conventional constant settings.

Only trusted, locally produced SB3 checkpoints should be loaded: their format
contains serialized Python objects. Learning efficacy is an empirical question.
"""
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from usv_sim.config import Config
from usv_research.residual import ResidualMission,rl_source_hash

SCENARIOS=['storm','sensor_fault','moving_front','intermittent']


def evaluator_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+rl_source_hash().encode()).hexdigest()


def evaluate_job(job):
    cfgdata,scenario,seed,spec=job
    cfg=Config(**cfgdata).validate();checkpoint=spec.get('checkpoint')
    if checkpoint:
        import torch
        from stable_baselines3 import PPO
        from usv_research.ppo import env_factory
        torch.set_num_threads(1)
        model=PPO.load(checkpoint,device='cpu')
        env=env_factory(cfg,4,False)()
        obs,_=env.reset(seed=seed,options={'world_seed':seed,'scenario':scenario})
    else:
        env=ResidualMission(cfg,False)
        obs=env.reset(seed=seed,options={'world_seed':seed,'scenario':scenario})
    done=False;actions=np.zeros(9,int);rewards=[]
    while not done:
        action=int(model.predict(obs,deterministic=True)[0]) if checkpoint else spec['action']
        actions[action]+=1
        if checkpoint:obs,reward,done,truncated,info=env.step(action);done=done or truncated
        else:obs,reward,done,info=env.step(action)
        rewards.append(float(reward))
    row=info['episode_metrics']
    row.update(policy=spec['name'],mean_step_reward=float(np.mean(rewards)),
        discounted_return=float(np.array(rewards)@(.99**np.arange(len(rewards)))),
        postwarmup_utility=float(row['delivered']-.025*row['energy_wh']/(cfg.n*100*cfg.duration_s/3600)),
        residual_actions=actions.tolist(),checkpoint_sha256=spec.get('sha256'),
        rl_source_sha256=rl_source_hash(),evaluator_sha256=evaluator_hash(),policy_spec=spec)
    if checkpoint:env.close()
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/bio_short.json');p.add_argument('--out',required=True)
    p.add_argument('--constants',nargs='*',type=int,default=[])
    p.add_argument('--checkpoints',nargs='*',default=[]);p.add_argument('--selection')
    p.add_argument('--seeds',type=int,default=3);p.add_argument('--seed-start',type=int,default=15000)
    p.add_argument('--scenarios',nargs='+',default=SCENARIOS);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--split',choices=['development','test','custom'],default='development');a=p.parse_args()
    if a.seeds<1 or a.workers<1:p.error('Use positive seeds/workers')
    cfg=Config.read(a.config);constants=list(a.constants)
    selection=None
    if a.selection:
        selection=json.loads(Path(a.selection).read_text())
        if selection['config']!=cfg.dict():raise ValueError('Selection and evaluation configurations differ')
        if set(range(a.seed_start,a.seed_start+a.seeds)) & set(selection['development_seeds']):
            raise ValueError('Final test seeds overlap development seeds')
        constants.extend([selection['selected_action'],4])
    specs=[]
    for action in sorted(set(constants)):
        if not 0<=action<9:p.error('Constant actions are 0..8')
        specs.append(dict(name=f'constant_{action}',action=action))
    for filename in a.checkpoints:
        path=Path(filename)
        meta=json.loads(path.parent.joinpath('manifest.json').read_text())
        if meta['signature']['rl_source_sha256']!=rl_source_hash():raise ValueError('Checkpoint causal source differs')
        if meta['signature']['stack']!=4 or meta['signature']['config']['n']!=cfg.n:
            raise ValueError('Checkpoint observation dimensions differ')
        seed=meta['signature']['training_seed']
        specs.append(dict(name=f'ppo_seed{seed}',checkpoint=str(path),training_seed=seed,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if not specs:p.error('Supply constants, checkpoints or a development selection')
    if len({s['name'] for s in specs})!=len(specs):p.error('Only one checkpoint per training seed per evaluation')
    dest=Path(a.out);dest.mkdir(parents=True,exist_ok=True)
    signature=dict(config=cfg.dict(),policies=specs,evaluator_sha256=evaluator_hash(),split=a.split,
        scenarios=a.scenarios,seeds=list(range(a.seed_start,a.seed_start+a.seeds)),selection=selection)
    manifest=dest/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['signature']!=signature:
            raise ValueError('Output is for different checkpoints/settings; use a new output directory')
    else:manifest.write_text(json.dumps(dict(signature=signature,arguments=vars(a)),indent=2))
    jobs=[]
    for scenario in a.scenarios:
        for seed in signature['seeds']:
            for spec in specs:
                path=dest/f'{spec["name"]}_{scenario}_{seed}.json'
                if not path.exists():jobs.append((cfg.dict(),scenario,seed,spec))
    started=time.perf_counter();print(f'{len(jobs)} pending paired evaluations',flush=True)
    def save(row,i):
        path=dest/f'{row["policy"]}_{row["scenario"]}_{row["seed"]}.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(row,indent=2,allow_nan=False));tmp.replace(path)
        if i%8==0 or i==len(jobs):print(f'{i}/{len(jobs)}; elapsed {time.perf_counter()-started:.1f}s',flush=True)
    if a.workers==1:
        for i,j in enumerate(jobs,1):save(evaluate_job(j),i)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for i,f in enumerate(as_completed([pool.submit(evaluate_job,j) for j in jobs]),1):save(f.result(),i)
    with (dest/'timing.jsonl').open('a') as f:
        f.write(json.dumps(dict(completed=len(jobs),wall_s=time.perf_counter()-started,workers=a.workers))+'\n')


if __name__=='__main__':main()
