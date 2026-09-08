"""PPO continuation with a recorded, distinct environment stream per chunk.

The policy/action/observation model is unchanged from v3. This runner addresses
repeated environment sequences on frequent resume; it is not bitwise resume.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from usv_sim.config import Config
from usv_research.ppo import env_factory
from usv_research.residual import rl_source_hash


def trainer_hash():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def stream_seed(training_seed,completed_steps):
    return int(np.random.SeedSequence([7331,training_seed,completed_steps]).generate_state(1)[0])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/train.json');p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=2048);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--workers',type=int,default=1);p.add_argument('--resume');a=p.parse_args()
    if min(a.steps,a.workers)<1 or a.seed<0:p.error('Positive steps/workers and nonnegative seed required')
    import torch,gymnasium,stable_baselines3
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback
    torch.set_num_threads(1)
    cfg=Config.read(a.config);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    signature=dict(config=cfg.dict(),stack=4,rl_source_sha256=rl_source_hash(),training_seed=a.seed)
    manifest=out/'manifest.json'
    if a.resume:
        old=json.loads(Path(a.resume).parent.joinpath('manifest.json').read_text())
        if old['signature']!=signature:raise ValueError('Checkpoint configuration/source/seed differs')
    elif manifest.exists():raise ValueError('Use --resume or a new output directory')
    fs=[env_factory(cfg,4,True,500000+a.seed*1000+i) for i in range(a.workers)]
    vec=SubprocVecEnv(fs) if a.workers>1 else DummyVecEnv(fs)
    model=PPO.load(a.resume,env=vec,device='cpu') if a.resume else PPO('MlpPolicy',vec,
        seed=a.seed,device='cpu',n_steps=128,batch_size=64,gamma=.99,learning_rate=3e-4,
        policy_kwargs={'net_arch':[128,128]},verbose=0)
    before=model.num_timesteps
    # Do this AFTER PPO construction/load, which may seed the vector environment.
    # VecEnv applies these seeds at the next explicit reset inside learn().
    stream=stream_seed(a.seed,before);vec.seed(stream)
    versions=dict(torch=torch.__version__,gymnasium=gymnasium.__version__,stable_baselines3=stable_baselines3.__version__)
    manifest.write_text(json.dumps(dict(signature=signature,versions=versions,arguments=vars(a),
        trainer_sha256=trainer_hash(),stream_rule='SeedSequence([7331, training_seed, completed_steps])'),indent=2)+'\n')
    callback=CheckpointCallback(save_freq=max(1,10000//a.workers),save_path=str(out),name_prefix='ppo')
    start=time.perf_counter()
    model.learn(a.steps,callback=callback,reset_num_timesteps=not bool(a.resume))
    model.save(out/'final');elapsed=time.perf_counter()-start
    row=dict(requested_steps=a.steps,actual_added_steps=model.num_timesteps-before,
        total_steps=model.num_timesteps,wall_s=elapsed,steps_per_second=(model.num_timesteps-before)/elapsed,
        checkpoint_sha256=hashlib.sha256((out/'final.zip').read_bytes()).hexdigest(),
        environment_stream_seed=stream,trainer_sha256=trainer_hash())
    with (out/'training_timing.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row,indent=2));vec.close()


if __name__=='__main__':main()
