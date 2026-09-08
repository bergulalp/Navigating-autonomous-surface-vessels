"""Visible, chunked PPO training and development-only constant selection."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def select(directory,out):
    from usv_research.analyze import load
    directory=Path(directory);data=load([directory])
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['signature']['split']!='development':raise ValueError('Only development data can select a setting')
    d=data[data.policy.str.startswith('constant_')]
    if set(d.policy)!=set(f'constant_{i}' for i in range(9)):raise ValueError('Evaluate all nine constant actions first')
    counts=d.groupby('policy').size()
    if counts.nunique()!=1:raise ValueError('Incomplete action comparisons')
    expected={(s,k) for s in manifest['signature']['scenarios'] for k in manifest['signature']['seeds']}
    for _,g in d.groupby('policy'):
        if set(zip(g.scenario,g.seed))!=expected:raise ValueError('Missing development missions')
    means=d.groupby('policy').mean_step_reward.mean()
    best=int(means.idxmax().split('_')[-1])
    result=dict(selected_action=best,criterion='mean_step_reward',development_seeds=manifest['signature']['seeds'],
        scenarios=manifest['signature']['scenarios'],config=manifest['signature']['config'],
        development_means=means.to_dict(),source_directory=str(directory),
        evaluator_sha256=manifest['signature']['evaluator_sha256'],
        development_manifest_sha256=hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest())
    target=Path(out);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and json.loads(target.read_text())!=result:raise ValueError('Use a new selection path; do not overwrite a frozen selection')
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(f'Development-selected conventional setting: action {best}; neutral economic action: 4')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('train');q.add_argument('--out',default='runs/windows_seed0')
    q.add_argument('--steps',type=int,default=20000);q.add_argument('--chunk',type=int,default=2048)
    q.add_argument('--workers',type=int,default=1);q.add_argument('--seed',type=int,default=0)
    q.add_argument('--config',default='configs/train.json')
    q=sub.add_parser('select');q.add_argument('--data',required=True);q.add_argument('--out',required=True)
    a=p.parse_args();os.chdir(ROOT)
    if a.command=='select':select(a.data,a.out);return
    if min(a.steps,a.chunk,a.workers)<1:p.error('Positive steps, chunk and workers required')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    checkpoint=out/'final.zip';timing=out/'training_timing.jsonl'
    if (out/'manifest.json').exists() and not checkpoint.exists():
        raise ValueError('Incomplete previous launch has no final checkpoint. Use a new --out directory.')
    remaining=a.steps;added=0;start=time.perf_counter()
    print('5.9.1. MARIN — PPO high-level parameter training',flush=True)
    print('Each step is 60 simulated seconds. Progress appears after each saved chunk.',flush=True)
    while remaining>0:
        requested=min(a.chunk,remaining)
        cmd=[sys.executable,'-m','usv_bio.ppo_train','--config',a.config,'--out',str(out),
             '--steps',str(requested),'--seed',str(a.seed),'--workers',str(a.workers)]
        if checkpoint.exists():cmd+=['--resume',str(checkpoint)]
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        print(f'Running next {requested} steps; checkpoint: {checkpoint}',flush=True)
        subprocess.run(cmd,check=True,env=env)
        log=json.loads(timing.read_text().splitlines()[-1]);actual=log['actual_added_steps']
        added+=actual;remaining=max(0,remaining-actual)
        elapsed=time.perf_counter()-start;rate=added/elapsed
        print(f'Saved. Added {added} steps this invocation; total {log["total_steps"]}; '
              f'{rate:.2f} steps/s; estimated remaining {remaining/max(rate,1e-9)/60:.1f} min',flush=True)
    print('Training complete. Evaluate on development missions before opening final test results.',flush=True)


if __name__=='__main__':main()
