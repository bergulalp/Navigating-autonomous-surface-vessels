"""Repeat the declared v4 experiments in a new output tree; never overwrite evidence."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['bio','learning'],required=True)
    p.add_argument('--out',default='runs/reproduce_v4');p.add_argument('--workers',type=int,default=1)
    a=p.parse_args();os.chdir(ROOT);dest=Path(a.out)
    if dest.resolve()==(ROOT/'results').resolve():raise ValueError('Use a new output tree, not frozen results')
    if a.workers<1:p.error('Positive workers required')
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def run(args):
        print('Running: '+' '.join(args),flush=True)
        subprocess.run([sys.executable,*args],check=True,env=env)
    if a.stage=='bio':
        for profile,seeds,start,scenarios in [
            ('bio_pilot',1,16000,['storm','moving_front']),
            ('bio_nominal',12,17000,['storm','sensor_fault','moving_front','intermittent']),
            ('bio_short',12,17000,['storm','sensor_fault','moving_front','intermittent']),
            ('bio_shifted',4,17500,['storm','sensor_fault','moving_front','intermittent']),
            ('bio_integration128',4,17700,['storm','moving_front'])]:
            config='bio_short' if profile=='bio_pilot' else profile
            run(['-m','usv_bio.experiment','--config',f'configs/{config}.json','--out',str(dest/profile),
                 '--seeds',str(seeds),'--seed-start',str(start),'--workers',str(a.workers),'--trace','--scenarios',*scenarios])
        return
    checkpoints=[]
    for seed in range(3):
        out=dest/f'ppo_seed{seed}';cp=out/'final.zip'
        if not cp.exists():run(['-m','usv_research.ppo','train','--config','configs/train.json','--out',str(out),
            '--steps','2048','--seed',str(seed),'--workers','1'])
        checkpoints.append(str(cp))
    run(['-m','usv_bio.learning_eval','--config','configs/bio_short.json','--out',str(dest/'development'),
         '--constants',*map(str,range(9)),'--checkpoints',*checkpoints,'--seeds','3','--seed-start','15000',
         '--split','development','--workers',str(a.workers)])
    run(['scripts/windows_rl.py','select','--data',str(dest/'development'),'--out',str(dest/'selection.json')])
    run(['-m','usv_bio.learning_eval','--config','configs/bio_short.json','--out',str(dest/'test'),
         '--selection',str(dest/'selection.json'),'--checkpoints',*checkpoints,'--seeds','8','--seed-start','18000',
         '--split','test','--workers',str(a.workers)])
    run(['scripts/analyze_v4_learning.py','--data',str(dest/'test'),'--selection',str(dest/'selection.json'),
         '--out',str(dest/'analysis')])


if __name__=='__main__':main()
