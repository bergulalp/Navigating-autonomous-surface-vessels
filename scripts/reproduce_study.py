"""Reproduce the frozen main and diagnostic studies. Uses restartable CLI jobs."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def suite(kind,workers):
    jobs=[]
    def add(config,out,policies,scenarios,seeds,seed_start,control,extra=()):
        jobs.append([sys.executable,'-m','usv_research.experiment','--config',config,'--control',control,
                     '--out',out,'--policies',*policies.split(),'--scenarios',*scenarios.split(),
                     '--seeds',str(seeds),'--seed-start',str(seed_start),'--workers',str(workers),'--trace',*extra])
    if kind in ('main','all'):
        for profile,config in [('nominal','configs/benchmark.json'),('short','configs/short_radio.json')]:
            add(config,'results/main_'+profile,'fixed_ring static_tuned adaptive economic fungal deadline_fusion greedy_sparse recruitment',
                'storm moving_front sensor_fault',20,7000,'configs/control_'+profile+'.json')
    if kind in ('supplements','all'):
        common='static_tuned economic fungal deadline_fusion'
        add('configs/short_radio.json','results/ablation_short','economic economic_no_cost fungal fungal_no_fusion deadline_fusion recruitment recruitment_no_social',
            'storm moving_front',10,7200,'configs/control_short.json')
        for price,label in [(.01,'low'),(.075,'high')]:
            add('configs/short_radio.json','results/price_'+label,'economic fungal deadline_fusion','storm moving_front',10,7200,
                'configs/control_short.json',('--energy-price',str(price)))
        add('configs/shifted_short.json','results/shifted_short',common,'storm moving_front sensor_fault',8,7400,'configs/control_short.json')
        add('configs/slow_short.json','results/slow_short',common,'moving_front',8,7500,'configs/control_short.json')
        add('configs/integration_128.json','results/integration_128','economic fungal deadline_fusion','storm moving_front',4,7600,'configs/control_nominal.json')
        for profile,config in [('nominal','configs/benchmark.json'),('short','configs/short_radio.json')]:
            add(config,'results/negative_'+profile,common,'calm blackout',6,7700,'configs/control_'+profile+'.json')
    return jobs


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',choices=['main','supplements','all'],default='main')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--show',action='store_true');a=p.parse_args()
    for command in suite(a.suite,a.workers):
        print(' '.join(command),flush=True)
        if not a.show:subprocess.run(command,cwd=ROOT,check=True)


if __name__=='__main__':main()
