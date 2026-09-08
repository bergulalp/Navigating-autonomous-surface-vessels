"""Restartable paired biological-mechanism experiments; v3 physics stays frozen."""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from usv_sim.config import Config, SCENARIOS, source_hash
from usv_research.control import ControlConfig
from usv_research.experiment import ResearchWorld, extension_hash
from .control import BioPolicy, BioConfig, POLICIES


def bio_hash():
    h=hashlib.sha256((source_hash()+extension_hash()).encode())
    for name in ('control.py','experiment.py'):
        h.update(name.encode());h.update(Path(__file__).with_name(name).read_bytes())
    return h.hexdigest()


def identity(config,bio,scenario,seed,policy):
    return hashlib.sha256(json.dumps(dict(config=config,bio=bio,scenario=scenario,seed=seed,
        policy=policy,source=bio_hash()),sort_keys=True).encode()).hexdigest()[:24]


def run_job(job):
    data,bio_data,scenario,seed,policy,trace=job
    cfg=Config(**data).validate();b=BioConfig(**bio_data)
    world=ResearchWorld(cfg,scenario,seed,'economic',bool(trace))
    world.policy=BioPolicy(policy,cfg,bio=b)
    row=world.run();row['policy']=policy
    row.update(bio_config=b.dict(),bio_sha256=bio_hash(),extension_sha256=extension_hash(),
        run_id=identity(data,bio_data,scenario,seed,policy))
    rows=[r for r in world.logs if r['t']>=cfg.warmup_s]
    row['delivered_p10']=float(np.quantile([r['delivered'] for r in rows],.1))
    row['accepted_moves']=sum(d['accepted'] for d in world.policy.diagnostics)
    row['mean_recruited_vessels']=float(np.mean([d.get('moving_vessels',0) for d in world.policy.diagnostics]))
    # This metric matches training's undiscounted interval reward exactly.
    row['training_utility']=float(np.mean([r['delivered'] for r in world.logs])-
        .025*row['energy_wh']/(cfg.n*100*cfg.duration_s/3600))
    row['postwarmup_utility']=row['delivered']-.025*row['energy_wh']/(cfg.n*100*cfg.duration_s/3600)
    if trace:
        world.save_trace(trace)
        Path(trace).with_suffix('.decisions.json').write_text(json.dumps(world.policy.diagnostics,indent=2))
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/bio_short.json');p.add_argument('--out',required=True)
    p.add_argument('--bio');p.add_argument('--seeds',type=int,default=8)
    p.add_argument('--seed-start',type=int,default=17000);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--policies',nargs='+',choices=POLICIES,default=list(POLICIES))
    p.add_argument('--scenarios',nargs='+',choices=SCENARIOS,default=['storm','sensor_fault','moving_front','intermittent'])
    p.add_argument('--trace',action='store_true');a=p.parse_args()
    if a.seeds<1 or a.workers<1:p.error('Positive seeds and workers required')
    cfg=Config.read(a.config);b=BioConfig(**(json.loads(Path(a.bio).read_text()) if a.bio else {}))
    if not 0<b.ode_step<=.02 or b.deliberation<=0:raise ValueError('Invalid ODE integration')
    dest=Path(a.out);dest.mkdir(parents=True,exist_ok=True)
    signature=dict(config=cfg.dict(),bio_config=b.dict(),bio_sha256=bio_hash())
    manifest=dest/'manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old['signature']!=signature:raise ValueError('Different source/configuration; use a new output directory')
    else:manifest.write_text(json.dumps(dict(signature=signature,arguments=vars(a),python=sys.version,platform=platform.platform()),indent=2))
    jobs=[]
    for s in a.scenarios:
        for seed in range(a.seed_start,a.seed_start+a.seeds):
            for policy in a.policies:
                key=identity(cfg.dict(),b.dict(),s,seed,policy)
                if (dest/(key+'.json')).exists():continue
                trace=str(dest/(key+'.npz')) if a.trace and seed==a.seed_start else None
                jobs.append((cfg.dict(),b.dict(),s,seed,policy,trace))
    start=time.perf_counter();print(f'{len(jobs)} pending missions; {a.workers} workers',flush=True)
    def save(row,i):
        target=dest/(row['run_id']+'.json');tmp=target.with_suffix('.tmp')
        tmp.write_text(json.dumps(row,indent=2,allow_nan=False));tmp.replace(target)
        if i%8==0 or i==len(jobs):print(f'{i}/{len(jobs)}; {time.perf_counter()-start:.1f}s; {row["policy"]}: C={row["delivered"]:.4f}',flush=True)
    if a.workers==1:
        for i,j in enumerate(jobs,1):save(run_job(j),i)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for i,f in enumerate(as_completed([pool.submit(run_job,j) for j in jobs]),1):save(f.result(),i)
    timing=dict(completed=len(jobs),wall_s=time.perf_counter()-start,workers=a.workers,cpu_count=os.cpu_count())
    with (dest/'timing.jsonl').open('a') as f:f.write(json.dumps(timing)+'\n')
    print(json.dumps(timing),flush=True)


if __name__=='__main__':main()
