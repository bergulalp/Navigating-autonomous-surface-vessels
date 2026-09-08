"""Select one static ring per known hardware profile using development data only."""
import argparse
import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from usv_sim.config import Config, SCENARIOS
from usv_research.control import ControlConfig
from usv_research.experiment import job_run, extension_hash


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--out', required=True)
    p.add_argument('--seeds', type=int, default=4); p.add_argument('--seed-start', type=int, default=110)
    p.add_argument('--workers', type=int, default=4)
    args=p.parse_args()
    cfg=Config.read(args.config); out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    radii=[.75,.8,.85,.9,.95,1.,1.05,1.1]
    jobs=[(cfg.dict(),ControlConfig(static_radius=r).dict(),s,seed,'static_tuned',None)
          for r in radii for s in SCENARIOS for seed in range(args.seed_start,args.seed_start+args.seeds)]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, f in enumerate(as_completed([pool.submit(job_run,j) for j in jobs]),1):
            row=f.result(); rows.append(row)
            (out/(row['run_id']+'.json')).write_text(json.dumps(row,indent=2,allow_nan=False))
            if i%40==0: print(f'{i}/{len(jobs)} static development episodes',flush=True)
    summary=[]
    for r in radii:
        rr=[x for x in rows if x['control']['static_radius']==r]
        C=np.mean([x['delivered'] for x in rr]); E=np.mean([x['energy_wh'] for x in rr])
        U=C-ControlConfig().energy_price*E/(cfg.n*100*cfg.duration_s/3600)
        summary.append(dict(radius=r,coverage=float(C),energy_wh=float(E),utility=float(U)))
    chosen=max(summary,key=lambda x:x['utility'])
    result=dict(selected=chosen,candidates=summary,scenarios=list(SCENARIOS),seeds=list(range(args.seed_start,args.seed_start+args.seeds)),
                selection='Equal scenario mixture; C-0.025*E/(N*100W*T/3600). All deployment energy included.',
                extension_sha256=extension_hash())
    (out/'selection.json').write_text(json.dumps(result,indent=2))
    (out/'selected_control.json').write_text(json.dumps(ControlConfig(static_radius=chosen['radius']).dict(),indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
