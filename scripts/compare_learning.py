"""Pair one saved learned policy with one comparator across identical missions."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from usv_research.analyze import load, paired

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('learned');p.add_argument('baseline');p.add_argument('--out',required=True)
    a=p.parse_args();left=load([a.learned]);right=load([a.baseline])
    if left.policy.nunique()!=1 or right.policy.nunique()!=1:
        raise ValueError('Each directory must contain exactly one policy')
    if set(zip(left.scenario,left.seed))!=set(zip(right.scenario,right.seed)):
        raise ValueError('Use exactly matching scenario/seed sets')
    fields=('n','duration_s','dt_s','report_ttl_s','radio_r50_calm_m','radar_r50_calm_m')
    l=left.set_index(['scenario','seed']);r=right.set_index(['scenario','seed'])
    for k in l.index:
        if l.loc[k,'config']!=r.loc[k,'config']:
            raise ValueError(f'Physical configuration differs for {k}')
    left['policy']='learned';right['policy']='baseline'
    left['dataset']=right['dataset']='paired_learning'
    d=pd.concat([left,right]);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    reference=d.config.map(lambda c:c['n']*100*c['duration_s']/3600)
    d['utility']=d.delivered-.025*d.energy_wh/reference
    rows=[]
    for m in ['delivered','energy_wh','utility','capped_aoi_s']:
        rows.append(dict(metric=m,**paired(d,'learned','baseline',m,True)))
    (out/'paired.json').write_text(json.dumps(rows,indent=2))
    d.groupby('policy')[['delivered','energy_wh','utility']].mean().to_csv(out/'means.csv')
    print(pd.DataFrame(rows).to_string(index=False))
    print('Intervals resample complete mission-seed blocks; this is one trained policy, not a multi-training-seed analysis.')

if __name__=='__main__':main()
