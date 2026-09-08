"""Episode-level estimates and paired uncertainty; no time-step pseudo-replication."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

COMPARISONS=[('economic','static_tuned'),('economic','adaptive'),('fungal','economic'),
             ('deadline_fusion','fungal'),('recruitment','greedy_sparse')]


def load(directories):
    rows=[]
    for directory in directories:
        directory=Path(directory)
        for p in sorted(directory.glob('*.json')):
            if p.name.endswith('.decisions.json'):continue
            r=json.loads(p.read_text())
            if not isinstance(r,dict) or 'delivered' not in r or 'seed' not in r:continue
            r['dataset']=directory.name
            rows.append(r)
    if not rows: raise ValueError('No episode records found')
    d=pd.DataFrame(rows)
    if d.duplicated(['dataset','scenario','policy','seed']).any():
        raise ValueError('Duplicate episode keys; do not mix settings in one dataset')
    return d


def interval(values, seed=9031):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(seed)
    draws=values[rng.integers(0,len(values),(20000,len(values)))].mean(axis=1)
    return tuple(float(x) for x in np.quantile(draws,[.025,.975]))


def contrast(a,b):
    d=a-b
    lo,hi=interval(d)
    rng=np.random.default_rng(9127)
    null=(d[None]*rng.choice([-1,1],(20000,len(d)))).mean(axis=1)
    p=(1+np.sum(np.abs(null)>=abs(d.mean())-1e-15))/(len(null)+1)
    return dict(n=len(d),difference=float(d.mean()),ci_low=lo,ci_high=hi,p_sign=float(p))


def paired(d,a,b,metric='delivered',pooled=False):
    keys=['dataset','scenario','seed']
    aa=d[d.policy==a][keys+[metric]]
    bb=d[d.policy==b][keys+[metric]]
    p=aa.merge(bb,on=keys,validate='one_to_one',suffixes=('_a','_b'))
    if p.empty:return None
    if pooled:
        expected=p[['dataset','scenario']].drop_duplicates().shape[0]
        counts=p.groupby('seed').size()
        if not (counts==expected).all():raise ValueError('Incomplete seed blocks in pooled contrast')
        p=p.groupby('seed')[[metric+'_a',metric+'_b']].mean().reset_index()
    return contrast(p[metric+'_a'].to_numpy(),p[metric+'_b'].to_numpy())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('directories',nargs='+');p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    d=load(args.directories)
    metrics=['delivered','energy_wh','capped_aoi_s','delivered_p10','graph_connected',
             'command_stale','early_report_rate','runtime_s','min_separation_m']
    rows=[]
    for (dataset,scenario,policy),g in d.groupby(['dataset','scenario','policy']):
        row=dict(dataset=dataset,scenario=scenario,policy=policy,n=len(g))
        for m in metrics:
            if m not in g:continue
            x=g[m].dropna().to_numpy()
            if not len(x):continue
            lo,hi=interval(x)
            row.update({m:float(x.mean()),m+'_lo':lo,m+'_hi':hi})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
    differences=[]
    for (dataset,scenario),g in d.groupby(['dataset','scenario']):
        for a,b in COMPARISONS:
            for metric in ('delivered','energy_wh'):
                r=paired(g,a,b,metric)
                if r:differences.append(dict(dataset=dataset,scenario=scenario,a=a,b=b,metric=metric,**r))
    pd.DataFrame(differences).to_csv(out/'paired.csv',index=False)
    pooled=[]
    for a,b in COMPARISONS:
        for metric in ('delivered','energy_wh'):
            r=paired(d,a,b,metric,True)
            if r:pooled.append(dict(a=a,b=b,metric=metric,**r))
    # Holm adjustment only for the three prespecified exploratory pooled C contrasts.
    family=[r for r in pooled if r['metric']=='delivered' and (r['a'],r['b']) in COMPARISONS[2:]]
    ordered=sorted(family,key=lambda x:x['p_sign']);last=0.
    for j,r in enumerate(ordered):
        last=max(last,min(1.,(len(ordered)-j)*r['p_sign']));r['p_holm']=last
    (out/'pooled.json').write_text(json.dumps(pooled,indent=2))
    scalar=[k for k in d.columns if k not in ('config','control','action_counts')]
    d[scalar].to_csv(out/'episodes.csv',index=False)
    print(pd.DataFrame(pooled).round(6).to_string(index=False))


if __name__=='__main__':main()
