"""Inspect saved-policy performance against a declared conventional comparator."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from usv_research.analyze import load,paired

METRICS=['mean_step_reward','delivered','energy_wh','discounted_return','postwarmup_utility']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',nargs='+',required=True);p.add_argument('--out',required=True)
    p.add_argument('--selection');p.add_argument('--baseline',default='constant_4');a=p.parse_args()
    baseline=a.baseline
    if a.selection:baseline='constant_'+str(json.loads(Path(a.selection).read_text())['selected_action'])
    d=load(a.data);d['dataset']='paired_rl'
    if d.duplicated(['policy','scenario','seed']).any():raise ValueError('Duplicate missions')
    if baseline not in set(d.policy):raise ValueError('Baseline missing')
    expected=set(zip(d[d.policy==baseline].scenario,d[d.policy==baseline].seed))
    for name,g in d.groupby('policy'):
        if set(zip(g.scenario,g.seed))!=expected:raise ValueError('Unmatched scenario/seed sets: '+name)
    for _,g in d.groupby(['scenario','seed']):
        if g.exogenous_sha256.nunique()!=1 or len({json.dumps(x,sort_keys=True) for x in g.config})!=1:
            raise ValueError('Physical pairing mismatch')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    means=d.groupby('policy')[METRICS+['capped_aoi_s','graph_connected']].mean()
    means.to_csv(out/'means.csv')
    d.groupby(['scenario','policy'])[METRICS].mean().to_csv(out/'scenario_means.csv')
    usage=[]
    for name,g in d.groupby('policy'):
        counts=np.array(g.residual_actions.to_list()).sum(axis=0)
        usage.append(dict(policy=name,distinct_actions=int((counts>0).sum()),
            **{f'action_{i}':int(v) for i,v in enumerate(counts)}))
    pd.DataFrame(usage).to_csv(out/'action_usage.csv',index=False)
    selected=[x for x in means.index if x.startswith('ppo_')]
    if not selected:selected=[x for x in means.index if x!=baseline]
    comparisons=[]
    for policy in selected:
        for metric in METRICS:
            r=paired(d,policy,baseline,metric,True)
            comparisons.append(dict(policy=policy,baseline=baseline,metric=metric,**r))
    aggregate=None
    if len(selected)>1 and all(s.startswith('ppo_') for s in selected):
        avg=d[d.policy.isin(selected)].groupby(['scenario','seed'])[METRICS].mean().reset_index()
        avg['policy']='ppo_training_mean';avg['dataset']='paired_rl'
        ag=pd.concat([avg,d[d.policy==baseline]],ignore_index=True)
        aggregate={m:paired(ag,'ppo_training_mean',baseline,m,True) for m in METRICS}
    report=dict(baseline=baseline,conditional_intervals='20,000 bootstrap resamples of complete mission-seed blocks; trained policies fixed',
        training_seeds=len([x for x in selected if x.startswith('ppo_')]),comparisons=comparisons,training_mean=aggregate)
    (out/'paired.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axs=plt.subplots(1,3,figsize=(11.4,3.7),layout='constrained')
    for ax,metric,scale,label in zip(axs,['mean_step_reward','delivered','energy_wh'],[1,100,1],
        ['Mean interval reward difference','Fresh-report coverage difference [pp]','Total energy difference [Wh]']):
        for i,policy in enumerate(selected):
            r=next(x for x in comparisons if x['policy']==policy and x['metric']==metric)
            x=r['difference']*scale;lo=r['ci_low']*scale;hi=r['ci_high']*scale
            ax.errorbar(x,i,xerr=[[max(0,x-lo)],[max(0,hi-x)]],fmt='o',capsize=3,color='#236B8E')
        ax.axvline(0,color='0.45',lw=1);ax.set_yticks(range(len(selected)),selected)
        ax.set_xlabel(label);ax.grid(axis='x',alpha=.2)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    fig.suptitle('Saved policies versus '+baseline+' — conditional 95% intervals')
    fig.savefig(out/'comparison.png',dpi=190);fig.savefig(out/'comparison.pdf');plt.close(fig)
    lines=['# 5.9.1. MARIN — read these learning results','',f'Comparator: **{baseline}**.',
        'Positive interval-reward and coverage differences favor the learned policy. Negative energy differences save energy.',
        'Intervals resample whole world-seed blocks and condition on these specific trained policies. They do not establish a population-level PPO advantage.','']
    for policy in selected:
        r=next(x for x in comparisons if x['policy']==policy and x['metric']=='mean_step_reward')
        finding=('identical on these missions' if r['difference']==r['ci_low']==r['ci_high']==0 else
                 ('positive on these missions' if r['ci_low']>0 else ('negative on these missions' if r['ci_high']<0 else 'inconclusive')))
        lines.append(f'- {policy}: reward difference {r["difference"]:+.5f}, 95% interval [{r["ci_low"]:+.5f}, {r["ci_high"]:+.5f}]: {finding}.')
        u=next(x for x in usage if x['policy']==policy)
        if u['distinct_actions']==1:
            action=next(i for i in range(9) if u[f'action_{i}']>0)
            lines.append(f'  It selected action {action} at every evaluated decision. This is constant behavior on this test set, not evidence of state-dependent adaptation.')
    lines+=['','Check `scenario_means.csv` and the coverage/energy panels before deciding that a reward gain is useful.',
        'Development results may guide tuning. After inspecting a final test set, reserve new world seeds for the next revision.']
    (out/'READ_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(means.round(5).to_string());print('\n'.join(lines))


if __name__=='__main__':main()
