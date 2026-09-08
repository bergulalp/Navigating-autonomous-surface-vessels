"""New mechanisms: paired main results and explicitly separate diagnostics."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from usv_research.analyze import load,paired
from usv_bio.control import POLICIES

PAIRS=[('reciprocal_handover','periodic_handover'),('reciprocal_handover','greedy_handover'),
       ('nonlinear_inhibition','linear_inhibition')]
LABELS={'economic':'Economic','periodic_handover':'Periodic handover','greedy_handover':'Greedy handover',
        'reciprocal_handover':'Reciprocal handover','linear_inhibition':'Linear inhibition',
        'nonlinear_inhibition':'Nonlinear inhibition'}


def analyze(d):
    rows=[]
    for a,b in PAIRS+[(p,'economic') for p in POLICIES if p!='economic']:
        for metric in ['delivered','energy_wh','postwarmup_utility','capped_aoi_s','graph_connected']:
            r=paired(d,a,b,metric,True)
            rows.append(dict(a=a,b=b,metric=metric,**r))
    family=[r for r in rows if r['metric']=='delivered' and (r['a'],r['b']) in PAIRS]
    last=0
    for j,r in enumerate(sorted(family,key=lambda x:x['p_sign'])):
        last=max(last,min(1,(len(family)-j)*r['p_sign']));r['p_holm']=last
    return rows


def main():
    out=ROOT/'results/analysis_v4_bio';out.mkdir(exist_ok=True)
    main=load([ROOT/'results/bio_nominal',ROOT/'results/bio_short'])
    if len(main)!=576:raise ValueError(f'Expected 576 main missions, got {len(main)}')
    metrics=['delivered','energy_wh','postwarmup_utility','capped_aoi_s','graph_connected',
             'runtime_s','mean_recruited_vessels','accepted_moves','delivered_p10']
    means=main.groupby('policy')[metrics].mean().reindex(POLICIES)
    means.to_csv(out/'means.csv');main.groupby(['dataset','scenario','policy'])[metrics].mean().to_csv(out/'cells.csv')
    all_results={'main':analyze(main)}
    for name,count in [('bio_shifted',96),('bio_integration128',48)]:
        d=load([ROOT/'results'/name])
        if len(d)!=count:raise ValueError(f'{name}: incomplete, {len(d)}/{count}')
        all_results[name]=analyze(d)
    (out/'contrasts.json').write_text(json.dumps(all_results,indent=2)+'\n')
    fig=plt.figure(figsize=(11.4,8.0),layout='constrained');gs=fig.add_gridspec(2,2)
    ax=fig.add_subplot(gs[0,0])
    colors=['#242424','#DC8C32','#53738D','#126B57','#9881C0','#4E307D']
    for i,policy in enumerate(POLICIES):
        row=means.loc[policy];ax.scatter(row.energy_wh,row.delivered*100,color=colors[i],s=52,label=LABELS[policy])
    ax.set_xlabel('Mean total energy [Wh]');ax.set_ylabel('Fresh-report coverage [%]')
    ax.set_title('(a) Main study: 576 missions');ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower right')
    ax=fig.add_subplot(gs[0,1])
    for i,(a,b) in enumerate(PAIRS):
        r=next(x for x in all_results['main'] if x['a']==a and x['b']==b and x['metric']=='delivered')
        x=100*r['difference'];lo=100*r['ci_low'];hi=100*r['ci_high']
        ax.errorbar(x,i,xerr=[[x-lo],[hi-x]],fmt='o',capsize=4,color='#126B57')
    ax.set_yticks(range(3),['Reciprocal − periodic','Reciprocal − greedy','Nonlinear − linear'],fontsize=9)
    ax.axvline(0,color='.5',lw=1);ax.set_xlabel('Coverage difference [percentage points]')
    ax.set_title('(b) Paired 95% intervals, 12 seed blocks');ax.grid(axis='x',alpha=.2)
    ax=fig.add_subplot(gs[1,:]);cols=[];values=[]
    for (profile,scenario),g in main.groupby(['dataset','scenario']):
        cols.append(profile.replace('bio_','')+'\n'+scenario.replace('_',' '))
        values.append([100*paired(g,a,b,'delivered',True)['difference'] for a,b in PAIRS])
    values=np.array(values).T;lim=max(abs(values).max(),.01)
    im=ax.imshow(values,cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto')
    ax.set_xticks(range(len(cols)),cols,fontsize=8)
    ax.set_yticks(range(3),['Reciprocal − periodic','Reciprocal − greedy','Nonlinear − linear'],fontsize=9)
    for i in range(3):
        for j in range(len(cols)):ax.text(j,i,f'{values[i,j]:+.2f}',ha='center',va='center',fontsize=9,color='white' if abs(values[i,j])>.6*lim else 'black')
    fig.colorbar(im,ax=ax,label='Coverage difference [pp]',shrink=.85)
    ax.set_title('(c) Scenario differences: means, not separate significance claims')
    fig.savefig(out/'bio_mechanisms.png',dpi=180);fig.savefig(out/'bio_mechanisms.pdf');plt.close(fig)
    print(means.round(5).to_string())
    print(pd.DataFrame(all_results['main']).query("metric == 'delivered'").to_string(index=False))


if __name__=='__main__':main()
