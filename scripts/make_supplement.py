"""Build diagnostic tables/figures from all frozen sensitivity and learning runs."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from usv_research.analyze import load, paired, contrast, interval
from make_paper import save, COLORS, NAMES

EXPECTED = {'ablation_short':140, 'price_low':60, 'price_high':60,
            'shifted_short':96, 'slow_short':32, 'integration_128':24,
            'negative_nominal':48, 'negative_short':48}


def main():
    plt.rcParams.update({'font.size':8.5, 'axes.spines.top':False,
                        'axes.spines.right':False, 'legend.frameon':False, 'pdf.fonttype':42})
    data = {}
    for name, expected in EXPECTED.items():
        data[name] = load([ROOT/'results'/name])
        if len(data[name]) != expected:
            raise ValueError(f'{name}: {len(data[name])}/{expected}; finish the frozen suite first')
    out = ROOT/'paper/generated'
    analysis = ROOT/'results/analysis_supplement'
    analysis.mkdir(exist_ok=True)
    rows = []
    def compare(d, label, a, b):
        r = {'condition':label, 'a':a, 'b':b}
        for metric in ('delivered','energy_wh'):
            v = paired(d, a, b, metric, pooled=True)
            for key,value in v.items(): r[metric+'_'+key] = value
        rows.append(r)
        return r
    mechanisms = []
    for a,b,label in [('economic','economic_no_cost','Energy penalty'),
                      ('fungal','fungal_no_fusion','Hydraulic fusion family'),
                      ('recruitment','recruitment_no_social','Social recruitment')]:
        mechanisms.append(compare(data['ablation_short'],label,a,b))
    sensitivity = []
    for name,label in [('shifted_short','Changed sensing/correlation'),
                       ('slow_short','Slower priority'),
                       ('integration_128','128 planning paths')]:
        for a,b in [('economic','static_tuned'),('fungal','economic'),('deadline_fusion','fungal')]:
            if b not in set(data[name].policy): continue
            sensitivity.append(compare(data[name],label,a,b))
    negatives = []
    neg = pd.concat([data['negative_nominal'],data['negative_short']])
    for scenario,label in [('calm','Calm'),('blackout','Temporary blackout')]:
        sub = neg[neg.scenario==scenario]
        for a,b in [('economic','static_tuned'),('fungal','economic'),('deadline_fusion','fungal')]:
            negatives.append(compare(sub,label,a,b))
    pd.DataFrame(rows).to_csv(analysis/'paired.csv',index=False)

    price_rows=[]
    for price,name in [(.01,'price_low'),(.025,'ablation_short'),(.075,'price_high')]:
        for policy in ['economic','fungal','deadline_fusion']:
            g=data[name][data[name].policy==policy].groupby('seed')[['delivered','energy_wh']].mean()
            row=dict(price=price,policy=policy,n=len(g))
            for metric in ('delivered','energy_wh'):
                lo,hi=interval(g[metric]);row.update({metric:g[metric].mean(),metric+'_lo':lo,metric+'_hi':hi})
            price_rows.append(row)
    prices=pd.DataFrame(price_rows);prices.to_csv(analysis/'prices.csv',index=False)
    fig,axs=plt.subplots(2,2,figsize=(7.05,5.5),layout='constrained')
    for j,metric in enumerate(['delivered','energy_wh']):
        ax=axs[0,j];ax.axvline(0,color='#555',lw=.8)
        for k,r in enumerate(mechanisms):
            scale=100 if j==0 else 1
            x=r[metric+'_difference']*scale;lo=r[metric+'_ci_low']*scale;hi=r[metric+'_ci_high']*scale
            ax.errorbar(x,k,xerr=[[x-lo],[hi-x]],fmt='o',color=COLORS[r['a']],capsize=3)
        ax.set_yticks(range(3),[r['condition'] for r in mechanisms] if j==0 else ['']*3)
        ax.invert_yaxis();ax.set_xlabel('Coverage change (points)' if j==0 else 'Energy change (Wh)')
        ax.set_title('Mechanism present minus removed')
        ax.grid(axis='x',alpha=.15)
    for j,metric in enumerate(['delivered','energy_wh']):
        ax=axs[1,j]
        for policy in ['economic','fungal','deadline_fusion']:
            g=prices[prices.policy==policy];scale=100 if j==0 else 1
            ax.errorbar(g.price,g[metric]*scale,
                        yerr=[(g[metric]-g[metric+'_lo'])*scale,(g[metric+'_hi']-g[metric])*scale],
                        fmt='o-',color=COLORS[policy],label=NAMES[policy],capsize=2)
        ax.set_xscale('log');ax.minorticks_off();ax.set_xticks([.01,.025,.075],['0.010','0.025','0.075'])
        ax.set_xlabel('Movement energy price');ax.set_ylabel('Delivered coverage (%)' if j==0 else 'Fleet energy (Wh)')
        ax.grid(alpha=.15)
    axs[1,0].legend(fontsize=7)
    save(fig,'ablation_price')

    # The evaluated checkpoint was saved at episode 50, before the separate resume test.
    learned=load([ROOT/'results/native_eval']);neutral=load([ROOT/'results/native_neutral'])
    if len(learned)!=18 or len(neutral)!=18:raise ValueError('Native evaluation incomplete')
    learned['dataset']='native_comparison';neutral['dataset']='native_comparison'
    learning=pd.concat([learned,neutral]);lr={}
    for metric in ['delivered','energy_wh']:
        lr[metric]=paired(learning,'native_residual','constant_4',metric,pooled=True)
    learning['utility']=learning.delivered-.025*learning.energy_wh/600
    lr['utility']=paired(learning,'native_residual','constant_4','utility',pooled=True)
    history=pd.read_json(ROOT/'results/native_residual_seed0/episodes.jsonl',lines=True)
    fig,axs=plt.subplots(1,2,figsize=(7.05,2.9),layout='constrained')
    axs[0].plot(history.episode,history.discounted_return,color='#2075bc',alpha=.3,lw=.8,label='Episode return')
    axs[0].plot(history.episode,history.discounted_return.rolling(10,min_periods=1).mean(),color='#2075bc',label='Trailing mean (10)')
    axs[0].axvline(50,color='#555',ls=':',label='Evaluated checkpoint')
    axs[0].set(xlabel='Training episode',ylabel='Discounted training return',title='Randomized training conditions')
    axs[0].legend(fontsize=6.5)
    for policy,color,label in [('constant_4','#30343b','Neutral economic'),('native_residual','#ce7437','Learned residual (50 episodes)')]:
        g=learning[learning.policy==policy].groupby('seed')[['delivered','energy_wh']].mean()
        x=g.energy_wh.mean();y=100*g.delivered.mean();xl,xh=interval(g.energy_wh);yl,yh=np.array(interval(g.delivered))*100
        axs[1].errorbar(x,y,xerr=[[x-xl],[xh-x]],yerr=[[y-yl],[yh-y]],fmt='o',color=color,label=label,capsize=3)
    axs[1].set(xlabel='Fleet energy (Wh)',ylabel='Delivered coverage (%)',title='18 held-out missions per policy')
    axs[1].legend(fontsize=6.5)
    for ax in axs:ax.grid(alpha=.15)
    save(fig,'learning')
    learning.to_csv(analysis/'learning_episodes.csv',index=False)
    (analysis/'learning_paired.json').write_text(json.dumps(lr,indent=2))

    def numbers(r,metric='delivered'):
        s=100 if metric=='delivered' else 1
        precision=3 if metric=='delivered' else 1
        return f"{s*r[metric+'_difference']:+.{precision}f} [{s*r[metric+'_ci_low']:.{precision}f}, {s*r[metric+'_ci_high']:.{precision}f}]"
    table=[]
    for r in sensitivity+negatives:
        names={'economic':'E','static_tuned':'S','fungal':'H','deadline_fusion':'D'}
        table.append(r['condition']+' & '+names[r['a']]+' -- '+names[r['b']]+' & '+str(r['delivered_n'])+' & '+numbers(r)+' & '+numbers(r,'energy_wh')+r' \\')
    (out/'supplement_table.tex').write_text('\n'.join(table)+'\n')
    txt=r'''\subsection{Mechanism, price and model checks}
The frozen diagnostic suite adds 508 missions. These are exploratory checks;
their intervals are pointwise and are not adjusted across all diagnostic tests.
The mechanism checks pair ten seed blocks across storm and moving-priority
conditions with the shorter radio. Figure~\ref{fig:ablation} reports both
coverage and realized energy.
'''
    for r in mechanisms:
        txt+=r['condition']+': mechanism present minus removed changes coverage by '+numbers(r)+' points and energy by '+numbers(r,'energy_wh')+' Wh.\n\n'
    txt+=r'''The no-fusion variant replaces mixed directions with branching-only step
lengths; this changes the proposal family as well as removing the hydraulic
term. It is not a perfectly isolated biochemical-mechanism intervention.
Price checks use the same ten seed blocks at three common energy prices.
These curves expose cost--coverage trade-offs, but do not constitute an
exactly energy-matched or globally optimized Pareto frontier.
\begin{figure*}[tbp]
\centering\includegraphics[width=.96\textwidth]{figures/ablation_price.pdf}
\caption{Exploratory mechanism and movement-price checks. Differences use
paired seed blocks; all bars are 95\% bootstrap intervals. Each price point
averages the same ten seeds across two scenarios.}
\label{fig:ablation}
\end{figure*}
\begin{table*}[tbp]
\centering\small\setlength{\tabcolsep}{4pt}
\caption{Diagnostic contrasts with 95\% paired intervals. E: economic; S: tuned
static; H: hydraulic fusion; D: deadline fusion. $n$ counts seed blocks.
Positive energy changes are costs. These are exploratory, pointwise intervals.}
\label{tab:sensitivity}
\begin{tabular}{@{}llrrr@{}}\toprule
Condition & Contrast & $n$ & \shortstack{Coverage change\\(points)} & \shortstack{Energy change\\(Wh)}\\\midrule
\inputtablerows{generated/supplement_table.tex}
\bottomrule\end{tabular}
\end{table*}
Table~\ref{tab:sensitivity} reports altered sensing/correlation, a slower
moving priority, 128-path planning, and calm/blackout controls. The altered-model
case jointly changes the actual detector to a probit curve and increases wave
memory/correlation; it cannot isolate those individual causes. Slower priority
travels at approximately 1.09 m/s, below the 3 m/s vessel limit. The 128-path
check uses only four independent seed blocks and compares policies within that
setting; it is not a paired numerical-convergence test against the 32-path main
study. Negative controls pool the two hardware profiles by seed. A complete
radio blackout prevents delivery irrespective of any formation policy.
The small default biological advantages are not robust across these checks.
Under changed sensing/correlation, both biological coverage intervals include
zero. With slower priority, deadline fusion loses 0.665 coverage points against
hydraulic fusion (interval [-0.934, -0.398]). In the four-seed 128-path check,
hydraulic fusion also underperforms economic control; this small experiment
does not identify numerical integration as the sole cause. Economic control
still improves coverage relative to tuned static in the changed-model and
slower-priority checks, while spending additional energy. These outcomes favor
economic control as the practical baseline and limit claims about biological
performance benefits.

\subsection{Learning pipeline and negative learning result}
A dependency-free linear-softmax REINFORCE policy chooses among nine combinations
of movement step and energy price around the economic controller. Four received
observation frames provide 332 input features for six vessels. The policy sees
cached states, belief summaries and ages, not private true sensor health.
Training uses domain-randomized 40-minute missions. One training seed was run for
50 episodes, evaluated, then resumed for five additional episodes to check
checkpoint continuity. The 50-episode checkpoint is retained separately; the
55-episode checkpoint is not the policy evaluated here.
'''
    c=lr['delivered'];e=lr['energy_wh'];u=lr['utility']
    txt+=(f"The 50-episode policy was tested on 18 missions, three scenarios and six paired seed blocks, against constant action 4, "
          f"which reproduces the neutral economic controller. Coverage changed by {100*c['difference']:+.3f} points "
          f"(95\\% interval [{100*c['ci_low']:.3f}, {100*c['ci_high']:.3f}]) and energy by {e['difference']:+.1f} Wh "
          f"([{e['ci_low']:.1f}, {e['ci_high']:.1f}]). The evaluation utility "
          r"$\bar C-0.025E/(600\,\mathrm{Wh})$ "
          f"changed by {u['difference']:+.5f} ([{u['ci_low']:.5f}, {u['ci_high']:.5f}]). "
          "There is no evidence here that learning improves the mission objective. The energy saving comes with a coverage loss. "
          "These small-sample intervals and one training seed do not characterize robust learning performance.\n")
    txt+=r'''\begin{figure*}[tbp]
\centering\includegraphics[width=.96\textwidth]{figures/learning.pdf}
\caption{Exploratory learning study. Left: randomized training returns, including
five resume-test episodes after the evaluated checkpoint. A training curve is
not evidence of improved held-out performance. Right: evaluation of the saved
50-episode policy against its neutral economic action; intervals resample six
seed blocks.}
\label{fig:learning}
\end{figure*}
The optional Gymnasium/Stable-Baselines3 PPO environment also passed the
environment checker. PPO was trained for 512 steps, saved, resumed for 128 more,
and loaded for two complete evaluation missions. This 640-step run verifies
the software path only; it is not a substantive PPO efficacy experiment.
The first 512 steps took 129.0 s on the provided CPU environment under concurrent
load (3.97 steps/s). Native training took 505.3 s for the first 50 episodes.
Local hardware should be timed before budgeting longer training.
'''
    (out/'supplement_results.tex').write_text(txt)
    summary={'diagnostic_episodes':sum(EXPECTED.values()),'contrasts':rows,'learning':lr,
             'minimum_separation_m':min(float(d.min_separation_m.min()) for d in data.values()),
             'collision_steps':sum(int(d.collision_steps.sum()) for d in data.values())}
    (out/'supplement_numeric.json').write_text(json.dumps(summary,indent=2))
    print(pd.DataFrame(rows)[['condition','a','b','delivered_difference','energy_wh_difference']].to_string(index=False))
    print('Generated diagnostics from 508 simulation missions and 36 learning-evaluation missions.')


if __name__=='__main__':main()
