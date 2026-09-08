"""Generate scientific figures and numerical LaTeX directly from episode records."""
import json
import sys
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from usv_research.analyze import load,interval,paired,COMPARISONS

ROOT=Path(__file__).resolve().parents[1]
NAMES={'fixed_ring':'Outer static','static_tuned':'Tuned static','adaptive':'Earlier adaptive',
       'economic':'Economic','fungal':'Hydraulic fungal','deadline_fusion':'Deadline fusion',
       'greedy_sparse':'Greedy sparse','recruitment':'Recruitment'}
COLORS=dict(zip(NAMES,['#999999','#30343b','#cb4945','#2075bc','#259369','#7048a5','#bc8742','#e07328']))
MARKERS=dict(zip(NAMES,['x','s','v','o','D','P','^','*']))
SCENARIOS={'storm':'Storm','moving_front':'Moving priority','sensor_fault':'Sensor fault'}


def save(fig,name):
    target=ROOT/'paper/figures';target.mkdir(parents=True,exist_ok=True)
    fig.savefig(target/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(target/(name+'.png'),dpi=210,bbox_inches='tight')
    plt.close(fig)


def main():
    plt.rcParams.update({'font.size':8.5,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'semibold','legend.frameon':False,'pdf.fonttype':42})
    d=load([ROOT/'results/main_nominal',ROOT/'results/main_short'])
    if len(d)!=960:raise ValueError(f'Main experiment incomplete: {len(d)}/960')
    out=ROOT/'paper/generated';out.mkdir(parents=True,exist_ok=True)
    pooled=json.loads((ROOT/'results/analysis_main/pooled.json').read_text())
    def pc(a,b,metric='delivered'):
        return next(x for x in pooled if x['a']==a and x['b']==b and x['metric']==metric)
    overall=d.groupby(['policy','seed'])[['delivered','energy_wh','capped_aoi_s','graph_connected','command_stale']].mean()
    mean=overall.groupby('policy').mean()
    table=[]
    for policy in NAMES:
        x=overall.loc[policy]
        lo,hi=interval(x.delivered)
        table.append(NAMES[policy]+f" & {100*x.delivered.mean():.2f} [{100*lo:.2f}, {100*hi:.2f}]"
                     +f" & {x.energy_wh.mean():.1f} & {x.capped_aoi_s.mean():.1f} & {100*x.graph_connected.mean():.1f}"
                     +f" & {100*x.command_stale.mean():.2f}"+r" \\")
    (out/'main_table.tex').write_text('\n'.join(table)+'\n')
    secondary=d.groupby('policy')[['delivered_p10','delivered_below_half_fraction','early_report_rate','runtime_s']].mean()
    secondary_rows=[]
    for policy in NAMES:
        s=secondary.loc[policy]
        secondary_rows.append(NAMES[policy]+f" & {100*s.delivered_p10:.2f} & {100*s.delivered_below_half_fraction:.2f}"
                              +f" & {100*s.early_report_rate:.2f} & {s.runtime_s:.2f}"+r" \\")
    (out/'secondary_table.tex').write_text('\n'.join(secondary_rows)+'\n')

    fig,axs=plt.subplots(3,2,figsize=(7.05,7.55),layout='constrained')
    for row,scenario in enumerate(SCENARIOS):
        for col,profile in enumerate(['nominal','short']):
            ax=axs[row,col];sub=d[(d.dataset=='main_'+profile)&(d.scenario==scenario)]
            for policy in NAMES:
                g=sub[sub.policy==policy];c=100*g.delivered.mean();e=g.energy_wh.mean()
                cl,ch=np.array(interval(g.delivered))*100;el,eh=interval(g.energy_wh)
                ax.errorbar(e,c,xerr=[[e-el],[eh-e]],yerr=[[c-cl],[ch-c]],color=COLORS[policy],
                            marker=MARKERS[policy],markersize=5,capsize=2,lw=.8,label=NAMES[policy])
            ax.set_title(f'{SCENARIOS[scenario]} | '+('nominal radio' if col==0 else 'shorter radio'))
            ax.set_xscale('log');ax.xaxis.set_major_formatter(ScalarFormatter());ax.minorticks_off()
            ax.set_xticks([400,800,1600]);ax.grid(alpha=.15)
            ax.set_ylabel('Delivered coverage (%)');ax.set_xlabel('Fleet energy (Wh, log scale)')
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=8)
    save(fig,'coverage_energy')

    fig,axs=plt.subplots(1,2,figsize=(7.05,3.35),layout='constrained')
    labels=['Economic - tuned static','Economic - earlier adaptive','Hydraulic fungal - economic',
            'Deadline fusion - hydraulic','Recruitment - greedy sparse']
    for j,metric in enumerate(['delivered','energy_wh']):
        ax=axs[j];ax.axvline(0,color='#555',lw=.8)
        for k,(a,b) in enumerate(COMPARISONS):
            r=pc(a,b,metric);scale=100 if j==0 else 1
            x=r['difference']*scale;lo=r['ci_low']*scale;hi=r['ci_high']*scale
            ax.errorbar(x,k,xerr=[[x-lo],[hi-x]],fmt='o',color=COLORS.get(a,'k'),capsize=3)
        ax.set_yticks(range(5),labels if j==0 else ['']*5);ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
        ax.set_xlabel('Coverage change (percentage points)' if j==0 else 'Energy change (Wh)')
    save(fig,'paired_effects')

    fig,axs=plt.subplots(1,2,figsize=(7.05,2.7),layout='constrained')
    for ax,profile in zip(axs,['nominal','short']):
        j=json.loads((ROOT/f'results/static_selection_{profile}/selection.json').read_text())
        r=pd.DataFrame(j['candidates'])
        ax.plot(r.radius,100*r.coverage,'o-',label='Delivered coverage')
        ax.plot(r.radius,100*r.utility,'s--',label='Selection utility × 100')
        chosen=j['selected']['radius'];ax.axvline(chosen,color='#333',ls=':',label=f'Selected {chosen:.2f}R')
        ax.set_title('Nominal radio' if profile=='nominal' else 'Shorter radio')
        ax.set_xlabel('Static ring radius / mission radius');ax.set_ylabel('Development score');ax.grid(alpha=.15)
        ax.legend(fontsize=7)
    save(fig,'static_selection')

    fig,axs=plt.subplots(2,2,figsize=(7.05,5.4),layout='constrained')
    trace_data={}
    sub=d[(d.dataset=='main_short')&(d.scenario=='moving_front')&(d.seed==7000)]
    for policy in ['static_tuned','economic','deadline_fusion']:
        r=sub[sub.policy==policy].iloc[0]
        z=np.load(ROOT/'results/main_short'/(r.run_id+'.npz'))
        trace_data[policy]=z
        time=z['t']/60
        cover=pd.Series(z['delivered']*100).rolling(5,min_periods=1).mean()
        axs[0,0].plot(time,cover,color=COLORS[policy],label=NAMES[policy])
        axs[0,1].plot(time,z['energy_wh'],color=COLORS[policy])
    axs[0,0].set(ylabel='Delivered coverage (%)',xlabel='Time (min)',title='One held-out mission: short radio')
    axs[0,0].legend(fontsize=7);axs[0,1].set(ylabel='Fleet energy (Wh)',xlabel='Time (min)',title='Movement has a cost')
    z=trace_data['deadline_fusion']
    for i in range(6):
        vessel_color=plt.get_cmap('tab10')(i)
        axs[1,0].plot(z['t']/60,np.linalg.norm(z['positions'][:,i],axis=1),lw=1,color=vessel_color)
        axs[1,1].plot(z['positions'][:,i,0]/1000,z['positions'][:,i,1]/1000,lw=1,color=vessel_color)
        axs[1,1].plot(z['positions'][-1,i,0]/1000,z['positions'][-1,i,1]/1000,'o',ms=4,color=vessel_color)
    axs[1,0].set(xlabel='Time (min)',ylabel='Vessel radius (m)',title='Deadline-fusion formation changes')
    aa=np.linspace(0,2*np.pi,200);axs[1,1].plot(2.5*np.cos(aa),2.5*np.sin(aa),'--',color='#aaa',lw=.8)
    axs[1,1].plot(0,0,'ks',ms=4);axs[1,1].set(xlabel='East (km)',ylabel='North (km)',title='Paths and final positions',aspect='equal')
    for ax in axs.ravel():ax.grid(alpha=.15)
    save(fig,'representative_mission')

    fig,ax=plt.subplots(figsize=(3.45,2.5),layout='constrained')
    times=np.arange(1,13);q=.7
    success=[sum(math.comb(t,k)*q**k*(1-q)**(t-k) for k in range(3,t+1)) for t in times]
    influence=[t*math.comb(t-1,2)*q**2*(1-q)**(t-3)/3 if t>=3 else 0 for t in times]
    ax.plot(times,success,'o-',label='Delivery probability',color='#2075bc')
    ax.plot(times,influence,'s-',label='Per-link deadline influence',color='#7048a5')
    ax.plot(times,[1/q**2]*len(times),'--',color='#259369',label='Hydraulic importance magnitude')
    ax.set(xlabel='Deadline (communication rounds)',ylabel='Probability / sensitivity')
    ax.legend(fontsize=6.5);ax.grid(alpha=.15)
    save(fig,'deadline_example')

    e=pc('economic','static_tuned');a=pc('economic','adaptive')
    energy_change=mean.loc['economic','energy_wh']/mean.loc['adaptive','energy_wh']-1
    bio=pc('deadline_fusion','fungal')
    abstract=(f"Maintaining useful sensing with a surface-vehicle team requires reports to arrive before a deadline while formation changes consume time and energy. "
       f"We develop a local economic controller and evaluate fungal branching/fusion, deadline-dependent link reinforcement, and ant-inspired recruitment as movement-proposal mechanisms. "
       f"A conditional Bernoulli sensitivity identity maps the effect of a radio link on timely delivery into a formation direction; it is an application of established reliability mathematics. "
       f"The simulator includes shared sensor visibility, correlated radio outages, one-hop-per-round caching, delayed commands and finite vessel response. "
       f"Across 960 held-out one-hour missions in six scenario/hardware cells, economic control changes delivered coverage by {100*e['difference']:+.2f} percentage points "
       f"(95\\% interval [{100*e['ci_low']:.2f}, {100*e['ci_high']:.2f}]) relative to a static ring selected using separate development data. "
       f"Relative to the earlier formation-library controller, coverage changes by {100*a['difference']:+.2f} points and mean fleet energy by {100*energy_change:+.1f}\\%. "
       f"Deadline fusion changes coverage by {100*bio['difference']:+.2f} points relative to hydraulic fusion "
       f"([{100*bio['ci_low']:.2f}, {100*bio['ci_high']:.2f}]). "
       f"The small biological gains are not robust across the diagnostic model and mission settings. "
       f"The work supplies a reproducible research testbed; synthetic outcomes do not establish field performance or first use of the biological ideas in robotics.\n")
    (out/'abstract.tex').write_text(abstract)

    results=r"""\section{Results}
\subsection{Main held-out comparison}
Table~\ref{tab:main} summarizes the equal-weight six-cell estimand. Intervals
resample 20 seed blocks, preserving each seed's six conditions.
Figure~\ref{fig:pareto} shows each scenario separately, including energy.
\begin{table*}[tbp]
\centering\small\setlength{\tabcolsep}{4pt}
\caption{Pooled main results: 20 independent seed blocks, 120 missions per policy.
Coverage intervals are 95\% bootstrap intervals. Other columns are means.
Connectivity is instantaneous all-node graph connectivity, not a guarantee.}
\label{tab:main}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Policy & \shortstack{Delivered coverage\\(\%, 95\% CI)} & \shortstack{Fleet energy\\(Wh)} & \shortstack{Capped AoI\\(s)} & \shortstack{Connected\\(\%)} & \shortstack{Stale commands\\(\%)}\\
\midrule
\inputtablerows{generated/main_table.tex}
\bottomrule
\end{tabular}
\end{table*}
\begin{figure*}[tbp]
\centering
\includegraphics[width=.96\textwidth]{figures/coverage_energy.pdf}
\caption{Coverage--energy outcomes in the six held-out conditions. Each point
summarizes 20 paired missions; bars show 95\% intervals of the mean. Energy uses
a logarithmic axis. Higher coverage and lower energy are preferable.}
\label{fig:pareto}
\end{figure*}
"""
    for left,right in COMPARISONS[:2]:
        r=pc(left,right);er=pc(left,right,'energy_wh')
        results+=f"\nCompared with {NAMES[right].lower()}, economic control changes coverage by {100*r['difference']:+.3f} percentage points " \
           +f"(95\\% CI [{100*r['ci_low']:.3f}, {100*r['ci_high']:.3f}]) and energy by {er['difference']:+.1f} Wh " \
           +f"([{er['ci_low']:.1f}, {er['ci_high']:.1f}]).\n"
    results+=r"""
\begin{figure*}[tbp]
\centering
\includegraphics[width=.96\textwidth]{figures/paired_effects.pdf}
\caption{Paired pooled changes. Positive coverage differences are beneficial;
positive energy differences represent extra cost. Intervals resample whole seed
blocks. Biological comparisons are exploratory.}
\label{fig:paired}
\end{figure*}
\subsection{Biological comparisons}
"""
    for left,right in COMPARISONS[2:]:
        r=pc(left,right);er=pc(left,right,'energy_wh');p=r.get('p_holm',r['p_sign'])
        ptext=r'$p_{\mathrm{Holm}}<0.001$' if p<.001 else f'$p_{{\\mathrm{{Holm}}}}={p:.3f}$'
        results+=f"\n{NAMES[left]} minus {NAMES[right].lower()} yields {100*r['difference']:+.3f} coverage points "\
                 +f"([{100*r['ci_low']:.3f}, {100*r['ci_high']:.3f}]; {ptext}) "\
                 +f"and {er['difference']:+.1f} Wh ([{er['ci_low']:.1f}, {er['ci_high']:.1f}]).\n"
    results+=r"""
These comparisons concern the complete proposal mechanisms under a common
acceptance objective. A pooled interval does not imply that every scenario
benefits, and equal energy prices do not impose exactly equal realized energy.
The price-sensitivity results provide additional trade-off context.
The effects are heterogeneous. Economic control versus tuned static gains
3.23 coverage points in nominal-radio moving priority and 1.94 points with
shorter radio, but changes coverage by -0.15 points in the shorter-radio
sensor-fault case. Deadline fusion versus hydraulic fusion gains 0.17 and
0.19 points in the two storm cells, while changing coverage by -0.03 and
-0.14 points in the two moving-priority cells. These descriptive cell patterns
suggest conditions for future hypotheses; they are not independently confirmed
subgroup discoveries or a reason to discard unfavorable cells.
\begin{figure*}[tbp]
\centering
\includegraphics[width=.94\textwidth]{figures/static_selection.pdf}
\caption{Static-design development data, excluded from held-out estimates.
The selected radius differs by radio profile. Coverage and the penalized
selection utility are plotted on a common score scale.}
\label{fig:static}
\end{figure*}
\subsection{Physical trajectories and secondary outcomes}
"""
    collisions=int(d.collision_steps.sum());sep=float(d.min_separation_m.min())
    results+=f"\nThe main runs recorded {collisions} collision steps below the simulator's 30 m threshold; "\
             +f"the minimum observed pair separation was {sep:.1f} m. This is a measured outcome in these runs, "\
             +"not a formal safety guarantee. Receiver AoI and command staleness are included in Table~\\ref{tab:main}; "\
             +"per-scenario secondary outcomes and reference-traffic metrics are retained in the raw data.\n"
    results+=r"""
\begin{figure*}[tbp]
\centering
\includegraphics[width=.94\textwidth]{figures/representative_mission.pdf}
\caption{Illustrative held-out seed 7000, moving priority, shorter radio.
The coverage display uses a trailing five-sample average of one-minute trace
samples. Vessel colors agree between the two lower panels; dots mark final
positions. This is one trajectory, not the evidence used to select a winning
policy or a substitute for the paired statistical comparisons.}
\label{fig:trajectory}
\end{figure*}
\input{generated/supplement_results.tex}
"""
    (out/'results.tex').write_text(results)
    (out/'numeric_summary.json').write_text(json.dumps(dict(pooled=pooled,means=mean.reset_index().to_dict('records'),
                                                          episodes=len(d),collisions=collisions,min_separation_m=sep),indent=2))
    print('Main scientific figures and manuscript values generated from 960 episodes.')


if __name__=='__main__':main()
