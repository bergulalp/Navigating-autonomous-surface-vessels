"""Generate manuscript numbers and tables from completed, paired v4 analyses."""
import json
import shutil
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from analyze_v4_bio import PAIRS,LABELS


def main():
    gen=ROOT/'paper/generated';figs=ROOT/'paper/figures'
    bio=json.loads((ROOT/'results/analysis_v4_bio/contrasts.json').read_text())
    rl=json.loads((ROOT/'results/analysis_v4_learning/paired.json').read_text())
    means=pd.read_csv(ROOT/'results/analysis_v4_bio/means.csv',index_col=0)
    learning=pd.read_csv(ROOT/'results/analysis_v4_learning/means.csv',index_col=0)
    selection=json.loads((ROOT/'results/rl_selection.json').read_text())
    def get(a,b,m='delivered',group='main'):
        return next(r for r in bio[group] if r['a']==a and r['b']==b and r['metric']==m)
    def ci(r,scale=100,digits=2):
        return f'{scale*r["difference"]:+.{digits}f} [{scale*r["ci_low"]:+.{digits}f}, {scale*r["ci_high"]:+.{digits}f}]'
    rp=get(*PAIRS[0]);rg=get(*PAIRS[1]);nl=get(*PAIRS[2])
    lines=[r'\subsection{Results of the new mechanism tests}',
        'The second study adds 732 model-based missions, including the pilot and declared diagnostics. '+
        f'Reciprocal versus periodic handover changes coverage by {ci(rp)} percentage points and total energy by '+
        f'{get(*PAIRS[0],"energy_wh")["difference"]:+.2f} Wh. Relative to greedy handover the coverage contrast is {ci(rg)} points. '+
        f'Nonlinear versus linear inhibition changes coverage by {ci(nl)} points and energy by {get(*PAIRS[2],"energy_wh")["difference"]:+.2f} Wh. '+
        'Brackets are paired 95\\% bootstrap intervals, not predictive bounds for a new vessel platform.',
        r'\begin{table}[t]\centering\small',
        r'\caption{New main-study means. Each policy has 96 missions across two radio profiles and four scenarios.}',
        r'\begin{tabular}{lrrr}\toprule Policy & $\bar C$ & $E$ [Wh] & AoI [s]\\\midrule']
    for policy,row in means.iterrows():
        lines.append(f'{LABELS[policy]} & {row.delivered:.4f} & {row.energy_wh:.1f} & {row.capped_aoi_s:.1f} '+r'\\')
    lines += [r'\bottomrule\end{tabular}\label{tab:v4means}\end{table}',
        r'\begin{figure*}[tbp]\centering',
        r'\includegraphics[width=.97\textwidth]{figures/bio_mechanisms.pdf}',
        r'\caption{Additional biological mechanisms with matched restrictions. Main-study coverage--energy means, paired coverage contrasts, and scenario-specific differences. A reduced movement budget can save energy while losing timely monitoring.}',
        r'\label{fig:v4bio}\end{figure*}',
        r'\begin{table*}[tbp]\centering\small',
        r'\caption{Prespecified new biological contrasts and small diagnostics. Coverage differences are in percentage points, with paired 95\% intervals. Main $p_H$ values apply Holm adjustment to three coverage tests; diagnostics are descriptive.}',
        r'\begin{tabular}{llrrr}\toprule Setting & Contrast & $\Delta C$ [95\% interval] & $\Delta E$ [Wh] & $p_H$\\\midrule']
    pairnames=['Reciprocal -- periodic','Reciprocal -- greedy','Nonlinear -- linear']
    for group,label in [('main','Main, 12 blocks'),('bio_shifted','Shifted, 4 blocks'),('bio_integration128','128 paths, 4 blocks')]:
        for (a,b),name in zip(PAIRS,pairnames):
            r=get(a,b,group=group);e=get(a,b,'energy_wh',group)
            pv=f'{r["p_holm"]:.3f}' if group=='main' else '--'
            lines.append(f'{label} & {name} & {ci(r)} & {e["difference"]:+.2f} & {pv} '+r'\\')
    lines += [r'\bottomrule\end{tabular}\label{tab:v4contrasts}\end{table*}',
        'These adaptations should be judged against their matched controls and against unrestricted economic control. '+
        f'Reciprocal handover versus economic control changes coverage by {ci(get("reciprocal_handover","economic"))} points; '+
        f'nonlinear inhibition versus economic control changes it by {ci(get("nonlinear_inhibition","economic"))} points. '+
        'A small main-study contrast alone does not establish a generally useful biological advantage. '+
        'The changed observation law and integration resolution are particularly relevant because all policies still optimize a sampled, imperfect prospective score.']
    (gen/'v4_results.tex').write_text('\n'.join(lines)+'\n')
    trace=json.loads((ROOT/'results/analysis_v4_bio/trace_diagnostics.json').read_text())
    td=trace['nonlinear_inhibition']
    (gen/'v4_trace.tex').write_text(
        f'A post-hoc inspection of the eight presaved nonlinear-inhibition main missions '
        f'(world seed 17000, two profiles and four scenarios) found that only '
        f'{100*td["fraction_maximum_above_threshold"]:.2f}\\% of {td["nonstale_decisions"]} nonstale decisions '
        f'had any proposal fraction above the 0.3 sigmoid midpoint; mean maximum support was '
        f'{td["mean_maximum_support"]:.3f}. The threshold was borrowed from a primarily binary-choice '
        'biological model, whereas these ten often similar movement proposals fragment support. '
        'This is a descriptive diagnosis, not a causal ablation: lower thresholds or grouping '
        'proposals into meaningful alternatives require a separate development and test experiment.\n')
    a=rl['training_mean'];action=selection['selected_action']
    from usv_research.residual import STEPS,PRICES
    train=[json.loads((ROOT/f'results/ppo_v4_single_seed{s}/training_timing.jsonl').read_text().splitlines()[-1]) for s in range(3)]
    lr=[f'The development set selected constant action {action}: a {STEPS[action//3]:g}-m proposal step and price {PRICES[action%3]:g}. '+
        f'Against that setting, the mean of the three short PPO policies changes mean interval reward by {ci(a["mean_step_reward"],1,5)}, '+
        f'post-warmup coverage by {ci(a["delivered"])} percentage points and energy by {ci(a["energy_wh"],1,2)} Wh. '+
        'These intervals resample eight mission-seed blocks while keeping the three learned policies fixed; they do not quantify performance across the population of possible training runs.',
        r'\begin{table}[t]\centering\small',
        r'\caption{Held-out short-PPO evaluation: 32 missions per policy, after 2048 steps per training seed.}',
        r'\begin{tabular}{lrrr}\toprule Policy & Mean reward & $\bar C$ & $E$ [Wh]\\\midrule']
    for policy,row in learning.iterrows():
        name=policy.replace('_',' ')
        lr.append(f'{name} & {row.mean_step_reward:.4f} & {row.delivered:.4f} & {row.energy_wh:.1f} '+r'\\')
    lr += [r'\bottomrule\end{tabular}\label{tab:v4rl}\end{table}',
        r'\begin{figure*}[tbp]\centering',
        r'\includegraphics[width=.98\textwidth]{figures/ppo_v4.pdf}',
        r'\caption{Each short PPO training seed compared with the development-selected constant setting on the same held-out world seeds. Intervals condition on the trained policies. Favorable energy differences are negative; favorable reward and coverage differences are positive.}',
        r'\label{fig:v4ppo}\end{figure*}',
        f'The three training runs completed at {min(x["steps_per_second"] for x in train):.2f}--{max(x["steps_per_second"] for x in train):.2f} steps/s on a shared CPU. '+
        'A separate 256-step, two-chunk run exercised the progress wrapper and optimizer continuation. '+
        'Neither that software check nor a 2048-step pilot establishes convergence. '+
        'The useful deliverable is a reproducible way to test learning against strong fixed settings, with no guarantee that extra training improves coverage or reward.']
    usage=pd.read_csv(ROOT/'results/analysis_v4_learning/action_usage.csv')
    notes=[]
    for _,u in usage[usage.policy.str.startswith('ppo_')].iterrows():
        if u.distinct_actions==1:
            action=next(i for i in range(9) if u[f'action_{i}']>0)
            notes.append(f'{u.policy.replace("_"," ")} used action {action} at every evaluated decision')
    if notes:
        lr.append('Action logs show that '+ '; '.join(notes)+'. This observed constant behavior provides no evidence of learned state-dependent adaptation, even when its score matches a strong fixed setting.')
    (gen/'v4_learning.tex').write_text('\n'.join(lr)+'\n')
    abstract=(
        'Surface-vehicle monitoring requires sensed reports to arrive on time while formation changes consume time and energy. '
        'We develop a local economic controller with correlated sensing and radio outages, temporal multihop delivery, delayed observations and finite movement. '
        'In 960 held-out one-hour missions, it improves delivered coverage by 0.95 percentage points relative to a separately selected static ring; '
        'relative to an earlier formation-library controller, coverage increases by 1.69 points and mean energy decreases by 62.5\\%. '
        'Small fungal-transport gains fail robustness checks. A further 576-mission main study tests reciprocal movement timing and nonlinear proposal inhibition against matched controls. '
        f'Reciprocal minus periodic handover changes coverage by {ci(rp)} points; nonlinear minus linear inhibition changes it by {ci(nl)} points. '
        'Additional model-shift and integration diagnostics retain negative comparisons. '
        f'Three 2048-step PPO pilots change mean interval reward by {ci(a["mean_step_reward"],1,5)} relative to a development-selected constant setting on unseen missions. '
        'Intervals are paired 95\\% bootstrap intervals conditional on the evaluated controllers. '
        'The contribution is an inspectable testbed and controlled mechanism comparisons, not a first-use claim, a field-validated system or a demonstrated general advantage of biological or learned control.')
    (gen/'v4_abstract.tex').write_text(abstract+'\n')
    shutil.copy2(ROOT/'results/analysis_v4_bio/bio_mechanisms.pdf',figs/'bio_mechanisms.pdf')
    shutil.copy2(ROOT/'results/analysis_v4_learning/comparison.pdf',figs/'ppo_v4.pdf')
    summary=dict(bio=bio,learning=rl,selected_constant=selection,training_timings=train)
    (gen/'v4_numeric_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Generated v4 manuscript figures, tables and abstract from completed analyses.')


if __name__=='__main__':main()
