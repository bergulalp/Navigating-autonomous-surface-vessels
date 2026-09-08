"""Plot arbitrary new experiment folders; no frozen-study episode count required."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from usv_research.analyze import load, interval
from make_paper import NAMES,COLORS

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directories',nargs='+');p.add_argument('--out',required=True)
    a=p.parse_args();d=load(a.directories);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    for dataset,dd in d.groupby('dataset'):
        scenarios=sorted(dd.scenario.unique());policies=sorted(dd.policy.unique(),key=lambda x:list(NAMES).index(x) if x in NAMES else 99)
        fig,axs=plt.subplots(len(scenarios),2,figsize=(9,max(2.7,.36*len(policies)+.9)*len(scenarios)),squeeze=False,layout='constrained')
        for row,scenario in enumerate(scenarios):
            sub=dd[dd.scenario==scenario]
            for col,metric in enumerate(['delivered','energy_wh']):
                ax=axs[row,col]
                for k,policy in enumerate(policies):
                    g=sub[sub.policy==policy]
                    if g.empty:continue
                    scale=100 if metric=='delivered' else 1
                    value=g[metric].mean()*scale;lo,hi=np.array(interval(g[metric]))*scale
                    ax.errorbar(value,k,xerr=[[value-lo],[hi-value]],fmt='o',color=COLORS.get(policy,'#2075bc'),capsize=3)
                ax.set_yticks(range(len(policies)),[NAMES.get(x,x) for x in policies] if col==0 else ['']*len(policies))
                ax.invert_yaxis();ax.set_xlabel('Delivered coverage (%)' if col==0 else 'Fleet energy (Wh)')
                if col==0:ax.set_xlim(0,100)
                ax.set_title(scenario.replace('_',' '));ax.grid(axis='x',alpha=.18)
        fig.suptitle(dataset+' | mission means and 95% bootstrap intervals',fontsize=12)
        for ext in ['png','pdf']:
            path=out/(dataset+'_overview.'+ext);fig.savefig(path,dpi=170,bbox_inches='tight');print(path)
        plt.close(fig)
    print('Use paired statistical contrasts for policy differences. A two-seed pilot is an execution check, not strong evidence.')

if __name__=='__main__':main()
