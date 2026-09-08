"""Post-hoc mechanism diagnostics on the presaved first-seed traces only."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def main():
    result={'scope':'Post-hoc descriptive analysis of presaved first main seed (17000), four scenarios and two profiles; not an independent ablation.'}
    for policy in ['linear_inhibition','nonlinear_inhibition']:
        values=[];missions=0
        for dataset in ['bio_nominal','bio_short']:
            for path in (ROOT/'results'/dataset).glob('*.json'):
                if path.name.endswith('.decisions.json') or path.name=='manifest.json':continue
                row=json.loads(path.read_text())
                if row.get('policy')!=policy or row.get('seed')!=17000:continue
                missions+=1
                for d in json.loads(path.with_suffix('.decisions.json').read_text()):
                    if not d.get('stale_hold',False):values.append(d['commitment'])
        f=np.array(values)
        result[policy]=dict(traced_missions=missions,nonstale_decisions=len(f),
            mean_maximum_support=float(f.max(axis=1).mean()),
            fraction_maximum_above_threshold=float(np.mean(f.max(axis=1)>.3)),
            maximum_support=float(f.max()),mean_total_committed=float(f.sum(axis=1).mean()))
    out=ROOT/'results/analysis_v4_bio/trace_diagnostics.json'
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
