"""Check frozen study completeness, causal provenance and saved-policy identity."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from usv_sim.config import source_hash
from usv_research.experiment import extension_hash, identity
from usv_research.residual import rl_source_hash
from usv_research.native_pg import Agent

EXPECTED={'development_v3':45,'static_selection_nominal':192,'static_selection_short':192,
          'main_nominal':480,'main_short':480,'ablation_short':140,'price_low':60,'price_high':60,
          'shifted_short':96,'slow_short':32,'integration_128':24,'negative_nominal':48,'negative_short':48}
PHYSICAL='24bfb8b9aac5b87f9f268595cf4462964bd3a01da9d8089f51ed740d780fee86'
EXTENSION='f000d324411431a807bac72657424c44472cc457a2b2f856442e941850c00046'
RL='5113616c5fa00215ac2ca5cfa63d5545c3ead1d5a52d320aad84b253498818e5'

def records(directory):
    rows=[]
    for path in sorted(directory.glob('*.json')):
        if path.name.endswith('.decisions.json'):continue
        row=json.loads(path.read_text())
        if isinstance(row,dict) and 'delivered' in row and 'seed' in row:rows.append((path,row))
    return rows

def require(condition,message):
    if not condition:raise ValueError(message)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checksums',action='store_true',help='Also verify every released file against RELEASE_MANIFEST.json')
    p.add_argument('--out',help='Optional JSON report path')
    a=p.parse_args()
    require(source_hash()==PHYSICAL,'Physical source differs from frozen release')
    require(extension_hash()==EXTENSION,'Controller/experiment source differs from frozen release')
    require(rl_source_hash()==RL,'Learning source differs from frozen release')
    inventory={};minimum=float('inf');collisions=0
    for name,count in EXPECTED.items():
        pairs=records(ROOT/'results'/name);require(len(pairs)==count,f'{name}: {len(pairs)}/{count} records')
        keys=set();tapes={};cells=set()
        for path,r in pairs:
            key=identity(r['config'],r['control'],r['scenario'],r['seed'],r['policy'])
            require(key==r['run_id']==path.stem,f'Run identity mismatch: {path}')
            require(key not in keys,f'Duplicate run: {path}');keys.add(key)
            require(r['source_sha256']==PHYSICAL and r['extension_sha256']==EXTENSION,f'Source mismatch: {path}')
            cell=(r['scenario'],r['seed']);cells.add((r['scenario'],r['seed'],r['policy']))
            tapes.setdefault(cell,set()).add(r['exogenous_sha256'])
            for m in ['delivered','energy_wh','capped_aoi_s','graph_connected','min_separation_m','runtime_s']:
                require(math.isfinite(r[m]),f'Nonfinite {m}: {path}')
            require(0<=r['delivered']<=1 and r['energy_wh']>=0,f'Invalid primary metric: {path}')
            require(0<=r['capped_aoi_s']<=10*r['config']['report_ttl_s']+1e-7,f'AoI cap error: {path}')
            minimum=min(minimum,r['min_separation_m']);collisions+=r['collision_steps']
        require(all(len(v)==1 for v in tapes.values()),f'Exogenous pairing mismatch: {name}')
        manifest=ROOT/'results'/name/'manifest.json'
        if manifest.exists():
            args=json.loads(manifest.read_text()).get('arguments',{})
            if all(k in args for k in ('scenarios','seed_start','seeds','policies')):
                expected={(s,k,v) for s in args['scenarios'] for k in range(args['seed_start'],args['seed_start']+args['seeds']) for v in args['policies']}
                require(cells==expected,f'Incomplete/wrong scenario/policy/seed cells: {name}')
        inventory[name]=count
    native_dir=ROOT/'results/native_residual_seed0'
    cp50=native_dir/'checkpoint_50.npz';agent,meta=Agent.load(cp50)
    require(agent.episodes==50 and meta['rl_source_sha256']==RL,'Wrong evaluated native checkpoint')
    final,meta=Agent.load(native_dir/'checkpoint.npz')
    history=[json.loads(x) for x in (native_dir/'episodes.jsonl').read_text().splitlines()]
    require(final.episodes==55 and len(history)==55,'Native resume evidence incomplete')
    require([r['episode'] for r in history]==list(range(1,56)),'Native training history is not continuous')
    native_tapes={}
    for name in ['native_eval','native_neutral']:
        pairs=records(ROOT/'results'/name);require(len(pairs)==18,f'Incomplete {name}')
        for _,r in pairs:
            require(r['checkpoint_sha256']==sha(cp50),'Native evaluation used a different checkpoint')
            require(r['rl_source_sha256']==RL and r['source_sha256']==PHYSICAL,'Native source mismatch')
            native_tapes.setdefault((r['scenario'],r['seed']),set()).add(r['exogenous_sha256'])
        inventory[name]=len(pairs)
    require(len(native_tapes)==18 and all(len(v)==1 for v in native_tapes.values()),'Native evaluation pairing mismatch')
    ppo_dir=ROOT/'results/ppo_smoke_seed0';pt=[json.loads(x) for x in (ppo_dir/'training_timing.jsonl').read_text().splitlines()]
    require(pt[-1]['total_steps']==640,'PPO resume test incomplete')
    require(pt[-1]['checkpoint_sha256']==sha(ppo_dir/'final.zip'),'PPO checkpoint differs from training log')
    pp=records(ROOT/'results/ppo_eval_smoke');require(len(pp)==2,'PPO saved-model evaluation incomplete')
    require(all(r['checkpoint_sha256']==sha(ppo_dir/'final.zip') and r['rl_source_sha256']==RL for _,r in pp),'PPO evaluation identity mismatch')
    check=json.loads((ROOT/'results/ppo_check/check.json').read_text())
    require(check['status']=='passed','PPO checker failed')
    inventory['ppo_eval_smoke']=2
    if a.checksums:
        files=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text())['files']
        for name,expected in files.items():
            path=ROOT/name
            require(path.is_file() and sha(path)==expected,f'File checksum mismatch: {name}')
    report=dict(status='passed',physical_sha256=PHYSICAL,extension_sha256=EXTENSION,rl_source_sha256=RL,
                episode_inventory=inventory,simulation_records=sum(EXPECTED.values()),
                learning_evaluation_records=38,native_training_episodes=55,ppo_smoke_steps=640,
                minimum_separation_m=minimum,collision_steps=collisions,
                file_checksums_verified=a.checksums)
    if a.out:Path(a.out).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
