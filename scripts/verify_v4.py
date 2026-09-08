"""Validate both studies, held-out pairing, source identity and trained checkpoints."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from usv_bio.experiment import bio_hash,identity
from usv_bio.learning_eval import evaluator_hash
from usv_bio.ppo_train import trainer_hash,stream_seed
from usv_research.residual import rl_source_hash

EXPECTED={'bio_pilot':12,'bio_nominal':288,'bio_short':288,'bio_shifted':96,'bio_integration128':48}
EVALUATIONS={'rl_constant_development':108,'rl_ppo_development':36,'rl_final_test':160}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def records(directory):
    result=[]
    for p in sorted(directory.glob('*.json')):
        if p.name.endswith('.decisions.json'):continue
        r=json.loads(p.read_text())
        if isinstance(r,dict) and 'delivered' in r and 'seed' in r:result.append((p,r))
    return result


def require(condition,message):
    if not condition:raise ValueError(message)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checksums',action='store_true');p.add_argument('--out');a=p.parse_args()
    legacy=subprocess.run([sys.executable,str(ROOT/'scripts/verify_release.py')],cwd=ROOT,capture_output=True,text=True,check=True)
    old=json.loads(legacy.stdout);protocol=json.loads((ROOT/'docs/V4_FROZEN_PROTOCOL.json').read_text())
    require(bio_hash()==protocol['bio_sha256'],'New controller differs from frozen protocol')
    require(rl_source_hash()==protocol['rl_source_sha256'],'Learning source differs from frozen protocol')
    counts={};minimum=float('inf');collisions=0
    for name,count in EXPECTED.items():
        rows=records(ROOT/'results'/name);require(len(rows)==count,f'{name}: {len(rows)}/{count}')
        manifest=json.loads((ROOT/'results'/name/'manifest.json').read_text());args=manifest['arguments']
        expected={(s,k,p) for s in args['scenarios'] for k in range(args['seed_start'],args['seed_start']+args['seeds']) for p in args['policies']}
        found=set();tapes={}
        for path,r in rows:
            require(r['bio_sha256']==bio_hash(),f'Causal hash mismatch {path}')
            require(r['config']==manifest['signature']['config'] and r['bio_config']==manifest['signature']['bio_config'],f'Configuration mismatch {path}')
            require(identity(r['config'],r['bio_config'],r['scenario'],r['seed'],r['policy'])==r['run_id']==path.stem,f'Run identity mismatch {path}')
            key=(r['scenario'],r['seed'],r['policy']);require(key not in found,f'Duplicate {path}');found.add(key)
            tapes.setdefault(key[:2],set()).add(r['exogenous_sha256'])
            for m in ['delivered','energy_wh','capped_aoi_s','graph_connected','postwarmup_utility']:
                require(math.isfinite(r[m]),f'Invalid {m}: {path}')
            require(0<=r['delivered']<=1 and r['energy_wh']>=0,f'Invalid metrics: {path}')
            minimum=min(minimum,r['min_separation_m']);collisions+=r['collision_steps']
        require(found==expected,f'Incomplete declared cells: {name}')
        require(all(len(v)==1 for v in tapes.values()),f'Exogenous pairing differs: {name}')
        counts[name]=len(rows)
    training={}
    for seed in [0,1,2]:
        folder=ROOT/f'results/ppo_v4_single_seed{seed}'
        log=[json.loads(x) for x in (folder/'training_timing.jsonl').read_text().splitlines()]
        require(len(log)==1 and log[-1]['total_steps']==2048,f'Wrong short training budget: {seed}')
        require(log[-1]['checkpoint_sha256']==sha(folder/'final.zip'),f'Checkpoint differs: {seed}')
        training[seed]=log[-1]
    smoke=ROOT/'results/rl_driver_smoke'
    log=[json.loads(x) for x in (smoke/'training_timing.jsonl').read_text().splitlines()]
    require([x['total_steps'] for x in log]==[128,256],'Chunked resume test incomplete')
    require(log[-1]['checkpoint_sha256']==sha(smoke/'final.zip'),'Driver checkpoint differs')
    require([x['environment_stream_seed'] for x in log]==[stream_seed(99,0),stream_seed(99,128)],'Chunk streams not correctly advanced')
    require(all(x['trainer_sha256']==trainer_hash() for x in log),'Driver training source differs')
    for name,count in EVALUATIONS.items():
        rows=records(ROOT/'results'/name);require(len(rows)==count,f'{name}: {len(rows)}/{count}')
        signature=json.loads((ROOT/'results'/name/'manifest.json').read_text())['signature']
        expected={(s,k,p['name']) for s in signature['scenarios'] for k in signature['seeds'] for p in signature['policies']}
        found=set();tapes={}
        for path,r in rows:
            require(r['evaluator_sha256']==evaluator_hash() and r['rl_source_sha256']==rl_source_hash(),f'Evaluator source mismatch: {path}')
            require(r['config']==signature['config'],f'Physical config differs: {path}')
            key=(r['scenario'],r['seed'],r['policy']);require(key not in found,f'Duplicate: {path}');found.add(key)
            tapes.setdefault(key[:2],set()).add(r['exogenous_sha256'])
            require(sum(r['residual_actions'])==40,f'Wrong decision count: {path}')
            if r['policy'].startswith('ppo_'):
                require(r['checkpoint_sha256']==training[r['policy_spec']['training_seed']]['checkpoint_sha256'],f'Wrong evaluated checkpoint: {path}')
            for metric in ['delivered','energy_wh','mean_step_reward','discounted_return']:
                require(math.isfinite(r[metric]),f'Invalid {metric}: {path}')
        require(found==expected,f'Incomplete learning cells: {name}')
        require(all(len(v)==1 for v in tapes.values()),f'Learning pairing differs: {name}')
        counts[name]=len(rows)
    selection=json.loads((ROOT/'results/rl_selection.json').read_text())
    test=json.loads((ROOT/'results/rl_final_test/manifest.json').read_text())['signature']
    require(not set(selection['development_seeds']) & set(test['seeds']),'Development/test overlap')
    require(selection==test['selection'],'Test did not use frozen selection')
    dev=records(ROOT/'results/rl_constant_development');totals={i:[] for i in range(9)}
    for _,r in dev:totals[int(r['policy'].split('_')[-1])].append(r['mean_step_reward'])
    require(selection['selected_action']==max(totals,key=lambda i:sum(totals[i])/len(totals[i])),'Selection criterion mismatch')
    require('\\author{5.9.1. MARIN' in (ROOT/'paper/main.tex').read_text(),'Manuscript project identity missing')
    require('name: "5.9.1. MARIN"' in (ROOT/'CITATION.cff').read_text(),'Citation identity missing')
    if a.checksums:
        manifest=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text())
        require(manifest['version']=='4.0.0','Wrong release manifest')
        for filename,digest in manifest['files'].items():
            require((ROOT/filename).is_file() and sha(ROOT/filename)==digest,f'File checksum differs: {filename}')
    report=dict(status='passed',project_identity='5.9.1. MARIN',version='4.0.0',legacy=old,
        new_episode_inventory=counts,new_model_based_missions=sum(EXPECTED.values()),new_learning_evaluations=sum(EVALUATIONS.values()),
        total_model_based_missions=old['simulation_records']+sum(EXPECTED.values()),
        ppo_new_research_steps=6144,ppo_new_driver_check_steps=256,training=training,
        bio_sha256=bio_hash(),evaluator_sha256=evaluator_hash(),new_minimum_separation_m=minimum,
        new_collision_steps=collisions,selected_constant_action=selection['selected_action'],
        file_checksums_verified=a.checksums,windows_host_tested=False)
    if a.out:Path(a.out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
