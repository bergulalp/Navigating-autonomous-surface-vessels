"""Package source, evidence and manuscript; exclude environments and source-paper copies."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]
EXCLUDE_PARTS={'__pycache__','.git','.venv','.venv-rl','.pytest_cache','research','qa','build','dist',
               'development_pilot','analysis_native','runs'}
EXCLUDE_SUFFIX={'.pyc','.aux','.bbl','.blg','.log','.fls','.fdb_latexmk','.out','.synctex.gz'}

def included(path):
    r=path.relative_to(ROOT)
    if any(x in EXCLUDE_PARTS or x.endswith('.egg-info') for x in r.parts):return False
    if path.suffix in EXCLUDE_SUFFIX or path.name.endswith('.synctex.gz'):return False
    if r.as_posix() in ['RELEASE_MANIFEST.json','docs/WORK_CHECKPOINT.md','docs/WORK_CHECKPOINT_V4.md','paper/build_console.txt']:return False
    if path.name.startswith('build_console'):return False
    if len(r.parts)>1 and r.parts[0]=='results' and r.parts[1].startswith(('my_','legacy_q_')):return False
    return True

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default=str(ROOT.parent/'MARIN_Swarm_v4.zip'))
    p.add_argument('--overleaf',default=str(ROOT.parent/'MARIN_Overleaf_v4.zip'))
    a=p.parse_args();target=Path(a.out).resolve()
    files=sorted(f for f in ROOT.rglob('*') if f.is_file() and included(f) and f.resolve()!=target)
    required=['README.md','paper/main.pdf','paper/generated/v4_results.tex','paper/generated/v4_learning.tex','results/release_validation_v4.json']
    if not all(ROOT.joinpath(f).is_file() for f in required):raise ValueError('Finish the manuscript and release validation first')
    manifest=ROOT/'RELEASE_MANIFEST.json'
    manifest.write_text(json.dumps(dict(version='4.0.0',date='2026-09-07',
        description='SHA-256 of released files; this manifest excludes itself.',
        files={f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files}),indent=2)+'\n')
    files.append(manifest)
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for f in files:z.write(f,'marin_swarm_v4/'+f.relative_to(ROOT).as_posix())
    print(json.dumps(dict(archive=str(target),files=len(files),bytes=target.stat().st_size,
                         sha256=hashlib.sha256(target.read_bytes()).hexdigest()),indent=2))
    overleaf=Path(a.overleaf).resolve();paper=ROOT/'paper'
    sources=[f for f in paper.rglob('*') if f.is_file() and
             (f.suffix in {'.tex','.bib'} or (f.suffix=='.pdf' and f.parent.name=='figures') or f.name=='OVERLEAF_README.md')]
    with zipfile.ZipFile(overleaf,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for f in sorted(sources):z.write(f,f.relative_to(paper).as_posix())
    print(json.dumps(dict(overleaf=str(overleaf),files=len(sources),bytes=overleaf.stat().st_size,
                         sha256=hashlib.sha256(overleaf.read_bytes()).hexdigest()),indent=2))

if __name__=='__main__':main()
