"""Use measured end-to-end throughput, including the chosen worker count."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('pilot', help='Output directory of a completed representative experiment')
p.add_argument('--episodes',type=int,default=1000)
a=p.parse_args()
timing=[json.loads(s) for s in (Path(a.pilot)/'timing.jsonl').read_text().splitlines()]
t=next(x for x in reversed(timing) if x['completed']>0)
seconds=t['wall_s_per_episode']*a.episodes
print(f"Pilot: {t['completed']} episodes in {t['wall_s']:.1f}s with {t['workers']} workers")
print(f"{a.episodes} similar episodes: {seconds/60:.1f} min ({seconds/3600:.2f} h)")
print('Planning range (0.7–2x): %.1f–%.1f min. Repeat the pilot for a different policy, grid, duration or worker count.'%(seconds*.7/60,seconds*2/60))
print('This estimates computation time, not time until an algorithm improves. Simulated mission duration and wall time differ.')
