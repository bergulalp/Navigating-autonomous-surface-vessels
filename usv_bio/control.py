"""Mechanism tests for generic surface monitoring, not biological replicas.

Reciprocity changes the time of eligibility for a proposal; cross-inhibition
changes commitment to proposal families. Neither changes sensing physics.
All state is maintained by the coordinator from the same cached observations.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.special import expit, softmax
from usv_sim.planning import mission_weights
from usv_research.control import LocalPolicy, ControlConfig, normalized, guidance_prediction

POLICIES = ('economic', 'periodic_handover', 'greedy_handover',
            'reciprocal_handover', 'linear_inhibition', 'nonlinear_inhibition')


@dataclass(frozen=True)
class BioConfig:
    settle_m: float = 35.0
    period_s: float = 120.0
    responsive_max_s: float = 240.0
    interdependence: float = 0.5
    inhibition: float = 1.0
    threshold: float = 0.3
    steepness: float = 10.0
    evidence_scale: float = 0.01
    deliberation: float = 2.0
    ode_step: float = 0.02
    def dict(self): return asdict(self)


def inhibition_update(f, quality, nonlinear, settings):
    """LES-inspired mean-field dynamics over 10 *virtual* option populations.

    f0 = 1-sum(f); pi = softmax(quality); r = 1/(1+4*pi).
    The nonlinear branch uses the linearly bounded sigmoid sigma_2 of
    March-Pons et al. (2025); the comparator uses sigma(f)=f.
    Explicit Euler is positivity preserving for these bounded rates and h<=.02.
    No clipping/renormalization is used to conceal integration failures.
    """
    f = np.asarray(f, float).copy()
    pi = softmax(quality)
    abandon = 1 / (1 + 4*pi)
    steps = int(np.ceil(settings.deliberation / settings.ode_step))
    h = settings.deliberation / steps
    for _ in range(steps):
        f0 = 1 - f.sum()
        sigma = f * expit(settings.steepness * (f-settings.threshold)) if nonlinear else f
        rhs = f0*((1-settings.interdependence)*pi+settings.interdependence*f)
        rhs -= abandon*f + settings.inhibition*f*(sigma.sum()-sigma)
        f += h*rhs
        if f.min() < -1e-12 or f.sum() > 1+1e-12:
            raise ArithmeticError('Commitment left its simplex; reduce the ODE step')
    return f


class BioPolicy(LocalPolicy):
    def __init__(self, name, cfg, control=None, bio=None):
        if name not in POLICIES: raise ValueError(name)
        super().__init__(name, cfg, control or ControlConfig())
        self.bio = bio or BioConfig()
        if name.endswith('handover') and cfg.n % 2:
            raise ValueError('This paired mechanism experiment requires an even vessel count')
        self.last_move = np.where(np.arange(cfg.n)%2 == 0, -120., -60.)
        self.next_move = np.where(np.arange(cfg.n)%2 == 0, 0., 60.)
        self.commitment = np.zeros(10)

    def eligibility(self, positions, priority, t):
        """A pair can start a new excursion only when both cached poses settle.

        This is a coordinator-side gate, not a distributed acknowledgement or
        a physical guarantee: stale telemetry and command latency still exist.
        Pair identities (0,1),(2,3),... are fixed before the mission.
        """
        mask = np.zeros(self.cfg.n)
        settled = np.linalg.norm(positions-self.goals, axis=1) <= self.bio.settle_m
        for i in range(0, self.cfg.n, 2):
            pair = np.array([i,i+1])
            if not settled[pair].all(): continue
            if self.name == 'periodic_handover':
                chosen = pair[int(t/self.cfg.decision_s) % 2]
            elif self.name == 'greedy_handover':
                chosen = pair[int(np.argmax(priority[pair]))]
            else:
                due = pair[self.next_move[pair] <= t+1e-9]
                if not len(due): continue
                chosen = due[int(np.argmin(self.next_move[due]))]
            mask[chosen] = 1
        return mask

    def register_moves(self, changed, t):
        # Accepted goal changes are the events. This does not measure tail beats
        # or pretend that a delayed command was already physically executed.
        for i in np.flatnonzero(changed):
            self.last_move[i] = t
            self.next_move[i] = t+self.bio.period_s
            j = i ^ 1
            lag = t-self.last_move[j]
            if self.cfg.decision_s <= lag <= self.bio.responsive_max_s:
                self.next_move[j] = self.last_move[j]+2*lag

    def decide(self, positions, moments, ages, severity, wind, t, scenario, forced_action=None):
        if self.name == 'economic':
            return super().decide(positions,moments,ages,severity,wind,t,scenario,forced_action)
        self.decisions += 1; self.action_counts[0] += 1
        cfg,c,b = self.cfg,self.settings,self.bio
        if np.max(ages) > cfg.command_timeout_s:
            self.diagnostics.append(dict(t=t,accepted=False,stale_hold=True,proposed=0))
            return self.goals.copy()
        horizon = min(c.horizon_s,max(cfg.decision_s,cfg.duration_s-t))
        future_w = mission_weights(self.planner.angles,t+horizon/2,cfg.duration_s,scenario,cfg.priority_turns)
        grad = np.zeros_like(positions)
        for i in range(cfg.n):
            for k in range(2):
                plus,minus = positions.copy(),positions.copy()
                plus[i,k] += c.gradient_delta_m; minus[i,k] -= c.gradient_delta_m
                grad[i,k] = (self.planner.field(plus,moments,severity,wind)@future_w-
                             self.planner.field(minus,moments,severity,wind)@future_w)/(2*c.gradient_delta_m)
        priority = np.linalg.norm(grad,axis=1)
        paired = self.name.endswith('handover')
        mask = self.eligibility(positions,priority,t) if paired else np.ones(cfg.n)
        direction = normalized(grad*mask[:,None])
        top = np.zeros(cfg.n); top[np.argsort(-priority,kind='stable')[:2]] = 1
        rotation = np.stack((-direction[:,1],direction[:,0]),axis=1)
        radial = positions/np.maximum(np.linalg.norm(positions,axis=1)[:,None],1e-12)
        directions = [direction*.5,direction,normalized(grad*top[:,None]*mask[:,None]),
                      normalized(direction+.35*rotation),normalized(direction-.35*rotation),
                      direction*.25,radial*mask[:,None],-radial*mask[:,None]]
        hold = positions.copy()
        if paired: hold[mask==0] = self.goals[mask==0]
        proposals = [self.goals.copy(),hold]
        for d in directions:
            goal = positions+c.step_m*d
            if paired: goal[mask==0] = self.goals[mask==0]
            proposals.append(goal)
        times = horizon*np.array([1/6,1/2,5/6])
        weights = [mission_weights(self.planner.angles,t+x,cfg.duration_s,scenario,cfg.priority_turns) for x in times]
        scores,covers,energies = [],[],[]
        for goal in proposals:
            path,energy = guidance_prediction(positions,goal,np.r_[times,horizon],cfg.max_speed_mps)
            dist = np.linalg.norm(path[:,:,None]-path[:,None,:],axis=-1)
            for d in dist: np.fill_diagonal(d,np.inf)
            if np.min(dist)<220 or np.max(np.linalg.norm(goal,axis=1))>1.4*cfg.mission_radius_m:
                scores.append(-1e6); covers.append(0.); energies.append(float(energy[-1])); continue
            cover = np.mean([self.planner.field(p,moments,severity,wind)@w for p,w in zip(path[:3],weights)])
            covers.append(float(cover)); energies.append(float(energy[-1]))
            scores.append(float(cover-c.energy_price*energy[-1]/(cfg.n*100*horizon/3600)))
        scores = np.array(scores)
        if self.name.endswith('inhibition'):
            quality = np.clip((scores-scores.max())/b.evidence_scale,-30,0)
            self.commitment = inhibition_update(self.commitment,quality,self.name=='nonlinear_inhibition',b)
            chosen = int(np.argmax(self.commitment))
        else: chosen = int(np.argmax(scores))
        # Identical modeled-improvement gate; it is not a guarantee of true gain.
        if scores[chosen] <= scores[0]+c.improvement_floor: chosen = 0
        changed = np.linalg.norm(proposals[chosen]-self.goals,axis=1)>1
        if chosen:
            self.switches += 1; self.goals = proposals[chosen].copy()
            if self.name=='reciprocal_handover': self.register_moves(changed,t)
        self.diagnostics.append(dict(t=t,accepted=bool(chosen),stale_hold=False,proposed=10,
            proposal=chosen,predicted_gain=float(scores[chosen]-scores[0]),
            predicted_coverage=covers[chosen],predicted_energy_wh=energies[chosen],
            moving_vessels=int(changed.sum()),eligible=mask.tolist(),
            commitment=self.commitment.tolist(),next_move=self.next_move.tolist()))
        return self.goals.copy()
