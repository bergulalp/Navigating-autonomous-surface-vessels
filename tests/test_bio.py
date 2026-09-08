"""Scientific invariants and a matched-control reduction, not output snapshots."""
import unittest
import numpy as np
from usv_sim.config import Config
from usv_research.experiment import ResearchWorld
from usv_research.control import LocalPolicy
from usv_bio.control import BioPolicy,BioConfig,inhibition_update


class BioTests(unittest.TestCase):
    def test_population_simplex_and_time_resolution(self):
        q=np.r_[0.,-np.ones(9)*3]
        for nonlinear in (False,True):
            x=inhibition_update(np.zeros(10),q,nonlinear,BioConfig())
            y=inhibition_update(np.zeros(10),q,nonlinear,BioConfig(ode_step=.01))
            self.assertGreaterEqual(x.min(),0);self.assertLessEqual(x.sum(),1)
            self.assertEqual(np.argmax(x),0)
            np.testing.assert_allclose(x,y,atol=.002)

    def test_reciprocal_phase_response(self):
        p=BioPolicy('reciprocal_handover',Config())
        p.last_move[1]=0
        p.register_moves(np.array([True,False,False,False,False,False]),60)
        self.assertEqual(p.next_move[1],120)  # T=2*tau
        p.register_moves(np.array([False,True,False,False,False,False]),120)
        self.assertEqual(p.next_move[0],180)

    def test_handover_cannot_start_with_unsettled_partner(self):
        for mode in ('periodic_handover','greedy_handover','reciprocal_handover'):
            p=BioPolicy(mode,Config());positions=p.goals.copy();positions[0,0]+=100
            mask=p.eligibility(positions,np.ones(6),120)
            self.assertEqual(mask[:2].sum(),0)
            self.assertTrue(np.all(mask.reshape(-1,2).sum(axis=1)<=1))

    def test_stale_observations_preserve_goals(self):
        for name in ('reciprocal_handover','nonlinear_inhibition'):
            cfg=Config();p=BioPolicy(name,cfg);old=p.goals.copy()
            result=p.decide(old,np.zeros((6,7)),np.full(6,181),.5,0,0,'storm')
            np.testing.assert_array_equal(result,old)

    def test_economic_reduction(self):
        cfg=Config(duration_s=360,warmup_s=60,grid_angles=8,target_count=2,planner_samples=4)
        a=ResearchWorld(cfg,'storm',123,'economic');b=ResearchWorld(cfg,'storm',123,'economic')
        a.policy=LocalPolicy('economic',cfg);b.policy=BioPolicy('economic',cfg)
        ra,rb=a.run(),b.run()
        for k in ('delivered','energy_wh','min_separation_m'):
            self.assertEqual(ra[k],rb[k])


if __name__=='__main__':unittest.main()
