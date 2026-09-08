import unittest
import numpy as np
from usv_sim.config import Config
from usv_research.experiment import ResearchWorld
from usv_research.control import LocalPolicy
from usv_research.residual import ResidualMission


class ResidualTests(unittest.TestCase):
    def test_neutral_action_preserves_economic_controller(self):
        cfg=Config(duration_s=180,warmup_s=30,grid_angles=8,planner_samples=4,target_count=3)
        env=ResidualMission(cfg,False)
        o=env.reset(seed=1,options={'world_seed':123,'scenario':'storm'})
        self.assertEqual(o.shape,(83,));self.assertTrue(np.isfinite(o).all())
        w=ResearchWorld(cfg,'storm',123,'economic');w.policy=LocalPolicy('economic',cfg)
        while w.index<w.steps:
            _,_,done,_=env.step(4);w.step_decision()
        self.assertTrue(done)
        np.testing.assert_array_equal(env.world.fleet.p,w.fleet.p)
        self.assertEqual(env.world.result()['delivered'],w.result()['delivered'])

    def test_hidden_health_not_in_residual_observation(self):
        env=ResidualMission(Config(grid_angles=8,planner_samples=4),False);env.reset(seed=20)
        old=env.observation();env.world.true_range[:]=.1;env.world.true_offset[:]=-4
        np.testing.assert_array_equal(old,env.observation())


if __name__=='__main__':unittest.main()
