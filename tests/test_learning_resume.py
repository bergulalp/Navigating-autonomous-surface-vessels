import tempfile
import unittest
from pathlib import Path
import numpy as np
from usv_research.native_pg import Agent


class LearningResumeTests(unittest.TestCase):
    def test_complete_episode_resume_restores_learning_and_rng(self):
        def rollout(agent):
            obs=np.array([.2,.8,-.4]);agent.reset(obs);rows=[]
            for k in range(3):
                action,x,p=agent.act(obs)
                rows.append((x,p,action,float(action==k)-.01*action))
                obs=obs+np.array([.01*action,0,.02])
            agent.update(rows)
            return [r[2] for r in rows]
        a=Agent(3,3,stack=2,seed=44);rollout(a)
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'checkpoint.npz';a.save(file,{'test':'resume'})
            b,meta=Agent.load(file)
            self.assertEqual(meta,{'test':'resume'})
            self.assertEqual(rollout(a),rollout(b))
            for name in ('w','adam_m','adam_v','baseline'):
                np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
            self.assertEqual(a.rng.bit_generator.state,b.rng.bit_generator.state)


if __name__=='__main__':unittest.main()
