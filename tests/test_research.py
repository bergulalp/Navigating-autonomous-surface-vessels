import unittest
import numpy as np
from usv_sim.config import Config
from usv_sim.belief import SensorBelief
from usv_sim.planning import ring, Planner, freshness_probability
from usv_research.control import Evaluator, guidance_prediction, hydraulic_potential, deadline_pivotality


class ResearchMathTests(unittest.TestCase):
    def test_deadline_influence_matches_exact_graph_enumeration(self):
        # Three reciprocal links x two rounds: enumerate all 64 graph paths.
        import itertools
        bits = np.array(list(itertools.product((False, True), repeat=6)))
        edges = [(0, 1), (0, 2), (1, 2)]
        graphs = np.tile(np.eye(3, dtype=bool), (2, 64, 1, 1))
        for k in range(2):
            for e, (i, j) in enumerate(edges):
                graphs[k, :, i, j] = graphs[k, :, j, i] = bits[:, 3*k+e]
        visible = np.zeros((2, 64, 2), bool); visible[0, :, 0] = True
        pd = np.array([[.7], [0.]])
        influence = deadline_pivotality(pd, visible, graphs, np.ones((2, 64)), np.ones(1))
        f = freshness_probability(pd, visible, graphs)[:, 0]
        for e, (i, j) in enumerate(edges):
            prob_up, prob_down = np.full(6, .5), np.full(6, .5)
            prob_up[[e, e+3]] += 1e-5; prob_down[[e, e+3]] -= 1e-5
            def expected(p): return float(f @ np.prod(np.where(bits, p, 1-p), axis=1))
            numerical = (expected(prob_up) - expected(prob_down)) / 2e-5
            self.assertAlmostEqual(influence[i, j], numerical, places=9)
            self.assertGreaterEqual(influence[i, j], 0)

    def test_temporal_evaluator_matches_validated_core(self):
        cfg = Config(grid_angles=8, planner_samples=8)
        p = ring(6, 2200)
        m = np.array([SensorBelief().moments() for _ in range(6)])
        w = np.arange(1, 25); w = w / w.sum()
        self.assertAlmostEqual(Planner(cfg).temporal_score(p, m, .6, .3, w),
                               float(Evaluator(cfg).field(p, m, .6, .3) @ w), places=12)

    def test_hydraulic_gradient_matches_finite_difference(self):
        cfg = Config()
        p = ring(6, 2200) + np.arange(12).reshape(6, 2) * 5
        b = np.arange(1, 7, dtype=float); b /= b.sum()
        _, g = hydraulic_potential(p, .7, cfg, b)
        for i in range(6):
            for j in range(2):
                a, z = p.copy(), p.copy(); a[i, j] += .01; z[i, j] -= .01
                numeric = (hydraulic_potential(a, .7, cfg, b)[0] - hydraulic_potential(z, .7, cfg, b)[0]) / .02
                self.assertAlmostEqual(g[i, j], numeric, places=9)

    def test_guidance_energy_integral_and_speed(self):
        p = np.zeros((3, 2)); goal = np.array([[0, 0], [80, 0], [800, 0]])
        times = np.arange(0, 300.001, .01)
        # Verify analytic integral with independent small-step integration.
        d = np.linalg.norm(goal, axis=1); energy = 0.
        for _ in times[:-1]:
            v = np.minimum(.025 * d, 3)
            d -= .01 * v; energy += float(18 * (v ** 3).sum() * .01 / 3600)
        path, e = guidance_prediction(p, goal, [0, 300])
        np.testing.assert_allclose(goal - path[-1], np.stack((d, np.zeros(3)), axis=1), atol=.025)
        self.assertAlmostEqual(e[-1], energy, delta=.02)
        np.testing.assert_allclose(path[0], p)


if __name__ == "__main__": unittest.main()
