import unittest
from dataclasses import replace
import numpy as np
from scipy.special import ndtr
from usv_sim.config import Config
from usv_sim.physics import footprints, forward_latest, reachable
from usv_sim.belief import SensorBelief
from usv_sim.planning import ring, information_score, freshness_probability
from usv_sim.world import World


class ScienceTests(unittest.TestCase):
    def test_shared_sensor_visibility_matches_sampler(self):
        cfg = Config(n=3)
        pd, radar, a = footprints(np.zeros((1, 2)), np.array([[200., 0]]), .9,
                                  np.ones(1), np.zeros(1), cfg)
        rng = np.random.default_rng(7)
        visible = rng.uniform(size=100000) < a[0]
        y = visible & (rng.uniform(size=100000) < pd[0, 0])
        self.assertAlmostEqual(float(y.mean()), float(a[0] * pd[0, 0]), delta=.005)
        self.assertLessEqual(float(a[0] * pd[0, 0]), float(a[0]))

    def test_shared_routes_have_joint_not_independent_failure(self):
        # Both observers depend on one failed last-hop edge: neither delivers.
        adj = np.eye(4, dtype=bool)
        adj[0, 2] = adj[2, 0] = True
        adj[1, 2] = adj[2, 1] = True
        self.assertEqual(reachable(adj, 3).tolist(), [False, False, False, True])
        adj[2, 3] = adj[3, 2] = True
        self.assertTrue(np.all(reachable(adj, 3)))

    def test_causal_packet_forwarding_requires_two_steps(self):
        adj = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=bool)
        stamp = np.array([[10.], [-np.inf], [-np.inf]])
        payload = np.array([[[42.]], [[0.]], [[0.]]])
        first, p1 = forward_latest(stamp, payload, adj)
        self.assertTrue(np.isneginf(first[2, 0]))
        second, p2 = forward_latest(first, p1, adj)
        self.assertEqual(second[2, 0], 10)
        self.assertEqual(p2[2, 0, 0], 42)

    def test_predictor_respects_time_order_of_links(self):
        pd = np.array([[1.], [0.]])
        visible = np.ones((2, 1, 2), dtype=bool)
        graphs = np.broadcast_to(np.eye(3, dtype=bool), (2, 1, 3, 3)).copy()
        graphs[0, 0, 0, 1] = graphs[0, 0, 1, 0] = True
        graphs[1, 0, 1, 2] = graphs[1, 0, 2, 1] = True
        self.assertEqual(float(freshness_probability(pd, visible, graphs)[0, 0]), 1)
        self.assertEqual(float(freshness_probability(pd, visible, graphs[::-1])[0, 0]), 0)

    def test_no_observations_do_not_increase_confidence(self):
        b = SensorBelief()
        before = b.moments()
        for _ in range(60):
            b.predict(5)
        after = b.moments()
        self.assertGreater(after[1], before[1])
        self.assertGreater(after[3], before[3])
        self.assertAlmostEqual(b.posterior.sum(), 1)
        self.assertTrue(np.all(b.posterior >= 0))

    def test_information_zero_when_parameters_known(self):
        cfg = Config(n=3)
        m = np.zeros((3, 7)); m[:, 0] = 1
        self.assertAlmostEqual(information_score(ring(3, 1000), m, .5, cfg), 0)

    def test_total_blackout_blocks_delivery_and_updates(self):
        cfg = Config(duration_s=360, warmup_s=60, n=3, grid_angles=16,
                     full_blackout=True, target_count=4)
        w = World(cfg, "storm", 3, "adaptive")
        w.control()
        for _ in range(6):
            w.step()
        before = sum(b.observations for b in w.beliefs)
        r = w.run()
        self.assertEqual(r["delivered"], 0)
        self.assertGreater(r["local_fresh"], 0)
        self.assertTrue(np.all(w.telemetry_times[-1] == 0))
        self.assertGreater(r["command_stale"], 0)
        # Only the initial manifest supports observations; none accumulate after it expires.
        self.assertEqual(sum(b.observations for b in w.beliefs), before)

    def test_reproducible_tapes_and_results(self):
        cfg = Config(duration_s=360, warmup_s=60, n=3, grid_angles=16, target_count=4)
        a, b, c = (World(cfg, "storm", 42, policy) for policy in ("adaptive", "adaptive", "fixed_ring"))
        self.assertEqual(a.exogenous_sha256, c.exogenous_sha256)
        ra, rb = a.run(), b.run()
        for key in ("delivered", "energy_wh", "range_rmse", "early_reports"):
            self.assertEqual(ra[key], rb[key])
        np.testing.assert_array_equal(a.fleet.p, b.fleet.p)

    def test_observation_does_not_read_current_truth(self):
        cfg = Config(n=3, full_blackout=True)
        w = World(cfg, "storm", 2, "adaptive")
        before = w.features().copy()
        w.fleet.p += 1000
        w.true_range[:] = .3
        np.testing.assert_array_equal(before, w.features())

    def test_outbound_report_cannot_be_early(self):
        cfg = Config(duration_s=600, warmup_s=0, n=3, grid_angles=16, target_count=4)
        w = World(cfg, "calm", 0, "fixed_ring")
        w.step_decision()
        w.target_eligible[:] = True
        w.target_crossing[:] = 200
        w.first_report[:] = [100, 180, 450, np.inf]
        self.assertEqual(w.result()["early_reports"], 1)
        self.assertEqual(w.result()["eligible_targets"], 4)


if __name__ == "__main__":
    unittest.main()
