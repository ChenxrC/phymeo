"""Frozen 4d quality/settling contracts; no device is imported by this suite."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
import guided_motion_metrics as original
import settling_scores as scores


def fixture():
    old, new, references, geometry = {}, {}, {}, {}
    for pose in (0, 15, 24):
        ids = np.full((16, 24), 9, dtype=np.uint32)
        ids[2:14, 2:10] = 156
        ids[2:14, 14:22] = 158
        previous = ids.copy()
        if pose in (15, 24):
            previous[4:12, 16:20] = 202
        geometry[pose] = dict(instance_id=ids, depth=np.full(ids.shape, 2.),
                              valid=np.ones(ids.shape, dtype=np.uint8),
                              previous_instance_id=previous,
                              previous_depth=np.ones(ids.shape),
                              previous_valid=np.ones(ids.shape, dtype=np.uint8),
                              blocker_ids=(202,))
        references[pose] = np.full((*ids.shape, 3), 100, dtype=np.uint8)
        old[pose] = {n: np.full_like(references[pose], v) for n, v in zip((1, 2, 4, 8, 16), (120, 180, 130, 115, 105))}
        new[pose] = {n: np.full_like(references[pose], v) for n, v in zip((1, 2, 4, 8, 16), (120, 115, 110, 108, 105))}
    return dict(old_rgb8=old, new_rgb8=new, reference_rgb8=references, geometry=geometry)


class SettlingScoresTests(unittest.TestCase):
    def test_registration_and_independent_domain_are_frozen(self):
        r = scores.score_settling(**fixture())
        self.assertEqual(r['registration']['poses'], [0, 15, 24])
        self.assertEqual(r['registration']['stages'], [1, 2, 4, 8, 16])
        self.assertEqual(r['registration']['required_reveal_poses'], [15])
        self.assertEqual(r['registration']['expected_reference_spp'], 1024)
        self.assertIn('display-linear', r['space'])
        self.assertEqual(r['status'], 'passed')
        self.assertTrue(r['quality_claim_supported'])

    def test_abs_rmse_matches_independent_transfer_and_signed_bias(self):
        r = scores.score_settling(**fixture())
        stat = r['per_pose']['15']['stages']['2']['fingers']['new']
        delta = ((115 / 255 + .055) / 1.055) ** 2.4 - ((100 / 255 + .055) / 1.055) ** 2.4
        self.assertAlmostEqual(stat['rmse'], delta)
        self.assertAlmostEqual(stat['mean_bias'], delta)
        self.assertAlmostEqual(stat['mean_bias_luminance'], delta)
        self.assertEqual(len(stat['mean_bias_rgb']), 3)

    def test_registered_reveal_and_right_finger_are_not_replaced_by_extra_pose(self):
        r = scores.score_settling(**fixture())
        self.assertEqual(r['per_stage']['2']['reveal']['count'], 32)
        self.assertEqual(r['per_stage']['2']['reveal_right_158']['count'], 32)
        self.assertEqual(r['per_pose']['24']['stages']['2']['reveal']['count'], 32)
        data = fixture()
        data['geometry'][15]['previous_instance_id'] = data['geometry'][15]['instance_id'].copy()
        r = scores.score_settling(**data)
        self.assertEqual(r['status'], 'insufficient_evidence')
        self.assertFalse(r['quality_claim_supported'])
        self.assertEqual(r['per_stage']['2']['reveal']['count'], 0)

    def test_n2_both_finger_gates_are_independent(self):
        data = fixture()
        for p in (0, 15, 24):
            data['new_rgb8'][p][2][:] = 125
        r = scores.score_settling(**data)
        self.assertFalse(r['gates']['n2_vs_new_n1']['passed'])
        self.assertTrue(r['gates']['n2_vs_old_n2']['passed'])
        self.assertEqual(r['status'], 'failed')
        data = fixture()
        for p in (0, 15, 24):
            data['old_rgb8'][p][2][:] = 116
        r = scores.score_settling(**data)
        self.assertTrue(r['gates']['n2_vs_new_n1']['passed'])
        self.assertFalse(r['gates']['n2_vs_old_n2']['passed'])

    def test_later_edge_regression_fails_its_own_gate(self):
        data = fixture()
        for p in (0, 15, 24):
            g = data['geometry'][p]
            mask = original.build_masks(**g)['edge']
            data['new_rgb8'][p][8][mask] = 220
        r = scores.score_settling(**data)
        self.assertFalse(r['gates']['edge_n8_vs_old']['passed'])
        self.assertEqual(r['status'], 'failed')

    def test_positive_adjacent_rise_reported_without_inventing_monotonic_gate(self):
        data = fixture()
        for p in (0, 15, 24):
            data['new_rgb8'][p][4][:] = 130
            data['old_rgb8'][p][4][:] = 220
        r = scores.score_settling(**data)
        self.assertEqual(r['status'], 'passed')
        self.assertTrue(r['progression']['new']['any_fingers_rise_over_10_percent'])
        self.assertGreater(r['progression']['new']['max_adjacent_rmse_rise'], 0.)
        self.assertEqual(r['progression']['new']['worst_transition'], [2, 4])

    def test_brightness_jumps_and_n1_drift_are_signed_and_not_motion_claims(self):
        r = scores.score_settling(**fixture())
        d = r['per_pose']['0']['transitions']['new']['1->2']['fingers']
        self.assertLess(d['mean_bias_luminance'], 0.)
        self.assertGreater(d['rmse'], 0.)
        self.assertEqual(d['rmse'], r['per_pose']['0']['from_n1']['new']['2']['fingers']['rmse'])
        self.assertIn('caller', r['geometry_scope'])
        self.assertIsNone(r['uncertainty']['reference_standard_error'])

    def test_zero_error_only_proves_preservation_not_point_eight_reduction(self):
        data = fixture()
        for p in (0, 15, 24):
            for n in (1, 2, 4, 8, 16):
                data['old_rgb8'][p][n][:] = 100
                data['new_rgb8'][p][n][:] = 100
        r = scores.score_settling(**data)
        self.assertEqual(r['status'], 'insufficient_evidence')
        self.assertFalse(r['gates']['n2_vs_old_n2']['passed'])
        self.assertTrue(r['gates']['n2_vs_new_n1']['passed'])
        self.assertTrue(r['gates']['edge_n2_vs_old']['passed'])
        self.assertIsNone(r['gates']['n2_vs_old_n2']['ratio'])
        self.assertFalse(r['progression']['new']['any_fingers_rise_over_10_percent'])

    def test_zero_then_positive_error_is_unbounded_adverse_rise(self):
        data = fixture()
        for p in (0, 15, 24):
            data['new_rgb8'][p][4][:] = 100
            data['new_rgb8'][p][8][:] = 105
        r = scores.score_settling(**data)
        transition = r['progression']['new']['adjacent'][2]
        self.assertEqual(transition['from_stage'], 4)
        self.assertTrue(transition['unbounded_relative_rise'])
        self.assertIsNone(transition['relative_rise'])
        self.assertTrue(transition['over_10_percent'])

    def test_pool_uses_pixel_weights_not_average_pose_rmse(self):
        data = fixture()
        data['geometry'][0]['instance_id'][2:8, 2:10] = 9
        data['new_rgb8'][0][2][:] = 110
        r = scores.score_settling(**data)
        rows = [r['per_pose'][str(p)]['stages']['2']['fingers'] for p in (0, 15, 24)]
        expected = (sum(x['new']['squared_error_sum'] for x in rows) / (3 * sum(x['count'] for x in rows))) ** .5
        self.assertAlmostEqual(r['per_stage']['2']['fingers']['new']['rmse'], expected)

    def test_missing_or_mismatched_input_never_silently_changes_registration(self):
        data = fixture(); del data['new_rgb8'][15][8]
        with self.assertRaises(ValueError): scores.score_settling(**data)
        data = fixture(); data['reference_rgb8'][24] = data['reference_rgb8'][24][:-1]
        with self.assertRaises(ValueError): scores.score_settling(**data)
        data = fixture(); data['new_rgb8'][0][2] = data['new_rgb8'][0][2].astype(np.float32)
        with self.assertRaises(TypeError): scores.score_settling(**data)

    def test_empty_finger_target_drops_claim_and_does_not_become_zero_error(self):
        data = fixture()
        for p in (0, 15, 24):
            data['geometry'][p]['instance_id'][:] = 9
        r = scores.score_settling(**data)
        self.assertEqual(r['status'], 'insufficient_evidence')
        self.assertIsNone(r['per_stage']['2']['fingers']['new']['rmse'])


if __name__ == '__main__':
    unittest.main()
