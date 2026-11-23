"""Independent contracts; synthetic arrays do not substitute for device evidence."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
import guided_motion_metrics as metrics


def inputs(pose='p0'):
    h, w = 16, 24
    ids = np.full((h, w), 9, dtype=np.uint32)
    ids[2:14, 2:10] = 156
    ids[2:14, 14:22] = 158
    previous = ids.copy()
    previous[4:12, 4:8] = 999
    rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    return dict(pose_id=pose, raw_rgb8=rgb + 20, filtered_rgb8=rgb + 5,
                reference_rgb8=rgb, instance_id=ids,
                depth=np.full((h, w), 2., dtype=np.float32),
                valid=np.ones((h, w), dtype=np.uint8),
                previous_instance_id=previous,
                previous_depth=np.ones((h, w), dtype=np.float32),
                previous_valid=np.ones((h, w), dtype=np.uint8), blocker_ids=(999,))


class GuidedMotionMetricsTests(unittest.TestCase):
    def test_inverse_srgb_is_display_linear_not_raw_radiance(self):
        result = metrics.srgb_to_display_linear(np.array([0, 10, 128, 255], dtype=np.uint8))
        self.assertAlmostEqual(result[0], 0.)
        self.assertAlmostEqual(result[1], (10 / 255) / 12.92)
        self.assertAlmostEqual(result[2], ((128 / 255 + .055) / 1.055) ** 2.4)
        self.assertEqual(result[3], 1.)

    def test_constant_id_erosion_never_wraps_or_merges_two_fingers(self):
        data = inputs()
        masks = metrics.build_masks(**{k: data[k] for k in (
            'instance_id', 'depth', 'valid', 'previous_instance_id',
            'previous_depth', 'previous_valid', 'blocker_ids')})
        self.assertEqual(int(masks['finger_156'].sum()), 60)
        self.assertEqual(int(masks['finger_158'].sum()), 60)
        self.assertFalse(masks['fingers'][0].any())
        self.assertFalse(masks['fingers'][-1].any())
        self.assertEqual(int(masks['reveal'].sum()), 32)

    def test_image_border_is_not_a_wrapped_geometric_edge(self):
        ids = np.ones((5, 5), dtype=np.uint32)
        masks = metrics.build_masks(instance_id=ids, depth=np.ones((5, 5)),
                                    valid=np.ones((5, 5), dtype=np.uint8))
        self.assertEqual(int(masks['edge'].sum()), 0)
        ids[:, 0] = 2
        masks = metrics.build_masks(instance_id=ids, depth=np.ones((5, 5)),
                                    valid=np.ones((5, 5), dtype=np.uint8))
        self.assertFalse(masks['edge'][:, -1].any())
        self.assertTrue(masks['edge'][:, :2].all())

    def test_reveal_excludes_background_invalid_wrong_id_and_nearer_surfaces(self):
        data = inputs()
        data['instance_id'][4, 4] = 0
        data['valid'][4, 5] = 0
        data['depth'][4, 6] = np.inf
        data['instance_id'][4, 7] = 999
        data['previous_valid'][5, 4] = 0
        data['previous_depth'][5, 5] = np.nan
        data['depth'][5, 6] = .5
        score = metrics.score_pose(**data)
        self.assertEqual(score['subsets']['reveal']['count'], 25)

    def test_exact_zero_error_mask_passes_without_zero_division(self):
        data = inputs()
        data['raw_rgb8'] = data['reference_rgb8'].copy()
        data['filtered_rgb8'] = data['reference_rgb8'].copy()
        score = metrics.score_pose(**data)
        report = metrics.summarize_scores([score], registered_pose_ids=['p0'],
                                          required_reveal_pose_ids=['p0'])
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['pooled']['fingers']['raw']['rmse'], 0.)
        self.assertTrue(report['gates']['fingers']['passed'])
        data['filtered_rgb8'] += 1
        score = metrics.score_pose(**data)
        report = metrics.summarize_scores([score], registered_pose_ids=['p0'],
                                          required_reveal_pose_ids=['p0'])
        self.assertEqual(report['status'], 'failed')
        self.assertIsNone(report['gates']['fingers']['rmse_ratio'])

    def test_all_empty_registered_targets_are_insufficient_not_perfect(self):
        data = inputs()
        data['instance_id'][:] = 9
        data['previous_instance_id'][:] = 9
        score = metrics.score_pose(**data)
        self.assertIsNone(score['subsets']['fingers']['raw']['rmse'])
        report = metrics.summarize_scores([score], registered_pose_ids=['p0'],
                                          required_reveal_pose_ids=['p0'])
        self.assertEqual(report['status'], 'insufficient_evidence')
        self.assertFalse(report['quality_claim_supported'])

    def test_missing_registered_pose_or_reveal_drops_claim(self):
        score = metrics.score_pose(**inputs())
        report = metrics.summarize_scores([score], registered_pose_ids=['p0', 'p1'],
                                          required_reveal_pose_ids=['p1'])
        self.assertEqual(report['status'], 'insufficient_evidence')
        self.assertIn('missing_pose:p1', report['insufficient_reasons'])

    def test_mean_bias_has_sign_and_uncertainty_is_not_iid_sem(self):
        score = metrics.score_pose(**inputs())
        for name in ('raw', 'filtered'):
            stat = score['subsets']['fingers'][name]
            self.assertGreater(stat['mean_bias'], 0)
            self.assertGreater(stat['mean_bias_luminance'], 0)
            self.assertEqual(len(stat['mean_bias_rgb']), 3)
            self.assertAlmostEqual(stat['pixel_error_sd'], 0., places=12)
        self.assertIsNone(score['uncertainty']['reference_standard_error'])
        self.assertEqual(score['subsets']['fingers']['rmse_change_sign'], 'improved')

    def test_pool_is_pixel_weighted_squared_error_not_mean_of_rmse(self):
        first = metrics.score_pose(**inputs('p0'))
        data = inputs('p1')
        data['raw_rgb8'][:] = 180
        data['filtered_rgb8'][:] = 110
        data['instance_id'][2:8, 2:10] = 9
        second = metrics.score_pose(**data)
        report = metrics.summarize_scores([first, second], registered_pose_ids=['p0', 'p1'],
                                          required_reveal_pose_ids=['p0'])
        a, b = [s['subsets']['fingers'] for s in (first, second)]
        expected = ((a['raw']['squared_error_sum'] + b['raw']['squared_error_sum']) /
                    (3 * (a['count'] + b['count']))) ** .5
        self.assertAlmostEqual(report['pooled']['fingers']['raw']['rmse'], expected)

    def test_integer_pose_identity_and_unchanged_later_batch_are_explicit(self):
        data = inputs(15)
        data['filtered_rgb8'] = data['raw_rgb8'].copy()
        score = metrics.score_pose(**data)
        report = metrics.summarize_scores([score], registered_pose_ids=[15],
                                          required_reveal_pose_ids=[15])
        self.assertEqual(score['pose_id'], 15)
        self.assertEqual(score['subsets']['fingers']['rmse_change_sign'], 'unchanged')
        self.assertTrue(score['subsets']['fingers']['filtered_equals_raw'])
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['gates']['fingers']['rmse_ratio'], 1.)

    def test_shape_type_and_nonfinite_foreground_rejected(self):
        for field, value in [('raw_rgb8', np.zeros((16, 24, 3), dtype=np.float32)),
                             ('reference_rgb8', np.zeros((1, 24, 3), dtype=np.uint8)),
                             ('valid', np.full((16, 24), 2, dtype=np.uint8))]:
            with self.subTest(field=field):
                data = inputs(); data[field] = value
                with self.assertRaises((TypeError, ValueError)):
                    metrics.score_pose(**data)
        data = inputs(); data['previous_depth'] = None
        with self.assertRaises(ValueError):
            metrics.score_pose(**data)

    def test_single_finger_missing_is_not_hidden_by_other_finger(self):
        data = inputs(); data['instance_id'][data['instance_id'] == 158] = 156
        report = metrics.summarize_scores([metrics.score_pose(**data)],
                                          registered_pose_ids=['p0'], required_reveal_pose_ids=['p0'])
        self.assertEqual(report['status'], 'insufficient_evidence')
        self.assertIn('pooled_target:finger_158', report['insufficient_reasons'])

    def test_worse_edge_is_not_hidden_by_better_finger_interior(self):
        data = inputs()
        masks = metrics.build_masks(instance_id=data['instance_id'], depth=data['depth'], valid=data['valid'])
        data['filtered_rgb8'][masks['edge']] = 240
        report = metrics.summarize_scores([metrics.score_pose(**data)],
                                          registered_pose_ids=['p0'], required_reveal_pose_ids=['p0'])
        self.assertTrue(report['gates']['fingers']['passed'])
        self.assertFalse(report['gates']['edge']['passed'])
        self.assertEqual(report['status'], 'failed')


if __name__ == '__main__':
    unittest.main()

