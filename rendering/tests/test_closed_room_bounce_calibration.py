"""Tests-first closed-room geometry, order energy and variance contracts."""
import importlib.util
import math
from pathlib import Path
import sys
import unittest

EXAMPLES=Path(__file__).resolve().parents[1]/'examples'
sys.path.insert(0,str(EXAMPLES))
spec=importlib.util.spec_from_file_location('closed_room_bounce_calibration',EXAMPLES/'closed_room_bounce_calibration.py')
chart=importlib.util.module_from_spec(spec);spec.loader.exec_module(chart)


class ClosedRoomTests(unittest.TestCase):
    def test_six_true_faces_form_a_closed_inward_box(self):
        walls=chart.geometry()[:6]
        self.assertEqual(len(walls),6)
        edges={}
        for wall in walls:
            points=chart.vertices(wall)
            self.assertGreater(chart.dot(wall['normal'],chart.sub((0.,0.,1.),points[0])),0.)
            for a,b in zip(points,points[1:]+points[:1]):
                edge=tuple(sorted((a,b)));edges[edge]=edges.get(edge,0)+1
        self.assertEqual(len(edges),12)
        self.assertEqual(set(edges.values()),{2})

    def test_camera_is_inside_and_all_primary_rays_have_a_valid_first_hit(self):
        self.assertTrue(chart.inside_room(chart.EYE))
        for y in range(chart.HEIGHT):
            for x in range(chart.WIDTH):
                identity,depth=chart.primary_hit(x,y)
                self.assertIn(identity,range(1,9))
                self.assertGreater(depth,chart.NEAR)
                self.assertLess(depth,chart.FAR)

    def test_fixed_white_and_red_rois_have_the_expected_finite_surface(self):
        for name,roi in chart.ROIS.items():
            expected=7 if name=='red' else 5
            for x,y in chart.roi_pixels(roi):
                self.assertEqual(chart.primary_hit(x,y)[0],expected,name)
        self.assertNotEqual(chart.ROIS['bright'],chart.ROIS['shadow'])

    def test_both_finite_emitters_are_inside_without_removing_wall_geometry(self):
        panels=chart.panels()
        self.assertEqual(len(panels),2)
        for p in panels:
            self.assertTrue(all(chart.inside_room(c) for c in chart.panel_corners(p)))
            self.assertGreater(p['radiance'],0)
            self.assertGreater(chart.dot(chart.cross(p['half_u'],p['half_v']),p['direction']),0)
        self.assertEqual(len(chart.geometry()[:6]),6)

    def test_reference_and_native_camera_axes_match_hand_derived_ray(self):
        self.assertEqual(chart.primary_ray(47.5,47.5),(0.,1.,0.))
        self.assertGreater(chart.primary_ray(60,48)[0],0)
        self.assertLess(chart.primary_ray(48,60)[2],0)

    def test_equal_sample_pair_difference_separates_variance_from_constant_bias(self):
        a=[1.,1.2,.8,1.];b=[1.,.8,1.2,1.]
        base=chart.pair_noise(a,b)
        shifted=chart.pair_noise([x+5 for x in a],[x+5 for x in b])
        self.assertAlmostEqual(base,shifted,places=14)
        self.assertEqual(chart.pair_noise([2.]*4,[2.]*4),0.)
        self.assertGreater(chart.errors([2.]*4,[1.]*4)['relative_mean_bias'],.9)

    def test_fixed_variance_reduction_gate_rejects_unchanged_noise(self):
        self.assertTrue(chart.variance_gate({1:1.,4:.5,16:.25})['passed'])
        self.assertFalse(chart.variance_gate({1:1.,4:1.,16:1.})['passed'])
        with self.assertRaises(ValueError):chart.variance_gate({1:0.,4:0.,16:0.})

    def test_order_gate_requires_actual_positive_second_bounce_energy(self):
        self.assertTrue(chart.order_gate({0:.1,1:.2,2:.25,6:.3},native={1:.18,2:.22})['passed'])
        self.assertFalse(chart.order_gate({0:.1,1:.2,2:.2,6:.3},native={1:.18,2:.18})['passed'])
        self.assertFalse(chart.order_gate({0:.1,1:.2,2:.15,6:.3},native={1:.18,2:.22})['passed'])

    def test_hdr_readback_contract_rejects_wrong_space_batch_or_byte_count(self):
        payload=dict(width=2,height=2,batches=4,primary_samples_per_pixel=64,format='RGBA32F_LE',
                     space='scene-linear-before-exposure',data=b'\0'*64)
        chart.validate_mean(payload,2,2,4)
        for key,value in (('data',b'\0'*63),('batches',3),('space','display-linear')):
            bad=dict(payload);bad[key]=value
            with self.assertRaises(ValueError):chart.validate_mean(bad,2,2,4)

    def test_source_contract_exposes_real_sdk_interior_scope_and_no_seed_fabrication(self):
        p=chart.protocol()
        self.assertEqual(p['pt_seeds'],[0,1,2,3])
        self.assertEqual(p['pt_depths'],[0,1,2,6])
        self.assertEqual(p['pt_spp'],512)
        self.assertEqual(p['native_seeds'],[0,1])
        self.assertEqual(p['native_batches'],[1,4,16])
        self.assertEqual(p['native_bounces'],[1,2])
        self.assertFalse(p['reference_uses_public_pt_admission'])
        self.assertEqual(p['fixed_energy_bias_limit'],.35)
        self.assertEqual(p['environment_radiance'],0.)

    def test_sensor_contract_is_byte_exact(self):
        before={c:b'1234' for c in chart.SENSORS}
        self.assertTrue(chart.sensor_equal(before,dict(before)))
        after=dict(before);after['INSTANCE_ID']=b'1235'
        self.assertFalse(chart.sensor_equal(before,after))


if __name__=='__main__':unittest.main()
