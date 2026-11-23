"""Frozen secondary-visibility control; HOST tests do not import a renderer."""
import importlib.util
from pathlib import Path
import unittest


def module():
    p=Path(__file__).with_name('study.py');s=importlib.util.spec_from_file_location('secondary_study',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


class SecondaryMotionContracts(unittest.TestCase):
    def test_primary_rays_are_exactly_unchanged_and_secondary_objects_are_offscreen(self):
        m=module()
        a=m.primary_witness(False);b=m.primary_witness(True)
        self.assertEqual(a,b);self.assertEqual(a['receiver_pixels'],96*96)
        self.assertEqual(a['secondary_primary_pixels'],0)
        self.assertEqual(a['min_depth'],1.);self.assertEqual(a['max_depth'],1.)

    def test_registered_gate_and_motion_are_fixed_before_results(self):
        m=module();p=m.protocol()
        self.assertEqual(p['roi'],[32,32,64,64]);self.assertEqual(p['opening_index'],8)
        self.assertEqual(p['frames'],16);self.assertEqual(p['reference_seeds'],list(range(100,132)))
        self.assertEqual(p['signal_min'],.01);self.assertEqual(p['stale_fraction_min'],.10)
        self.assertEqual(m.geometry(False)[0],m.geometry(True)[0])
        self.assertEqual(m.geometry(False)[1],m.geometry(True)[1])
        self.assertNotEqual(m.geometry(False)[2],m.geometry(True)[2])

    def test_weak_signal_does_not_claim_ghosting(self):
        m=module()
        closed=[[.001,0.,0.]]*32;opened=[[.002,0.,0.]]*32
        value=m.score(closed,opened,[[.001,0.,0.]]*2,[[.002,0.,0.]]*2)
        self.assertEqual(value['outcome'],'insufficient_secondary_signal')
        self.assertFalse(value['measured_stale_indirect'])

    def test_real_old_color_and_no_old_color_are_distinguished(self):
        m=module();closed=[[0.,0.,0.]]*32;opened=[[.1,0.,0.]]*32
        bad=m.score(closed,opened,[[.05,0.,0.]]*2,[[.1,0.,0.]]*2)
        self.assertTrue(bad['measured_stale_indirect']);self.assertAlmostEqual(bad['incremental_old_fraction'],.5)
        clean=m.score(closed,opened,[[.1,0.,0.]]*2,[[.1,0.,0.]]*2)
        self.assertFalse(clean['measured_stale_indirect'])
        self.assertEqual(clean['outcome'],'no_registered_stale_signal')

    def test_seed_disagreement_and_reference_uncertainty_are_not_a_positive(self):
        m=module();closed=[[0.,0.,0.]]*32;opened=[[.1,0.,0.]]*32
        mixed=m.score(closed,opened,[[.01,0.,0.],[.15,0.,0.]],[[.1,0.,0.]]*2)
        self.assertFalse(mixed['measured_stale_indirect'])
        noisy=m.score(closed,[[0.,0.,0.]]*31+[[3.2,0.,0.]],[[.02,0.,0.]]*2,[[.1,0.,0.]]*2)
        self.assertEqual(noisy['outcome'],'insufficient_secondary_signal')

    def test_missing_nonfinite_and_zero_reference_rejected_or_insufficient(self):
        m=module()
        for reference in ([],[[float('nan'),0.,0.]]*32):
            with self.assertRaises(ValueError):m.score(reference,[[.1,0.,0.]]*32,[[0.,0.,0.]]*2,[[0.,0.,0.]]*2)
        zero=m.score([[0.,0.,0.]]*32,[[0.,0.,0.]]*32,[[0.,0.,0.]]*2,[[0.,0.,0.]]*2)
        self.assertEqual(zero['outcome'],'insufficient_secondary_signal')


if __name__=='__main__':unittest.main()