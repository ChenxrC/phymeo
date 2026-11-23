"""Independent registration/statistics checks for the bounded secondary-light trial."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


def experiment():
    path=Path(__file__).resolve().parents[1]/'examples/secondary_light_budget.py'
    spec=importlib.util.spec_from_file_location('secondary_light_budget',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class SecondaryBudgetExperiment(unittest.TestCase):
    def test_registration_preserves_original_budgets_and_separate_evidence(self):
        p=experiment().protocol()
        self.assertEqual(p['seeds'],[0,1,2,3]);self.assertEqual(p['batches'],16)
        self.assertEqual(p['budgets'],[[8,8],[8,2]])
        self.assertEqual(p['relative_room_energy_limit'],.05)
        self.assertEqual(p['primary_samples_per_batch'],16)
        self.assertFalse(p['panda_quality_proven_by_this_trial'])
        self.assertEqual(p['thin_indirect']['seeds'],[0])
        self.assertEqual(p['thin_indirect']['bounces'],2)
        self.assertFalse(p['thin_indirect']['quality_acceptance_claimed'])

    def test_energy_reports_seed_spread_and_insufficient_dark_reference(self):
        m=experiment();same=m.energy_comparison([1.,1.,1.,1.],[1.,1.,1.,1.])
        self.assertEqual(same['status'],'passed')
        self.assertEqual(same['relative_change'],0.)
        bad=m.energy_comparison([1.]*4,[2.]*4)
        self.assertEqual(bad['status'],'failed')
        dark=m.energy_comparison([0.]*4,[0.]*4)
        self.assertEqual(dark['status'],'insufficient')
        noisy=m.energy_comparison([0.,0.,0.,1.],[0.,0.,0.,1.])
        self.assertEqual(noisy['status'],'insufficient')

    def test_no_mismatched_seed_or_nonfinite_statistics(self):
        m=experiment()
        for left,right in (([1.],[1.]*4),([1.]*4,[float('nan')]*4),([],[])):
            with self.assertRaises(ValueError):m.energy_comparison(left,right)

    def test_sensors_and_primary_rgb_guard_catches_wrong_main_budget(self):
        m=experiment();a={'RGB':b'\x01\x02\x03','FORWARD_DEPTH':b'd','INSTANCE_ID':b'i','VALID':b'v'}
        self.assertTrue(m.identical_primary(a,dict(a)))
        self.assertFalse(m.identical_primary(a,dict(a,RGB=b'\x02\x02\x03')))
        self.assertFalse(m.identical_primary(a,dict(a,INSTANCE_ID=b'j')))
        self.assertTrue(m.wrong_primary_detected(a,dict(a,RGB=b'\x02\x02\x03')))
        self.assertFalse(m.wrong_primary_detected(a,dict(a)))
        self.assertFalse(m.wrong_primary_detected(a,dict(a,INSTANCE_ID=b'j')))
        self.assertFalse(m.wrong_primary_detected(a,dict(a,RGB=b'\x02\x02\x03',INSTANCE_ID=b'j')))

    def test_cases_use_true_room_thin_occluder_and_mirror_material(self):
        import sys
        root=Path(__file__).resolve().parents[1]/'examples'
        sys.path.insert(0,str(root))
        try:
            m=experiment()
            with TemporaryDirectory() as directory:
                room=m.build_case(Path(directory)/'room','room')
                self.assertEqual(len(room[0].instances),8)
                thin=m.build_case(Path(directory)/'thin','thin',phase=1)
                self.assertEqual(thin[1].nodes[-1].world_pose.position_m, (.12,0.,0.))
                indirect=m.build_case(Path(directory)/'thin-indirect','thin-indirect',phase=-1)
                self.assertEqual(indirect[1].nodes[-1].world_pose.position_m,(-.12,0.,0.))
                self.assertEqual(indirect[0].meshes[-1].id,thin[0].meshes[-1].id)
                mirror=m.build_case(Path(directory)/'mirror','mirror')
                material=next(x for x in mirror[0].materials if x.id=='back')
                self.assertEqual((material.metallic,material.roughness),(1.,.04))
                single=m.build_case(Path(directory)/'single','single')
                self.assertEqual(len(single[0].lights),1)
                self.assertEqual(single[3].panel_strength,0.)
        finally:sys.path.remove(str(root))


if __name__=='__main__':unittest.main()
