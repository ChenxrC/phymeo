"""Opt-in real area-grid variants; a failed quality target is a saved counterexample."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from test_area_light_quadrature_experiment import experiment


@unittest.skipUnless(os.environ.get('PHYMEO_NATIVE_AREA_DLL'), 'set PHYMEO_NATIVE_AREA_DLL for real grid variants')
class AreaGridDevice(unittest.TestCase):
    def test_analytic_profile_finite_visibility_and_same_source_sensors(self):
        module = experiment()
        library = Path(os.environ['PHYMEO_NATIVE_AREA_DLL'])
        original = Path(os.environ['PHYMEO_NATIVE_AREA_BASELINE_DLL'])
        args = SimpleNamespace(library=library,shaders=Path(os.environ.get('PHYMEO_NATIVE_AREA_SHADERS',str(library.parent))),
            baseline_library=original,baseline_shaders=Path(os.environ.get('PHYMEO_NATIVE_AREA_BASELINE_SHADERS',str(original.parent))),
            device_index=int(os.environ.get('PHYMEO_NATIVE_DEVICE_INDEX','1')))
        with TemporaryDirectory() as directory:
            report = dict(registration=module.load_registration())
            module.analytic(args,Path(directory),report)
            self.assertTrue(report['default_output_preserved'])
            self.assertEqual(len(report['analytic_captures']),48)
            self.assertEqual(len(report['backface_controls']),2)
            self.assertTrue(all(r['sensor_equivalent'] for r in report['analytic_captures']))
            measurements = [r for r in report['analytic_captures'] if r['grid']==8 and 'error_acceptance' in r]
            self.assertEqual(len(measurements),6)
            self.assertIsInstance(report['quality_passed'],bool)
            # Do not redefine quality acceptance to make the test pass. This is
            # a pipeline/integrity test; quality failures remain in the report.
            self.assertEqual(report['quality_passed'],all(r['all_preregistered_gates_passed'] for r in measurements))


if __name__ == '__main__':
    unittest.main()
