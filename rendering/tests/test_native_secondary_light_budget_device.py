"""Opt-in public Vulkan smoke; no implicit device execution during HOST runs."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from test_secondary_light_budget_experiment import experiment


@unittest.skipUnless(os.environ.get('PHYMEO_NATIVE_SECONDARY_DLL'),'explicit secondary-budget device slot required')
class SecondaryBudgetDevice(unittest.TestCase):
    def test_matched_direct_thin_mirror_and_zero_controls(self):
        module=experiment()
        with TemporaryDirectory() as temporary:
            library=Path(os.environ['PHYMEO_NATIVE_SECONDARY_DLL'])
            report=module.run(library,Path(temporary)/'result',
                shaders=Path(os.environ.get('PHYMEO_NATIVE_SECONDARY_SHADERS',str(library.parent))),
                device_index=int(os.environ.get('PHYMEO_NATIVE_DEVICE_INDEX','1')),validation=True,
                cases=('direct','thin','mirror','zero','single'))
            self.assertEqual(report['status'],'complete')
            self.assertTrue(report['functional_passed'])
            self.assertTrue(report['all_cleanup_passed'])
            self.assertFalse(report['panda_acceptance_claimed'])
            self.assertIsNone(report['room_quality'])
            self.assertTrue(report['wrong_primary_negative_control_detected'])


if __name__=='__main__':unittest.main()
