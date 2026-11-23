"""HOST-only interpretation of native MIS capability and active-upload receipts."""
from copy import deepcopy
import unittest

from phymeo_rendering.backends._native_environment import ResidentEnvironmentInfo
from phymeo_rendering.backends._native_indirect import indirect_declaration
from phymeo_rendering.backends._native_progressive import ProgressiveHistory
from phymeo_rendering.backends._native_resident import ResidentDevice
from phymeo_rendering.backends.native import _NativeDelegate
from test_native_resident import Recorder, fixture


def native_info(compatible=0, active=0, *, loaded=1):
    value = ResidentEnvironmentInfo()
    value.environment_abi_version = 1
    value.supported = value.shader_compatible = 1
    value.loaded = loaded
    value.reserved[0] = compatible
    value.reserved[1] = active
    return value


def decoded(value):
    device = ResidentDevice.__new__(ResidentDevice)
    device._remember_environment_info(value)
    return device._environment_info


class EnvironmentMisDiagnostics(unittest.TestCase):
    def test_genuine_legacy_reserved_zero_remains_valid(self):
        result = decoded(native_info())
        self.assertIs(result['importance_shader_compatible'], False)
        self.assertIs(result['importance_active'], False)
        self.assertEqual(result['loaded'], 1)

    def test_compatible_is_distinct_from_active_nonconstant_upload(self):
        capable = decoded(native_info(1, 0))
        active = decoded(native_info(1, 1))
        self.assertIs(capable['importance_shader_compatible'], True)
        self.assertIs(capable['importance_active'], False)
        self.assertIs(active['importance_shader_compatible'], True)
        self.assertIs(active['importance_active'], True)
        self.assertIn('configuration', active['importance_scope'])

    def test_inconsistent_or_unknown_flags_do_not_replace_last_good_receipt(self):
        device = ResidentDevice.__new__(ResidentDevice)
        device._remember_environment_info(native_info(1, 1))
        saved = deepcopy(device._environment_info)
        for value in (native_info(0, 1), native_info(1, 1, loaded=0),
                      native_info(2, 0), native_info(1, 2)):
            with self.subTest(flags=tuple(value.reserved)):
                with self.assertRaisesRegex(RuntimeError, 'importance'):
                    device._remember_environment_info(value)
                self.assertEqual(device._environment_info, saved)

    def delegate(self, information, *, bounces=1, enabled=True, progressive=True):
        scene, mesh = fixture()
        recorder = Recorder()
        old_info = recorder.info
        recorder.info = lambda: dict(old_info(), resident_environment_info=information)
        declaration = indirect_declaration('diffuse_bounce' if enabled else 'none', 7)
        declaration['bounces'] = bounces if enabled else 0
        history = ProgressiveHistory(bounces) if progressive else None
        return _NativeDelegate(scene, {'shared': mesh}, recorder, 1., resident=True,
                               shadow_mode='ray_query', indirect_info=declaration,
                               progressive_history=history)

    def test_public_active_estimator_is_eight_plus_eight_and_one_secondary_proposal(self):
        owner = self.delegate(decoded(native_info(1, 1)), bounces=2)
        before = deepcopy(owner.indirect_info)
        result = owner.diagnostics()
        sampling = result['environment_sampling']
        self.assertEqual(sampling['mode'], 'cosine-environment-balance-mis')
        self.assertIs(sampling['active'], True)
        self.assertEqual(sampling['primary_cosine_samples'], 8)
        self.assertEqual(sampling['primary_environment_samples'], 8)
        self.assertEqual(sampling['second_hop_proposal'], 'equal-probability cosine/environment mixture')
        self.assertIn('balance MIS', result['indirect']['estimator'])
        self.assertNotIn('4x4 stratified', result['indirect']['estimator'])
        self.assertEqual(owner.indirect_info, before)
        self.assertEqual(result['progressive']['batches'], 0)

    def test_cdf_and_progressive_descriptions_match_actual_modes(self):
        information = decoded(native_info(1, 1))
        self.assertIn('cubemap CDF', information['importance_mass_scope'])
        self.assertNotIn('four-child hierarchy', information['importance_mass_scope'])
        for bounces in (1, 2):
            for mode in ('none', 'guided', 'guided_settling'):
                with self.subTest(bounces=bounces, spatial_filter=mode):
                    owner = self.delegate(information, bounces=bounces)
                    if mode != 'none':
                        owner.spatial_info = {'mode': mode}
                    result = owner.diagnostics()
                    indirect = result['indirect']
                    self.assertIn('adjacent FP32 cubemap CDF difference',
                                  result['environment_sampling']['density'])
                    self.assertIn('cubemap CDF', indirect['estimator'])
                    self.assertIn('cosine/environment', indirect['geometry_policy'])
                    self.assertIn('oriented true face', indirect['geometry_policy'])
                    self.assertIn('denominator stays 16', indirect['geometry_policy'])
                    self.assertIn('final hit never recurses', indirect['terminal_lighting'])
                    self.assertIn('two diffuse bounces' if bounces == 2 else 'one diffuse bounce',
                                  indirect['terminal_lighting'])
                    if bounces == 2:
                        self.assertIn('cosine/environment mixture', indirect['terminal_lighting'])
                    self.assertIn('state-reset GPU linear mean', indirect['limitations'])
                    self.assertIn('raw RGBA32F', indirect['limitations'])
                    self.assertIn('display-only ' + mode, indirect['limitations'])
                    for stale in ('no denoiser', 'no accumulation',
                                  'no raw-HDR', 'no multiple diffuse'):
                        self.assertNotIn(stale, indirect['limitations'])
                    self.assertIn('0-based batch', indirect['seed_scope'])

    def test_compatible_constant_and_old_uploads_keep_cosine_estimator(self):
        for information in (None, decoded(native_info()), decoded(native_info(1, 0))):
            with self.subTest(information=information):
                owner = self.delegate(information)
                result = owner.diagnostics()
                sampling = result['environment_sampling']
                self.assertIs(sampling['active'], False)
                self.assertEqual(sampling['mode'], 'stratified-cosine')
                self.assertEqual((sampling['primary_cosine_samples'], sampling['primary_environment_samples']), (16, 0))
                self.assertEqual(result['indirect']['estimator'], owner.indirect_info['estimator'])
                self.assertEqual(sampling['native_evidence_available'], information is not None)

    def test_no_indirect_does_not_claim_cosine_or_mis_work(self):
        result = self.delegate(decoded(native_info()), enabled=False, progressive=False).diagnostics()
        sampling = result['environment_sampling']
        self.assertEqual(sampling['mode'], 'not-applicable')
        self.assertEqual((sampling['primary_cosine_samples'], sampling['primary_environment_samples']), (0, 0))
        self.assertIs(sampling['active'], False)
        self.assertNotIn('estimator', result['indirect'])

    def test_live_receipt_changes_are_not_hidden_by_constructor_capabilities(self):
        information = decoded(native_info(1, 0, loaded=0))
        owner = self.delegate(information)
        self.assertIs(owner.diagnostics()['environment_sampling']['active'], False)
        information.update(decoded(native_info(1, 1)))
        self.assertIs(owner.diagnostics()['environment_sampling']['active'], True)


if __name__ == '__main__':
    unittest.main()
