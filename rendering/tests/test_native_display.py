"""Optional resident display API; independent of device/GPU availability."""
from types import SimpleNamespace
import unittest
from phymeo_rendering.backends import BackendUnavailableError
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.backends._native_display import configure_display


class Function:
    def __init__(self, result):
        self.result, self.calls = result, []
    def __call__(self, *args):
        self.calls.append(args)
        return self.result


def device(version=1, compatible=1, status=0):
    functions = SimpleNamespace(pr_resident_display_transform_abi_version=Function(version),
        pr_resident_display_transform_shader_compatible=Function(compatible),
        pr_resident_set_display_transform=Function(status))
    return SimpleNamespace(_lib=functions, _handle=123, _error=lambda: 'fixture display rejection')


class NativeDisplayTests(unittest.TestCase):
    def test_public_default_and_explicit_choice(self):
        self.assertEqual(NativeBackend().display_transform, 'clip')
        self.assertEqual(NativeBackend(resident=True, display_transform='pbr-neutral').display_transform, 'pbr-neutral')
        self.assertEqual(NativeBackend(resident=True, display_transform='clip').display_transform, 'clip')

    def test_public_unknown_and_legacy_neutral_reject(self):
        for value in (None, True, 1, '', 'neutral', 'PBR-Neutral'):
            with self.subTest(value=value), self.assertRaisesRegex((ValueError, TypeError), 'display_transform'):
                NativeBackend(display_transform=value)
        with self.assertRaisesRegex(ValueError, 'resident'):
            NativeBackend(display_transform='pbr-neutral')

    def test_clip_never_touches_optional_library_symbols(self):
        class Untouchable:
            def __getattr__(self, name):
                raise AssertionError('default display accessed '+name)
        result=configure_display(Untouchable(), 'clip')
        self.assertEqual(result['display_transform'], 'clip')

    def test_neutral_selects_exact_abi_and_mode(self):
        target=device()
        result=configure_display(target, 'pbr-neutral')
        self.assertEqual(result['display_transform'], 'pbr-neutral')
        self.assertEqual(result['display_transform_abi'], 1)
        self.assertEqual(target._lib.pr_resident_set_display_transform.calls, [(123, 1)])

    def test_missing_symbols_explicitly_reject_without_fallback(self):
        for missing in ('pr_resident_display_transform_abi_version',
                        'pr_resident_display_transform_shader_compatible', 'pr_resident_set_display_transform'):
            target=device(); delattr(target._lib, missing)
            with self.subTest(missing=missing), self.assertRaisesRegex(BackendUnavailableError, 'display'):
                configure_display(target, 'pbr-neutral')

    def test_old_version_or_shader_cannot_reach_setter(self):
        for version, compatible in ((0,1),(2,1),(1,0),(1,-1)):
            target=device(version, compatible)
            with self.subTest(version=version, compatible=compatible), self.assertRaises(BackendUnavailableError):
                configure_display(target, 'pbr-neutral')
            self.assertEqual(target._lib.pr_resident_set_display_transform.calls, [])

    def test_actual_setter_error_is_not_downgraded_to_clip(self):
        with self.assertRaisesRegex(BackendUnavailableError, 'fixture display rejection'):
            configure_display(device(status=-1), 'pbr-neutral')

    def test_private_unknown_choice_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'display_transform'):
            configure_display(device(), 'unknown')


if __name__ == '__main__':
    unittest.main()
