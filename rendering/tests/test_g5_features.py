"""Source-external feature acceptance plumbing; no device or project subprocess."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "examples/texture_calibration.py", "examples/_panda_scene.py",
    "examples/reference_hdri_probe.py", "examples/roughness_calibration.py",
    "examples/native_ibl_calibration.py", "tests/test_environment.py",
    "tools/_g5_installed.py",
)


def features():
    spec = importlib.util.spec_from_file_location("g5_feature_smoke", ROOT/"tools/g5_feature_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class G5FeatureBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rendering = self.root/"candidate"/"rendering"
        for relative in FILES:
            path = self.rendering/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("# fixture identity: "+relative+"\n").encode())

    def test_copies_exact_independent_fixtures_and_records_every_digest(self):
        target = self.root/"outside"/"fixture"
        manifest = features().copy_feature_bundle(self.rendering, target)
        self.assertEqual(manifest["schema"], "phymeo-g5-feature-fixture-v1")
        self.assertEqual(set(manifest["files"]), set(FILES))
        for relative in FILES:
            raw = (self.rendering/relative).read_bytes()
            self.assertEqual((target/relative).read_bytes(), raw)
            self.assertEqual(manifest["files"][relative], hashlib.sha256(raw).hexdigest())
        self.assertEqual(features().verify_feature_bundle(target), manifest)
        self.assertFalse((target/"src").exists())

    def test_modified_or_missing_fixture_is_rejected_before_execution(self):
        target = self.root/"outside"/"fixture"
        features().copy_feature_bundle(self.rendering, target)
        original = (target/FILES[0]).read_bytes()
        (target/FILES[0]).write_bytes(original+b"# modified\n")
        with self.assertRaisesRegex(ValueError, "digest|SHA"):
            features().verify_feature_bundle(target)
        (target/FILES[0]).write_bytes(original)
        (target/FILES[-1]).unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            features().verify_feature_bundle(target)

    def test_bundle_cannot_live_in_source_or_smuggle_additional_sources(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            features().copy_feature_bundle(self.rendering, self.rendering/"artifacts"/"fixture")
        target = self.root/"outside"/"fixture"
        features().copy_feature_bundle(self.rendering, target)
        path = target/"manifest.json"
        manifest = json.loads(path.read_text())
        manifest["files"]["../product.py"] = "a"*64
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "fixture|member"):
            features().verify_feature_bundle(target)

    def test_installed_commands_preserve_isolated_python_and_bound_native_library(self):
        module = features()
        for kind in ("host", "uv-native", "uv-sapien-raster", "uv-sapien-pt", "ibl-native", "ibl-sapien"):
            with self.subTest(kind=kind):
                command = module.feature_command(
                    Path("C:/outside/env/Scripts/python.exe"), Path("C:/outside/feature_launcher.py"), kind,
                    bundle=Path("C:/outside/fixture"), output=Path("C:/outside/captures")/kind,
                    source_root=Path("C:/candidate"), library=Path("C:/outside/native-build/phymeo_native.dll"),
                    device_index=1, sapien_device="cuda:0")
                self.assertEqual(command[1], "-I")
                self.assertNotIn("-c", command)
                self.assertEqual(command[command.index("--kind")+1], kind)
                self.assertEqual(command[command.index("--forbid-source")+1], str(Path("C:/candidate")))
                if kind in ("uv-native", "ibl-native"):
                    self.assertEqual(command[command.index("--library")+1],
                                     str(Path("C:/outside/native-build/phymeo_native.dll")))
                    self.assertEqual(command[command.index("--device-index")+1], "1")
        with self.assertRaisesRegex(ValueError, "library"):
            module.feature_command(Path("python"), Path("launcher.py"), "ibl-native",
                                   bundle=self.root/"fixture", output=self.root/"capture",
                                   source_root=self.rendering.parent)

    def test_environment_owns_temp_but_preserves_windows_profile_and_loader(self):
        original = {"PYTHONPATH": "C:/source", "PYTHONHOME": "C:/old",
                    "USERPROFILE": "C:/Users/real-user", "LOCALAPPDATA": "C:/Users/real-user/AppData/Local",
                    "APPDATA": "C:/Users/real-user/AppData/Roaming", "HOMEDRIVE": "C:", "HOMEPATH": "/Users/real-user",
                    "PATH": "C:/Windows/System32", "VK_LAYER_PATH": "C:/sdk/validation", "TEMP": "C:/old-temp"}
        before = dict(original)
        environment = features().feature_environment(self.root/"external-run", original)
        self.assertEqual(original, before)
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("PYTHONHOME", environment)
        for key in ("USERPROFILE", "LOCALAPPDATA", "APPDATA", "HOMEDRIVE", "HOMEPATH", "PATH", "VK_LAYER_PATH"):
            self.assertEqual(environment[key], original[key])
        self.assertEqual(environment["PIP_NO_INDEX"], "1")
        self.assertEqual(environment["TEMP"], environment["TMP"])
        self.assertEqual(environment["TEMP"], environment["TMPDIR"])
        self.assertTrue(Path(environment["TEMP"]).resolve().is_relative_to((self.root/"external-run").resolve()))


if __name__ == "__main__":
    unittest.main()
