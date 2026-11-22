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
    "examples/reference_neutral_calibration.py", "examples/native_resident_smoke.py",
    "tests/fixtures/g1/table.gltf", "tests/fixtures/g1/block.gltf",
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



class G5ExtendedFeatureTests(unittest.TestCase):
    def test_default_plan_requires_all_measured_features_and_keeps_basic_paths_explicit(self):
        module = features()
        plan = module.plan_features(("native-raster", "sapien-raster", "sapien-pt"))
        self.assertEqual(set(plan), {"host", "uv-native", "uv-sapien-raster", "uv-sapien-pt",
                                    "ibl-native", "ibl-sapien", "roughness-native",
                                    "roughness-sapien-raster", "roughness-sapien-pt",
                                    "neutral-sapien", "ray-query-native"})
        self.assertTrue(all(row["status"] == "required" for row in plan.values()))
        selected = module.plan_features(("native-raster",))
        self.assertEqual(selected["ray-query-native"]["status"], "required")
        self.assertEqual(selected["neutral-sapien"]["status"], "not_verified")
        self.assertIn("path", selected["neutral-sapien"]["reason"])

    def test_explicit_skip_requires_a_reason_and_never_masquerades_as_a_pass(self):
        module = features()
        paths = ("native-raster", "sapien-raster", "sapien-pt")
        plan = module.plan_features(paths, skip_features=("ray-query-native:not in this release",))
        self.assertEqual(plan["ray-query-native"],
                         {"status": "not_verified", "reason": "not in this release", "explicit_skip": True})
        self.assertEqual(plan["roughness-native"]["status"], "required")
        for invalid in (("ray-query-native",), ("ray-query-native:",), ("unknown:reason",),
                        ("host:reason",), ("neutral-sapien:a", "neutral-sapien:b")):
            with self.subTest(skip=invalid), self.assertRaises(ValueError):
                module.plan_features(paths, skip_features=invalid)

    def test_new_installed_commands_bind_native_library_or_reference_device(self):
        module = features()
        for kind in ("roughness-native", "roughness-sapien-raster", "roughness-sapien-pt",
                     "neutral-sapien", "ray-query-native"):
            with self.subTest(kind=kind):
                command = module.feature_command("python", "launcher.py", kind, bundle="bundle",
                    output="output", source_root="source", library="installed.dll", device_index=7,
                    sapien_device="cuda:0")
                self.assertEqual(command[1], "-I")
                self.assertEqual(command[command.index("--kind")+1], kind)
                if kind in ("roughness-native", "ray-query-native"):
                    self.assertEqual(command[command.index("--library")+1], "installed.dll")
                    self.assertEqual(command[command.index("--device-index")+1], "7")
                else:
                    self.assertEqual(command[command.index("--sapien-device")+1], "cuda:0")
        with self.assertRaisesRegex(ValueError, "library"):
            module.feature_command("python", "launcher.py", "ray-query-native", bundle="bundle",
                                   output="output", source_root="source")

    def test_bounded_optional_bundle_omits_only_explicitly_unrequested_protocols(self):
        module = features()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rendering = root/"source"/"rendering"
            for relative in FILES:
                path = rendering/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(("# exact original "+relative).encode())
            plan = module.plan_features(("native-raster", "sapien-raster", "sapien-pt"),
                skip_features=("neutral-sapien:deferred release", "ray-query-native:deferred release"))
            enabled = [kind for kind, row in plan.items() if row["status"] == "required"]
            # Absent, unselected examples cannot break an older basic candidate.
            (rendering/"examples/reference_neutral_calibration.py").unlink()
            (rendering/"examples/native_resident_smoke.py").unlink()
            target = root/"outside"/"bundle"
            manifest = module.copy_feature_bundle(rendering, target, enabled=enabled)
            self.assertNotIn("examples/reference_neutral_calibration.py", manifest["files"])
            self.assertNotIn("examples/native_resident_smoke.py", manifest["files"])
            self.assertEqual(set(manifest["enabled_features"]), set(enabled))
            self.assertEqual(module.verify_feature_bundle(target), manifest)
            # Expanding the claim without adding its original files must fail.
            manifest["enabled_features"].append("neutral-sapien")
            (target/"manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "fixture|member"):
                module.verify_feature_bundle(target)

    def test_saved_feature_results_cannot_claim_different_chart_or_less_coverage(self):
        import copy
        module = features()
        roughness = {"passed": True, "backend": "native", "role": "roughness",
                     "cells": [{"max_absolute_rgb8": 1} for _ in range(16)],
                     "sample_count": 784, "max_absolute_rgb8": 1, "oracle_color_span": 70.,
                     "sensor_differences": {"forward_depth_m": False},
                     "limits": {"max_rgb8": 3, "min_span": 40}}
        module.validate_feature_result("roughness-native", roughness)
        for change in ({"role": "color"}, {"cells": roughness["cells"][:-1]},
                       {"sample_count": 735}, {"max_absolute_rgb8": 4},
                       {"oracle_color_span": 0.}, {"limits": {"max_rgb8": 4, "min_span": 40}},
                       {"sensor_differences": {"depth": True}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                module.validate_feature_result("roughness-native", dict(roughness, **change))
        neutral = {"schema": "phymeo-reference-neutral-result-v1", "status": "passed",
                   "captures": [{"path": path, "display_transform": transform, "exposure": 8.}
                                for path in ("raster", "pt") for transform in ("clip", "pbr-neutral")],
                   "raster_sensors_unchanged": True, "raw_readbacks_unchanged": {"raster": True, "pt": True},
                   "pt_foreground_recovered_pixels": 100}
        module.validate_feature_result("neutral-sapien", neutral)
        for key, value in (("captures", neutral["captures"][:3]), ("raster_sensors_unchanged", False),
                           ("pt_foreground_recovered_pixels", 63),
                           ("raw_readbacks_unchanged", {"raster": True, "pt": False})):
            bad = copy.deepcopy(neutral); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.validate_feature_result("neutral-sapien", bad)

    def test_reused_protocol_arguments_are_exact_and_do_not_lower_acceptance_gates(self):
        module = features()
        for kind, backend in (("roughness-native", "native"), ("roughness-sapien-raster", "sapien-raster"),
                              ("roughness-sapien-pt", "sapien-pt")):
            script, result, argv = module.protocol_command(kind, "capture", library="installed.dll",
                                                          device_index=7, sapien_device="cuda:0")
            self.assertEqual(script, "texture_calibration.py")
            self.assertEqual(result, "result.json")
            self.assertEqual(argv[argv.index("--role")+1], "roughness")
            self.assertEqual(argv[argv.index("--backend")+1], backend)
            self.assertNotIn("--tolerance", argv)
        script, result, argv = module.protocol_command("neutral-sapien", "capture", sapien_device="cuda:0")
        self.assertEqual((script, result), ("reference_neutral_calibration.py", "summary.json"))
        self.assertEqual(argv, ["--output", "capture", "--device", "cuda:0"])

if __name__ == "__main__":
    unittest.main()
