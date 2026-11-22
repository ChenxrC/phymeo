"""G5 host checks; device/build execution belongs to the separate offline driver."""
from __future__ import annotations

import base64
import csv
import fnmatch
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch
import zipfile

from phymeo_rendering import panda_visuals

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "assets" / "panda"
PREFIX = "phymeo_rendering/_panda_assets/"
DIST = "phymeo_rendering-0.1.0a0.dist-info/"


def driver():
    spec = importlib.util.spec_from_file_location("g5_reproduce", ROOT / "tools" / "g5_reproduce.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wheel_bytes(files):
    files = dict(files)
    stream = io.StringIO(newline="")
    rows = csv.writer(stream)
    for name, data in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rows.writerow((name, "sha256=" + digest, len(data)))
    rows.writerow((DIST + "RECORD", "", ""))
    files[DIST + "RECORD"] = stream.getvalue().encode()
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return result.getvalue()


class G5PackagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = {}
        for path in (ROOT / "src" / "phymeo_rendering").rglob("*"):
            if path.is_file() and path.suffix in {".py", ".cpp", ".h", ".inc", ".vert", ".frag", ".glsl"}:
                cls.files[path.relative_to(ROOT / "src").as_posix()] = path.read_bytes()
        cmake = ROOT / "src/phymeo_rendering/backends/_native/CMakeLists.txt"
        cls.files[cmake.relative_to(ROOT / "src").as_posix()] = cmake.read_bytes()
        for path in DATA.rglob("*"):
            if path.is_file():
                cls.files[PREFIX + path.relative_to(DATA).as_posix()] = path.read_bytes()
        cls.files[DIST + "WHEEL"] = b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        cls.files[DIST + "METADATA"] = (
            b"Metadata-Version: 2.4\nName: phymeo-rendering\nVersion: 0.1.0a0\n"
            b"Provides-Extra: sapien\nRequires-Dist: sapien==3.0.3; extra == 'sapien'\n"
            b"Requires-Dist: numpy==1.26.4; extra == 'sapien'\n\n"
        )

    def inspect(self, files=None):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "phymeo_rendering-0.1.0a0-py3-none-any.whl"
            path.write_bytes(wheel_bytes(self.files if files is None else files))
            return driver().inspect_wheel(path, ROOT)

    def test_package_maps_existing_assets_without_repository_copy(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        settings = config["tool"]["setuptools"]
        self.assertEqual(settings["package-dir"]["phymeo_rendering._panda_assets"], "assets/panda")
        self.assertFalse((ROOT / "src/phymeo_rendering/_panda_assets").exists())
        patterns = settings["package-data"]["phymeo_rendering._panda_assets"]
        for path in DATA.rglob("*"):
            if path.is_file():
                name = path.relative_to(DATA).as_posix()
                with self.subTest(resource=name):
                    self.assertTrue(any(fnmatch.fnmatchcase(name, p) for p in patterns))

    def test_all_native_includes_and_shader_sources_are_declared(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        patterns = config["tool"]["setuptools"]["package-data"]["phymeo_rendering.backends"]
        base = ROOT / "src/phymeo_rendering/backends"
        required = [p.relative_to(base).as_posix() for folder in ("_native", "_shaders")
                    for p in (base / folder).rglob("*") if p.is_file()]
        # A shared include must remain included even on a revision before it exists.
        required += ["_native/future.inc", "_shaders/future.glsl"]
        for name in required:
            with self.subTest(source=name):
                self.assertTrue(any(fnmatch.fnmatchcase(name, p) for p in patterns))

    def test_default_panda_directory_uses_installed_location_before_source(self):
        self.assertEqual(panda_visuals._default_asset_dir(), DATA)
        with tempfile.TemporaryDirectory() as temp:
            package = Path(temp) / "lib/site-packages/phymeo_rendering"
            assets = package / "_panda_assets"
            assets.mkdir(parents=True)
            (assets / "manifest.json").write_text("corrupted", encoding="utf-8")
            with patch.object(panda_visuals, "__file__", str(package / "panda_visuals.py")):
                self.assertEqual(panda_visuals._default_asset_dir(), assets)
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    panda_visuals.panda_visual_assets()
        # Existing explicit override and source-checkout use keep their identities.
        self.assertEqual(panda_visuals.panda_visual_assets(), panda_visuals.panda_visual_assets(DATA))

    def test_complete_source_wheel_is_inspected_without_optional_backend(self):
        report = self.inspect()
        self.assertEqual(report["schema"], "phymeo-rendering-wheel-audit-v1")
        self.assertEqual(report["panda"]["manifest_sha256"], panda_visuals._MANIFEST_SHA256)
        self.assertEqual(report["panda"]["instances"], 59)
        self.assertIn("phymeo_rendering/backends/_native/resident.inc", report["native_sources"])
        self.assertEqual(report["platform_binaries"], [])
        self.assertEqual(report["base_dependencies"], [])

    def test_source_wheel_rejects_missing_include_shader_or_asset(self):
        examples = ("phymeo_rendering/backends/_native/resident.inc",
                    "phymeo_rendering/backends/_shaders/native.vert",
                    PREFIX + "NOTICE", PREFIX + "original/hand.dae.gz")
        for name in examples:
            with self.subTest(missing=name):
                files = dict(self.files)
                del files[name]
                with self.assertRaisesRegex(ValueError, "missing"):
                    self.inspect(files)

    def test_source_wheel_rejects_forged_asset_even_with_valid_record(self):
        for name in (PREFIX + "manifest.json", PREFIX + "NOTICE", PREFIX + "original/hand.dae.gz"):
            with self.subTest(corrupt=name):
                files = dict(self.files)
                files[name] += b"\ncorruption"
                with self.assertRaisesRegex(ValueError, "digest|SHA256"):
                    self.inspect(files)

    def test_pure_tag_never_hides_compiled_platform_or_shader_binary(self):
        for suffix in ("dll", "pyd", "so", "dylib", "spv"):
            with self.subTest(suffix=suffix):
                files = dict(self.files)
                files["phymeo_rendering/backends/_native/hidden." + suffix] = b"binary"
                with self.assertRaisesRegex(ValueError, "binary"):
                    self.inspect(files)

    def test_wheel_record_and_base_dependency_claim_are_checked(self):
        files = dict(self.files)
        files[DIST + "METADATA"] += b"Requires-Dist: sapien==3.0.3\n"
        # Insert into headers, not the optional metadata description body.
        files[DIST + "METADATA"] = files[DIST + "METADATA"].replace(b"\n\n", b"\n")
        with self.assertRaisesRegex(ValueError, "base dependenc"):
            self.inspect(files)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.whl"
            raw = wheel_bytes(self.files)
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                with zipfile.ZipFile(path, "w") as changed:
                    for name in archive.namelist():
                        data = archive.read(name)
                        if name.endswith("panda_visuals.py"):
                            data += b"\n# changed after RECORD\n"
                        changed.writestr(name, data)
            with self.assertRaisesRegex(ValueError, "RECORD"):
                driver().inspect_wheel(path, ROOT)

    def test_source_external_workdir_rejects_source_and_ancestor(self):
        for path in (ROOT, ROOT / "artifacts/g5", ROOT.parent):
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "outside"):
                    driver().validate_external_workdir(path, ROOT.parent)
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(driver().validate_external_workdir(Path(temp) / "run", ROOT.parent),
                             (Path(temp) / "run").resolve())

    @unittest.skipUnless(os.environ.get("PHYMEO_G5_WHEEL"), "built wheel path is opt-in")
    def test_actual_built_wheel_matches_current_sources_and_pinned_assets(self):
        report = driver().inspect_wheel(Path(os.environ["PHYMEO_G5_WHEEL"]), ROOT)
        self.assertEqual(report["platform_binaries"], [])
        self.assertEqual(report["panda"]["instances"], 59)


if __name__ == "__main__":
    unittest.main()
