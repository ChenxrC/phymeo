"""SHA-bound feature fixtures and an isolated installed-wheel acceptance launcher.

No backend is imported by the coordinator helpers. This script is copied outside
the checkout and invoked with the same new venv's Python -I. It runs the original
UV/HDRI protocols without modifying their sources, inputs, or acceptance gates.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import runpy
import struct
import sys
import unittest


FEATURE_FILES = (
    "examples/texture_calibration.py", "examples/_panda_scene.py",
    "examples/reference_hdri_probe.py", "examples/roughness_calibration.py",
    "examples/native_ibl_calibration.py", "tests/test_environment.py",
    "tools/_g5_installed.py",
)
KINDS = ("host", "uv-native", "uv-sapien-raster", "uv-sapien-pt", "ibl-native", "ibl-sapien")


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def copy_feature_bundle(rendering_root, destination):
    rendering = Path(rendering_root).resolve()
    target = Path(destination).resolve()
    source = rendering.parent
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("feature fixture must remain outside the source checkout")
    originals = {}
    for relative in FEATURE_FILES:
        original = (rendering/relative).resolve()
        if not original.is_relative_to(rendering) or not original.is_file():
            raise ValueError(f"missing or escaped feature fixture: {relative}")
        originals[relative] = original.read_bytes()
    target.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "phymeo-g5-feature-fixture-v1", "source_root": str(source), "files": {}}
    for relative, raw in originals.items():
        path = target/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
        manifest["files"][relative] = hashlib.sha256(raw).hexdigest()
    _save(target/"manifest.json", manifest)
    return manifest


def verify_feature_bundle(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema") != "phymeo-g5-feature-fixture-v1"
            or set(manifest.get("files", {})) != set(FEATURE_FILES)):
        raise ValueError("unexpected feature fixture members")
    for relative, expected in manifest["files"].items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"missing or escaped feature fixture member: {relative}")
        if _digest(path) != expected:
            raise ValueError(f"feature fixture SHA256 digest differs: {relative}")
    return manifest


def feature_environment(work_dir, inherited):
    environment = dict(inherited)
    for key in ("PYTHONPATH", "PYTHONHOME"):
        environment.pop(key, None)
    temporary = Path(work_dir).resolve()/"temporary"
    temporary.mkdir(parents=True, exist_ok=True)
    for key in ("TEMP", "TMP", "TMPDIR"):
        environment[key] = str(temporary)
    environment.update(PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONDONTWRITEBYTECODE="1")
    # Keep Windows profile/AppData and loader search path intact. Driver caches
    # there are external runtime resources, never accepted as wheel contents.
    return environment


def feature_command(python, launcher, kind, *, bundle, output, source_root,
                    library=None, device_index=1, sapien_device="cuda:0"):
    if kind not in KINDS:
        raise ValueError("unknown installed feature kind")
    command = [str(python), "-I", str(launcher), "--kind", kind, "--bundle", str(bundle),
               "--output", str(output), "--forbid-source", str(source_root)]
    if kind in ("uv-native", "ibl-native"):
        if library is None:
            raise ValueError("native feature requires its newly built library")
        command += ["--library", str(library), "--device-index", str(device_index)]
    if kind in ("uv-sapien-raster", "uv-sapien-pt", "ibl-sapien"):
        command += ["--sapien-device", sapien_device]
    return command


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _dependencies():
    prefix = Path(sys.prefix).resolve()
    observed = {}
    for name, distribution, version in (("numpy", "numpy", "1.26.4"),
                                        ("PIL", "Pillow", "12.2.0"), ("sapien", "sapien", "3.0.3")):
        if name not in sys.modules:
            continue
        module = sys.modules[name]
        path = Path(module.__file__).resolve()
        actual = importlib.metadata.version(distribution)
        if not path.is_relative_to(prefix) or actual != version:
            raise ValueError(f"{distribution} escaped the fixed external environment")
        observed[distribution] = {"version": actual, "file": str(path)}
    return observed


def _host(bundle, output):
    from phymeo_rendering.assets import TextureMap
    from phymeo_rendering.environment import RadianceEnvironment, load_environment, write_sapien_faces
    from phymeo_rendering.mesh_data import load_mesh
    from phymeo_rendering.texture_data import load_texture

    output.mkdir(parents=True, exist_ok=False)
    module = _module("g5_copied_environment_tests", bundle/"tests/test_environment.py")
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    (output/"environment-host.log").write_text(stream.getvalue(), encoding="utf-8")
    if not result.wasSuccessful() or result.testsRun != 6:
        raise RuntimeError("copied environment HOST protocol did not pass all six tests")
    uv = _module("g5_copied_texture_calibration", bundle/"examples/texture_calibration.py")
    scenes, locations = uv.plates(output/"uv-inputs")
    source = scenes[0].materials[0].base_color_map
    decoded = load_texture(source)
    if (decoded.width, decoded.height, decoded.rgb8) != (2, 2, uv.TEXELS.tobytes()):
        raise ValueError("installed decoder changed the sealed UV source pixels")
    for index, mesh in enumerate(scenes[0].meshes):
        expected = tuple(struct.unpack("<f", struct.pack("<f", value))[0] for value in uv.COORDINATES[index % 8])
        if load_mesh(mesh).texcoords != (expected,)*4:
            raise ValueError("installed mesh reader changed the sealed FLOAT VEC2 UV inputs")
    try:
        load_texture(TextureMap(source.uri, "0"*64))
    except ValueError as error:
        if "SHA256" not in str(error):
            raise
    else:
        raise ValueError("installed image decoder accepted an incorrect source hash")
    hdri = _module("g5_copied_hdri_fixture", bundle/"examples/reference_hdri_probe.py")
    raw = hdri.calibration_hdr()
    path = output/"calibration.hdr"
    path.write_bytes(raw)
    environment = RadianceEnvironment(path.as_uri(), _digest(path), yaw_degrees=90.)
    panorama = load_environment(environment)
    faces = output/"faces"
    faces.mkdir()
    paths, evidence = write_sapien_faces(environment, faces)
    # These are real decoded/prepared files, not a claim that GPU sampling ran.
    try:
        importlib.metadata.version("sapien")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise ValueError("independent feature HOST stage already contains SAPIEN")
    return {"status": "passed", "environment_host_tests": result.testsRun,
            "uv_plates": len(locations), "uv_source_sha256": source.sha256,
            "uv_rgb8_sha256": hashlib.sha256(decoded.rgb8).hexdigest(),
            "environment_shape": list(panorama.shape), "environment": evidence,
            "face_files": {p.name: _digest(p) for p in paths},
            "scope": "installed NumPy/Pillow preparation without SAPIEN or a render device"}


def _run_original(path, argv):
    old = list(sys.argv)
    try:
        sys.argv = [str(path), *argv]
        try:
            runpy.run_path(str(path), run_name="__main__")
        except SystemExit as error:
            if error.code not in (0, None):
                raise RuntimeError(f"sealed acceptance script exited with {error.code}") from error
    finally:
        sys.argv = old


def installed_feature(args):
    bundle, output, source = Path(args.bundle).resolve(), Path(args.output).resolve(), Path(args.forbid_source).resolve()
    if any(path.is_relative_to(source) for path in (bundle, output)):
        raise ValueError("installed feature input/output must remain outside the source checkout")
    if output.exists():
        raise ValueError("feature output must be a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = verify_feature_bundle(bundle)
    report_path = output.with_name(output.name+".installed.json")
    if report_path.exists():
        raise ValueError("feature launcher report must be new")
    report = {"schema": "phymeo-g5-installed-feature-v1", "kind": args.kind, "status": "running",
              "fixture_manifest_sha256": _digest(bundle/"manifest.json"), "fixture": manifest,
              "python": sys.executable, "output": str(output)}
    binding = _module("g5_copied_install_binding", bundle/"tools/_g5_installed.py")
    # -I drops the script directory. Add only the SHA-bound independent examples,
    # so their sibling fixture imports work without any product-source path.
    sys.path.insert(0, str(bundle/"examples"))
    try:
        report["before"] = binding.installed_binding(source)
        if args.kind == "host":
            report["result"] = _host(bundle, output)
        else:
            argv = ["--output", str(output)]
            if args.kind.startswith("uv-"):
                backend = {"uv-native": "native", "uv-sapien-raster": "sapien-raster",
                           "uv-sapien-pt": "sapien-pt"}[args.kind]
                if backend != "native" and args.sapien_device != "cuda:0":
                    raise ValueError("sealed UV reference protocol explicitly uses cuda:0")
                script, result_file = "texture_calibration.py", "result.json"
                argv += ["--backend", backend]
            elif args.kind == "ibl-native":
                script, result_file = "native_ibl_calibration.py", "summary.json"
            elif args.kind == "ibl-sapien":
                script, result_file = "reference_hdri_probe.py", "summary.json"
                argv += ["--device", args.sapien_device]
            else:
                raise ValueError("unknown installed feature kind")
            if args.kind in ("uv-native", "ibl-native"):
                library = Path(args.library).resolve()
                if not library.is_file() or library.is_relative_to(source):
                    raise ValueError("native library must be the source-external build")
                argv += ["--library", str(library), "--device-index", str(args.device_index)]
                report["native_library"] = {"file": str(library), "sha256": _digest(library)}
            report["script"] = script
            report["argv"] = argv
            _run_original(bundle/"examples"/script, argv)
            result = json.loads((output/result_file).read_text(encoding="utf-8"))
            if not (result.get("status") == "passed" or result.get("passed") is True):
                raise ValueError("sealed feature result did not pass")
            report["result"] = {"file": result_file, "sha256": _digest(output/result_file), "status": "passed"}
        report["dependencies"] = _dependencies()
        report["after"] = binding.installed_binding(source)
        verify_feature_bundle(bundle)
        report["files"] = {p.relative_to(output).as_posix(): _digest(p) for p in sorted(output.rglob("*")) if p.is_file()}
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        _save(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--forbid-source", required=True)
    parser.add_argument("--library")
    parser.add_argument("--device-index", type=int, default=1)
    parser.add_argument("--sapien-device", default="cuda:0")
    args = parser.parse_args()
    installed_feature(args)


if __name__ == "__main__":
    main()
