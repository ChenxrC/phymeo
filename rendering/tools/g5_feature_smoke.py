"""SHA-bound feature fixtures and an isolated installed-wheel acceptance launcher.

No backend is imported by the coordinator helpers. This script is copied outside
the checkout and invoked with the same new venv's Python -I. It runs the original
UV/HDRI/roughness/Neutral/shadow protocols without changing their acceptance gates.
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


BASE_FEATURE_FILES = (
    "examples/texture_calibration.py", "examples/_panda_scene.py",
    "examples/reference_hdri_probe.py", "examples/roughness_calibration.py",
    "examples/native_ibl_calibration.py", "tests/test_environment.py",
    "tools/_g5_installed.py",
)
NEUTRAL_FILES = ("examples/reference_neutral_calibration.py",)
RAY_QUERY_FILES = ("examples/native_resident_smoke.py", "tests/fixtures/g1/table.gltf",
                   "tests/fixtures/g1/block.gltf")
FEATURE_FILES = BASE_FEATURE_FILES + NEUTRAL_FILES + RAY_QUERY_FILES
KINDS = ("host", "uv-native", "uv-sapien-raster", "uv-sapien-pt", "ibl-native", "ibl-sapien",
         "roughness-native", "roughness-sapien-raster", "roughness-sapien-pt",
         "neutral-sapien", "ray-query-native")
NATIVE_KINDS = ("uv-native", "ibl-native", "roughness-native", "ray-query-native")


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def plan_features(requested_paths, *, skip_features=()):
    paths = set(requested_paths)
    if not paths <= {"native-raster", "sapien-raster", "sapien-pt"}:
        raise ValueError("unknown requested feature path")
    required_paths = {"host": set(), "ibl-sapien": {"sapien-pt"},
                      "neutral-sapien": {"sapien-raster", "sapien-pt"}}
    for kind in KINDS:
        if kind in NATIVE_KINDS:
            required_paths[kind] = {"native-raster"}
        elif kind.endswith("-sapien-raster"):
            required_paths[kind] = {"sapien-raster"}
        elif kind.endswith("-sapien-pt"):
            required_paths[kind] = {"sapien-pt"}
    plan = {kind: ({"status": "required"} if required_paths[kind] <= paths else
                   {"status": "not_verified", "reason": "required device path not selected",
                    "explicit_skip": False}) for kind in KINDS}
    seen = set()
    for option in skip_features:
        kind, separator, reason = option.partition(":")
        if not separator or not reason.strip() or kind not in KINDS or kind == "host" or kind in seen:
            raise ValueError("skip-feature requires a unique known device feature KIND:REASON")
        seen.add(kind)
        plan[kind] = {"status": "not_verified", "reason": reason.strip(), "explicit_skip": True}
    return plan


def _fixture_members(enabled):
    values = tuple(enabled)
    if not values or len(values) != len(set(values)) or not set(values) <= set(KINDS):
        raise ValueError("unexpected feature fixture selection")
    return (BASE_FEATURE_FILES + (NEUTRAL_FILES if "neutral-sapien" in values else ())
            + (RAY_QUERY_FILES if "ray-query-native" in values else ()))


def copy_feature_bundle(rendering_root, destination, *, enabled=None):
    rendering = Path(rendering_root).resolve()
    target = Path(destination).resolve()
    source = rendering.parent
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("feature fixture must remain outside the source checkout")
    enabled = tuple(KINDS if enabled is None else enabled)
    originals = {}
    for relative in _fixture_members(enabled):
        original = (rendering/relative).resolve()
        if not original.is_relative_to(rendering) or not original.is_file():
            raise ValueError(f"missing or escaped feature fixture: {relative}")
        originals[relative] = original.read_bytes()
    target.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "phymeo-g5-feature-fixture-v1", "source_root": str(source),
                "enabled_features": list(enabled), "files": {}}
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
    # Older fixture manifests contain precisely the original seven protocols.
    members = (_fixture_members(manifest["enabled_features"]) if "enabled_features" in manifest
               else BASE_FEATURE_FILES)
    if (manifest.get("schema") != "phymeo-g5-feature-fixture-v1"
            or set(manifest.get("files", {})) != set(members)):
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
    if kind in NATIVE_KINDS:
        if library is None:
            raise ValueError("native feature requires its newly built library")
        command += ["--library", str(library), "--device-index", str(device_index)]
    if kind not in NATIVE_KINDS and kind != "host":
        command += ["--sapien-device", sapien_device]
    return command


def protocol_command(kind, output, *, library=None, device_index=1, sapien_device="cuda:0"):
    """Exact argv for existing independent examples; no acceptance overrides."""
    argv = ["--output", str(output)]
    if kind.startswith(("uv-", "roughness-")):
        backend = kind.split("-", 1)[1]
        if backend != "native" and sapien_device != "cuda:0":
            raise ValueError("sealed chart reference protocol explicitly uses cuda:0")
        script, result_file = "texture_calibration.py", "result.json"
        argv += ["--backend", backend]
        if kind.startswith("roughness-"):
            argv += ["--role", "roughness"]
    elif kind == "ibl-native":
        script, result_file = "native_ibl_calibration.py", "summary.json"
    elif kind in ("ibl-sapien", "neutral-sapien"):
        script = "reference_hdri_probe.py" if kind == "ibl-sapien" else "reference_neutral_calibration.py"
        result_file = "summary.json"
        argv += ["--device", sapien_device]
    else:
        raise ValueError("feature has no standalone example command")
    if kind in NATIVE_KINDS:
        if library is None:
            raise ValueError("native feature requires its newly built library")
        argv += ["--library", str(library), "--device-index", str(device_index)]
    return script, result_file, argv


def validate_feature_result(kind, result):
    """Bind the saved protocol to the requested feature, never adjust its gates."""
    import math

    if not (result.get("status") == "passed" or result.get("passed") is True):
        raise ValueError("sealed feature result did not pass")
    if kind.startswith(("uv-", "roughness-")):
        role = "roughness" if kind.startswith("roughness-") else "color"
        cells = result.get("cells", [])
        maximum, span = result.get("max_absolute_rgb8", math.inf), result.get("oracle_color_span", -math.inf)
        if (result.get("backend") != kind.split("-", 1)[1] or result.get("role", "color") != role
                or len(cells) != 16 or result.get("sample_count") != 16*49
                or result.get("limits") != {"max_rgb8": 3, "min_span": 40}
                or not math.isfinite(maximum) or not 0 <= maximum <= 3
                or not math.isfinite(span) or span < 40
                or any(not 0 <= cell.get("max_absolute_rgb8", math.inf) <= 3 for cell in cells)
                or any(result.get("sensor_differences", {}).values())):
            raise ValueError("sealed chart feature/coverage/acceptance gates differ")
    elif kind == "neutral-sapien":
        captures = result.get("captures", [])
        expected = {(path, transform) for path in ("raster", "pt") for transform in ("clip", "pbr-neutral")}
        if (result.get("schema") != "phymeo-reference-neutral-result-v1" or len(captures) != 4
                or {(row.get("path"), row.get("display_transform")) for row in captures} != expected
                or any(row.get("exposure") != 8. for row in captures)
                or result.get("raster_sensors_unchanged") is not True
                or result.get("raw_readbacks_unchanged") != {"raster": True, "pt": True}
                or result.get("pt_foreground_recovered_pixels", 0) < 64):
            raise ValueError("sealed Neutral four-exposure coverage/acceptance gates differ")
    elif kind == "ray-query-native":
        if (result.get("shadow_mode") != "ray_query" or result.get("primary_visibility") != "raster"
                or result.get("ray_query_enabled") is not True or len(result.get("captures", [])) != 3):
            raise ValueError("installed public hardware-shadow evidence missing")


def _ray_query(bundle, output, library, device_index):
    """Reuse the original public moving-shadow/depth test with installed hardware mode."""
    from dataclasses import replace
    from urllib.parse import urlparse
    from phymeo_rendering.backends.native import NativeBackend
    from phymeo_rendering.outputs import Channel

    output.mkdir(parents=True, exist_ok=False)
    original = _module("g5_copied_resident_smoke", bundle/"examples/native_resident_smoke.py")
    png = _module("g5_copied_ray_png", bundle/"examples/reference_hdri_probe.py")
    scene = original._scene()
    # Only relocate the two byte-identical, SHA-bound mesh URIs. Scene, poses,
    # analytic depths, brightness bounds and original assertions stay unchanged.
    scene = replace(scene, meshes=tuple(replace(mesh, uri=(bundle/"tests/fixtures/g1"/
        Path(urlparse(mesh.uri).path).name).as_uri()) for mesh in scene.meshes))
    backend = NativeBackend(library_path=library, shader_directory=library.parent,
                            device_index=device_index, validation=True, resident=True, shadow_mode="ray_query")
    captures, sessions = [], []

    class RecordedSession:
        def __init__(self, owner):
            self.owner = owner
        def capture(self, frame, request):
            capture = self.owner.capture(frame, request)
            stem = f"capture-{len(captures):02d}"
            row = {"tick": frame.key.tick, "planes": {}, "diagnostics": self.owner.diagnostics()}
            for plane in capture.planes:
                filename = stem+"."+plane.channel.name.lower()+".raw"
                (output/filename).write_bytes(plane.data)
                row["planes"][plane.channel.value] = {"file": filename, "sha256": _digest(output/filename)}
                if plane.channel is Channel.RGB:
                    png._png(output/(stem+".png"), plane.data, request.camera.width, request.camera.height)
            captures.append(row)
            return capture
        def diagnostics(self):
            return self.owner.diagnostics()
        def close(self):
            self.owner.close()

    class RecordedBackend:
        def open_scene(self, value):
            owner = backend.open_scene(value)
            sessions.append(owner)
            return RecordedSession(owner)

    report = {"schema": "phymeo-g5-public-ray-query-v1", "status": "running", "shadow_mode": "ray_query",
              "primary_visibility": "raster", "captures": captures,
              "scope": "public installed hardware ray-query shadows; original moving-shadow, analytic depth, ID and validity gates; no native PT"}
    try:
        report["original_moving_shadow"] = original.moving_shadow(RecordedBackend(), scene)
        report["ray_query_enabled"] = all(row["diagnostics"]["device"]["resident_ray_query_info"]["enabled"] == 1
                                           and row["diagnostics"]["shadow_mode"] == "ray_query" for row in captures)
        report["status"] = "passed"
        validate_feature_result("ray-query-native", report)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for owner in sessions:
            owner.close()
        _save(output/"summary.json", report)
    return report


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
    if args.kind not in manifest.get("enabled_features", KINDS[:6]):
        raise ValueError("feature was not selected in the bound fixture manifest")
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
            library = None
            if args.kind in NATIVE_KINDS:
                library = Path(args.library).resolve()
                if not library.is_file() or library.is_relative_to(source):
                    raise ValueError("native library must be the source-external build")
                report["native_library"] = {"file": str(library), "sha256": _digest(library)}
            if args.kind == "ray-query-native":
                report["script"] = "native_resident_smoke.py:moving_shadow"
                result_file = "summary.json"
                result = _ray_query(bundle, output, library, args.device_index)
            else:
                script, result_file, argv = protocol_command(args.kind, output, library=library,
                    device_index=args.device_index, sapien_device=args.sapien_device)
                report["script"] = script
                report["argv"] = argv
                _run_original(bundle/"examples"/script, argv)
                result = json.loads((output/result_file).read_text(encoding="utf-8"))
            validate_feature_result(args.kind, result)
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
