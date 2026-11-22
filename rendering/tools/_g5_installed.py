"""Copied to a new external directory by g5_reproduce; no checkout imports."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import sys
import unittest


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def installed_binding(source):
    import phymeo_rendering
    package = Path(phymeo_rendering.__file__).resolve().parent
    prefix = Path(sys.prefix).resolve()
    if not package.is_relative_to(prefix) or package.is_relative_to(source):
        raise ValueError("rendering did not import from the clean external environment")
    for entry in sys.path:
        if entry and Path(entry).resolve().is_relative_to(source):
            raise ValueError("source checkout present on isolated sys.path")
    for name, module in tuple(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if name == "phymeo" or name.startswith("phymeo."):
            raise ValueError("physical engine imported during rendering")
        if name.startswith("phymeo_rendering") and file and not Path(file).resolve().is_relative_to(package):
            raise ValueError("mixed installed/source rendering modules")
    return {"package_directory": str(package), "python": sys.executable, "prefix": str(prefix),
            "python_version": sys.version, "distribution_version": importlib.metadata.version("phymeo-rendering"),
            "sys_path": list(sys.path)}


def probe(args):
    source = Path(args.forbid_source).resolve()
    bundle = Path(args.bundle).resolve()
    output = Path(args.output).resolve()
    if bundle.is_relative_to(source) or output.is_relative_to(source):
        raise ValueError("copied acceptance input/output must remain outside the checkout")
    # -I deliberately drops the script directory. Only copied fixture code is
    # added; neither rendering/src nor any repository checkout is added.
    sys.path.insert(0, str(bundle))
    report = {"schema": "phymeo-rendering-installed-probe-v1", "mode": args.mode, "status": "running"}
    try:
        report.update(installed_binding(source))
        if args.mode == "base":
            from phymeo_rendering.panda_visuals import panda_visual_assets
            from phymeo_rendering.backends.native import NativeBackend  # noqa: F401: verify lazy import
            from phymeo_rendering.backends.sapien import SapienBackend  # noqa: F401: verify lazy import
            if any(name in sys.modules for name in ("sapien", "numpy", "PIL")):
                raise ValueError("base import loaded an optional backend dependency")
            distributions = {d.metadata["Name"].lower() for d in importlib.metadata.distributions()}
            if distributions & {"sapien", "numpy", "pillow"}:
                raise ValueError("base proof environment already contains optional backend packages")
            meshes, materials, instances = panda_visual_assets()
            package = Path(report["package_directory"])
            expected = package / "_panda_assets"
            from urllib.parse import unquote, urlparse
            for mesh in meshes:
                # URI decoding also handles Windows file:///C:/... paths.
                from urllib.request import url2pathname
                path = Path(url2pathname(unquote(urlparse(mesh.uri).path))).resolve()
                if not path.is_relative_to(expected) or not path.is_file():
                    raise ValueError("default Panda asset escaped the installed wheel")
            report["panda"] = {"meshes": len(meshes), "materials": len(materials), "instances": len(instances),
                               "asset_directory": str(expected),
                               "manifest_sha256": hashlib.sha256((expected / "manifest.json").read_bytes()).hexdigest()}
            report["optional_backends_installed"] = False
        else:
            from integration._g1_table import table_scene, table_frames
            from integration._r0_table import exercise_table
            from phymeo_rendering.cameras import Camera
            from phymeo_rendering.outputs import Channel
            from phymeo_rendering.pipeline import CaptureRequest, RenderPath
            from phymeo_rendering.scene import Pose
            if args.mode == "native-raster":
                from phymeo_rendering.backends.native import NativeBackend
                factory = lambda: NativeBackend(library_path=args.library, shader_directory=args.shader_directory,
                                                device_index=args.native_device_index, validation=True)
                path, label, sensors = RenderPath.RASTER, "native-raster", True
            else:
                from phymeo_rendering.backends.sapien import SapienBackend, SapienConfig
                path = RenderPath.PATH_TRACE if args.mode == "sapien-pt" else RenderPath.RASTER
                factory = lambda: SapienBackend(SapienConfig(path=path, device=args.sapien_device,
                                                             samples_per_pixel=64, max_bounces=4))
                label, sensors = args.mode, path is RenderPath.RASTER
            frame = table_frames()[0]
            camera = Camera("capabilities", frame.key, Pose((0., 0., 1.0625)),
                            640, 480, 500., 500., 320., 240., .1, 5.)
            session = factory().open_scene(table_scene())
            try:
                supported = []
                for request_path in RenderPath:
                    for size in range(1, len(Channel) + 1):
                        for channels in itertools.combinations(Channel, size):
                            request = CaptureRequest(camera, request_path, channels)
                            supported.append({"path": request_path.value, "channels": [c.value for c in channels],
                                              "supports": session.supports(request), "scope": "queried F0 configuration"})
                report["capabilities"] = supported
            finally:
                session.close()
            report["capability_session"] = session.diagnostics()
            os.environ["PHYMEO_R0_OUTPUT"] = str(output)
            exercise_table(unittest.TestCase(), factory, label, path, sensors=sensors)
            metrics_path = output / label / "metrics.json"
            report["f0"] = {"metrics": str(metrics_path), "sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
                            "executed_channels": [c.value for c in Channel] if sensors else [Channel.RGB.value],
                            "path": path.value, "source_steps": 0,
                            "checks": "fixed interior probes/background; lighting/shadow change; foreign frame and unsupported path rejection; close"}
            report["output_files"] = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted((output / label).rglob("*")) if p.is_file()}
        report.update(installed_binding(source))
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save(output / (args.mode + ".json"), report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("base", "native-raster", "sapien-raster", "sapien-pt"))
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--forbid-source", required=True)
    parser.add_argument("--library")
    parser.add_argument("--shader-directory")
    parser.add_argument("--native-device-index", type=int, default=0)
    parser.add_argument("--sapien-device", default="cuda:0")
    probe(parser.parse_args())


if __name__ == "__main__":
    main()
