"""Bound full-distance8 rigid / marker100 thin69 check; no new reference renders.

Default: four captures, each endpoint in a fresh owner. --presentation recorded
actually presents original poses 0..15 per backend, with no skipped predecessor.
Each prepared runtime lives in its own process. This script is not a benchmark.
"""
import argparse
from dataclasses import asdict
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")


def add_examples(args):
    # Explicit author dependencies precede inherited PYTHONPATH. Renderer itself
    # must subsequently come only from activate(prepared), before fixture import.
    sys.path.insert(0, str(args.examples.resolve(strict=True)))


def source_pair(args):
    from preview_lowres_filter.build_source import verify, markers
    from preview_lowres_filter.runtime import check_seal
    _, base = verify(args.baseline)
    _, filtered = check_seal(args.prepared)
    policy = base["contract"]
    require(policy["paths_per_anchor"] == 8 and policy["tile"] == 4
            and policy["bounces"] == 1 and policy["area_light_grid"] == 2
            and policy["rigid_history"]["marker"] == 97
            and not any(k in policy for k in ("lowres_filter", "local_diffuse", "edge_rescue")),
            "unaltered full-distance8 rigid parent required")
    require(filtered["lowres_parent_sha256"] == digest(args.baseline / "source-manifest.json"),
            "marker100 is not derived from the supplied baseline")
    expected = dict(filtered["contract"])
    require(expected.pop("lowres_filter")["marker"] == 100 and expected == policy,
            "paired transport/history configuration differs")
    binaries = {}
    for name, root in (("baseline", args.baseline), ("filtered", args.prepared)):
        binaries[name] = dict(dll_sha256=digest(root / "build/phymeo_native.dll"),
            spv_sha256={p.name: digest(p) for p in sorted((root / "build").glob("*.spv"))})
    old, new = binaries["baseline"]["spv_sha256"], binaries["filtered"]["spv_sha256"]
    require(old and set(old) == set(new) and all(old[k] == new[k] for k in old
            if k != "resident_spatial.comp.spv"), "original ray/history SPIR-V differs")
    marker = markers((args.prepared / "build/resident_spatial.comp.spv").read_bytes())
    require(marker.get("100") == dict(name="PR_PREVIEW_LOWRES_FILTER_ABI", value=[1]),
            "actual marker100 SPIR-V missing")
    return dict(baseline_manifest_sha256=digest(args.baseline / "source-manifest.json"),
        filtered_manifest_sha256=digest(args.prepared / "source-manifest.json"), binaries=binaries,
        unchanged_ray_history_shaders=True, marker100=marker["100"], contract=filtered["contract"])


def worker(args):
    add_examples(args)
    binding = source_pair(args)
    from preview_compact_rigid_history.runtime import RigidHistoryBackend, activate
    from preview_compact_rigid_history.state import bind_presentations, frame_digest
    prepared = args.baseline if args.worker == "baseline" else args.prepared
    activate(prepared)
    # Reuse the already-tested exact fixture loader / display-error definition.
    sys.path.insert(0, str(HERE))
    from capture import module
    coverage = module(HERE / "coverage.py", "rigid_thin_coverage")
    fixture = module(args.fixture, "rigid_thin_original_fixture")
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest, RenderPath
    import numpy as np
    from PIL import Image
    reference = read(args.reference / "summary.json")
    require(reference["schema"] == "phymeo-native-reveal-preview-v1"
            and reference["status"] == "complete" and reference["reference_batches"] == 64
            and reference["fixture_sha256"] == digest(args.fixture), "sealed reference/fixture differs")
    folder = args.output / args.worker
    folder.mkdir(parents=True, exist_ok=False)
    result = dict(schema="phymeo-rigid-thin69-capture-v1", status="preparing", mode=args.worker,
        presentation=args.presentation, source_binding=binding, frames={}, presented=[], owners=[],
        reference_summary_sha256=digest(args.reference / "summary.json"),
        fixture_sha256=digest(args.fixture), script_sha256=digest(__file__),
        limitations=["Authored optical fixture, not dynamics.",
            "Fresh endpoints test current output only; recorded mode consumes every original pose 0..15.",
            "Pose14 has no saved 1024-spp reference; no high-sample error is reported for it.",
            "Eight proposals per actual low anchor, not full-frame spp; history age is not spp.",
            "Displayed finite-reference errors are not an exact light-transport oracle."],
        timing_scope="four-channel synchronous capture only; no warmups/FPS claim; PNG and diagnostics outside timer")
    save(folder / "summary.json", result)
    owner = None
    try:
        with TemporaryDirectory(prefix="phymeo-rigid-thin-") as temp:
            assets = (args.assets or args.fixture.parent.parent / "assets/panda").resolve(strict=True)
            visual = fixture.original.panda_visual_assets
            with patch.object(fixture.original, "panda_visual_assets", partial(visual, assets)):
                nodes = fixture.build_motion(Path(temp))
            require(len(nodes) == 25, "registered 25-pose fixture changed")
            scene = nodes[0][0]
            material_id = next(i.material_id for i in scene.instances if i.id == 158)
            material = next(m for m in scene.materials if m.id == material_id)
            require(tuple(material.base_color) == (.7, .7, .7) and material.metallic == 0
                    and material.roughness == 1, "registered receiver material differs")
            inputs = {str(i): dict(frame=asdict(f), camera=asdict(c)) for i, (_, f, c) in enumerate(nodes)}
            save(folder / "inputs.json", inputs)
            source_sha = sha256(canonical(inputs)).hexdigest()
            result["presentation_source_sha256"] = source_sha
            result["inputs_sha256"] = digest(folder / "inputs.json")
            Backend = RigidHistoryBackend
            if args.worker == "filtered":
                from preview_lowres_filter.runtime import backend_class
                Backend = backend_class(folder / "debug")
            backend = Backend(prepared, device_index=args.device_index, seed=0, studio=None,
                exposure=1., display_transform="pbr-neutral", reflection_mode="environment", reuse_history=True)
            sequences = [(14,), (15,)] if args.presentation == "fresh" else [tuple(range(16))]
            geometry, images = {}, {}
            for indices in sequences:
                records = bind_presentations([nodes[i][1] for i in indices], source_sha, start_index=indices[0])
                owner = backend.open_scene(scene)
                owner_record = dict(indices=list(indices))
                result["owners"].append(owner_record)
                try:
                    for ordinal, (i, presentation) in enumerate(zip(indices, records)):
                        _, frame, camera = nodes[i]
                        request = CaptureRequest(camera, RenderPath.RASTER, tuple(Channel))
                        started = perf_counter()
                        shot = owner.capture(frame, request, presentation=presentation)
                        wall_ms = (perf_counter() - started) * 1000
                        require(shot.frame_key == frame.key and shot.camera == camera
                                and shot.time_s == frame.time_s, "public capture input binding differs")
                        planes = {p.channel: p.data for p in shot.planes}
                        require(set(planes) == set(Channel), "all four public planes required")
                        sensors = {c.value: sha256(v).hexdigest() for c, v in planes.items() if c != Channel.RGB}
                        saved = reference["frames"][str(i)]
                        require(sensors == saved["sensors"] and asdict(frame.key) == saved["key"]
                                and frame.time_s == saved["time"], "sealed geometry/clock differs at pose " + str(i))
                        diag = owner.diagnostics()
                        require(diag["decoded_geometry_sha256"] == reference["diagnostics"]["decoded_geometry_sha256"]
                                and diag["actual"]["paths_per_anchor"] == 8 and diag["actual"]["invalid"] == 0
                                and diag["rigid_history"]["marker"] == 97
                                and diag["rigid_history"]["committed_rgb"] == ordinal + 1
                                and diag["presentation"] == presentation, "actual method/presentation differs")
                        if args.worker == "filtered":
                            require(diag["lowres_filter"]["marker"] == 100, "filtered method not active")
                        if ordinal == 0:
                            require(diag["history"]["actual"]["last_reset"]
                                    and diag["history"]["actual"]["accepted_pixels"] == 0,
                                    "fresh owner reused unpresented history")
                        item = dict(pose_index=i, key=asdict(frame.key), time=frame.time_s,
                            frame_sha256=frame_digest(frame), camera_sha256=sha256(canonical(asdict(camera))).hexdigest(),
                            presentation=presentation, sensors=sensors, capture_ms=wall_ms,
                            rgb_sha256=sha256(planes[Channel.RGB]).hexdigest(), actual=diag["actual"],
                            history=diag["history"], rigid_history=diag["rigid_history"])
                        result["presented"].append(item)
                        if i in (14, 15):
                            h, w = camera.height, camera.width
                            rgb = np.frombuffer(planes[Channel.RGB], np.uint8).reshape(h, w, 3).copy()
                            images[i] = rgb
                            geometry[i] = dict(ids=np.frombuffer(planes[Channel.INSTANCE_ID], "<u4").reshape(h, w).copy(),
                                depth=np.frombuffer(planes[Channel.FORWARD_DEPTH], "<f4").reshape(h, w).copy(),
                                valid=np.frombuffer(planes[Channel.VALID], np.uint8).reshape(h, w).copy())
                            Image.fromarray(rgb).save(folder / f"pose-{i}.png")
                            result["frames"][str(i)] = dict(item, png_sha256=digest(folder / f"pose-{i}.png"))
                            result["diagnostics"] = diag
                finally:
                    owner.close()
                    closed = owner.diagnostics()
                    owner_record["closed_diagnostics"] = closed
                    device = closed["device"]
                    require(device["closed"] and all(device[k] == 0 for k in
                        ("live_contexts", "live_frames", "live_resources", "validation_errors"))
                        and device["resident_telemetry"]["final_after_destroy"]
                        and all(device["resident_telemetry"][k] == 0 for k in
                            ("device_local_bytes_current", "host_visible_bytes_current")), "native cleanup failed")
                    owner = None
            mask = coverage.reveal_mask(geometry[14], geometry[15], blocker=202) & (geometry[15]["ids"] == 158)
            expected_mask = np.array(Image.open(args.reference / "exposed-right-finger.png"))
            require(int(mask.sum()) == 69 and np.array_equal(np.uint8(mask) * 255, expected_mask),
                    "original fixed 69-pixel reveal changed")
            Image.fromarray(np.uint8(mask) * 255).save(folder / "exposed-right-finger.png")
            refs = []
            for seed in (1, 2):
                path = args.reference / f"seed-{seed}-pose-15.png"
                rgb = np.array(Image.open(path))
                expected = reference["references"][f"{seed}-15"]
                require(rgb.shape == images[15].shape and rgb.dtype == np.uint8 and expected["spp"] == 1024
                        and sha256(rgb.tobytes()).hexdigest() == expected["rgb_sha256"], "independent reference changed")
                refs.append(rgb)
            valid = (geometry[15]["valid"] != 0) & np.isfinite(geometry[15]["depth"]) & (geometry[15]["depth"] > 0)
            masks = dict(registered_reveal69=mask, current_right_finger=valid & (geometry[15]["ids"] == 158))
            result["errors"] = {key: dict(pixels=int(m.sum()),
                versus_independent_references=[coverage.display_error(images[15][m], ref[m]) for ref in refs],
                reference_disagreement=coverage.display_error(refs[0][m], refs[1][m])) for key, m in masks.items()}
            pixels = [dict(x=int(x), y=int(y), rgb8=images[15][y, x].tolist(),
                          reference_rgb8=[ref[y, x].tolist() for ref in refs]) for y, x in zip(*np.where(mask))]
            save(folder / "pixels.json", pixels)
            result["pixels_sha256"] = digest(folder / "pixels.json")
            result["mask_sha256"] = digest(folder / "exposed-right-finger.png")
            result["status"] = "actual_capture_and_reference_binding_passed"
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if owner is not None:
            owner.close()
        save(folder / "summary.json", result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("baseline", "prepared", "examples", "fixture", "reference", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--device-index", type=int, default=1)
    parser.add_argument("--presentation", choices=("fresh", "recorded"), default="fresh")
    parser.add_argument("--worker", choices=("baseline", "filtered"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for key in ("baseline", "prepared", "examples", "fixture", "reference"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
        return
    add_examples(args)
    binding = source_pair(args)
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(schema="phymeo-rigid-thin69-paired-v1", status="running", source_binding=binding,
        presentation=args.presentation, profile_order=["baseline", "filtered"],
        planned_captures=4 if args.presentation == "fresh" else 32, modes={},
        scope="Same sealed optical fixture and reference; no re-rendered references or retuned ROI/budget.")
    save(args.output / "summary.json", result)
    try:
        for mode in result["profile_order"]:
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", mode]
            for key in ("baseline", "prepared", "examples", "fixture", "reference", "output", "assets"):
                value = getattr(args, key)
                if value is not None:
                    command.extend(["--" + key, str(value)])
            command.extend(["--device-index", str(args.device_index), "--presentation", args.presentation])
            with (args.output / (mode + ".stdout.log")).open("wb") as stdout, (args.output / (mode + ".stderr.log")).open("wb") as stderr:
                code = subprocess.run(command, stdout=stdout, stderr=stderr, check=False).returncode
            require(code == 0, mode + " child failed; original stdout/stderr and partial receipt retained")
            path = args.output / mode / "summary.json"
            report = read(path)
            require(report["status"] == "actual_capture_and_reference_binding_passed", "incomplete child receipt")
            result["modes"][mode] = dict(file=str(path), sha256=digest(path), errors=report["errors"],
                                       presented_captures=len(report["presented"]))
            save(args.output / "summary.json", result)
        a, b = [read(args.output / mode / "summary.json") for mode in result["profile_order"]]
        require(a["presentation_source_sha256"] == b["presentation_source_sha256"], "paired original input states differ")
        for i in ("14", "15"):
            require(all(a["frames"][i][k] == b["frames"][i][k] for k in
                        ("frame_sha256", "camera_sha256", "sensors", "presentation")), "paired frame/geometry differs")
        require(a["mask_sha256"] == b["mask_sha256"], "paired registered mask differs")
        result["status"] = "actual_pair_bound_no_quality_gate_invented"
        result["all_requested_geometry_exact"] = True
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save(args.output / "summary.json", result)
    print(str(args.output / "summary.json"))


if __name__ == "__main__":
    main()
