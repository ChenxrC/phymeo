"""Exercise the installed overnight product and verify the bytes of every imported Phymeo module."""

import argparse
import gzip
import json
import sys
from importlib.metadata import distribution
from pathlib import Path
from zipfile import ZipFile


def robot():
    from vision_guided_place import run

    results = []
    for kind in ("box", "sphere"):
        trial = run(kind, 0.055)
        state = trial.pop("full_run")
        assert state["qualified"] and state["accepted_steps"] == 3600
        assert abs(trial["calibration_error_m"]) < 0.005 and state["max_final_pad_N"] < 0.001
        results.append(
            dict(
                **trial,
                accepted_steps=state["accepted_steps"],
                final_table_n=state["mean_final_table_N"],
                placement_error_m=state["target_error_m"],
                final_twist=state["events"]["end"]["payload_velocity_m_s"],
            )
        )
    return results


def cloth():
    from phymeo.experiments.convex_tool_transfer import build_tool_transfer, run_tool_transfer

    record = run_tool_transfer(build_tool_transfer())
    assert record["qualified"] and record["status"] == "completed"
    release = record["events"][-1]
    assert release["kind"] == "independent_flight" and release["step"] - release["start_step"] >= 32
    return record


def device(ph, np, cache):
    ball = ph.FreeSphere
    hit = ph.SphereIslandWorld(
        (
            (81, ball(3, 0.3125, (0.375, 0.5, 0), gravity=(0, 0, 0))),
            (4, ball(1, 0.3125, (0, 0, 0), (1.2, 1.6, 0), gravity=(0, 0, 0))),
        ),
        restitution=0.5,
    )
    far = ph.SphereIslandWorld(((81, ball(1, 0.25, (0, 0, 5), gravity=(0, 0, 0))),))
    with ph.TyxSphereIslandBatch([(3, far), (87, hit)], "cuda", cache_root=cache) as batch:
        frame = batch.step(1 / 128)
        assert frame[87].accepted and frame[3].accepted
        (event,) = frame[87].contacts
        assert not frame[3].contacts and abs(event.normal_impulse_n_s - 2.25) < 8e-6
        np.testing.assert_allclose(frame[87].state.bodies[81].velocity, [0.45, 0.6, 0], atol=5e-6)
        measured = dict(
            impulse_ns=event.normal_impulse_n_s,
            energy_change_j=frame[87].contact_kinetic_change_j,
            stats=batch.stats(),
            world_keys=list(frame),
        )
        batch.reset([87])
        assert batch.snapshot()[3].tick == 1 and batch.snapshot()[87].tick == 0
    key = ph.CollisionKey(0, 2, 0)
    plane = ph.CollisionKey(0, 5, 0)
    optics = ph.CollisionModel(
        (
            ph.CollisionShape(key, "sphere", (0,), radius=0.5),
            ph.CollisionShape(plane, "plane", (0,), normal=(0, 0, 7)),
        )
    )
    poses = {key: ph.CollisionMotion([[0, 0, -2]]), plane: ph.CollisionMotion([[0, 0, -4]])}
    camera = ph.DepthCamera(3, 3, np.pi / 2)
    with ph.TyxRaycaster("cuda", cache_root=cache) as ray_operator:
        frame = camera.capture(optics, poses, operator=ray_operator)
        assert frame.keys[frame.shape_index[1, 1]] == key
        assert frame.keys[frame.shape_index[0, 0]] == plane
        np.testing.assert_allclose(frame.ray_depth_m[1, 1], 1.5, atol=1e-5)
        np.testing.assert_allclose(frame.forward_depth_m[0, 0], 4.0, atol=1e-5)
        measured["camera_stats"] = ray_operator.stats()
    return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-site", type=Path, required=True)
    parser.add_argument("--expected-wheel", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("robot", "cloth", "cuda"), required=True)
    args = parser.parse_args()
    site = args.wheel_site.resolve(strict=True)
    outside = Path(__file__).resolve().parent
    root = args.checkout.resolve(strict=True)
    assert __debug__ and sys.flags.isolated
    assert (
        Path.cwd().resolve() == outside
        and site.parent == outside
        and not outside.is_relative_to(root)
    )
    if args.out.exists():
        raise ValueError("choose a new receipt path")
    sys.path[:0] = [str(site), str(outside)]
    import numpy as np
    from check_newton_stage import verified_wheel

    import phymeo as ph

    assert Path(ph.__file__).resolve().is_relative_to(site)
    installed = distribution("phymeo")
    assert Path(installed.locate_file("")).resolve() == site
    wheel = verified_wheel(installed, args.expected_wheel)
    if args.mode == "robot":
        product = robot()
    elif args.mode == "cloth":
        record = cloth()
        stored = args.out.with_suffix(".trace.json.gz")
        with gzip.open(stored, "xt", encoding="utf8") as stream:
            json.dump(record, stream, allow_nan=False, separators=(",", ":"))
        product = dict(trace=stored.name, events=record["events"], status=record["status"])
    else:
        product = device(ph, np, outside / "cache")
    paths = set()
    for name, module in tuple(sys.modules.items()):
        if (name == "phymeo" or name.startswith("phymeo.")) and getattr(module, "__file__", None):
            paths.add(Path(module.__file__).resolve())
    with ZipFile(args.expected_wheel) as archive:
        for file in paths:
            assert (
                file.is_relative_to(site)
                and archive.read(file.relative_to(site).as_posix()) == file.read_bytes()
            )
    report = dict(wheel=wheel, mode=args.mode, imported_modules=len(paths), product=product)
    with args.out.open("x", encoding="utf8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(args.mode, "verified", len(paths), "installed module files")


if __name__ == "__main__":
    main()
