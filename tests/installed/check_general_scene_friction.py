"""Outside-checkout HOST smoke for the installed general scene friction bridge.

Copy this consumer outside the checkout and run there with Python -I. The
original declaration-only example and affine-jaws fixture are copied byte for
byte beside the new output before explicit loading. Only public scene methods
advance physics; no repository test or private solver is called by this script.

Frozen windows: F01's first 64 steps (+0.5 N, h=1/1024), F04 affine's original
64 steps (h=1/1024), and F05's one legal fast-tilt rejection (h=1/128). These are
packaging checks, not the full source suite, F07, GPU physics, or a benchmark.
"""

import argparse
import gzip
import hashlib
from importlib.metadata import distribution
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
import traceback
from urllib.parse import urlsplit
from zipfile import ZipFile

import numpy as np

STEPS = 64
H = 1 / 1024
DECLARATIONS = (
    "examples/general_scene_friction.py",
    "tests/fixtures/urdf/floating_mimic_jaws.urdf",
)
LIMITS = (
    "HOST public scene smoke: two original 64-step subwindows and one motion "
    "rejection. No F07 success, arbitrary rotating-contact acceptance, CUDA "
    "physics, performance, or complete source-suite claim."
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def close(actual, expected, atol):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0)


def archive_identity(installed, wheel):
    assert wheel.suffix.lower() == ".whl", "expected actual wheel archive"
    sha = digest(wheel)
    raw = installed.read_text("direct_url.json")
    assert raw is not None, "install the local wheel directly; retain direct_url.json"
    direct = json.loads(raw)
    origin = urlsplit(direct["url"])
    assert origin.scheme == "file" and origin.path.lower().endswith(".whl")
    archive = direct["archive_info"]
    recorded = archive.get("hashes", {}).get("sha256")
    if recorded is None:
        algorithm, _, recorded = archive.get("hash", "").partition("=")
        assert algorithm == "sha256"
    assert recorded.lower() == sha, "installed archive SHA differs from expected wheel"
    return dict(path=str(wheel), sha256=sha, installed_from=direct["url"], version=installed.version)


def module_provenance(site, wheel):
    rows = []
    with ZipFile(wheel) as archive:
        for name, module in sorted(tuple(sys.modules.items())):
            if name != "phymeo" and not name.startswith("phymeo."):
                continue
            file = getattr(module, "__file__", None)
            assert file is not None, f"missing product module file: {name}"
            path = Path(file).resolve(strict=True)
            assert path.is_relative_to(site), f"non-wheel module {name}: {path}"
            member = path.relative_to(site).as_posix()
            content = path.read_bytes()
            assert archive.read(member) == content, f"installed bytes differ: {member}"
            rows.append(dict(module=name, path=str(path), wheel_member=member,
                             sha256=hashlib.sha256(content).hexdigest()))
    assert rows
    return rows


def copy_declarations(checkout, output):
    copies, rows = output / "declarations", []
    for relative in DECLARATIONS:
        source = (checkout / relative).resolve(strict=True)
        assert source.is_relative_to(checkout)
        target = copies / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        content = source.read_bytes()
        with target.open("xb") as stream:
            stream.write(content)
        target = target.resolve(strict=True)
        assert not target.is_relative_to(checkout)
        assert target.read_bytes() == content
        rows.append(dict(source=str(source), copied=str(target), bytes=len(content),
                         sha256=hashlib.sha256(content).hexdigest()))
    return copies, rows


def verify_declarations(rows):
    for row in rows:
        assert digest(Path(row["source"])) == row["sha256"], "source declaration changed"
        assert digest(Path(row["copied"])) == row["sha256"], "copied declaration changed"


def load_example(copies, checkout):
    path = (copies / DECLARATIONS[0]).resolve(strict=True)
    assert not path.is_relative_to(checkout)
    name = "installed_general_scene_friction_declarations"
    assert name not in sys.modules
    spec = spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == path
    assert module.H == H and module.KT == 200.0
    return module


def snapshot(example, scene):
    return example.public_payload(dict(
        state=scene.state, reading=scene.observe(), geometry=scene.geometry(),
        events=scene.contact_events, history=scene.tangent_history,
        receipt=scene.tangent_receipt, last_frame=scene.last_frame,
    ))


def check_energy(frame, previous_energy):
    history, receipt, events = frame["history"], frame["receipt"], frame["events"]
    assert receipt.accepted
    assert receipt.material_motion_error_m <= 1e-6
    assert receipt.application_motion_error_m <= 1e-6
    assert receipt.material_application_error_m <= 1e-6
    assert abs(receipt.energy_residual_j) <= 1e-8
    close(receipt.per_station_energy_residual_j, 0, 1e-8)
    assert history.commit_time_s == events.commit_time_s == frame["reading"].time_s
    assert history.sample_time_s == events.sample_time_s == frame["reading"].load_time_s
    actual_work = 0.0
    for index, record in enumerate(history.active):
        sample, update = record.sample, record.update
        matches = []
        for event in events.active:
            if (event.identity.first, event.identity.second) != (sample.first.shape, sample.second.shape):
                continue
            for component in event.components:
                for site, material in zip(component.load.sites, component.material, strict=True):
                    if (np.allclose(material.first.local_point_m, sample.first.local_point_m, atol=1e-11, rtol=0)
                            and np.allclose(material.second.local_point_m, sample.second.local_point_m, atol=1e-11, rtol=0)):
                        matches.append((site, component.load, component.law))
        assert len(matches) == 1, "material address must match one actual load site"
        site, load, law = matches[0]
        traction = np.asarray(site.force_on_second_n) - site.force_n * np.asarray(load.normal)
        elastic = np.asarray(update.elastic_displacement_m)
        close(traction, -200 * elastic, 1e-9)
        assert np.linalg.norm(traction) <= law.friction * site.force_n + 1e-9
        close(update.stored_energy_j, 100 * np.dot(elastic, elastic), 1e-12)
        actual_work += float(traction @ receipt.actual_application_displacement_m[index])
    close(actual_work, receipt.actual_work_j, 1e-9)
    energy = sum(record.update.stored_energy_j for record in history.active)
    assert actual_work + energy - previous_energy <= 1e-8
    return energy


def check_paddles(example, case, frame, index):
    reading, entry = frame["reading"], frame["entry"]
    row = case.scene.rigid_rows[case.world, example.BOX]
    assert len(frame["history"].active) == 2
    assert all(record.identity.generation == 1 for record in frame["history"].active)
    close(sum(load.force_n for load in reading.loads.contacts), 4, 1e-9)
    close(reading.external.rigid.force[row], [.5, 0, 0], 0)
    close(reading.rigid.velocity[row] - entry.rigid.velocity[row],
          H * (reading.external.rigid.force[row] + reading.loads.inputs.rigid.force[row]), 1e-10)
    momentum_increment = reading.rigid.velocity[row, 0] - entry.rigid.velocity[row, 0]
    for address, robot in reading.robots.items():
        close(2 * (robot.v - entry.robots[address].v), H * robot.contact_effort, 1e-10)
        momentum_increment += 2 * (robot.v[0] - entry.robots[address].v[0])
    close(momentum_increment, H * .5, 1e-10)
    close(reading.rigid.angular_velocity[row], 0, 1e-9)
    if index == 1:
        # Independent two 2 kg P tools / one 1 kg box linear first-step oracle.
        force = -200 * H**2 * .5 / (1 + 2.5 * 200 * H**2)
        acceleration = .5 + 2 * force
        close(reading.rigid.velocity[row], [H * acceleration, 0, 0], 2e-11)
        close(reading.rigid.position[row], [H**2 * acceleration, 0, 0], 2e-11)
        for robot in reading.robots.values():
            close(robot.contact_effort_full, [-force], 1e-9)
            close(robot.v, [-H * force / 2], 2e-11)
            close(robot.q, [-H**2 * force / 2], 2e-11)
        displacement = [[H**2 * (.5 + 2.5 * force), 0, 0]] * 2
        close(frame["receipt"].actual_material_displacement_m, displacement, 2e-11)
        close(frame["receipt"].actual_application_displacement_m, displacement, 2e-11)


def check_affine(example, case, frame):
    reading = frame["reading"]
    row = reading.robots[case.world, example.LOWER_TOOL]
    plate = reading.robots[case.world, example.UPPER_TOOL]
    assert row.q.shape == (8,) and row.v.shape == (7,)
    assert row.full_q.shape == (9,) and row.full_v.shape == (8,)
    assert row.contact_effort_full.shape == (8,) and row.contact_effort.shape == (7,)
    assert row.link_wrenches_world.shape == (3, 6)
    assert len(frame["history"].active) == 8
    close(sum(load.force_n for load in reading.loads.contacts), 4, 1e-8)
    close(row.full_v[3:6], 0, 1e-9)
    close(row.full_q[3:7], [0, 0, 0, 1], 1e-9)
    close(5 * row.full_v[0] + row.full_v[6] + 2 * row.full_v[7] + 3 * plate.v[0], 0, 1e-9)
    close(row.full_q[8], -row.full_q[7] + .004, 2e-11)
    close(row.full_v[7], -row.full_v[6], 2e-11)
    close(row.mimic_loads.constraint_effort_full[:6], 0, 1e-9)
    close(row.mimic_loads.constraint_effort_full[6] - row.mimic_loads.constraint_effort_full[7], 0, 1e-9)
    assert abs(row.mimic_loads.constraint_power_W) <= 1e-9
    assert all(abs(item.constraint_power_W) <= 1e-9 for item in row.mimic_loads.row_forces)
    close(row.link_wrenches_world[:2], 0, 1e-9)
    close(row.link_wrenches_world.sum(axis=0)[:3] + plate.link_wrenches_world.sum(axis=0)[:3], 0, 1e-9)
    for event in frame["events"].active:
        for owner in (event.first_owner, event.second_owner):
            if owner.body == example.LOWER_TOOL:
                assert owner.kind == "robot_link" and owner.link_name == "right_jaw" and owner.link_index == 2


def write_gzip(path, value):
    with path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            compressed.write(json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8"))


def run_window(example, copies, output, label, result):
    data = dict(case=label, backend="host", status="running", dt_s=H, frames=[])
    case = None
    result.update(status="running", accepted_steps=0, checked_steps=0, dt_s=H, backend="host")
    try:
        directory = output / "generated-assets" / label
        case = (example.make_two_paddles(directory) if label == "F01-first64" else
                example.make_floating_clamp(directory, fixture_dir=copies / "tests/fixtures/urdf", mimic=True))
        scene = case.scene
        data["parameters"] = example.public_payload(case.parameters)
        data["registrations"] = example.public_payload(case.registry.registrations)
        data["declared_actions"] = example.public_payload(case.actions)
        data["before_enable"] = snapshot(example, scene)
        normals = scene.sample()
        data["initial_normal_sample"] = example.public_payload(normals)
        close(sum(load.force_n for load in normals.contacts), 4, 1e-9)
        expected_sites = 2 if label == "F01-first64" else 8
        assert sum(len(load.sites) for load in normals.contacts) == expected_sites
        close([site.force_n for load in normals.contacts for site in load.sites],
              [2.0 if expected_sites == 2 else .5] * expected_sites, 1e-9)
        scene.enable_coupled_tangent(case.registry, stiffness_n_per_m=example.KT)
        assert scene.tangent_history.step_index == 0 and scene.tangent_receipt is None
        data["after_enable"] = snapshot(example, scene)
        energy = 0.0
        for index in range(1, STEPS + 1):
            external = None
            entry = scene.state
            if label == "F01-first64":
                external = scene.inputs()
                external.rigid.force[scene.rigid_rows[case.world, example.BOX], 0] = .5
                reading = scene.step(H, actions=case.actions, external=external)
            else:
                reading = scene.step(H, actions=case.actions)
            result["accepted_steps"] = index
            frame = example.capture_frame(case, entry, reading)
            frame.update(state=scene.state, loads=reading.loads, last_frame=scene.last_frame,
                         actual_input=dict(dt_s=H, actions=case.actions, external=external))
            saved = dict(index=index, checks_complete=False, public=example.public_payload(frame))
            data["frames"].append(saved)
            assert reading.step_number == index and reading.time_s == index * H
            assert reading.load_time_s == (index - 1) * H
            if label == "F01-first64":
                check_paddles(example, case, frame, index)
            else:
                check_affine(example, case, frame)
            energy = check_energy(frame, energy)
            saved["checks_complete"] = True
            result["checked_steps"] = index
        if label == "F04-affine64":
            assert abs(reading.robots[case.world, example.LOWER_TOOL].q[0]) > 1e-7
        else:
            assert abs(reading.rigid.position[scene.rigid_rows[case.world, example.BOX], 0]) > 1e-6
        result.update(status="passed", physical_time_s=STEPS * H, final_spring_energy_j=energy)
        data["status"] = "passed"
    except Exception as error:
        result["status"] = data["status"] = "failed"
        data["failure"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            if case is not None:
                data["final_public"] = snapshot(example, case.scene)
        finally:
            try:
                if case is not None:
                    case.close()
            finally:
                trace = output / f"{label}.json.gz"
                write_gzip(trace, data)
                result.update(trace=str(trace), trace_sha256=digest(trace), trace_bytes=trace.stat().st_size)


def run_rejection(example, output, result):
    data = dict(case="F05-legal-fast-tilt", backend="host", status="running", dt_s=1 / 128)
    result.update(status="running", backend="host", attempted_steps=1, accepted_steps=0, dt_s=1 / 128)
    try:
        with example.make_rotating_tool(output / "generated-assets" / "F05", tilted=True) as case:
            scene = case.scene
            data["parameters"] = example.public_payload(case.parameters)
            data["registrations"] = example.public_payload(case.registry.registrations)
            normals = scene.sample()
            data["initial_normal_sample"] = example.public_payload(normals)
            close(sum(load.force_n for load in normals.contacts), 2, 1e-9)
            assert scene.observe().robots[case.world, example.LOWER_TOOL].v[0] == 5
            scene.enable_coupled_tangent(case.registry, stiffness_n_per_m=example.KT)
            before, history, receipt = snapshot(example, scene), scene.tangent_history, scene.tangent_receipt
            data["before"] = before
            data["actual_input"] = example.public_payload(dict(dt_s=1 / 128, actions=case.actions, external=None))
            try:
                scene.step(1 / 128, actions=case.actions)
            except ValueError as error:
                data["rejection"] = dict(type=type(error).__name__, message=str(error))
                assert any(word in str(error) for word in ("motion", "material", "application", "station", "anchor"))
                assert "velocity outside" not in str(error)
            else:
                result["accepted_steps"] = 1
                raise AssertionError("legal fast tilt must reject actual motion, not publish a step")
            finally:
                data["after"] = snapshot(example, scene)
            assert data["after"] == before
            assert scene.tangent_history is history and scene.tangent_receipt is receipt
            assert scene.observe().step_number == 0 and scene.observe().time_s == 0
            data["status"] = result["status"] = "passed"
            result.update(complete_public_rollback=True, history_identity_preserved=True,
                          rejection=data["rejection"])
    except Exception as error:
        data["status"] = result["status"] = "failed"
        data["failure"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        trace = output / "F05-legal-fast-tilt.json.gz"
        write_gzip(trace, data)
        result.update(trace=str(trace), trace_sha256=digest(trace), trace_bytes=trace.stat().st_size)


def absolute_path(value):
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("an absolute path is required")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", "--outdir", dest="output_dir", type=absolute_path, required=True)
    parser.add_argument("--wheel-site", type=absolute_path, required=True)
    parser.add_argument("--expected-wheel", type=absolute_path, required=True)
    parser.add_argument("--checkout", type=absolute_path, required=True)
    args = parser.parse_args()
    site, checkout = args.wheel_site.resolve(strict=True), args.checkout.resolve(strict=True)
    wheel, script = args.expected_wheel.resolve(strict=True), Path(__file__).resolve(strict=True)
    output = args.output_dir.resolve()
    assert __debug__ and sys.flags.isolated, "run the copied consumer with python -I"
    assert Path.cwd().resolve() == script.parent, "run outside checkout from the copied script directory"
    assert not script.is_relative_to(checkout) and not site.is_relative_to(checkout)
    assert not output.is_relative_to(checkout) and not output.is_relative_to(site)
    assert "phymeo" not in sys.modules
    source_script = checkout / "tests/installed/check_general_scene_friction.py"
    assert script.read_bytes() == source_script.read_bytes(), "consumer copy differs from checkout source"
    output.mkdir(parents=True, exist_ok=False)
    record = dict(status="running", backend="host", errors=[], cases={}, limits=LIMITS,
                  script=dict(path=str(script), sha256=digest(script), source=str(source_script)),
                  versions=dict(python=sys.version, numpy=np.__version__),
                  gates=dict(material_motion_m=1e-6, application_motion_m=1e-6,
                             energy_residual_j=1e-8, first_step_state_atol=2e-11,
                             force_atol_n=1e-9, physical_windows="F01 first64; F04 original64; F05 one rejection"))
    sys.path.insert(0, str(site))
    try:
        import phymeo as ph

        assert Path(ph.__file__).resolve(strict=True).is_relative_to(site)
        installed = distribution("phymeo")
        assert Path(installed.locate_file("")).resolve() == site
        record["wheel"] = archive_identity(installed, wheel)
        record["modules_before"] = module_provenance(site, wheel)
        copies, record["declarations"] = copy_declarations(checkout, output)
        example = load_example(copies, checkout)
        for label in ("F01-first64", "F04-affine64"):
            result = record["cases"].setdefault(label, {})
            run_window(example, copies, output, label, result)
        run_rejection(example, output, record["cases"].setdefault("F05-legal-fast-tilt", {}))
        record["status"] = "passed"
    except Exception as error:
        record["status"] = "failed"
        record["errors"].append(dict(stage="wheel_import_or_window", type=type(error).__name__,
                                     message=str(error), traceback=traceback.format_exc()))
    try:
        assert digest(source_script) == record["script"]["sha256"], "consumer source changed"
        if "declarations" in record:
            verify_declarations(record["declarations"])
        if "phymeo" in sys.modules:
            record["modules_after"] = module_provenance(site, wheel)
    except Exception as error:
        record["status"] = "failed"
        record["errors"].append(dict(stage="final_provenance", type=type(error).__name__,
                                     message=str(error), traceback=traceback.format_exc()))
    record["generated_inputs"] = [dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size)
                                  for path in sorted((output / "generated-assets").rglob("*.urdf"))]
    trace = output / "general-scene-friction-wheel.receipt.json.gz"
    write_gzip(trace, record)
    summary = dict(status=record["status"], raw_exit=0 if record["status"] == "passed" else 1,
                   backend="host", cases=record["cases"], wheel=record.get("wheel"),
                   verified_module_count=len(record.get("modules_after", ())),
                   receipt=str(trace), receipt_sha256=digest(trace), errors=record["errors"], limits=LIMITS)
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    print(json.dumps(summary, allow_nan=False))
    return summary["raw_exit"]


if __name__ == "__main__":
    raise SystemExit(main())
