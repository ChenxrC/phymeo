"""Outside-checkout C4 wheel smoke: one public mixed batch, not the full matrix.

Copy this single script next to a separately installed wheel site and run with
Python -I. All product access is through ``import phymeo as ph``. The script
contains its own declared fixtures and FP64 rigid-transform derivative oracle;
it imports no repository test/reference script or private Phymeo module.

Fixed window: 40 steps of h=1/1024; old FreeSphere/PrismaticSlider plus complete
2R and N9 R/P trees. The 2R uses the frozen (.37,-.51)/(.5,-.25) pose and the
frozen (1,1/6) external effort; N9 uses the original alternating-chain input.
This combines existing mechanisms for packaging, without changing C4 gates.
"""

import argparse
import gzip
import hashlib
import json
import sys
import traceback
from dataclasses import fields, is_dataclass
from importlib.metadata import distribution
from itertools import combinations, product
from math import cos, sin
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZipFile

import numpy as np

H = 1 / 1024
WINDOW_STEPS = 40
RTOL = 5e-5
ATOL = 5e-6
WINDOW_ATOL = 2e-4
TREE_KEYS = (53, 91)


def json_value(value):
    if is_dataclass(value):
        return {field.name: json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": repr(value)}
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def close(actual, expected, *, atol=ATOL):
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=atol)


def archive_identity(installed, expected):
    wheel = expected.resolve(strict=True)
    if wheel.suffix.lower() != ".whl":
        raise ValueError("--expected-wheel must name the actual installed wheel archive")
    with wheel.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    direct_text = installed.read_text("direct_url.json")
    if direct_text is None:
        raise AssertionError("install the local wheel directly, retaining direct_url.json")
    direct = json.loads(direct_text)
    origin = urlsplit(direct["url"])
    assert origin.scheme == "file" and origin.path.lower().endswith(".whl")
    archive = direct["archive_info"]
    recorded = archive.get("hashes", {}).get("sha256")
    if recorded is None:
        algorithm, _, recorded = archive.get("hash", "").partition("=")
        assert algorithm == "sha256"
    assert recorded.lower() == digest
    return dict(
        path=str(wheel), sha256=digest, installed_from=direct["url"], version=installed.version
    )


def module_provenance(site, wheel):
    rows = []
    with ZipFile(wheel) as archive:
        for name, module in sorted(tuple(sys.modules.items())):
            if name != "phymeo" and not name.startswith("phymeo."):
                continue
            file = getattr(module, "__file__", None)
            assert file is not None, f"no wheel source for imported {name}"
            path = Path(file).resolve(strict=True)
            assert path.is_relative_to(site), f"non-wheel module {name}: {path}"
            member = path.relative_to(site).as_posix()
            content = path.read_bytes()
            assert archive.read(member) == content, f"installed bytes differ: {member}"
            rows.append(
                dict(
                    module=name,
                    path=str(path),
                    wheel_member=member,
                    sha256=hashlib.sha256(content).hexdigest(),
                )
            )
    assert rows
    return rows


def skew(axis):
    x, y, z = axis
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def rotation(axis, angle):
    k = skew(axis)
    return np.eye(3) + sin(angle) * k + (1 - cos(angle)) * (k @ k)


def vee(matrix):
    return (
        np.array(
            [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]]
        )
        / 2
    )


def oracle(data, q, v):
    """Independent homogeneous-transform partial/time derivatives, qdd=0.

    Original identity static joint rotations suffice for these frozen fixtures.
    This is neither a finite difference nor a Phymeo dynamics/FK readback.
    """
    n = len(q)
    transforms, derivatives, first, second = [], [], [], []
    linear, angular = np.zeros((n, 3, n)), np.zeros((n, 3, n))
    com, omega, alpha, acceleration = (np.zeros((n, 3)) for _ in range(4))
    for body in range(n):
        static = np.eye(4)
        static[:3, 3] = data["offsets"][body]
        motion, derivative, curvature = np.eye(4), np.zeros((4, 4)), np.zeros((4, 4))
        axis = data["axes"][body]
        if data["kinds"][body] == "revolute":
            motion[:3, :3] = rotation(axis, q[body])
            derivative[:3, :3] = skew(axis) @ motion[:3, :3]
            curvature[:3, :3] = skew(axis) @ derivative[:3, :3]
        else:
            motion[:3, 3] = axis * q[body]
            derivative[:3, 3] = axis
        local, dl, ddl = static @ motion, static @ derivative, static @ curvature
        parent = data["parents"][body]
        base = np.eye(4) if parent < 0 else transforms[parent]
        partial = np.zeros((n, 4, 4)) if parent < 0 else derivatives[parent]
        timed = np.zeros((4, 4)) if parent < 0 else first[parent]
        timed2 = np.zeros((4, 4)) if parent < 0 else second[parent]
        frame = base @ local
        df = partial @ local
        df[body] += base @ dl
        dt = timed @ local + v[body] * (base @ dl)
        ddt = timed2 @ local + 2 * v[body] * (timed @ dl) + v[body] ** 2 * (base @ ddl)
        transforms.append(frame)
        derivatives.append(df)
        first.append(dt)
        second.append(ddt)
        point = np.r_[data["com"][body], 1.0]
        com[body] = (frame @ point)[:3]
        linear[body] = (df @ point)[:, :3].T
        r = frame[:3, :3]
        for column in range(n):
            angular[body, :, column] = vee(df[column, :3, :3] @ r.T)
        omega[body] = vee(dt[:3, :3] @ r.T)
        alpha[body] = vee(ddt[:3, :3] @ r.T + dt[:3, :3] @ dt[:3, :3].T)
        acceleration[body] = (ddt @ point)[:3]
    mass, bias = np.zeros((n, n)), np.zeros(n)
    for body in range(n):
        r = transforms[body][:3, :3]
        inertia = r @ np.diag(data["inertia"][body]) @ r.T
        mass += (
            data["mass"][body] * (linear[body].T @ linear[body])
            + angular[body].T @ inertia @ angular[body]
        )
        bias += linear[body].T @ (data["mass"][body] * (acceleration[body] - data["gravity"]))
        bias += angular[body].T @ (
            inertia @ alpha[body] + np.cross(omega[body], inertia @ omega[body])
        )
    frames = np.asarray(transforms)
    return dict(
        position=frames[:, :3, 3],
        rotation=frames[:, :3, :3],
        com=com,
        mass=mass,
        bias=bias,
        linear=linear,
        angular=angular,
    )


def model_data(kind):
    if kind == "2R":
        return dict(
            parents=np.array([-1, 0]),
            kinds=("revolute",) * 2,
            offsets=np.array([[0.0, 0, 0], [1.0, 0, 0]]),
            axes=np.array([[0.0, 0, 1]] * 2),
            mass=np.ones(2),
            com=np.array([[0.5, 0, 0]] * 2),
            inertia=np.array([[1 / 24, 1 / 24, 1 / 12]] * 2),
            gravity=np.zeros(3),
        )
    if kind == "N9":
        return dict(
            parents=np.arange(9) - 1,
            kinds=tuple("revolute" if i % 2 == 0 else "prismatic" for i in range(9)),
            offsets=np.array([[1 / 8, 1 / 16, 0]] * 9),
            axes=np.array(
                [((0, 0, 1), (1, 0, 0), (0, 1, 0))[i % 3] for i in range(9)], dtype=float
            ),
            mass=np.array([1 + (i % 3) / 2 for i in range(9)]),
            com=np.array([[1 / 16, 1 / 32, 0]] * 9),
            inertia=np.array([[1 / 64, 1 / 32, 3 / 64]] * 9),
            gravity=np.zeros(3),
        )
    return dict(
        parents=np.array([-1]),
        kinds=("prismatic",),
        offsets=np.zeros((1, 3)),
        axes=np.array([[1.0, 0, 0]]),
        mass=np.array([2.0]),
        com=np.zeros((1, 3)),
        inertia=np.array([[1 / 8] * 3]),
        gravity=np.zeros(3),
    )


def program(ph, data, q, v):
    n = len(data["parents"])
    model = ph.RobotModel(
        data["parents"],
        data["kinds"],
        data["offsets"],
        np.tile([0.0, 0, 0, 1], (n, 1)),
        data["axes"],
        data["mass"],
        data["com"],
        data["inertia"],
        data["gravity"],
    )
    drive = ph.DrivenRobotProgram(
        model, kp=np.zeros(n), kd=np.zeros(n), effort_limit=np.full(n, 8.0), delay_steps=0
    )
    return drive, drive.initial_state(ph.RobotState(np.asarray(q), np.asarray(v)))


def make_worlds(ph):
    rows, worlds, controls = {}, [], {}
    declarations = (
        (91, "N9", [(i % 3 - 1) / (8 if i % 2 == 0 else 64) for i in range(9)], [0.0] * 9),
        (7, "sphere", [0.25, -0.125, 0.5], [0.125, 0, -0.0625]),
        (53, "2R", [0.37, -0.51], [0.5, -0.25]),
        (23, "slider", [1 / 8], [-1 / 16]),
    )
    for key, kind, q, v in declarations:
        if kind == "sphere":
            spec = ph.FreeSphere(2.0, 1 / 16, position=q, velocity=v, gravity=(0, 0, 0))
            effort, data = np.array([0.25, 0, 0]), None
            controls[key] = effort
        else:
            data = model_data(kind)
            drive, initial = program(ph, data, q, v)
            effort = (
                np.array([1.0, 1 / 6])
                if kind == "2R"
                else np.array([(-1) ** i / 16 for i in range(9)])
                if kind == "N9"
                else np.array([0.25])
            )
            spec = (
                ph.PrismaticSlider(drive, initial)
                if kind == "slider"
                else ph.FixedRobotTree(drive, initial)
            )
            n = len(q)
            controls[key] = drive.inputs(
                np.zeros(n),
                np.zeros(n),
                feedforward=np.zeros(n),
                loads=ph.RobotInputs(effort, np.zeros((n, 3)), np.zeros((n, 3))),
            )
        worlds.append((key, spec))
        rows[key] = dict(kind=kind, model=data, q=np.array(q), v=np.array(v), effort=effort)
    return worlds, controls, rows


def box(low, high):
    return np.array(list(product(*zip(low, high, strict=True))), dtype=float)


def make_geometry(ph, rows, snapshots):
    shapes, bindings, materials, excluded = [], [], {}, []
    for world, row in rows.items():
        kind = row["kind"]
        if kind == "sphere":
            declarations = ((None, "sphere", np.zeros((1, 3)), 1 / 16, (101,)),)
        elif kind == "slider":
            declarations = (
                (0, "capsule", np.array([[-0.5, 0, 0], [0.5, 0, 0]]), 1 / 32, (201, 202)),
            )
        elif kind == "2R":
            declarations = (
                (0, "sphere", np.array([[0.25, -0.125, 0]]), 1 / 16, (101,)),
                (1, "capsule", np.array([[-0.5, 0, 0], [0.5, 0, 0]]), 1 / 32, (201, 202)),
            )
        else:
            local = (
                box([-0.25, -0.125, -0.0625], [0.25, 0.125, 0.0625])
                @ rotation([0, 0, 1], np.pi / 4).T
            )
            declarations = ((8, "hull", local, 0.0, tuple(range(301, 309))),)
        dynamic_keys = []
        for number, (group, shape_kind, local, radius, ids) in enumerate(declarations):
            key = ph.CollisionKey(world, 100 + (0 if group is None else group), 11 + number)
            shapes.append(ph.CollisionShape(key, shape_kind, ids, radius, revision=7))
            if kind == "sphere":
                binding = ph.ResidentSphereBinding(key, world)
            elif kind == "slider":
                binding = ph.ResidentSliderBinding(
                    key,
                    world,
                    ph.CollisionMotion(local),
                    origin_m=(0, 0, 0),
                    orientation_xyzw=(0, 0, 0, 1),
                    axis_local=(1, 0, 0),
                )
            else:
                binding = ph.ResidentTreeBinding(
                    key, world, group, local, topology=snapshots[world].topology
                )
            bindings.append(binding)
            materials[key] = dict(
                world=world,
                group=group,
                local_points_m=local,
                radius_m=radius,
                vertex_ids=ids,
                kind=shape_kind,
                revision=7,
                dynamic=True,
            )
            dynamic_keys.append(key)
        excluded.extend(combinations(dynamic_keys, 2))
        for body, low, high in ((900, [-4.0] * 3, [4.0] * 3), (901, [20.0] * 3, [21.0] * 3)):
            key = ph.CollisionKey(world, body, 1)
            points = box(low, high)
            shapes.append(ph.CollisionShape(key, "hull", tuple(range(8)), revision=3))
            bindings.append(ph.ResidentStaticBinding(key, ph.CollisionMotion(points)))
            materials[key] = dict(
                world=world,
                local_points_m=points,
                radius_m=0.0,
                vertex_ids=tuple(range(8)),
                kind="hull",
                revision=3,
                dynamic=False,
            )
    # Registration is intentionally reverse ordered; keys are model-local and
    # repeat across independent worlds, never array indices.
    return (
        ph.CollisionModel(tuple(reversed(shapes)), tuple(excluded)),
        tuple(reversed(bindings)),
        materials,
    )


def collision_record(result):
    return {
        name: json_value(getattr(result, name))
        for name in (
            "topology_generation",
            "state_epoch",
            "world_keys",
            "ticks",
            "shape_keys",
            "shape_revisions",
            "bounds",
            "registered_pairs",
            "candidates",
        )
    }


def check_geometry(result, rows, snapshots, materials):
    expected_bounds = {}
    kinematics = {}
    for world, row in rows.items():
        if row["kind"] != "sphere":
            state = snapshots[world]
            kinematics[world] = oracle(row["model"], state.position, state.velocity)
    assert result.shape_keys == tuple(sorted(materials))
    assert result.shape_revisions == tuple(materials[key]["revision"] for key in result.shape_keys)
    for index, key in enumerate(result.shape_keys):
        material = materials[key]
        points = material["local_points_m"]
        if material["dynamic"]:
            if rows[key.world]["kind"] == "sphere":
                points = np.asarray([snapshots[key.world].position])
            else:
                pose = kinematics[key.world]
                group = material["group"]
                points = pose["position"][group] + points @ pose["rotation"][group].T
        radius = material["radius_m"]
        low, high = points.min(0) - radius, points.max(0) + radius
        actual = np.asarray(result.bounds[index], dtype=float)
        assert np.isfinite(actual).all()
        # Strict material containment uses the ACTUAL stored q, with independent
        # FP64 geometry. No product diagnostic or measured bound defines truth.
        assert np.all(actual[:3] <= low) and np.all(actual[3:] >= high), str(key)
        expected_bounds[key] = (low, high)
    registered, candidates = [], []
    for first, second in combinations(result.shape_keys, 2):
        if (
            first.world != second.world
            or materials[first]["dynamic"] == materials[second]["dynamic"]
        ):
            continue
        registered.append((first, second))
        a, b = expected_bounds[first], expected_bounds[second]
        if np.all(a[0] <= b[1]) and np.all(b[0] <= a[1]):
            candidates.append((first, second))
    assert result.registered_pairs == tuple(registered)
    assert result.candidates == tuple(candidates)
    assert len(candidates) == 5  # one broad containing support per actual shape
    assert result.world_keys == tuple(sorted(rows))
    assert result.ticks.tolist() == [snapshots[world].tick for world in result.world_keys]
    return {str(key): dict(low_m=low, high_m=high) for key, (low, high) in expected_bounds.items()}


def advance_oracle(rows, expected):
    dynamics = {}
    for world, row in rows.items():
        q, v = expected[world]
        if row["kind"] == "sphere":
            acceleration = row["effort"] / 2
            terms = None
        else:
            terms = oracle(row["model"], q, v)
            if row["kind"] == "2R":
                c, s = cos(q[1]), sin(q[1])
                mass = np.array([[5 / 3 + c, 1 / 3 + c / 2], [1 / 3 + c / 2, 1 / 3]])
                bias = np.array([-s * (2 * v[0] * v[1] + v[1] ** 2) / 2, s * v[0] ** 2 / 2])
                np.testing.assert_allclose(terms["mass"], mass, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(terms["bias"], bias, rtol=1e-12, atol=1e-12)
                assert abs(mass[0, 1]) > 0.1
            acceleration = np.linalg.solve(terms["mass"], row["effort"] - terms["bias"])
        next_v = v + H * acceleration
        expected[world] = (q + H * next_v, next_v)
        dynamics[world] = dict(
            entry_q=q,
            entry_v=v,
            acceleration=acceleration,
            mass=None if terms is None else terms["mass"],
            bias=None if terms is None else terms["bias"],
        )
    return dynamics


def check_state(rows, actual, expected, tree_frames, summary, tick):
    for world, row in rows.items():
        state = actual[world]
        assert state.tick == tick
        close(state.position, expected[world][0], atol=WINDOW_ATOL)
        close(state.velocity, expected[world][1], atol=WINDOW_ATOL)
        q, v = np.asarray(state.position), np.asarray(state.velocity)
        if row["kind"] == "sphere":
            energy = float(v @ v)  # 1/2 * 2 kg
            close(state.applied_effort, row["effort"])
        else:
            terms = oracle(row["model"], q, v)
            energy = float(0.5 * v @ terms["mass"] @ v)
            close(
                state.applied_effort, np.zeros(len(q))
            )  # zero PD/FF; real external effort separate
            if world in TREE_KEYS:
                assert len(state.position) == (2 if world == 53 else 9)
                close(state.generalized_applied, row["effort"])
                assert state.history == () and np.asarray(state.control).shape == (len(q), 10)
                frame = tree_frames[world]
                assert frame.tick == tick and frame.joint_types == row["model"]["kinds"]
                np.testing.assert_array_equal(frame.position, state.position)
                np.testing.assert_array_equal(frame.velocity, state.velocity)
                close(frame.link_position_m, terms["position"])
                close(frame.link_rotation, terms["rotation"])
                close(frame.com_position_m, terms["com"])
                assert set(summary[world]) == {"kinetic_energy_j"}
        close(summary[world]["kinetic_energy_j"], energy)


def expect_rejection(function):
    try:
        function()
    except (ValueError, TypeError) as error:
        return dict(type=type(error).__name__, message=str(error))
    raise AssertionError("invalid public input/result was not rejected")


def run_window(ph, mode, output, record):
    worlds, controls, rows = make_worlds(ph)
    record.update(h_s=H, continuous_window_steps=WINDOW_STEPS, input=rows, trace=[])
    with ph.TyxWorldBatch(worlds, mode, cache_root=output / "cache") as batch:
        batch.set_controls(controls)
        initial = batch.snapshot()
        model, bindings, materials = make_geometry(ph, rows, initial)
        record.update(
            initial=initial,
            collision_model=model,
            original_material={str(key): value for key, value in materials.items()},
        )
        expected = {
            key: (np.asarray(value.position), np.asarray(value.velocity))
            for key, value in initial.items()
        }
        with batch.prepare_collision(model, bindings) as plan:
            beginning = plan.query()
            record["initial_geometry"] = collision_record(beginning)
            check_geometry(beginning, rows, initial, materials)
            previous = initial
            checkpoint20 = result20 = replay_expected = None
            for tick in range(1, WINDOW_STEPS + 1):
                frame = dict(tick=tick, entry=previous, checks_complete=False)
                record["trace"].append(frame)
                accepted = batch.step(H)
                frame["accepted"] = accepted
                actual = batch.snapshot()
                frame["actual"] = actual
                tree_frames = batch.tree_kinematics(TREE_KEYS)
                frame["tree_kinematics"] = tree_frames
                summary = batch.summary()
                frame["summary"] = summary
                geometry = plan.query()
                frame["geometry"] = collision_record(geometry)
                dynamics = advance_oracle(rows, expected)
                frame.update(
                    oracle_dynamics=dynamics,
                    oracle_state={
                        key: dict(q=q.copy(), v=v.copy()) for key, (q, v) in expected.items()
                    },
                    batch_stats=batch.stats(),
                    plan_stats=plan.stats(),
                )
                assert tuple(accepted) == (91, 7, 53, 23) and all(accepted.values())
                check_state(rows, actual, expected, tree_frames, summary, tick)
                frame["oracle_material_bounds"] = check_geometry(geometry, rows, actual, materials)
                frame["checks_complete"] = True
                if tick == 20:
                    checkpoint20, result20 = actual, geometry
                if tick == 21:
                    replay_expected = (actual, geometry, tree_frames)
                previous = actual
            record["window_final"] = previous
            final_epoch = batch.stats()["state_epoch"]
            # Invalid public restore must not publish a partial bank or advance epoch.
            record["invalid_restore"] = expect_rejection(lambda: batch.restore({91: previous[53]}))
            assert batch.snapshot() == previous and batch.stats()["state_epoch"] == final_epoch
            plan.require_current(geometry)
            # Restore the original held commands/history with the actual step-20
            # checkpoint, then independently reproduce the already saved step21.
            batch.restore(checkpoint20)
            restored = batch.snapshot()
            assert restored == checkpoint20
            assert batch.stats()["state_epoch"] == final_epoch + 1
            stale = expect_rejection(lambda: plan.require_current(result20))
            restored_geometry = plan.query()
            np.testing.assert_array_equal(restored_geometry.bounds, result20.bounds)
            assert restored_geometry.candidates == result20.candidates
            assert batch.step(H) == {91: True, 7: True, 53: True, 23: True}
            repeated = batch.snapshot()
            repeated_frames = batch.tree_kinematics(TREE_KEYS)
            repeated_geometry = plan.query()
            record["restore_replay"] = dict(
                checkpoint=checkpoint20,
                restored=restored,
                stale_result_rejection=stale,
                restored_geometry=collision_record(restored_geometry),
                actual=repeated,
                expected_original_step21=replay_expected[0],
                tree_kinematics=repeated_frames,
                geometry=collision_record(repeated_geometry),
                batch_stats=batch.stats(),
                plan_stats=plan.stats(),
            )
            assert repeated == replay_expected[0]
            np.testing.assert_array_equal(repeated_geometry.bounds, replay_expected[1].bounds)
            assert repeated_geometry.candidates == replay_expected[1].candidates
            for key in TREE_KEYS:
                np.testing.assert_array_equal(
                    repeated_frames[key].link_position_m, replay_expected[2][key].link_position_m
                )
                np.testing.assert_array_equal(
                    repeated_frames[key].link_rotation, replay_expected[2][key].link_rotation
                )
            check_geometry(repeated_geometry, rows, repeated, materials)
            record["checks"] = dict(
                full_40_step_window=True,
                complete_tree_rows=True,
                coupled_state_and_energy=True,
                independent_fk_and_all_material_bounds=True,
                exact_registered_candidates=True,
                invalid_restore_no_publish=True,
                same_tick_old_result_rejected=True,
                restored_step21_exact_replay=True,
            )
            record["batch_stats_before_close"] = batch.stats()
            assert batch.stats()["completed_steps"] == 41
            assert batch.stats()["accepted_world_steps"] == 164
            assert batch.stats()["rejected_world_steps"] == 0
            if mode == "cuda":
                assert batch.stats()["tree_state_downloaded_bytes"] > 0
                assert batch.stats()["submitted_kernels"] > 0
                assert batch.stats()["device_allocations"] > 0
    record["batch_stats_after_close"] = batch.stats()
    assert batch.stats()["collision_plan_live_count"] == 0
    assert batch.stats()["device_owned_bytes"] == 0
    record["checks"]["resource_close"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wheel-site", type=Path, required=True)
    parser.add_argument("--expected-wheel", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    args = parser.parse_args()
    site = args.wheel_site.resolve(strict=True)
    checkout = args.checkout.resolve(strict=True)
    script = Path(__file__).resolve(strict=True)
    output = args.output_dir.resolve()
    assert __debug__ and sys.flags.isolated, "run the copied consumer with python -I"
    assert Path.cwd().resolve() == script.parent
    assert not script.is_relative_to(checkout) and not site.is_relative_to(checkout)
    assert not output.is_relative_to(checkout)
    assert "phymeo" not in sys.modules
    output.mkdir(parents=True, exist_ok=False)
    record = dict(
        status="running",
        mode=args.mode,
        errors=[],
        script=dict(path=str(script), sha256=hashlib.sha256(script.read_bytes()).hexdigest()),
        limits="Wheel smoke only; current-pose broad phase, no response/CCD, mimic, floating root, performance or complete C4-matrix claim.",
        gates=dict(
            rtol=RTOL,
            state_atol=ATOL,
            window_atol=WINDOW_ATOL,
            material_containment="strict at actual stored q",
            candidate_domain="same-world dynamic/static",
        ),
    )
    sys.path.insert(0, str(site))
    try:
        import phymeo as ph

        assert Path(ph.__file__).resolve().is_relative_to(site)
        installed = distribution("phymeo")
        assert Path(installed.locate_file("")).resolve() == site
        record["wheel"] = archive_identity(installed, args.expected_wheel)
        record["modules_before"] = module_provenance(site, args.expected_wheel)
        names = (
            "TyxWorldBatch",
            "FixedRobotTree",
            "FreeSphere",
            "PrismaticSlider",
            "ResidentTreeBinding",
            "ResidentSphereBinding",
            "ResidentSliderBinding",
            "ResidentStaticBinding",
        )
        assert all(hasattr(ph, name) for name in names), "public resident inputs are missing"
        record["public_api"] = names
        record["versions"] = dict(
            numpy=np.__version__,
            tyx=distribution("tyx-compute").version,
            python=sys.version,
            phymeo=installed.version,
        )
        run_window(ph, args.mode, output, record)
        record["status"] = "passed"
    except Exception as error:
        record["status"] = "failed"
        record["errors"].append(
            dict(
                stage="wheel_import_or_window",
                type=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
            )
        )
    # Verify every late-loaded product module even if a physical check failed.
    try:
        if "phymeo" in sys.modules:
            record["modules_after"] = module_provenance(site, args.expected_wheel)
    except Exception as error:
        record["status"] = "failed"
        record["errors"].append(
            dict(
                stage="final_module_provenance",
                type=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
            )
        )
    trace_path = output / "resident-tree-wheel.trace.json.gz"
    with gzip.open(trace_path, "xt", encoding="utf-8") as stream:
        json.dump(json_value(record), stream, allow_nan=False, separators=(",", ":"))
    summary = dict(
        status=record["status"],
        raw_exit=0 if record["status"] == "passed" else 1,
        mode=args.mode,
        attempted_window_steps=len(record.get("trace", ())),
        full_window_steps=sum(
            bool(frame.get("checks_complete")) for frame in record.get("trace", ())
        ),
        checks=record.get("checks", {}),
        wheel=record.get("wheel"),
        verified_module_count=len(record.get("modules_after", ())),
        batch_stats=record.get("batch_stats_after_close", record.get("batch_stats_before_close")),
        trace=str(trace_path),
        errors=record["errors"],
        limits=record["limits"],
    )
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(json_value(summary), stream, indent=2, allow_nan=False)
    print(json.dumps(json_value(summary), allow_nan=False))
    return summary["raw_exit"]


if __name__ == "__main__":
    raise SystemExit(main())
