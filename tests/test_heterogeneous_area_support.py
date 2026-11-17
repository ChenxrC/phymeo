"""I3l preregistered heterogeneous support: one saved CUDA trajectory only.

The scalar BE oracle and FP64 rollout below do not call surface, row-assembly
or solver helpers. The native sphere is a different model/law baseline; no
area stiffness, plateau or energy accuracy claim is applied to that world.
"""

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest

import experiments.area_pressure.heterogeneous_support as support

H, DURATION, STEPS, GRAVITY = 0.001, 0.45, 450, 9.81
KEYS = (41, 7, 305, 11)
AREA_KEYS = (41, 7, 305)
WORLD_MODELS = {"41": "soft", "7": "offset", "305": "soft", "11": "native"}
REGISTERED = {
    "soft": dict(
        mass=0.30, damping=70.0, pressure=100000.0, area=4e-4, stiffness=4000.0, x=0.0, half_y=0.01
    ),
    "offset": dict(
        mass=0.45,
        damping=90.0,
        pressure=150000.0,
        area=3e-4,
        stiffness=4500.0,
        x=0.004,
        half_y=0.0075,
    ),
    "native": dict(
        mass=0.30, damping=70.0, pressure=None, area=None, stiffness=None, x=0.0, half_y=0.01
    ),
}
INITIAL = {41: (0.002, 0.0), 7: (0.003, 0.0), 305: (0.001, -0.03), 11: (0.002, 0.0)}


def _phase(step):
    return ("gravity", 0.0) if step < 150 else ("loaded", -1.5) if step < 300 else ("release", 7.0)


def _assert_manifest(payload):
    assert payload["schema"] == "heterogeneous-area-support-v1"
    assert payload["h"] == H and payload["duration"] == DURATION
    assert list(payload["world_order"]) == list(KEYS)
    assert set(payload["models"]) == set(REGISTERED)
    assert set(payload["worlds"]) == set(WORLD_MODELS)
    assert payload["stages"] == [
        {"end_s": 0.15, "force_n": 0.0},
        {"end_s": 0.30, "force_n": -1.5},
        {"end_s": 0.45, "force_n": 7.0},
    ]
    for model, expected in REGISTERED.items():
        actual = payload["models"][model]
        assert actual["mass_kg"] == expected["mass"]
        assert actual["damping_ns_m"] == expected["damping"]
        assert actual["center_pressure_pa"] == expected["pressure"]
        assert actual["reference_stiffness_n_m"] == expected["stiffness"]
        assert actual["plate_center_x_m"] == expected["x"]
        assert actual["body_half_height_m"] == 0.01
        assert list(actual["plate_half_size_m"]) == [0.01, expected["half_y"], 0.0025]
        assert actual["shape"] == ("sphere" if model == "native" else "cube")
    for key, model in WORLD_MODELS.items():
        row = payload["worlds"][key]
        assert row["model"] == model
        assert len(row["initial_q"]) == len(row["initial_v"]) == 1
        declaration = payload["models"][model]
        gap = (
            declaration["body_origin_z_m"]
            + row["initial_q"][0]
            - 0.01
            - (declaration["plate_origin_z_m"] + 0.0025)
        )
        assert gap == pytest.approx(INITIAL[int(key)][0], rel=0, abs=1e-15)
        assert row["initial_v"][0] == INITIAL[int(key)][1]


def _finite(value):
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return True


def _be_scalar(mass, damping, stiffness, delta, velocity, applied):
    effective_mass = mass + H * damping
    free_velocity = (mass * velocity + H * (applied - mass * GRAVITY)) / effective_mass
    force = (
        max(0.0, stiffness * (delta - H * free_velocity) / (1 + H * H * stiffness / effective_mass))
        if delta >= 0
        else 0.0
    )
    return force, free_velocity + H * force / effective_mass


def _energy(mass, stiffness, gap, velocity):
    # The datum is the body BOTTOM relative to the support top. An arbitrary
    # constant center-height potential must not inflate the 2% denominator.
    return (
        0.5 * mass * velocity * velocity
        + 0.5 * stiffness * max(-gap, 0.0) ** 2
        + mass * GRAVITY * gap
    )


def _array(value, width):
    return np.asarray(value, dtype=np.float64).reshape(-1, width)


def _surface_area(triangles):
    # Direct 3D triangle measure, independent of the quadrature constructor.
    return sum(
        0.5
        * float(
            np.linalg.norm(
                np.cross(
                    np.asarray(t["vertices"])[1] - t["vertices"][0],
                    np.asarray(t["vertices"])[2] - t["vertices"][0],
                )
            )
        )
        for t in triangles
    )


def _audit(payload):
    _assert_manifest(payload)
    assert payload["mode"] == "cuda" and payload["status"] == "complete"
    assert len(payload["records"]) == STEPS
    assert set(payload["initial"]) == set(WORLD_MODELS)
    assert set(payload["model_body_names"]) == set(REGISTERED)
    names = {model: tuple(values) for model, values in payload["model_body_names"].items()}
    body_ids = {
        model: (values.index("pressure"), values.index("plate")) for model, values in names.items()
    }
    previous = payload["initial"]
    references = {}
    metrics = {}
    for key in KEYS:
        model = WORLD_MODELS[str(key)]
        declared = payload["worlds"][str(key)]
        initial = previous[str(key)]
        assert initial["tick"] == 0 and initial["time"] == 0
        np.testing.assert_array_equal(
            initial["q"], np.asarray(declared["initial_q"], np.float32).astype(float)
        )
        np.testing.assert_array_equal(
            initial["v"], np.asarray(declared["initial_v"], np.float32).astype(float)
        )
        references[key] = (float(declared["initial_q"][0]), float(declared["initial_v"][0]))
        metrics[key] = dict(
            model=model,
            force=[],
            gap_in=[],
            gap_out=[],
            velocity=[],
            moment_y=[],
            empty=[],
            force_error=[],
            velocity_error=[],
            energy_residual=[],
            fp64_reference=[],
            fp64_abs_errors=[],
            area_errors=[],
            reaction_errors=[],
        )
    for index, record in enumerate(payload["records"]):
        phase, applied = _phase(index)
        assert record["step"] == index and record["phase"] == phase and record["force_n"] == applied
        assert record["elapsed_s"] >= 0 and set(record["worlds"]) == set(WORLD_MODELS)
        arrays = record["arrays"]
        catalogs = record["catalogs"]
        assert set(catalogs) == set(WORLD_MODELS)
        entry_meta = np.asarray(arrays["entry_meta"])
        area_meta = np.asarray(arrays["area_meta"])
        assert np.asarray(arrays["area_declared_mask"])[:, 0].tolist() == [1, 1, 1, 0]
        for row, key in enumerate(KEYS):
            world, model = str(key), WORLD_MODELS[str(key)]
            values, declared = record["worlds"][world], payload["models"][model]
            before, after = values["input"], values["output"]
            assert before["tick"] == index and after["tick"] == index + 1
            for field in ("q", "v", "act", "time", "tick"):
                assert before[field] == previous[world][field]
            assert 0 <= before["time"] < after["time"]
            assert abs(after["time"] - (index + 1) * H) <= 2e-7
            np.testing.assert_array_equal(arrays["entry_q"][row][:1], before["q"])
            np.testing.assert_array_equal(arrays["entry_v"][row][:1], before["v"])
            np.testing.assert_array_equal(arrays["q"][row][:1], after["q"])
            np.testing.assert_array_equal(arrays["v"][row][:1], after["v"])
            assert entry_meta[row, :3].tolist() == [1, index, index + 1]
            assert entry_meta[row, 5] == 0
            clock = arrays["entry_clock"][row]
            assert float(clock[0]) - float(clock[1]) == before["time"]
            assert float(clock[2]) - float(clock[3]) == after["time"]
            a, b = body_ids[model]
            entry_positions = np.asarray(arrays["entry_body_position"][row], float)
            output_positions = np.asarray(arrays["body_position"][row], float)
            gap_in = float(entry_positions[a, 2] - 0.01 - (entry_positions[b, 2] + 0.0025))
            gap_out = float(output_positions[a, 2] - 0.01 - (output_positions[b, 2] + 0.0025))
            np.testing.assert_array_equal(entry_positions[b], output_positions[b])
            pressure_wrench, plate_wrench = (
                values["body_wrenches"][body] for body in ("pressure", "plate")
            )
            for wrench in (pressure_wrench, plate_wrench):
                assert wrench["input_tick"] == index and wrench["output_tick"] == index + 1
                assert wrench["time_s"] == before["time"]
                np.testing.assert_array_equal(wrench["reference_point_world"], entry_positions[a])
            force = float(pressure_wrench["total_force_world"][2])
            moment_y = float(pressure_wrench["total_moment_world"][1])
            force_reaction = (
                np.asarray(pressure_wrench["total_force_world"]) + plate_wrench["total_force_world"]
            )
            moment_reaction = (
                np.asarray(pressure_wrench["total_moment_world"])
                + plate_wrench["total_moment_world"]
            )
            measured = metrics[key]
            measured["reaction_errors"].append(
                float(max(np.max(np.abs(force_reaction)), np.max(np.abs(moment_reaction))))
            )
            measured["force"].append(force)
            measured["gap_in"].append(gap_in)
            measured["gap_out"].append(gap_out)
            measured["velocity"].append(float(after["v"][0]))
            measured["moment_y"].append(moment_y)
            measured["empty"].append(not pressure_wrench["contacts"])
            assert values["dynamics"]["applied"] == [applied]
            if key not in AREA_KEYS:
                assert values["geometry"] == [] and values["area"] == [] and catalogs[world] == []
                assert area_meta[row].tolist() == [0, 0, 0, 0]
                continue
            mass, damping, stiffness = (
                REGISTERED[model]["mass"],
                REGISTERED[model]["damping"],
                REGISTERED[model]["stiffness"],
            )
            expected_force, expected_velocity = _be_scalar(
                mass, damping, stiffness, -gap_in, before["v"][0], applied
            )
            measured["force_error"].append(
                dict(
                    absolute=abs(force - expected_force),
                    bound=0.001 + 0.001 * abs(expected_force),
                    expected=expected_force,
                )
            )
            measured["velocity_error"].append(abs(after["v"][0] - expected_velocity))
            assert len(values["geometry"]) == len(values["area"]) == len(catalogs[world]) == 1
            geometry, pair, catalog = values["geometry"][0], values["area"][0], catalogs[world][0]
            surface = geometry["surface"]
            assert geometry["pressure_domain"] == "full"
            assert pair["body_ids"] == [a, b]
            assert pair["identity"] == surface["identity"] == catalog["identity"]
            assert pair["stamp"] == surface["stamp"] == catalog["stamp"]
            stamp = pair["stamp"]
            assert (
                stamp["world_id"] == world
                and stamp["step_index"] == index
                and stamp["sample_time_s"] == before["time"]
            )
            assert stamp["configuration_id"]
            assert (
                pair["sample_ids"] == catalog["sample_ids"]
                and pair["source_ids"] == catalog["source_ids"]
            )
            count = len(pair["normal_force"])
            assert area_meta[row].tolist() == [
                1,
                count,
                sum(f > 0 for f in pair["normal_force"]),
                0,
            ]
            assert (
                len(pair["dA"]) == len(pair["sample_ids"]) == 3 * len(surface["triangles"]) == count
            )
            represented_area = float(sum(pair["dA"]))
            triangle_area = _surface_area(surface["triangles"])
            expected_area = REGISTERED[model]["area"] if surface["triangles"] else 0.0
            measured["area_errors"].append(
                max(abs(represented_area - expected_area), abs(triangle_area - expected_area))
            )
            words = np.asarray(arrays["area_measure_words"][row][:count], dtype="<i4")
            np.testing.assert_array_equal(np.frombuffer(words.tobytes(), dtype="<f8"), pair["dA"])
            np.testing.assert_allclose(
                pressure_wrench["area_force_world"],
                pressure_wrench["total_force_world"],
                rtol=0,
                atol=1e-12,
            )
            assert not any(pressure_wrench["native_force_world"])
            if count:
                assert gap_in <= 0
                np.testing.assert_allclose(
                    pair["normals"], np.tile((0, 0, 1), (count, 1)), rtol=0, atol=1e-12
                )
                np.testing.assert_allclose(
                    _array(pair["force_on_a"], 3) + _array(pair["force_on_b"], 3),
                    0,
                    rtol=0,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    _array(pair["points"], 3)[:, 2],
                    entry_positions[b, 2] + 0.0025,
                    rtol=0,
                    atol=1e-12,
                )
            else:
                assert gap_in >= 0 and force == 0
            assert abs(force - sum(pair["normal_force"])) <= 2e-5
            assert abs(values["qfrc_constraint"][0] - sum(pair["normal_force"])) <= 2e-5
            np.testing.assert_allclose(
                values["dynamics"]["mass_matrix"], [[mass]], rtol=0, atol=2e-6
            )
            np.testing.assert_allclose(
                values["dynamics"]["bias"], [mass * GRAVITY], rtol=0, atol=2e-6
            )
            np.testing.assert_allclose(
                values["dynamics"]["passive"], [-damping * before["v"][0]], rtol=0, atol=2e-6
            )
            residual = (
                _energy(mass, stiffness, gap_out, after["v"][0])
                - _energy(mass, stiffness, gap_in, before["v"][0])
                - applied * (output_positions[a, 2] - entry_positions[a, 2])
                + H * damping * after["v"][0] ** 2
            )
            measured["energy_residual"].append(float(residual))
            # Independent FP64 state is never overwritten by product outputs.
            rq, rv = references[key]
            reference_gap = (
                declared["body_origin_z_m"] + rq - 0.01 - (declared["plate_origin_z_m"] + 0.0025)
            )
            rf, rvo = _be_scalar(mass, damping, stiffness, -reference_gap, rv, applied)
            rqo = rq + H * rvo
            measured["fp64_reference"].append(
                dict(
                    step=index,
                    input_q=rq,
                    input_v=rv,
                    input_gap_m=reference_gap,
                    force_n=rf,
                    output_q=rqo,
                    output_v=rvo,
                )
            )
            measured["fp64_abs_errors"].append(
                dict(q=abs(after["q"][0] - rqo), v=abs(after["v"][0] - rvo), force=abs(force - rf))
            )
            references[key] = (rqo, rvo)
        previous = {key: values["output"] for key, values in record["worlds"].items()}
    checked = dict(
        schema="heterogeneous-area-support-checked-v1",
        gates={"finite": _finite(payload), "complete": len(payload["records"]) == STEPS},
        worlds={},
    )
    for key, measured in metrics.items():
        model = measured["model"]
        forces, gaps, velocities = measured["force"], measured["gap_out"], measured["velocity"]
        positive = [i for i, value in enumerate(forces) if value > 0]
        gates = dict(
            initial_empty=measured["empty"][0],
            born=bool(positive) and any(not v for v in measured["empty"]),
            final_separated=all(measured["empty"][410:])
            and all(f == 0 for f in forces[410:])
            and min(gaps[410:]) > 0,
            reaction=max(measured["reaction_errors"]) <= 2e-6,
        )
        result = dict(
            gates=gates,
            peak_force_n=max(forces),
            final_gap_m=gaps[-1],
            final_velocity_m_s=velocities[-1],
            first_positive_step=positive[0] if positive else None,
            last_positive_step=positive[-1] if positive else None,
            max_reaction_error=max(measured["reaction_errors"]),
        )
        if key in AREA_KEYS:
            mass, stiffness = REGISTERED[model]["mass"], REGISTERED[model]["stiffness"]
            initial_energy = _energy(
                mass, stiffness, measured["gap_in"][0], payload["initial"][str(key)]["v"][0]
            )
            assert initial_energy > 0
            residual_ratio = sum(max(v, 0.0) for v in measured["energy_residual"]) / initial_energy
            maximum_depth = max(0.0, -min(measured["gap_in"]), -min(gaps))
            gates.update(
                force_oracle=all(
                    error["absolute"] <= error["bound"] for error in measured["force_error"]
                ),
                velocity_oracle=max(measured["velocity_error"]) <= 3e-5,
                depth=maximum_depth < 0.0025,
                true_area=max(measured["area_errors"]) <= 1e-12 + 1e-9 * REGISTERED[model]["area"],
                release_gap=min(gaps[410:]) > 0.001,
                energy=residual_ratio <= 0.02,
            )
            plateaus = []
            for start, stop, applied in ((120, 150, 0.0), (270, 300, -1.5)):
                target = mass * GRAVITY - applied
                mean_force = float(np.mean(forces[start:stop]))
                mean_depth = -float(np.mean(gaps[start:stop]))
                maximum_speed = max(abs(v) for v in velocities[start:stop])
                ok = (
                    abs(mean_force - target) <= 0.03 * target
                    and abs(mean_depth - target / stiffness) <= 0.03 * target / stiffness
                    and maximum_speed <= 0.002
                )
                gates[f"plateau_{start}_{stop}"] = bool(ok)
                plateaus.append(
                    dict(
                        input_step_range=[start, stop],
                        target_force_n=target,
                        mean_force_n=mean_force,
                        target_depth_m=target / stiffness,
                        mean_depth_m=mean_depth,
                        max_abs_velocity_m_s=maximum_speed,
                    )
                )
            moment_errors = [
                abs(my + REGISTERED[model]["x"] * force)
                for my, force in zip(measured["moment_y"], forces, strict=True)
            ]
            if model == "offset":
                gates["eccentric_moment"] = all(
                    error <= 2e-6 + 2e-4 * abs(0.004 * force)
                    for error, force in zip(moment_errors, forces, strict=True)
                )
            result.update(
                plateaus=plateaus,
                max_depth_m=maximum_depth,
                max_force_error_n=max(e["absolute"] for e in measured["force_error"]),
                max_velocity_error_m_s=max(measured["velocity_error"]),
                max_area_error_m2=max(measured["area_errors"]),
                max_moment_error_nm=max(moment_errors),
                initial_positive_energy_j=initial_energy,
                positive_energy_residual_fraction=residual_ratio,
                max_fp64_errors={
                    name: max(e[name] for e in measured["fp64_abs_errors"])
                    for name in ("q", "v", "force")
                },
                fp64_reference=measured["fp64_reference"],
                fp64_abs_errors=measured["fp64_abs_errors"],
                per_step_force_errors=measured["force_error"],
                per_step_energy_residual_j=measured["energy_residual"],
            )
        checked["worlds"][str(key)] = result
    return checked


def test_registered_declarations_preserve_models_assets_and_independent_initial_states():
    import mujoco as mj

    specs, worlds, assets, manifest = support.case_declarations()
    _assert_manifest(manifest)
    assert tuple(spec.key for spec in specs) == ("soft", "offset", "native")
    assert tuple(world.world for world in worlds) == KEYS
    assert tuple(world.model_key for world in worlds) == ("soft", "offset", "soft", "native")
    assert set(assets) == {"soft", "offset"}
    for spec in specs:
        expected, model = REGISTERED[spec.key], spec.compilation.fresh_native_model()
        assert model.nq == model.nv == 1 and model.nu == model.na == 0
        assert (
            int(model.jnt_type[0]) == int(mj.mjtJoint.mjJNT_SLIDE) and not model.jnt_limited.any()
        )
        np.testing.assert_allclose(model.dof_damping, [expected["damping"]], rtol=0, atol=0)
        np.testing.assert_allclose(model.opt.gravity, (0, 0, -GRAVITY), rtol=0, atol=0)
        assert model.opt.timestep == H
        body, plate = int(model.body("pressure").id), int(model.body("plate").id)
        assert model.body_mass[body] == pytest.approx(expected["mass"], rel=1e-14)
        geoms = np.flatnonzero(model.geom_bodyid == body)
        assert len(geoms) == 1
        if spec.key == "native":
            assert spec.area_pressure_pairs == ()
            assert model.geom_type[geoms[0]] == mj.mjtGeom.mjGEOM_SPHERE
            assert model.geom_size[geoms[0], 0] == 0.01 and model.geom_condim[geoms[0]] == 1
            assert not model.geom_friction.any()
            continue
        assert spec.area_pressure_pairs == (("pressure", "plate"),)
        assert not model.geom_contype.any() and not model.geom_conaffinity.any()
        (asset,) = assets[spec.key]
        assert (asset.identity.compliant_body, asset.identity.rigid_body) == ("pressure", "plate")
        mesh = asset.pressure_local
        assert mesh.pressure_domain == "full" and len(mesh.cells) == 12 and len(mesh.node_ids) == 9
        assert sorted(mesh.nodal_pressure) == [0.0] * 8 + [expected["pressure"]]
        assert (
            np.max(mesh.world_vertices[:, 2]) == 0.01 and np.min(mesh.world_vertices[:, 2]) == -0.01
        )
        assert (
            np.max(mesh.world_vertices[:, 0]) == 0.02 and np.min(mesh.world_vertices[:, 0]) == -0.02
        )
        face = asset.rigid_local
        assert len(face.faces) == 2
        world_vertices = face.world_vertices + model.body_pos[plate]
        assert np.mean(world_vertices[:, 0]) == pytest.approx(expected["x"], abs=1e-15)
        assert np.ptp(world_vertices[:, 0]) == pytest.approx(0.02, abs=1e-15)
        assert np.ptp(world_vertices[:, 1]) == pytest.approx(2 * expected["half_y"], abs=1e-15)
        np.testing.assert_allclose(
            world_vertices[:, 2],
            manifest["models"][spec.key]["plate_origin_z_m"] + 0.0025,
            rtol=0,
            atol=1e-15,
        )
        with pytest.raises(ValueError):
            mesh.world_vertices.setflags(write=True)
    for world in worlds:
        np.testing.assert_array_equal(
            world.initial_state.q, manifest["worlds"][str(world.world)]["initial_q"]
        )
        np.testing.assert_array_equal(
            world.initial_state.v, manifest["worlds"][str(world.world)]["initial_v"]
        )
    assert manifest["worlds"]["41"]["initial_q"] != manifest["worlds"]["305"]["initial_q"]


def test_run_path_and_mode_preflight_do_not_construct_or_replace(tmp_path, monkeypatch):
    def forbidden_declarations():
        raise AssertionError("invalid run input reached model construction")

    monkeypatch.setattr(support, "case_declarations", forbidden_declarations)
    existing = tmp_path / "existing.json.gz"
    existing.write_bytes(b"preserve these original bytes")
    with pytest.raises(FileExistsError):
        support.run_case(existing, mode="cuda")
    assert existing.read_bytes() == b"preserve these original bytes"
    for path, mode in (
        (Path("relative.json.gz"), "cuda"),
        (tmp_path / "wrong.json", "cuda"),
        (tmp_path / "invalid.json.gz", "invalid"),
    ):
        with pytest.raises(ValueError):
            support.run_case(path, mode=mode)
    assert not (tmp_path / "wrong.json").exists() and not (tmp_path / "invalid.json.gz").exists()


@pytest.mark.cuda
def test_one_saved_heterogeneous_area_support_window_meets_registered_gates(tmp_path):
    trajectory = tmp_path / "trajectory.json.gz"
    payload = support.run_case(trajectory, mode="cuda")  # Exactly one 450-step call.
    assert trajectory.is_file()
    with gzip.open(trajectory, "rt", encoding="utf-8") as stream:
        stored = json.load(stream)
    assert stored == json.loads(json.dumps(payload, allow_nan=False))
    checked = _audit(stored)
    summary = tmp_path / "checked-summary.json"
    summary.write_text(
        json.dumps(checked, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    failures = [name for name, passed in checked["gates"].items() if not passed]
    failures.extend(
        f"world={world}:{name}"
        for world, result in checked["worlds"].items()
        for name, passed in result["gates"].items()
        if not passed
    )
    assert not failures, "; ".join(failures)
