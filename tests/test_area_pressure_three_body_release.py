"""I3o independent two-row complementarity oracle and fixed CUDA release.

The oracle enumerates active sets algebraically; it never calls the production
intersection, quadrature, Jacobian, row-assembly or constraint-solver helpers.
"""

import gzip
import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import experiments.area_pressure.three_body_release as experiment

H = 0.001
MASSES = np.array((0.5, 1.25, 0.75))
MASS = np.diag(MASSES)
J = np.array(((1.0, -1.0, 0.0), (1.0, 0.0, -1.0)))
K = np.array((4500.0, 9000.0))
AREAS = np.array((0.000225, 0.00045))
NAMES = ("pressure", "left", "right")
PATCHES = ("left", "right")
MODEL = "three-body-release"
WORLD = 13


def _depth(q):
    q = np.asarray(q, dtype=np.float64)
    return q[1:] - (0.002 + q[0])


def _oracle(q, v):
    """Frozen-geometry BE with all three masses and the full cross-pair block."""
    v = np.asarray(v, dtype=np.float64)
    inverse = np.linalg.inv(MASS)
    block = np.diag(1 / K) + H * H * J @ inverse @ J.T
    depth = _depth(q)
    rhs = depth - H * (J @ v)
    available = depth >= 0  # Never invent a predictive row for a separated pair.
    candidates = []
    for active in ((), (0,), (1,), (0, 1)):
        if any(not available[i] for i in active):
            continue
        force = np.zeros(2)
        if len(active) == 1:
            (i,) = active
            force[i] = rhs[i] / block[i, i]
        elif len(active) == 2:
            a, b, c = block[0, 0], block[0, 1], block[1, 1]
            determinant = a * c - b * b
            assert determinant > 0
            force[:] = (
                (c * rhs[0] - b * rhs[1]) / determinant,
                (a * rhs[1] - b * rhs[0]) / determinant,
            )
        residual = block @ force - rhs
        inactive = [i for i in range(2) if available[i] and i not in active]
        if np.any(force < 0) or any(residual[i] < -1e-12 for i in inactive):
            continue
        if any(abs(residual[i]) > 1e-12 for i in active):
            continue
        candidates.append(force)
    assert candidates, "independent SPD active-set oracle found no feasible branch"
    for candidate in candidates[1:]:
        np.testing.assert_allclose(candidate, candidates[0], rtol=0, atol=1e-12)
    force = candidates[0]
    return force, v + H * inverse @ J.T @ force, block, rhs


def _energy(q, v):
    v = np.asarray(v, dtype=np.float64)
    return 0.5 * float(v @ MASS @ v) + 0.5 * float(K @ np.maximum(_depth(q), 0.0) ** 2)


def _load(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _audit(payload):
    reverse = payload["reverse_first_step"]
    count = 1 if reverse else 40
    assert payload["schema"] == "three-body-area-release-v1"
    assert payload["status"] == "complete" and payload["mode"] == "cuda"
    assert payload["h"] == H and payload["steps"] == count
    assert (
        payload["accepted_steps"] == payload["complete_records"] == len(payload["records"]) == count
    )
    assert payload["body_order"] == list(NAMES)
    assert payload["pair_order"] == list(reversed(PATCHES) if reverse else PATCHES)
    assert payload["mass_kg"] == list(MASSES)
    names = tuple(payload["body_names"])
    body_ids = [names.index(name) for name in NAMES]
    initial = payload["initial"]
    reference_q, reference_v = np.array((-0.003, 0.0, 0.0)), np.zeros(3)
    previous = initial
    initial_energy = _energy(initial["q"], initial["v"])
    assert initial_energy > 0
    total_initial_momentum = float(MASSES @ initial["v"])
    metrics = dict(
        force_errors=[],
        velocity_errors=[],
        position_errors=[],
        momentum_errors=[],
        positive_energy_residuals=[],
        fp64_reference=[],
        fp64_errors=[],
        geometric_pairs=[],
        loaded_pairs=[],
    )
    first_forces = None
    for step, record in enumerate(payload["records"]):
        assert (
            record["step"] == step and record["state_accepted"] and record["observation_complete"]
        )
        assert record["readback_calls"] == 1
        before, after, receipt = record["input"], record["output"], record["receipt"]
        assert before == previous and before["tick"] == step and after["tick"] == step + 1
        assert receipt["input_time"] == before["time"] and receipt["output_time"] == after["time"]
        assert receipt["contact_time"] == before["time"]
        assert receipt["contact_phase"] == "tyx-euler-input-constraint-evaluation"
        assert before["time"] < after["time"]
        np.testing.assert_array_equal(receipt["input_q"], before["q"])
        np.testing.assert_array_equal(receipt["input_v"], before["v"])
        values, catalogs = record["arrays"], record["catalogs"][str(WORLD)]
        assert values["entry_meta"][0] == [1, step, step + 1, 0, 0, 0]
        assert values["area_declared_mask"] == [[1]]
        assert not np.any(values["entry_contact_ids"]) and not np.any(values["entry_contact_data"])
        np.testing.assert_array_equal(values["entry_q"][0][:3], before["q"])
        np.testing.assert_array_equal(values["entry_v"][0][:3], before["v"])
        np.testing.assert_array_equal(values["q"][0][:3], after["q"])
        np.testing.assert_array_equal(values["v"][0][:3], after["v"])
        dynamics = receipt["dynamics"]
        np.testing.assert_allclose(dynamics["mass_matrix"], MASS, rtol=0, atol=2e-6)
        for name in ("bias", "passive", "applied", "smooth_acceleration"):
            np.testing.assert_allclose(dynamics[name], 0, rtol=0, atol=1e-12)
        raw_pairs = receipt["raw_contact_fields"]["area_pressure"]
        assert len(raw_pairs) == len(record["geometry"]) == len(catalogs) == 2
        force = np.zeros(2)
        geometric, loaded = [], []
        samples = 0
        for ordinal, (pair, geometry, catalog) in enumerate(
            zip(raw_pairs, record["geometry"], catalogs, strict=True)
        ):
            patch = pair["identity"]["rigid_body"]
            index = PATCHES.index(patch)
            assert patch == payload["pair_order"][ordinal]
            assert pair["body_ids"] == [body_ids[0], body_ids[index + 1]]
            assert pair["identity"] == catalog["identity"] == geometry["surface"]["identity"]
            assert pair["stamp"] == catalog["stamp"] == geometry["surface"]["stamp"]
            assert (
                pair["stamp"]["step_index"] == step
                and pair["stamp"]["sample_time_s"] == before["time"]
            )
            assert (
                pair["sample_ids"] == catalog["sample_ids"]
                and pair["source_ids"] == catalog["source_ids"]
            )
            quad = pair["source_quadrature"]
            for key in ("identity", "stamp", "sample_ids", "source_ids", "dA"):
                assert quad[key] == pair[key]
            assert all(tuple(ids) == ("bottom", "face") for ids in pair["source_ids"])
            n = len(pair["normal_force"])
            assert n in (0, 3) and len(pair["dA"]) == n
            assert n == 3 * len(geometry["surface"]["triangles"])
            force[index] = sum(pair["normal_force"])
            if n:
                geometric.append(patch)
                np.testing.assert_allclose(
                    pair["dA"], np.full(3, AREAS[index] / 3), rtol=1e-9, atol=1e-12
                )
                np.testing.assert_allclose(
                    pair["normal_jacobian"], np.tile(J[index], (3, 1)), rtol=0, atol=2e-6
                )
                np.testing.assert_allclose(
                    pair["normals"], np.tile((0, 0, 1), (3, 1)), rtol=0, atol=1e-12
                )
                np.testing.assert_allclose(pair["stiffness"], K[index] / 3, rtol=2e-5, atol=2e-5)
                expected_points_z = values["entry_body_position"][0][body_ids[index + 1]][2]
                np.testing.assert_allclose(
                    np.asarray(pair["points"])[:, 2], expected_points_z, rtol=0, atol=2e-8
                )
                expected_fa = np.asarray(pair["normal_force"])[:, None] * np.array((0.0, 0.0, 1.0))
                np.testing.assert_allclose(pair["force_on_a"], expected_fa, rtol=0, atol=1e-12)
                np.testing.assert_allclose(
                    pair["torque_on_a_world_origin"],
                    np.cross(pair["points"], expected_fa),
                    rtol=0,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    np.asarray(pair["force_on_a"]) + pair["force_on_b"], 0, rtol=0, atol=1e-12
                )
                np.testing.assert_allclose(
                    np.asarray(pair["torque_on_a_world_origin"]) + pair["torque_on_b_world_origin"],
                    0,
                    rtol=0,
                    atol=1e-12,
                )
                words = np.asarray(
                    values["area_measure_words"][0][samples : samples + n], dtype="<i4"
                )
                np.testing.assert_array_equal(
                    np.frombuffer(words.tobytes(), dtype="<f8"), pair["dA"]
                )
                for local, ids in enumerate(values["area_ids"][0][samples : samples + n]):
                    assert ids == [1, ordinal, samples + local, body_ids[0], body_ids[index + 1], 1]
            if force[index] > 0:
                loaded.append(patch)
            samples += n
        assert values["area_meta"][0] == [
            1,
            samples,
            sum(f > 0 for p in raw_pairs for f in p["normal_force"]),
            0,
        ]
        assert all(row["constraint_type"] == 7 for row in receipt["constraints"])
        assert len(receipt["constraints"]) == samples
        np.testing.assert_allclose(receipt["qfrc_constraint"], J.T @ force, rtol=4e-4, atol=2e-5)
        body_wrenches = record["body_wrenches"]
        expected_fz = J.T @ force
        for index, name in enumerate(NAMES):
            wrench = body_wrenches[name]
            assert wrench["input_tick"] == step and wrench["output_tick"] == step + 1
            assert wrench["reference_point_world"] == [0.0, 0.0, 0.0]
            assert not any(wrench["native_force_world"]) and not any(wrench["native_moment_world"])
            np.testing.assert_allclose(
                wrench["total_force_world"], (0, 0, expected_fz[index]), rtol=4e-4, atol=2e-5
            )
        for key in ("total_force_world", "total_moment_world"):
            np.testing.assert_allclose(
                np.sum([body_wrenches[name][key] for name in NAMES], axis=0), 0, rtol=0, atol=2e-6
            )
        expected_f, expected_v, _, _ = _oracle(before["q"], before["v"])
        metrics["force_errors"].append(
            dict(
                absolute=np.abs(force - expected_f).tolist(),
                bound=(0.001 + 0.001 * np.abs(expected_f)).tolist(),
            )
        )
        metrics["velocity_errors"].append(
            float(np.max(np.abs(np.asarray(after["v"]) - expected_v)))
        )
        metrics["position_errors"].append(
            float(
                np.max(
                    np.abs(
                        np.asarray(after["q"])
                        - (np.asarray(before["q"]) + H * np.asarray(after["v"]))
                    )
                )
            )
        )
        metrics["momentum_errors"].append(abs(float(MASSES @ after["v"]) - total_initial_momentum))
        metrics["positive_energy_residuals"].append(
            max(0.0, _energy(after["q"], after["v"]) - _energy(before["q"], before["v"]))
        )
        reference_f, next_v, _, _ = _oracle(reference_q, reference_v)
        next_q = reference_q + H * next_v
        metrics["fp64_reference"].append(
            dict(
                input_q=reference_q.tolist(),
                input_v=reference_v.tolist(),
                force=reference_f.tolist(),
                output_q=next_q.tolist(),
                output_v=next_v.tolist(),
            )
        )
        metrics["fp64_errors"].append(
            dict(
                q=float(np.max(np.abs(np.asarray(after["q"]) - next_q))),
                v=float(np.max(np.abs(np.asarray(after["v"]) - next_v))),
                force=float(np.max(np.abs(force - reference_f))),
            )
        )
        reference_q, reference_v = next_q, next_v
        metrics["geometric_pairs"].append(geometric)
        metrics["loaded_pairs"].append(loaded)
        if first_forces is None:
            first_forces = force.copy()
        previous = after
    gates = dict(
        force=all(
            np.all(np.asarray(row["absolute"]) <= row["bound"]) for row in metrics["force_errors"]
        ),
        velocity=max(metrics["velocity_errors"]) <= 3e-5,
        position=max(metrics["position_errors"]) <= 2e-7,
        momentum=max(metrics["momentum_errors"]) <= 2e-5,
        energy=sum(metrics["positive_energy_residuals"]) <= 0.02 * initial_energy,
    )
    if not reverse:
        geometric_counts = [len(names) for names in metrics["geometric_pairs"]]
        loaded_counts = [len(names) for names in metrics["loaded_pairs"]]
        single = [
            i for i, count in enumerate(geometric_counts) if count == 1 and loaded_counts[i] == 1
        ]
        empty = [i for i, count in enumerate(geometric_counts) if count == 0]
        gates["two_single_empty"] = bool(
            geometric_counts[0] == loaded_counts[0] == 2
            and single
            and empty
            and single[0] < empty[0]
            and geometric_counts[-1] == loaded_counts[-1] == 0
        )
    coupled, _, block, rhs = _oracle(initial["q"], initial["v"])
    wrong_independent = rhs / np.diag(block)
    wrong_error = float(np.max(np.abs(wrong_independent - coupled)))
    gates["distinguishes_independent_pairs"] = bool(
        wrong_error > np.max(0.001 + 0.001 * np.abs(coupled))
        and np.max(np.abs(first_forces - coupled)) < wrong_error / 2
    )
    return dict(
        gates=gates,
        initial_energy_j=initial_energy,
        first_force_n=first_forces.tolist(),
        first_coupled_force_n=coupled.tolist(),
        wrong_independent_force_n=wrong_independent.tolist(),
        metrics=metrics,
    )


def test_fixed_declarations_geometry_and_four_set_oracle_are_independent():
    import mujoco as mj

    specs, worlds, assets, manifest = experiment.case_declarations()
    assert manifest["h"] == H and manifest["steps"] == 40 and manifest["constraint_capacity"] == 64
    assert manifest["mass_kg"] == list(MASSES) and manifest["body_order"] == list(NAMES)
    model = specs[0].compilation.fresh_native_model()
    assert model.nq == model.nv == 3 and model.nu == model.na == 0
    assert (
        not model.dof_damping.any()
        and not model.dof_frictionloss.any()
        and not model.opt.gravity.any()
    )
    assert not model.jnt_limited.any() and np.all(model.jnt_type == mj.mjtJoint.mjJNT_SLIDE)
    assert not model.geom_contype.any() and not model.geom_conaffinity.any()
    np.testing.assert_array_equal(worlds[0].initial_state.q, (-0.003, 0, 0))
    np.testing.assert_array_equal(worlds[0].initial_state.v, (0, 0, 0))
    for index, name in enumerate(NAMES):
        assert model.body_mass[int(model.body(name).id)] == pytest.approx(MASSES[index])
    for index, asset in enumerate(assets[MODEL]):
        assert asset.identity.rigid_body == PATCHES[index]
        assert asset.pressure_local.pressure_domain == "full"
        assert len(asset.pressure_local.cells) == 4 and len(asset.pressure_local.node_ids) == 5
        np.testing.assert_array_equal(asset.pressure_local.nodal_pressure, (0, 0, 0, 0, 400000))
        p = asset.rigid_local.world_vertices
        area = 0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))
        assert area == pytest.approx(AREAS[index], rel=1e-12)
    force, _, block, _ = _oracle((-0.003, 0, 0), (0, 0, 0))
    h, depth = Fraction(1, 1000), Fraction(1, 1000)
    a = Fraction(1, 4500) + h * h * (2 + Fraction(4, 5))
    b = 2 * h * h
    c = Fraction(1, 9000) + h * h * (2 + Fraction(4, 3))
    expected = (depth * (c - b) / (a * c - b * b), depth * (a - b) / (a * c - b * b))
    np.testing.assert_allclose(force, list(map(float, expected)), rtol=0, atol=1e-12)
    assert block[0, 1] == pytest.approx(H * H / 0.5, rel=0, abs=1e-18)
    for q, v, index in (((-0.003, 0, -0.002), (0, 0, 0), 0), ((-0.003, -0.002, 0), (0, 0, 0), 1)):
        f, _, bmat, rhs = _oracle(q, v)
        assert f[index] == rhs[index] / bmat[index, index] and f[1 - index] == 0
    assert not _oracle((-0.003, 0, 0), (2, 0, 0))[0].any()
    assert not _oracle((0, 0, 0), (-100, 0, 0))[0].any()


def test_fresh_output_preflight_and_late_export_failure_preserve_accepted_prefix(
    tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid input reached declarations")

    monkeypatch.setattr(experiment, "case_declarations", forbidden)
    old = tmp_path / "existing.json.gz"
    old.write_bytes(b"unchanged")
    with pytest.raises(FileExistsError):
        experiment.run_case(old)
    assert old.read_bytes() == b"unchanged"
    for output, kwargs in (
        (Path("relative.json.gz"), {}),
        (tmp_path / "bad.json", {}),
        (tmp_path / "mode.json.gz", {"mode": "interpret"}),
        (tmp_path / "flag.json.gz", {"reverse_first_step": "yes"}),
    ):
        with pytest.raises(ValueError):
            experiment.run_case(output, **kwargs)

    # Protocol-only fake accepts one transition, then observation fails. No
    # fake numeric output is used as a force, geometry or CUDA correctness oracle.
    class Workspace:
        binding = {}
        model_metadata = {MODEL: {"body_names": ("world", *NAMES)}}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def refresh(self, **kwargs):
            return object()

    class Batch:
        checkpoint_revision = 0

        def __init__(self, *args, **kwargs):
            self.tick = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def prepare_device_workspace(self, **kwargs):
            return Workspace()

        def integration_snapshot(self):
            return {
                WORLD: SimpleNamespace(
                    q=np.array((-0.003, 0.0, 0.0)),
                    v=np.zeros(3),
                    act=np.zeros(0),
                    time=self.tick * H,
                    tick=self.tick,
                )
            }

        @property
        def last_step(self):
            return SimpleNamespace(receipts={WORLD: {"protocol_fake": True}})

    class Scene:
        asset_sha256 = "protocol-only"
        last_geometry = {WORLD: ()}

        def __init__(self, batch, assets):
            self.batch, self.assets = batch, assets

        def step(self, h, **kwargs):
            self.batch.tick += 1
            self.batch.checkpoint_revision += 1
            return {WORLD: True}

    class Reader:
        def __init__(self, *args, **kwargs):
            pass

        def measure_many(self, *args, **kwargs):
            raise RuntimeError("late-observation-failure")

    monkeypatch.setattr(experiment, "case_declarations", lambda **kwargs: ((), (), {}, {}))
    monkeypatch.setattr(experiment, "MuJoCoTyxHeterogeneousBatch", Batch)
    monkeypatch.setattr(experiment, "MuJoCoTyxHeterogeneousAreaPressureScene", Scene)
    monkeypatch.setattr(experiment, "BodyContactWrenchReader", Reader)
    destination = tmp_path / "failed-prefix.json.gz"
    with pytest.raises(RuntimeError, match="late-observation-failure"):
        experiment.run_case(destination)
    failed = _load(destination)
    assert (
        failed["status"] == "failed"
        and failed["accepted_steps"] == 1
        and failed["complete_records"] == 0
    )
    assert len(failed["records"]) == 1
    (row,) = failed["records"]
    assert row["state_accepted"] and not row["observation_complete"]
    assert row["input"]["tick"] == 0 and row["output"]["tick"] == 1
    assert row["receipt"] == {"protocol_fake": True}


@pytest.mark.cuda
def test_forty_step_release_and_one_reversed_step_match_coupled_oracle(tmp_path):
    normal_path, reversed_path = tmp_path / "release.json.gz", tmp_path / "reverse-first.json.gz"
    normal = experiment.run_case(normal_path)
    reversed_case = experiment.run_case(reversed_path, reverse_first_step=True)
    assert _load(normal_path) == normal and _load(reversed_path) == reversed_case
    checked = {"forward": _audit(normal), "reverse_first": _audit(reversed_case)}
    first, reverse = normal["records"][0], reversed_case["records"][0]
    np.testing.assert_allclose(first["output"]["q"], reverse["output"]["q"], rtol=0, atol=2e-7)
    np.testing.assert_allclose(first["output"]["v"], reverse["output"]["v"], rtol=0, atol=3e-5)
    np.testing.assert_allclose(
        checked["forward"]["first_force_n"],
        checked["reverse_first"]["first_force_n"],
        rtol=0.001,
        atol=0.001,
    )
    (tmp_path / "checked-summary.json").write_text(
        json.dumps(checked, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    failures = [
        f"{case}:{name}"
        for case, result in checked.items()
        for name, passed in result["gates"].items()
        if not passed
    ]
    assert not failures, "; ".join(failures)
