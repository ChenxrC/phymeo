"""I3x historical force -> original native integrators, not live publication.

Exactly one FREE batch and one scalar-tree batch are integrated. Saved actual
forces are assembled into Q on HOST for this diagnostic. No friction solve,
pack kernel, native mj_step, Scene advance or state publication is performed.
"""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from test_tyx_area_friction_entry_pack import (
    FIXTURE_SHA,
    _decode,
    _equations,
    _fixture,
    _plain,
    _problem,
)

from phymeo.backends.mujoco_tyx import _floating_kernels, _tree_kernels
from phymeo.backends.tyx_area_friction import TyxAreaFrictionWorkspace

PACK_SHA = "883dc7603c69d5789ea1e697ec8d64be33cb85c998bd244e4e05aba1a6b235a0"
OUTPUTS = (
    "qnext",
    "vnext",
    "tnext",
    "factor",
    "rhs",
    "forward",
    "acceleration",
    "integration_acc",
    "constraint",
    "rows",
    "status",
)


def _load():
    path = Path(__file__).parent / "fixtures" / "area_friction_entry" / "pack-pipelines.json.gz"
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PACK_SHA
    payload = _decode(json.loads(gzip.decompress(raw)))
    assert payload["schema"] == "area-friction-entry-pack-v1" and payload["status"] == "PASS"
    assert payload["fixture_sha256"] == FIXTURE_SHA and len(payload["pipelines"]) == 2
    return _fixture(), payload


def _native_inputs(mj, original, pipeline, free):
    world, count, nv = ("7", 3, 6) if free else ("41", 1, 1)
    model = mj.MjModel.from_xml_string(original["xml"]["A" if free else "B"])
    assert (model.nq, model.nv) == ((7, 6) if free else (1, 1))
    assert not np.any(model.jnt_limited)
    receipt, initial = original["receipt"]["receipts"][world], original["initial"][world]
    assert initial["tick"] == 0 and initial["time"] == receipt["input_time"] == 0.0
    source = pipeline["source_after"]
    dynamics = receipt["dynamics"]
    assert dynamics["time"] == receipt["input_time"]
    for name in ("q", "v"):
        np.testing.assert_array_equal(receipt["input_" + name], initial[name])
        np.testing.assert_array_equal(dynamics["input_" + name], receipt["input_" + name])
    for lane in range(count):
        np.testing.assert_array_equal(
            source["source_mass"][lane, :nv, :nv], dynamics["mass_matrix"].astype(np.float32)
        )
        for name in ("v", "bias", "passive", "applied"):
            expected = dynamics["input_v" if name == "v" else name].astype(np.float32)
            np.testing.assert_array_equal(source["source_" + name][lane, :nv], expected)
        assert not source["source_status"][lane] and not pipeline["raw"]["meta"][lane, 0]
    h = float(np.float32(receipt["h"]))
    assert h == pipeline["constants"]["h"] == float(np.float32(model.opt.timestep))
    joints = np.empty((model.njnt, 4), np.int32)
    kinds = {
        int(mj.mjtJoint.mjJNT_FREE): 3,
        int(mj.mjtJoint.mjJNT_BALL): 2,
        int(mj.mjtJoint.mjJNT_HINGE): 1,
        int(mj.mjtJoint.mjJNT_SLIDE): 0,
    }
    for i in range(model.njnt):
        joints[i] = (
            kinds[int(model.jnt_type[i])],
            model.jnt_qposadr[i],
            model.jnt_dofadr[i],
            model.jnt_bodyid[i],
        )
    vectors = source["source_vectors"].copy()
    np.testing.assert_array_equal(vectors[1, :nv], model.dof_damping.astype(np.float32))
    # The pack fixture stores only consumed vector fields. Native integration
    # additionally needs the actual static unbounded-joint limits and qpos0.
    vectors[4, : model.nq] = model.qpos0
    vectors[5, :nv] = model.dof_armature
    vectors[7], vectors[8] = -np.inf, np.inf
    q = np.zeros((count, 8), np.float32)
    q[:, : model.nq] = receipt["input_q"]
    force = np.zeros((count, nv), np.float64)
    for lane in range(count):
        problem = _problem(pipeline["packed"], lane)
        contacts = pipeline["raw"]["contact"][lane, : len(problem.samples)]
        for sample, stored in zip(problem.samples, contacts, strict=True):
            g = sample.jacobian_a.astype(float) - sample.jacobian_b.astype(float)
            force[lane] += g.T @ stored[9:12].astype(float)
    applied = source["source_applied"][:count].copy()
    applied[:, :nv] = (applied[:, :nv].astype(float) + force).astype(np.float32)
    inputs = dict(
        vectors=vectors,
        joints=joints,
        mass=source["source_mass"][:count].copy(),
        bias=source["source_bias"][:count].copy(),
        passive=source["source_passive"][:count].copy(),
        applied=applied,
        q=q,
        v=source["source_v"][:count].copy(),
        clock=np.zeros((count, 2), np.float32),
    )
    return model, inputs, force, h


def _launch(workspace, host, model, h, free):
    import tyx as tx

    count = len(host["q"])
    buffers = {}
    shapes = dict(
        qnext=(count, 8),
        vnext=(count, 8),
        tnext=(count, 2),
        factor=(count, 8, 8),
        rhs=(count, 8),
        forward=(count, 8),
        acceleration=(count, 8),
        integration_acc=(count, 8),
        constraint=(count, 8),
        rows=(count, 20),
        status=(count,),
    )
    with tx.use_runtime(workspace._runtime):
        for name, array in host.items():
            buffer = tx.cuda_empty(array.shape, array.dtype)
            workspace._resources.add_buffer(buffer)
            buffer.copy_from(array)
            buffers[name] = buffer
        for name, shape in shapes.items():
            dtype = np.int32 if name == "status" else np.float32
            buffer = tx.cuda_empty(shape, dtype)
            workspace._resources.add_buffer(buffer)
            buffer.copy_from(np.zeros(shape, dtype))
            buffers[name] = buffer
        shared = tuple(
            buffers[name]
            for name in (
                "vectors",
                "mass",
                "bias",
                "passive",
                "applied",
                "q",
                "v",
                "clock",
                "qnext",
                "vnext",
                "tnext",
                "factor",
                "rhs",
                "forward",
                "acceleration",
                "integration_acc",
                "constraint",
                "rows",
                "status",
            )
        )
        if free:
            _floating_kernels.solve(
                count, model.nv, h, *shared, model.nq, model.njnt, buffers["joints"], mode="cuda"
            )
        else:
            _tree_kernels.solve(
                count, model.nv, model.nq, model.njnt, h, buffers["joints"], *shared, mode="cuda"
            )
        workspace._runtime.sync()
        return dict(
            outputs={name: buffers[name].to_numpy() for name in OUTPUTS},
            unchanged_inputs={name: buffers[name].to_numpy() for name in host},
        )


def _close(actual, expected, absolute, relative):
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    assert np.all(np.abs(a - b) <= absolute + relative * np.abs(b))


def _audit(mj, model, pipeline, host, generalized, h, record, free):
    outputs, count, nv = record["outputs"], len(host["q"]), model.nv
    np.testing.assert_array_equal(outputs["status"], np.zeros(count, np.int32))
    for name, array in host.items():
        assert record["unchanged_inputs"][name].tobytes() == array.tobytes()
    expected_clock = np.tile(np.array((h, 0.0), np.float32), (count, 1))
    np.testing.assert_array_equal(outputs["tnext"], expected_clock)
    audits = record["actual_velocity_certificates"] = []
    for lane in range(count):
        problem = _problem(pipeline["packed"], lane)
        velocity = outputs["vnext"][lane, :nv].astype(float)
        prior_root = pipeline["raw"]["vectors"][lane, 1, :nv].astype(float)
        ns = len(problem.samples)
        actual = pipeline["raw"]["contact"][lane, :ns]
        e = _equations(problem, velocity)
        audits.append(e)
        e["saved_root_velocity"] = prior_root
        e["actual_integrated_velocity"] = velocity
        e["velocity_max_error"] = float(np.max(np.abs(velocity - prior_root)))
        _close(velocity, prior_root, 5e-5, 5e-5)
        assert e["residual_norm"] <= 2e-5 * e["scale"]
        _close(actual[:, 0], e["fn"], 0.002, 0.002)
        _close(actual[:, 6:9], e["ft"], 0.002, 0.002)
        _close(actual[:, 9:12], e["force"], 0.002, 0.002)
        _close(actual[:, 12:15], e["moment"], 1e-5, 0.002)
        for sample, stored in zip(problem.samples, actual, strict=True):
            assert stored[0] >= 0
            assert (
                np.linalg.norm(stored[6:9].astype(float))
                <= float(sample.mu) * float(stored[0]) + 1e-5
            )
            relative = (
                sample.jacobian_a.astype(float) - sample.jacobian_b.astype(float)
            ) @ velocity
            n = sample.normal_world.astype(float)
            tangent = relative - n * (n @ relative) / (n @ n)
            assert float(stored[6:9].astype(float) @ tangent) <= 1e-14
        mass = host["mass"][lane, :nv, :nv].astype(float)
        damping = host["vectors"][1, :nv].astype(float)
        vin = host["v"][lane, :nv].astype(float)
        rhs = (
            host["applied"][lane, :nv].astype(float)
            + host["passive"][lane, :nv].astype(float)
            - host["bias"][lane, :nv].astype(float)
        )
        response = mass + h * np.diag(damping)
        physical_acc, integration_acc = np.linalg.solve(mass, rhs), np.linalg.solve(response, rhs)
        _close(h * outputs["acceleration"][lane, :nv], h * physical_acc, 5e-5, 5e-5)
        _close(h * outputs["integration_acc"][lane, :nv], h * integration_acc, 5e-5, 5e-5)
        _close(velocity, vin + h * integration_acc, 5e-5, 5e-5)
        np.testing.assert_allclose(
            velocity, vin + h * outputs["integration_acc"][lane, :nv], rtol=2e-6, atol=2e-7
        )
        stored_residual = e["response"] @ velocity - e["rhs"] - h * generalized[lane]
        stored_norm = np.sqrt(
            max(0.0, float(stored_residual @ np.linalg.solve(e["response"], stored_residual)))
        )
        assert stored_norm <= 2e-5 * e["scale"]
        expected_q = host["q"][lane, : model.nq].astype(float).copy()
        mj.mj_integratePos(model, expected_q, velocity, h)
        qout = outputs["qnext"][lane, : model.nq].astype(float)
        if free:
            np.testing.assert_allclose(qout[:3], expected_q[:3], rtol=0, atol=2e-7)
            sign = 1 if qout[3:] @ expected_q[3:] >= 0 else -1
            np.testing.assert_allclose(sign * qout[3:], expected_q[3:], rtol=0, atol=3e-7)
            assert abs(float(qout[3:] @ qout[3:]) - 1) <= 2e-6
        else:
            np.testing.assert_allclose(qout, expected_q, rtol=0, atol=2e-7)
            m, d, v0 = float(mass[0, 0]), float(damping[0]), float(vin[0])
            right, doubled = m * v0 / (m + h * d), (m - h * d) * v0 / (m + h * d)
            _close(velocity, np.array([right]), 5e-5, 5e-5)
            assert abs(velocity[0] - doubled) > 1e-3
            e["damping_oracle"] = dict(correct=right, repeated=doubled)
        e.update(
            physical_acceleration=physical_acc,
            integration_acceleration=integration_acc,
            expected_q=expected_q,
            stored_force_residual=stored_residual,
            stored_force_residual_norm=stored_norm,
            strict_host_gate=e["residual_norm"] <= 1e-10 * e["scale"],
        )


@pytest.mark.cuda
def test_saved_forces_through_two_native_integrators(tmp_path, monkeypatch):
    import mujoco as mj

    evidence = dict(
        schema="area-friction-native-integration-bridge-v1",
        records=[],
        original_entry_sha256=FIXTURE_SHA,
        saved_pack_sha256=PACK_SHA,
        scope="historical-native-integration-only-no-scene-or-force-publication",
        force_assembly="HOST sum of actual saved full G.T@F; total_applied RN32 once",
        clock_basis="registered fresh zero clock; no saved raw clock words were available",
    )
    try:
        original, saved = _load()
        evidence["original_sources"] = original["sources"]

        def forbidden(*args, **kwargs):
            raise AssertionError("I3x must not solve friction or step a native scene")

        monkeypatch.setattr(mj, "mj_step", forbidden)
        with TyxAreaFrictionWorkspace(capacity=3, mode="cuda") as workspace:
            monkeypatch.setattr(workspace, "solve_many", forbidden)
            monkeypatch.setattr(workspace, "_kernel", forbidden)
            for index, free in enumerate((True, False)):
                pipeline = saved["pipelines"][index]
                model, host, generalized, h = _native_inputs(mj, original, pipeline, free)
                count = len(host["q"])
                labels = pipeline["labels"][:count]
                assert all("synthetic" not in label for label in labels)
                record = dict(
                    kernel="floating.solve" if free else "tree.solve",
                    labels=labels,
                    h=h,
                    nq=model.nq,
                    nv=model.nv,
                    inputs=host,
                    generalized_force=generalized,
                    saved_packed=pipeline["packed"],
                    saved_solver_raw=pipeline["raw"],
                )
                evidence["records"].append(record)
                record.update(_launch(workspace, host, model, h, free))
                _audit(mj, model, pipeline, host, generalized, h, record, free)
            assert workspace.last_results == ()
        evidence.update(status="PASS", native_kernel_calls=2, friction_solver_calls=0)
    except BaseException as error:
        evidence.update(status="FAIL", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        with gzip.open(
            tmp_path / "native-integration-bridge.json.gz", "xt", encoding="utf-8"
        ) as stream:
            json.dump(_plain(evidence), stream, allow_nan=False)
