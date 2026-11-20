"""Original RR same-entry CUDA packing, including samples beyond the old 128 cap.

Only the production pack kernel runs on CUDA. CPU MuJoCo supplies the original
entry coefficients and independent point Jacobians. No solver, state integration,
triangle-surface substitute, or complete RR Scene behavior is exercised here.
"""

from types import SimpleNamespace

import mujoco as mj
import numpy as np
import pytest
from test_tyx_area_friction_entry_pack import INPUT_NAMES, SOURCE_NAMES, _device_source

from experiments.area_pressure.pusht_entry_area import build_area
from experiments.area_pressure.pusht_entry_rows import assemble_entry, load_entries
from phymeo.backends._tyx_area_friction_entry import pack_entry
from phymeo.backends.mujoco_tyx._resources import DeviceResources
from phymeo.training.manipulation_tasks import manipulation_scenes

SAMPLE_CAPACITY = 256


def _source(model, entry):
    rows = assemble_entry(model, entry)
    data = rows["data"]
    np.testing.assert_array_equal(data.qpos, entry["q"])
    np.testing.assert_array_equal(data.qvel, entry["v"])
    np.testing.assert_array_equal(data.ctrl, entry["ctrl"])
    area = build_area(model, data)
    assert area["body_ids"] == (model.body("object").id, model.body("tool").id)
    assert area["quadrature"]["method"] == "cropped-exterior-adaptive-midpoint-v1"
    nv, nb, ns = model.nv, model.nbody, len(area["points"])
    assert (model.nq, nv, model.nu, nb) == (9, 8, 2, 5)
    source = dict(
        source_status=np.zeros(1, np.int32),
        source_mass=rows["mass"][None].astype(np.float32),
        source_vectors=np.zeros((9, nv), np.float32),
        source_v=data.qvel[None].astype(np.float32),
        source_bias=data.qfrc_bias[None].astype(np.float32),
        source_passive=data.qfrc_passive[None].astype(np.float32),
        source_applied=np.zeros((1, nv), np.float32),
        motion=np.zeros((1, nb, 7, 3), np.float32),
        body_jacobian=np.zeros((1, nb, 6, nv), np.float32),
        area_input=np.zeros((1, SAMPLE_CAPACITY, 8), np.float32),
        area_pair_index=np.zeros((1, SAMPLE_CAPACITY), np.int32),
        area_sizes=np.array([ns], np.int32),
        area_body_ids=np.array([area["body_ids"]], np.int32),
        pair_laws=np.array([[area["mu"], area["slip"]]], np.float32),
    )
    source["source_vectors"][1] = model.dof_damping
    external = data.qfrc_actuator.copy() + data.qfrc_applied
    for body in range(1, nb):
        wrench = data.xfrc_applied[body]
        if np.any(wrench):
            mj.mj_applyFT(model, data, wrench[:3], wrench[3:], data.xipos[body], body, external)
    source["source_applied"][0] = external
    np.testing.assert_allclose(
        external + data.qfrc_passive - data.qfrc_bias, rows["base_load"], rtol=0, atol=1e-14
    )
    source["motion"][0, :, 0] = data.xpos
    source["motion"][0, :, 5] = data.xipos
    for body in range(nb):
        linear, angular = np.zeros((3, nv)), np.zeros((3, nv))
        mj.mj_jac(model, data, linear, angular, data.xipos[body], body)
        source["body_jacobian"][0, body] = np.vstack((linear, angular))
    source["area_input"][0, :ns] = np.column_stack(
        (area["points"], area["normals"], area["stiffness"], area["initial_gap"])
    )
    assert set(source) == set(SOURCE_NAMES)
    return source, area


def _launch(runtime, resident, output, status, h, monkeypatch):
    import tyx as tx

    def forbidden(*args, **kwargs):
        raise AssertionError("production pack attempted an intermediate HOST readback")

    with tx.use_runtime(runtime):
        with monkeypatch.context() as guard:
            guard.setattr(tx.CudaArray, "to_numpy", forbidden)
            pack_entry(
                1,
                1,
                8,
                8,
                5,
                1,
                SAMPLE_CAPACITY,
                h,
                *(resident[name] for name in SOURCE_NAMES),
                *(output[name] for name in INPUT_NAMES),
                status,
                sample_capacity=SAMPLE_CAPACITY,
                mode="cuda",
            )
            runtime.sync()
        np.testing.assert_array_equal(status.to_numpy(), [0])
        return {name: output[name].to_numpy() for name in INPUT_NAMES}


def _check_pack(packed, source, area, h):
    ns = int(source["area_sizes"][0])
    np.testing.assert_array_equal(packed["sizes"], [[8, ns, 24, 12]])
    np.testing.assert_array_equal(packed["steps"], np.array([h], np.float32))
    np.testing.assert_array_equal(packed["mass"], source["source_mass"])
    np.testing.assert_array_equal(packed["inputs"][0, 0], source["source_vectors"][1])
    np.testing.assert_array_equal(packed["inputs"][0, 1], source["source_v"][0])
    damping = source["source_vectors"][1].astype(float)
    velocity = source["source_v"][0].astype(float)
    force = (
        source["source_applied"][0].astype(float)
        + source["source_passive"][0].astype(float)
        - source["source_bias"][0].astype(float)
        + damping * velocity
    )
    np.testing.assert_allclose(packed["inputs"][0, 2], force, rtol=2e-6, atol=2e-7)
    np.testing.assert_array_equal(packed["samples"][0, :ns, :8], source["area_input"][0, :ns])
    np.testing.assert_array_equal(
        packed["samples"][0, :ns, 8:], np.tile(source["pair_laws"][0], (ns, 1))
    )
    for name, body in zip(("ja", "jb"), area["body_ids"], strict=True):
        com_j = source["body_jacobian"][0, body].astype(float)
        offsets = source["area_input"][0, :ns, :3].astype(float) - source["motion"][
            0, body, 5
        ].astype(float)
        expected = com_j[:3][None] + np.cross(com_j[3:].T[None], offsets[:, None]).transpose(
            0, 2, 1
        )
        # First isolate arithmetic on the actual FP32 source. Then compare each
        # sample to the independent FP64 mj_jac at the original world point.
        np.testing.assert_allclose(packed[name][0, :ns], expected, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(
            packed[name][0, :ns],
            area["jac_a" if name == "ja" else "jac_b"],
            rtol=2e-6,
            atol=2e-7,
        )
        assert not np.any(packed[name][0, ns:])
    assert not np.any(packed["samples"][0, ns:])


@pytest.mark.cuda
def test_rr_peak_237_points_and_withdrawal_reuse_shared_cuda_pack(monkeypatch):
    import tyx as tx

    scene = next(scene for scene in manipulation_scenes() if scene.key == "pusht-planar")
    model = mj.MjModel.from_xml_string(scene.mjcf)
    entries = load_entries()
    peak, withdrawn = entries["peak"], entries["withdrawn"]
    assert (peak["tick"], withdrawn["tick"]) == (1255, 1767)
    assert abs(peak["time"] - 1.255) < 2e-6
    source, area = _source(model, peak)
    empty, empty_area = _source(model, withdrawn)
    assert source["area_sizes"].tolist() == [237]
    assert empty["area_sizes"].tolist() == [0]
    assert area["quadrature"]["clearance_m"] < 0 < empty_area["quadrature"]["clearance_m"]
    assert np.any(source["source_v"]) and not np.array_equal(source["source_v"], empty["source_v"])
    assert np.any(area["jac_a"][:, :, 2:]) and np.any(area["jac_b"][:, :, :2])
    assert not np.any(area["jac_a"][:, :, :2]) and not np.any(area["jac_b"][:, :, 2:])
    h = float(np.float32(model.opt.timestep))
    layouts = dict(
        sizes=(1, 4),
        mass=(1, 8, 8),
        inputs=(1, 3, 8),
        steps=(1,),
        samples=(1, SAMPLE_CAPACITY, 10),
        ja=(1, SAMPLE_CAPACITY, 3, 8),
        jb=(1, SAMPLE_CAPACITY, 3, 8),
    )
    resources = DeviceResources()
    try:
        runtime = tx.Runtime(backend="cuda")
        resources.set_runtime(runtime)
        owner = SimpleNamespace(_runtime=runtime, _resources=resources, capacity=1)
        resident, status = _device_source(owner, source)
        output = {}
        with tx.use_runtime(runtime):
            for name, shape in layouts.items():
                initial = np.full(shape, -77, np.int32 if name == "sizes" else np.float32)
                buffer = tx.cuda_empty(shape, initial.dtype)
                resources.add_buffer(buffer)
                buffer.copy_from(initial)
                output[name] = buffer
        packed = _launch(runtime, resident, output, status, h, monkeypatch)
        _check_pack(packed, source, area, h)
        # No silent truncation at the previous 128-sample boundary: all 109
        # later samples have positive stiffness and both real movable-body J's.
        assert np.all(packed["samples"][0, 128:237, 6] > 0)
        assert np.any(packed["ja"][0, 128:237]) and np.any(packed["jb"][0, 128:237])
        with tx.use_runtime(runtime):
            for name in SOURCE_NAMES:
                np.testing.assert_array_equal(resident[name].to_numpy(), source[name])
            # Reuse every input and output allocation at the real withdrawn
            # entry. Old scratch point bytes remain, but the true count is zero.
            empty["area_input"][:] = source["area_input"]
            empty["area_pair_index"][:] = source["area_pair_index"]
            for name in SOURCE_NAMES:
                resident[name].copy_from(empty[name])
        cleared = _launch(runtime, resident, output, status, h, monkeypatch)
        _check_pack(cleared, empty, empty_area, h)
        for name in ("samples", "ja", "jb"):
            assert not np.any(cleared[name])  # Includes the formerly live [128:237] tail.
        with tx.use_runtime(runtime):
            for name in SOURCE_NAMES:
                np.testing.assert_array_equal(resident[name].to_numpy(), empty[name])
    finally:
        resources.close()
