"""Saved moving-state contact sign and force-lever precision regressions."""

import json
from pathlib import Path

import mujoco as mj
import numpy as np
import pytest
import tyx as tx

from phymeo.backends.mujoco_tyx import _mesh_contact_kernels as kernels
from phymeo.backends.mujoco_tyx._mesh_catalog import MeshQueryCatalog, compile_mesh_query_catalog
from phymeo.backends.mujoco_tyx._mesh_contacts import MeshContactWorkspace
from phymeo.backends.mujoco_tyx._mesh_query import MeshQueryWorkspace

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.tyx
@pytest.mark.parametrize(
    "variant", ["original", "reverse", "permute", "rotate", "translate", "overlap"]
)
def test_signed_feature_can_disprove_rounded_overlap(tyx_mode, tmp_path, variant):
    case = json.loads((FIXTURES / "m16_near_zero_separation.json").read_text())
    points = [np.asarray(case[k], np.float32) for k in ("a", "b")]
    pos, rot, axis = (np.asarray(case[k]) for k in ("positions", "rotations", "axis"))
    expected = case["signed_sat"]
    if variant == "reverse":
        points.reverse()
        pos, rot = pos[::-1].copy(), rot[::-1].copy()
    elif variant == "permute":
        points = [p[::-1].copy() for p in points]
    elif variant == "rotate":
        transform = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        pos, rot = pos @ transform.T, transform @ rot
    elif variant == "translate":
        pos += [0.125, -0.25, 0.0625]
    elif variant == "overlap":
        pos[1] -= 1e-8 * axis
        expected -= 1e-8
    cat = MeshQueryCatalog(
        np.arange(2, dtype=np.int32),
        np.array([1, 2], np.int32),
        np.array([0, len(points[0]), sum(map(len, points))], np.int32),
        np.concatenate(points),
        np.array([[0, 1]], np.int32),
        np.zeros(2, np.float32),
        2,
    )
    with MeshQueryWorkspace(cat, capacity=2, mode=tyx_mode, cache_root=tmp_path) as query:
        result = query.query(pos[None], rot[None])
        raw = result.read()
        assert raw["meta"][0, 0] == 1 and raw["meta"][0, 9] == 7
        gap = float(raw["gap"][0, 0])
        assert (gap > 0) == (expected > 0)
        assert gap == pytest.approx(expected, abs=2e-12)
        with result.device_view():
            pass


@pytest.mark.tyx
@pytest.mark.parametrize("index", range(3))
def test_saved_contact_lever_consumes_rigid_low_lanes(tyx_mode, tmp_path, index):
    case = json.loads((FIXTURES / "m16_contact_motion_precision.json").read_text())[index]
    n = len(case["qvel"])
    host = dict(
        counts=np.ones(1, np.int32),
        kinds=np.asarray(case["kinds"], np.int32),
        params=np.asarray(case["params"], np.float32),
        pairs=np.array([[0, 1]], np.int32),
        geometry=np.asarray([case["geometry"]], np.float32),
        point_low=np.asarray([case["point_low"]], np.float32),
        com=np.zeros((1, 3, 3), np.float32),
        com_low=np.zeros((1, 3, 3), np.float32),
        bj=np.zeros((1, 3, 6, n), np.float32),
        bj_low=np.zeros((1, 3, 6, n), np.float32),
        qvel=np.asarray([case["qvel"]], np.float32),
        frame=np.zeros((1, 3, 3), np.float32),
        jac=np.zeros((1, 3, n), np.float32),
        aref=np.zeros((1, 3), np.float32),
        D=np.zeros((1, 3), np.float32),
        R=np.zeros((1, 3), np.float32),
        friction=np.zeros((1, 2), np.float32),
        mu=np.zeros(1, np.float32),
        eligible=np.zeros(1, np.int32),
        law=np.zeros((1, 10), np.float32),
        status=np.zeros(1, np.int32),
    )
    for name, source in [
        ("com", "com"),
        ("com_low", "com_low"),
        ("bj", "body_jac"),
        ("bj_low", "body_jac_low"),
    ]:
        host[name][0, 1:] = case[source]
    runtime = tx.Runtime(backend="cuda" if tyx_mode == "cuda" else "cpu", cache_root=tmp_path)
    with runtime.activate():
        device = {k: tx.cuda.array(v) if tyx_mode == "cuda" else v for k, v in host.items()}
        try:
            errors = []
            for precision in [0, 1]:
                kernels.assemble_contacts(
                    1,
                    1,
                    n,
                    precision,
                    case["h"],
                    case["impratio"],
                    mj.mjMINVAL,
                    mj.mjMINIMP,
                    mj.mjMAXIMP,
                    mj.mjMINMU,
                    *device.values(),
                    mode=tyx_mode,
                )
                runtime.sync()
                jac = device["jac"].to_numpy() if tyx_mode == "cuda" else device["jac"].copy()
                errors.append(float(np.max(abs(jac[0] - np.asarray(case["expected_jac"])))))
            assert errors[1] < 1e-9
            assert errors[0] > 10 * errors[1]
        finally:
            for value in device.values():
                if hasattr(value, "free"):
                    value.free()


@pytest.mark.tyx
def test_motion_precision_schema_device_reuse_and_nonfinite_rejection(tyx_mode, tmp_path):
    m = mj.MjModel.from_xml_string("""<mujoco><option cone="elliptic"/>
      <worldbody><body><freejoint/><geom type="box" size=".1 .1 .1"/></body>
      <body pos="1 0 0"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
      </worldbody></mujoco>""")
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    cat = compile_mesh_query_catalog(mj, m)
    motion = dict(
        com=d.xipos[None].astype(np.float32),
        body_jac=np.zeros((1, 3, 6, 12), np.float32),
        qvel=np.zeros((1, 12), np.float32),
    )
    lows = dict(
        com_low=np.zeros_like(motion["com"]), body_jac_low=np.zeros_like(motion["body_jac"])
    )
    with (
        MeshQueryWorkspace(cat, capacity=4, mode=tyx_mode, cache_root=tmp_path) as q,
        MeshContactWorkspace(mj, m, cat, capacity=4, mode=tyx_mode, cache_root=tmp_path) as c,
    ):
        result = q.query(d.geom_xpos[None], d.geom_xmat.reshape(1, 2, 3, 3))
        for bad in [
            dict(motion, com_low=lows["com_low"]),
            dict(motion, **dict(lows, com_low=lows["com_low"].astype(float))),
        ]:
            with pytest.raises(ValueError):
                c.assemble(result, bad)
        with c.runtime.activate():
            device = {
                k: tx.cuda.array(v) if tyx_mode == "cuda" else v.copy()
                for k, v in dict(motion, **lows).items()
            }
            try:
                first = c.assemble_device(result, device)
                with first.device_view():
                    with pytest.raises(RuntimeError, match="borrowed"):
                        c.assemble_device(result, device)
                assert first.read()["status"].tolist() == [0]
                with pytest.raises(ValueError, match="shape/dtype"):
                    c.assemble_device(result, dict(device, body_jac_low=device["com_low"]))
                with pytest.raises(RuntimeError, match="expired"):
                    first.read()
            finally:
                for value in device.values():
                    if hasattr(value, "free"):
                        value.free()
        for name in lows:
            bad = {k: v.copy() for k, v in lows.items()}
            bad[name].flat[0] = np.nan
            invalid = c.assemble(result, dict(motion, **bad))
            with pytest.raises(ValueError, match="invalid"):
                with invalid.device_view():
                    pass
            # Omitting optional low lanes after bad host data must not reuse them.
            with c.assemble(result, motion).device_view():
                pass
