"""I3w resident-layout pack -> unchanged CUDA solver, without a HOST bridge.

The checked-in gzip is an actual accepted I3v entry, not a new scene execution.
One explicitly synthetic lane gives body B a nonzero Jacobian. This verifies a
kernel seam only: no live integration or Scene transaction is claimed.
"""

import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import phymeo.backends._tyx_area_friction_entry as entry_module
from phymeo.backends._tyx_area_friction_entry import pack_entry
from phymeo.backends.tyx_area_friction import TyxAreaFrictionWorkspace

FIXTURE_SHA = "50583f6e9f6875a77e09c3c3fa645588aa30c33b7b7d4b6113b1d1731bccd1b6"
INPUT_NAMES = ("sizes", "mass", "inputs", "steps", "samples", "ja", "jb")
OUTPUT_NAMES = ("meta", "scalars", "vectors", "matrices", "contact", "trace_meta", "trace")
SOURCE_NAMES = (
    "source_status",
    "source_mass",
    "source_vectors",
    "source_v",
    "source_bias",
    "source_passive",
    "source_applied",
    "motion",
    "body_jacobian",
    "area_input",
    "area_pair_index",
    "area_sizes",
    "area_body_ids",
    "pair_laws",
)
C, P = 128, 8


def test_declared_sample_capacity_changes_only_per_point_storage():
    from phymeo.backends.mujoco_tyx._area_friction_layout import scratch_layout

    default = scratch_layout(2, 8)
    for capacity in (129, 238, 512):
        wide = scratch_layout(2, 8, sample_capacity=capacity)
        for name, (shape, dtype) in default.items():
            expected = (
                (2, capacity, *shape[2:]) if name in ("samples", "ja", "jb", "contact") else shape
            )
            assert wide[name] == (expected, dtype)
    for invalid in (True, 0, 513):
        with pytest.raises(ValueError, match="sample capacity"):
            scratch_layout(2, 8, sample_capacity=invalid)


def _decode(value):
    if isinstance(value, dict) and set(value) == {"dtype", "shape", "hex"}:
        dtype = np.dtype(value["dtype"])
        assert dtype.str in ("<f4", "<f8", "<i4", "<i8", "|b1")
        array = np.frombuffer(bytes.fromhex(value["hex"]), dtype=dtype)
        return array.reshape(tuple(value["shape"]))
    if isinstance(value, dict):
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _fixture():
    path = Path(__file__).parent / "fixtures" / "area_friction_entry" / "entry.json.gz"
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA
    payload = _decode(json.loads(gzip.decompress(raw)))
    assert payload["status"] == "PASS" and payload["schema"] == "mujoco-area-friction-entry-v1"
    assert payload["problem_dimensions"] == [
        dict(nv=6, samples=18),
        dict(nv=6, samples=18),
        dict(nv=1, samples=0),
    ]
    return payload


def _plain(value):
    if isinstance(value, np.ndarray):
        return dict(dtype=value.dtype.str, shape=list(value.shape), hex=value.tobytes().hex())
    if isinstance(value, np.generic):
        return _plain(np.asarray(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(type(value).__name__)


def _save(path, evidence):
    with gzip.open(path, "xt", encoding="utf-8") as stream:
        json.dump(_plain(evidence), stream, allow_nan=False)


def _source(fixture, free):
    key, w, nv = ("7", 4, 6) if free else ("41", 1, 1)
    receipt = fixture["receipt"]["receipts"][key]
    dynamics, area = receipt["dynamics"], receipt["raw_contact_fields"]["area_pressure"][0]
    nb = dynamics["body_com"].shape[0]
    result = dict(
        source_status=np.zeros(w, np.int32),
        source_mass=np.zeros((w, P, P), np.float32),
        source_vectors=np.zeros((9, P), np.float32),
        source_v=np.zeros((w, P), np.float32),
        source_bias=np.zeros((w, P), np.float32),
        source_passive=np.zeros((w, P), np.float32),
        source_applied=np.zeros((w, P), np.float32),
        motion=np.zeros((w, nb, 7, 3), np.float32),
        body_jacobian=np.zeros((w, nb, 6, P), np.float32),
        area_input=np.zeros((w, C, 8), np.float32),
        area_pair_index=np.zeros((w, C), np.int32),
        area_sizes=np.zeros(w, np.int32),
        area_body_ids=np.array([area["body_ids"], area["body_ids"]], np.int32),
        pair_laws=np.array(((0.5, 0.001), (0, 0.001)), np.float32),
    )
    result["source_mass"][:, :nv, :nv] = dynamics["mass_matrix"]
    result["source_vectors"][1, :nv] = fixture["prepared"][0 if free else 2]["binding"]["damping"]
    for field in ("v", "bias", "passive", "applied"):
        result["source_" + field][:, :nv] = dynamics["input_v" if field == "v" else field]
    result["motion"][:, :, 0] = dynamics["body_position"]
    result["motion"][:, :, 5] = dynamics["body_com"]
    result["body_jacobian"][:, :, :, :nv] = dynamics["jacobian_com"]
    ns = len(area["stiffness"])
    result["area_sizes"][:] = ns
    result["area_input"][:, :ns, :3] = area["points"]
    result["area_input"][:, :ns, 3:6] = area["normals"]
    result["area_input"][:, :ns, 6] = area["stiffness"]
    result["area_input"][:, :ns, 7] = area["surrogate_gap"]
    labels = ["actual-slider-empty-D40"]
    if free:
        result["area_pair_index"][1, :ns] = 1
        result["area_sizes"][2] = 0
        ba, bb = area["body_ids"]
        result["motion"][3, bb, 5] = (0.01, -0.02, 0.005)
        result["body_jacobian"][3, bb] = np.float32(0.25) * result["body_jacobian"][3, ba]
        labels = [
            "actual-free-mu.5",
            "actual-free-mu0",
            "actual-free-empty-counterfactual",
            "synthetic-layout-only-nonzero-JB",
        ]
    constants = dict(
        capacity=w,
        count=w,
        nv=nv,
        width=P,
        nbody=nb,
        npair=2,
        area_capacity=C,
        h=float(np.float32(receipt["h"])),
    )
    return result, constants, labels


def _expected_pack(source, constants):
    """Independent cross-product algebra on the actual packed source fields."""
    w, nv, h = constants["count"], constants["nv"], constants["h"]
    expected = dict(
        sizes=np.zeros((w, 4), np.int32),
        mass=np.zeros((w, 6, 6)),
        inputs=np.zeros((w, 3, 6)),
        steps=np.full(w, h),
        samples=np.zeros((w, 128, 10)),
        ja=np.zeros((w, 128, 3, 6)),
        jb=np.zeros((w, 128, 3, 6)),
    )
    for lane in range(w):
        ns = int(source["area_sizes"][lane])
        expected["sizes"][lane] = (nv, ns, 24, 12)
        expected["mass"][lane, :nv, :nv] = source["source_mass"][lane, :nv, :nv]
        damping, velocity = (
            source["source_vectors"][1, :nv].astype(float),
            source["source_v"][lane, :nv].astype(float),
        )
        force = (
            source["source_applied"][lane, :nv].astype(float)
            + source["source_passive"][lane, :nv].astype(float)
            - source["source_bias"][lane, :nv].astype(float)
            + damping * velocity
        )
        expected["inputs"][lane, :, :nv] = np.array((damping, velocity, force))
        for sample in range(ns):
            pair = source["area_pair_index"][lane, sample]
            row = source["area_input"][lane, sample].astype(float)
            expected["samples"][lane, sample] = np.r_[row, source["pair_laws"][pair]]
            for name, body in zip(("ja", "jb"), source["area_body_ids"][pair], strict=True):
                j = source["body_jacobian"][lane, body, :, :nv].astype(float)
                offset = row[:3] - source["motion"][lane, body, 5].astype(float)
                expected[name][lane, sample, :, :nv] = j[:3] + np.cross(j[3:].T, offset).T
    return expected


def _device_source(workspace, source):
    import tyx as tx

    buffers = {}
    with tx.use_runtime(workspace._runtime):
        for name, array in source.items():
            buffer = tx.cuda_empty(array.shape, array.dtype)
            workspace._resources.add_buffer(buffer)
            buffer.copy_from(array)
            buffers[name] = buffer
        status = tx.cuda_empty((workspace.capacity,), np.int32)
        workspace._resources.add_buffer(status)
        status.copy_from(np.full(workspace.capacity, -77, np.int32))
    return buffers, status


def _call(workspace, resident, pack_status, constants, monkeypatch):
    """No D2H or HOST repack is permitted between pack and the original solve."""
    import tyx as tx

    output = workspace._buffers
    arguments = tuple(
        constants[name]
        for name in ("capacity", "count", "nv", "width", "nbody", "npair", "area_capacity", "h")
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("intermediate HOST readback/repack")

    with tx.use_runtime(workspace._runtime):
        with monkeypatch.context() as guard:
            guard.setattr(tx.CudaArray, "to_numpy", forbidden)
            guard.setattr(workspace, "_prepare", forbidden)
            pack_entry(
                *arguments,
                *(resident[name] for name in SOURCE_NAMES),
                *(output[name] for name in INPUT_NAMES),
                pack_status,
                mode="cuda",
            )
            workspace._kernel(
                workspace.capacity,
                constants["count"],
                *(output[name] for name in workspace._layouts),
                mode="cuda",
            )
            workspace._runtime.sync()
        # These are the first readbacks; retain full inputs, scratch and padding.
        return dict(
            pack_status=pack_status.to_numpy(),
            source_status=resident["source_status"].to_numpy(),
            source_after={name: resident[name].to_numpy() for name in SOURCE_NAMES},
            packed={name: output[name].to_numpy() for name in INPUT_NAMES},
            raw={name: output[name].to_numpy() for name in OUTPUT_NAMES},
        )


def _problem(packed, lane):
    nv, ns = map(int, packed["sizes"][lane, :2])
    samples = []
    for index in range(ns):
        row = packed["samples"][lane, index]
        samples.append(
            SimpleNamespace(
                point_world=row[:3],
                normal_world=row[3:6],
                stiffness_n_m=row[6],
                phi_m=row[7],
                mu=row[8],
                slip_speed_m_s=row[9],
                jacobian_a=packed["ja"][lane, index, :, :nv],
                jacobian_b=packed["jb"][lane, index, :, :nv],
            )
        )
    return SimpleNamespace(
        mass_matrix=packed["mass"][lane, :nv, :nv],
        damping=packed["inputs"][lane, 0, :nv],
        input_velocity=packed["inputs"][lane, 1, :nv],
        applied_force=packed["inputs"][lane, 2, :nv],
        h_s=packed["steps"][lane],
        samples=tuple(samples),
    )


def _equations(problem, velocity):
    v, h = np.asarray(velocity, dtype=float), float(problem.h_s)
    m = problem.mass_matrix.astype(float)
    a = m + h * np.diag(problem.damping.astype(float))
    rhs = m @ problem.input_velocity.astype(float) + h * problem.applied_force.astype(float)
    free, total, jacobian = np.linalg.solve(a, rhs), np.zeros(len(v)), a.copy()
    fn_rows, ft_rows, forces, moments, normal_rows = [], [], [], [], []
    for s in problem.samples:
        n = s.normal_world.astype(float)
        g = s.jacobian_a.astype(float) - s.jacobian_b.astype(float)
        normal_row = n @ g
        projection = np.eye(3) - np.outer(n, n) / (n @ n)
        vt = projection @ g @ v
        k, mu = float(s.stiffness_n_m), float(s.mu)
        fn = max(0.0, -k * (float(s.phi_m) + h * normal_row @ v))
        speed = np.sqrt(vt @ vt + float(s.slip_speed_m_s) ** 2)
        ft = -mu * fn * vt / speed
        force = fn * n + ft
        derivative = (
            np.outer(n - mu * vt / speed, -k * h * normal_row if fn > 0 else np.zeros(len(v)))
            - mu * fn * (np.eye(3) / speed - np.outer(vt, vt) / speed**3) @ projection @ g
        )
        total += g.T @ force
        jacobian -= h * g.T @ derivative
        fn_rows.append(fn)
        ft_rows.append(ft)
        forces.append(force)
        moments.append(np.cross(s.point_world.astype(float), force))
        normal_rows.append(normal_row)
    residual = a @ v - rhs - h * total
    norm = np.sqrt(max(0.0, float(residual @ np.linalg.solve(a, residual))))
    ns = len(problem.samples)
    return dict(
        response=a,
        rhs=rhs,
        free=free,
        residual=residual,
        residual_norm=norm,
        scale=1 + np.sqrt(max(0.0, float(free @ a @ free))),
        jacobian=jacobian,
        generalized=total,
        fn=np.asarray(fn_rows),
        ft=np.asarray(ft_rows).reshape(ns, 3),
        force=np.asarray(forces).reshape(ns, 3),
        moment=np.asarray(moments).reshape(ns, 3),
        normal_rows=np.asarray(normal_rows).reshape(ns, len(v)),
    )


def _close(actual, expected, absolute, relative):
    assert np.shape(actual) == np.shape(expected)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    assert np.all(np.abs(actual - expected) <= absolute + relative * np.abs(expected))


def _check_pipeline(record, source, constants):
    expected, packed, raw = _expected_pack(source, constants), record["packed"], record["raw"]
    np.testing.assert_array_equal(record["pack_status"], np.zeros(constants["capacity"], np.int32))
    np.testing.assert_array_equal(record["source_status"], source["source_status"])
    for name in SOURCE_NAMES:
        np.testing.assert_array_equal(record["source_after"][name], source[name])
    for name in ("sizes", "mass", "steps", "samples"):
        np.testing.assert_array_equal(packed[name], expected[name].astype(packed[name].dtype))
    np.testing.assert_array_equal(
        packed["inputs"][:, :2], expected["inputs"][:, :2].astype(np.float32)
    )
    _close(packed["inputs"][:, 2], expected["inputs"][:, 2], 2e-5, 3e-4)
    for name in ("ja", "jb"):
        _close(packed[name], expected[name], 2e-6, 2e-5)
    certificates = []
    for lane in range(constants["count"]):
        p = _problem(packed, lane)
        nv, ns = len(p.input_velocity), len(p.samples)
        assert raw["meta"][lane, 0] == 0 and 2 <= raw["meta"][lane, 2] <= 290
        v, contact = raw["vectors"][lane, 1, :nv], raw["contact"][lane, :ns]
        e = _equations(p, v)
        assert e["residual_norm"] <= 2e-5 * e["scale"]
        assert raw["scalars"][lane, 0] <= raw["scalars"][lane, 1]
        _close(raw["vectors"][lane, 0, :nv], e["free"], 5e-5, 5e-5)
        _close(contact[:, 0], e["fn"], 0.002, 0.002)
        _close(contact[:, 6:9], e["ft"], 0.002, 0.002)
        _close(contact[:, 9:12], e["force"], 0.002, 0.002)
        _close(contact[:, 12:15], e["moment"], 1e-5, 0.002)
        np.testing.assert_allclose(
            raw["matrices"][lane, 2, :nv, :nv], e["jacobian"], rtol=2e-5, atol=2e-7
        )
        actual_q = np.zeros(nv)
        for sample, actual in zip(p.samples, contact, strict=True):
            assert (
                actual[0] >= 0
                and np.linalg.norm(actual[6:9].astype(float))
                <= float(sample.mu) * float(actual[0]) + 1e-5
            )
            actual_q += (
                sample.jacobian_a.astype(float) - sample.jacobian_b.astype(float)
            ).T @ actual[9:12].astype(float)
        stored_residual = e["response"] @ v.astype(float) - e["rhs"] - float(p.h_s) * actual_q
        stored_norm = np.sqrt(
            max(0.0, float(stored_residual @ np.linalg.solve(e["response"], stored_residual)))
        )
        assert stored_norm <= 2e-5 * e["scale"]
        if ns == 0:
            _close(v, e["free"], 5e-5, 5e-5)
        elif all(float(s.mu) == 0 for s in p.samples):
            j = e["normal_rows"]
            k, phi = (
                np.array([float(s.stiffness_n_m) for s in p.samples]),
                np.array([float(s.phi_m) for s in p.samples]),
            )
            h = float(p.h_s)
            target = np.linalg.solve(
                e["response"] + h * h * j.T @ (k[:, None] * j), e["rhs"] - h * j.T @ (k * phi)
            )
            normal = -k * (phi + h * j @ target)
            assert np.all(normal > 0)
            _close(v, target, 5e-5, 5e-5)
            _close(contact[:, 0], normal, 0.002, 0.002)
            e.update(normal_oracle_velocity=target, normal_oracle_force=normal)
        assert not np.any(packed["samples"][lane, ns:])
        assert not np.any(packed["ja"][lane, ns:]) and not np.any(packed["jb"][lane, ns:])
        assert not np.any(packed["ja"][lane, :, :, nv:]) and not np.any(
            packed["jb"][lane, :, :, nv:]
        )
        assert not np.any(packed["inputs"][lane, :, nv:])
        assert not np.any(raw["vectors"][lane, :, nv:]) and not np.any(raw["contact"][lane, ns:])
        assert not np.any(raw["matrices"][lane, :, nv:, :]) and not np.any(
            raw["matrices"][lane, :, :, nv:]
        )
        e.update(
            stored_force_residual=stored_residual,
            stored_force_residual_norm=stored_norm,
            strict_host_gate=e["residual_norm"] <= 1e-10 * e["scale"],
        )
        certificates.append(e)
    record.update(expected_pack=expected, certificates=certificates)


def test_fixed_entry_fixture_binding_and_independent_layout(monkeypatch):
    fixture = _fixture()
    source, constants, labels = _source(fixture, True)
    assert constants["nv"] == 6 and constants["width"] == 8 and len(labels) == 4
    expected = _expected_pack(source, constants)
    assert np.any(expected["jb"][3]) and not np.any(expected["jb"][:3])
    assert not np.array_equal(expected["ja"][3] - expected["jb"][3], expected["ja"][3])
    slider, one, _ = _source(fixture, False)
    assert one["nv"] == 1 and slider["area_sizes"].tolist() == [0]
    assert slider["source_vectors"][1, 0] == 40
    # Metadata admission must precede the private kernel; these HOST calls
    # deliberately never execute an interpret kernel or a numerical solve.
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("invalid metadata reached the pack kernel")

    monkeypatch.setattr(entry_module, "_pack_entry_kernel", forbidden)
    for kind in ("width", "nv", "count", "dtype", "shape", "output-alias", "source-alias"):
        target = {
            name: value.astype(np.int32 if name == "sizes" else np.float32)
            for name, value in expected.items()
        }
        status = np.full(4, -77, np.int32)
        dims = dict(constants)
        if kind == "width":
            dims["width"] = 7
        elif kind == "nv":
            dims["nv"] = 7
        elif kind == "count":
            dims["count"] = dims["capacity"] + 1
        elif kind == "dtype":
            target["steps"] = target["steps"].astype(np.float64)
        elif kind == "shape":
            target["samples"] = target["samples"][:, :127].copy()
        elif kind == "output-alias":
            target["ja"] = target["jb"]
        else:
            status = source["source_status"]
        before = {name: value.tobytes() for name, value in target.items()}
        before_status = status.tobytes()
        with pytest.raises(ValueError):
            pack_entry(
                *(
                    dims[name]
                    for name in (
                        "capacity",
                        "count",
                        "nv",
                        "width",
                        "nbody",
                        "npair",
                        "area_capacity",
                        "h",
                    )
                ),
                *(source[name] for name in SOURCE_NAMES),
                *(target[name] for name in INPUT_NAMES),
                status,
                mode="interpret",
            )
        assert calls == [] and status.tobytes() == before_status
        assert all(value.tobytes() == before[name] for name, value in target.items())


@pytest.mark.cuda
def test_two_resident_pipelines_without_intermediate_host(tmp_path, monkeypatch):
    fixture = _fixture()
    evidence = dict(schema="area-friction-entry-pack-v1", fixture_sha256=FIXTURE_SHA, pipelines=[])
    try:
        for free in (True, False):
            source, constants, labels = _source(fixture, free)
            record = dict(constants=constants, labels=labels, source=source)
            evidence["pipelines"].append(record)
            with TyxAreaFrictionWorkspace(capacity=constants["capacity"], mode="cuda") as workspace:
                resident, status = _device_source(workspace, source)
                record.update(_call(workspace, resident, status, constants, monkeypatch))
                assert workspace.last_results == ()  # No public publication contract is exercised.
                _check_pipeline(record, source, constants)
                if free:
                    assert np.any(record["packed"]["jb"][3])
                else:
                    p = _problem(record["packed"], 0)
                    m, d, h, v0 = (
                        float(p.mass_matrix[0, 0]),
                        float(p.damping[0]),
                        float(p.h_s),
                        float(p.input_velocity[0]),
                    )
                    correct, repeated = m * v0 / (m + h * d), (m - h * d) * v0 / (m + h * d)
                    actual = float(record["raw"]["vectors"][0, 1, 0])
                    assert abs(actual - correct) <= 5e-5 * (1 + abs(correct))
                    assert abs(actual - repeated) > 1e-3
                    record["damping_oracle"] = dict(
                        correct=correct, repeated=repeated, actual=actual
                    )
        evidence["status"] = "PASS"
    except BaseException as error:
        evidence.update(status="FAIL", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        _save(tmp_path / "entry-pack-pipelines.json.gz", evidence)


@pytest.mark.cuda
def test_last_lane_pack_errors_survive_solver_metadata_reset(tmp_path, monkeypatch):
    import tyx as tx

    base, constants, labels = _source(_fixture(), True)
    evidence = dict(
        schema="area-friction-entry-pack-errors-v1", fixture_sha256=FIXTURE_SHA, cases=[]
    )
    cases = (
        ("upstream", 1),
        ("negative-count", 2),
        ("129-count", 2),
        ("pair-id", 2),
        ("body-id", 2),
        ("normal", 3),
        ("mu", 3),
        ("slip-speed", 3),
        ("stiffness", 3),
        ("assembled-overflow", 4),
        ("nonfinite", 3),
        ("phi", 3),
        ("h", 3),
    )
    try:
        with TyxAreaFrictionWorkspace(capacity=4, mode="cuda") as workspace:
            resident, status = _device_source(workspace, base)
            original_publication = workspace.last_results
            for name, code in cases:
                source = {key: value.copy() for key, value in base.items()}
                case_constants = dict(constants)
                lane = 3
                if name == "upstream":
                    source["source_status"][lane] = 37
                    source["source_mass"][lane, 0, 0] = np.nan
                    source["area_sizes"][lane] = 129
                elif name == "negative-count":
                    source["area_sizes"][lane] = -1
                elif name == "129-count":
                    source["area_sizes"][lane] = 129
                elif name == "pair-id":
                    source["area_pair_index"][lane, 0] = 2
                elif name in ("body-id", "mu", "slip-speed"):
                    source["area_pair_index"][1] = 0
                    source["area_pair_index"][lane, :18] = 1
                    if name == "body-id":
                        source["area_body_ids"][1, 0] = constants["nbody"]
                    else:
                        source["pair_laws"][1, 0 if name == "mu" else 1] = -1 if name == "mu" else 0
                elif name == "normal":
                    source["area_input"][lane, 0, 3:6] = 0
                elif name == "stiffness":
                    source["area_input"][lane, 0, 6] = 0
                elif name == "nonfinite":
                    source["source_bias"][lane, 0] = np.nan
                elif name == "phi":
                    source["area_input"][lane, 0, 7] = 0.001
                elif name == "h":
                    case_constants["h"] = 0.0
                else:
                    source["source_applied"][lane, 0] = np.float32(3e38)
                    source["source_passive"][lane, 0] = np.float32(3e38)
                record = dict(
                    name=name,
                    expected_status=code,
                    constants=case_constants,
                    labels=labels,
                    source=source,
                )
                evidence["cases"].append(record)
                with tx.use_runtime(workspace._runtime):
                    for key, array in source.items():
                        resident[key].copy_from(array)
                record.update(_call(workspace, resident, status, case_constants, monkeypatch))
                # Static pair tables are shared and completely validated in
                # every lane, including an empty lane. Their corruption is a
                # whole-batch error, not merely a selected-sample error.
                failed = range(4) if name in ("body-id", "mu", "slip-speed", "h") else (lane,)
                expected_status = [code if i in failed else 0 for i in range(4)]
                assert record["pack_status"].tolist() == expected_status
                np.testing.assert_array_equal(record["source_status"], source["source_status"])
                for i in failed:
                    for array in record["packed"].values():
                        assert not np.any(array[i])
                    assert record["raw"]["meta"][i, 0] == 1
                    assert record["raw"]["meta"][i, 2] == 0
                    assert not np.any(record["raw"]["vectors"][i])
                    assert not np.any(record["raw"]["contact"][i])
                assert workspace.last_results is original_publication
            retry = dict(constants=constants, labels=labels, source=base)
            evidence["clean_retry"] = retry
            with tx.use_runtime(workspace._runtime):
                for key, array in base.items():
                    resident[key].copy_from(array)
            retry.update(_call(workspace, resident, status, constants, monkeypatch))
            _check_pipeline(retry, base, constants)
            assert workspace.last_results is original_publication
            evidence["status"] = "PASS"
    except BaseException as error:
        evidence.update(status="FAIL", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        _save(tmp_path / "entry-pack-errors.json.gz", evidence)
