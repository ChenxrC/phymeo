"""C9 ordinary floating trees: frozen public tests, AUTHORING_UNEXECUTED.

Only old APIs are imported at collection. The first node establishes a physical
7x7 HOST premise before looking up the new resident public API. Execution belongs
to the parent; all literal masses, loads and tolerances are declared inputs.
"""

import gzip
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass

import numpy as np
import pytest

import phymeo as ph
from phymeo.core.rigid import RigidBodyModel
from phymeo.core.robot import RobotModel
from phymeo.dynamics.floating_robot import FloatingRobotModel, FloatingRobotState
from phymeo.solvers.floating_tree import FloatingRobotInputs, floating_dynamics_terms

H = 1 / 1024
ATOL, RTOL, WINDOW_ATOL = 5e-6, 5e-5, 2e-4
WORLD = 503


def _json(value):
    if is_dataclass(value):
        return {field.name: _json(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def _save(directory, name, payload):
    directory.mkdir(parents=True, exist_ok=True)
    with gzip.open(directory / (name + ".json.gz"), "wt", encoding="utf-8") as stream:
        json.dump(_json(payload), stream, allow_nan=False, separators=(",", ":"))


def _close(actual, expected, *, atol=ATOL, rtol=RTOL):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def _p03_host():
    root = RigidBodyModel(np.array([0.5]), np.full((1, 3), 0.5), np.zeros(3))
    child = RobotModel(
        np.array([-1]), ("prismatic",), np.zeros((1, 3)),
        np.array([[0.0, 0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]),
        np.array([1.0]), np.zeros((1, 3)), np.full((1, 3), 0.25), np.zeros(3),
    )
    model = FloatingRobotModel(root, np.zeros(3), child)
    state = FloatingRobotState(np.array([0.0, 0, 0, 0, 0, 0, 1, 0]), np.zeros(7))
    return model, state


@pytest.mark.tyx
def test_p03_public_floating_tree_keeps_complete_mass_and_internal_recoil(tyx_mode, tmp_path):
    model, state = _p03_host()
    expected_mass = np.diag([3.0, 3, 3, 0.75, 0.75, 0.75, 1])
    expected_mass[0, 6] = expected_mass[6, 0] = 1
    expected_a = np.array([-1.5, 0, 0, 0, 0, 0, 4.5])
    internal = FloatingRobotInputs(np.array([3.0]), np.zeros((2, 3)), np.zeros((2, 3)))
    host = floating_dynamics_terms(model, state, internal)
    assert model.body_count == 2 and host.mass_matrix.shape == (7, 7)
    _close(host.mass_matrix, expected_mass, atol=1e-12, rtol=0)
    _close(host.bias, np.zeros(7), atol=1e-12, rtol=0)
    _close(host.applied, [0, 0, 0, 0, 0, 0, 3], atol=1e-12, rtol=0)
    _close(host.acceleration, expected_a, atol=1e-12, rtol=0)
    # Mxx=[[3,1],[1,1]] gives (-3/2,9/2), with child world ax=3.
    assert 2 * expected_a[0] + expected_a[0] + expected_a[6] == 0
    record = dict(
        case="P03", mode=tyx_mode, dt_s=H, model=model, initial=state,
        literal_mass=expected_mass, literal_acceleration=expected_a,
        host_premise=host, drive=[[0.0, 0.0, 3.0]], external=None,
    )
    try:
        tree = ph.FloatingRobotTree(
            model, state, kp=np.zeros(1), kd=np.zeros(1), effort_limit=np.array([8.0])
        )
        packet = tree.inputs(np.array([[0.0, 0.0, 3.0]]))
        with ph.TyxWorldBatch([(WORLD, tree)], tyx_mode, cache_root=tmp_path) as batch:
            batch.set_controls({WORLD: packet})
            entry = batch.snapshot()[WORLD]
            assert len(entry.position) == 8 and len(entry.velocity) == 7
            assert batch.step(H) == {WORLD: True}
            accepted = batch.snapshot()[WORLD]
            geometry = batch.tree_kinematics()[WORLD]
            diagnostic = batch.tree_diagnostics()[WORLD]
            record.update(
                entry=entry, accepted=accepted, geometry=geometry,
                diagnostic=diagnostic, energy=batch.summary(), stats=batch.stats(),
            )
            assert accepted.tick == entry.tick + 1
            _close(diagnostic.mass, expected_mass, atol=2e-5)
            _close(diagnostic.acceleration, expected_a)
            _close(diagnostic.drive, [3.0])
            _close(accepted.velocity, H * expected_a)
            _close(accepted.position[:3], [-1.5 * H * H, 0, 0])
            _close(accepted.position[3:7], [0, 0, 0, 1])
            _close(accepted.position[7:], [4.5 * H * H])
            _close(geometry.linear_velocity_m_s, [[-1.5 * H, 0, 0], [3 * H, 0, 0]])
            _close(geometry.angular_velocity_rad_s, np.zeros((2, 3)))
            _close((2 * geometry.linear_velocity_m_s[0] + geometry.linear_velocity_m_s[1]) / 3, np.zeros(3), atol=ATOL, rtol=0)
            assert accepted.control == entry.control and not accepted.history
            assert (geometry.joint_count, geometry.body_count, geometry.velocity_count, geometry.configuration_count) == (1, 2, 7, 8)
            assert batch.stats()["floating_world_count"] == 1
            assert batch.stats()["submitted_kernels"] > 0
    except Exception as error:
        record["failure"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        _save(tmp_path, "p03-public-floating", record)
# The following tests share only data/serialization helpers with the first node.
# They do not change its already-frozen physical input or acceptance gates.


def _reference():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "reference_newton/robot/resident_floating_reference.py"
    spec = importlib.util.spec_from_file_location("c9_independent_time_jets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model(data):
    c = data["children"]
    root = RigidBodyModel(np.array([1/data["root_mass"]]), data["root_inertia"][None], data["gravity"])
    children = None if not len(c["parents"]) else RobotModel(c["parents"], c["kinds"], c["offsets"], c["orientations"], c["axes"], c["mass"], c["com"], c["inertia"], data["gravity"])
    return FloatingRobotModel(root, data["root_com"], children, root_inertia_frame_xyzw=data["root_inertia_frame"])


def _tree(data, *, delay=0, kp=None, kd=None, cap=None, iterations=512, model=None):
    n = len(data["q"])-7
    return ph.FloatingRobotTree(
        _model(data) if model is None else model,
        FloatingRobotState(data["q"], data["v"]),
        kp=np.zeros(n) if kp is None else np.asarray(kp, float),
        kd=np.zeros(n) if kd is None else np.asarray(kd, float),
        effort_limit=np.full(n, 64.) if cap is None else np.asarray(cap, float),
        delay_steps=delay, control_max_iterations=iterations,
    )


def _packet(tree, data, q=None, drive=None):
    external = _reference().external(data, data["q"] if q is None else q)
    return tree.inputs(data["drive"] if drive is None else drive, FloatingRobotInputs(*external)), external


def _audit(data, entry, accepted, geometry, diagnostic, summary, selected, external, *, kp=None, kd=None, cap=None):
    reference = _reference()
    q, v, terms = reference.advance(data, np.asarray(entry.position), np.asarray(entry.velocity), selected, external, H, kp=kp, kd=kd, caps=cap)
    n, b, d = len(data["q"])-7, len(data["q"])-6, len(data["v"])
    assert accepted.tick == entry.tick+1 and accepted.control == entry.control
    assert len(accepted.position) == n+7 and len(accepted.velocity) == d
    assert len(accepted.applied_effort) == n and len(accepted.generalized_applied) == d
    assert (geometry.joint_count, geometry.body_count, geometry.velocity_count, geometry.configuration_count) == (n,b,d,n+7)
    assert diagnostic.entry_tick == entry.tick and diagnostic.exit_tick == accepted.tick
    _close(diagnostic.entry_position, entry.position)
    _close(diagnostic.entry_velocity, entry.velocity)
    for name in ("mass", "bias", "linear_jacobian", "angular_jacobian", "external", "acceleration", "drive"):
        _close(getattr(diagnostic, name), terms[name], atol=2e-5 if name == "mass" else ATOL)
    _close(accepted.position, q); _close(accepted.velocity, v)
    expected_geometry = reference.mechanics(data, np.asarray(accepted.position), np.asarray(accepted.velocity))
    for name, other in (("link_position_m","position"),("link_rotation","rotation"),("com_position_m","com_position"),("linear_velocity_m_s","com_velocity"),("angular_velocity_rad_s","angular_velocity")):
        _close(getattr(geometry, name), expected_geometry[other])
    _close(summary["kinetic_energy_j"], expected_geometry["energy"])
    assert diagnostic.equation_residual <= 2e-5
    assert abs(np.dot(accepted.position[3:7], accepted.position[3:7])-1) <= 1e-5
    # Sign continuity, rather than a positive-w convention, is the frozen drift rule.
    assert np.dot(accepted.position[3:7], entry.position[3:7]) >= 0
    full_drive = np.r_[np.zeros(6), diagnostic.drive]
    _close(diagnostic.mass@diagnostic.acceleration+diagnostic.bias, diagnostic.external+full_drive, atol=2e-5)
    return dict(same_entry_reference=terms, actual_end_reference=expected_geometry, predicted_position=q, predicted_velocity=v,
                generalized_entry_power=float(np.dot(diagnostic.external+full_drive, entry.velocity)),
                actual_kinetic_energy=summary["kinetic_energy_j"])


def _window(data, mode, directory, *, steps=40, delay=0, kp=None, kd=None, cap=None, model=None):
    reference = _reference()
    tree = _tree(data, delay=delay, kp=kp, kd=kd, cap=cap, model=model)
    expected_q, expected_v = data["q"].copy(), data["v"].copy()
    n = len(data["q"])-7
    initial_drive = np.column_stack((data["q"][7:], np.zeros(n), np.zeros(n)))
    rows, failure = [], None
    with ph.TyxWorldBatch([(WORLD, tree)], mode, cache_root=directory) as batch:
        try:
            for step in range(steps):
                current = batch.snapshot()[WORLD]
                packet, external = _packet(tree, data, np.asarray(current.position))
                batch.set_controls({WORLD: packet})
                entry = batch.snapshot()[WORLD]
                selected = initial_drive if delay and step == 0 else data["drive"]
                expected_q, expected_v, trajectory_terms = reference.advance(data, expected_q, expected_v, selected, external, H, kp=kp, kd=kd, caps=cap)
                assert batch.step(H) == {WORLD: True}
                accepted, geometry, diagnostic = batch.snapshot()[WORLD], batch.tree_kinematics()[WORLD], batch.tree_diagnostics()[WORLD]
                summary = batch.summary()[WORLD]
                row = dict(entry=entry, accepted=accepted, geometry=geometry, diagnostic=diagnostic, summary=summary,
                           packet=packet, selected_drive=selected, expected_position=expected_q.copy(), expected_velocity=expected_v.copy(), trajectory_reference=trajectory_terms,
                           audit=_audit(data, entry, accepted, geometry, diagnostic, summary, selected, external, kp=kp, kd=kd, cap=cap), stats=batch.stats())
                rows.append(row)
                # At a huge common translation, compare displacement from the exact
                # declared anchor, not a forgiving world-coordinate relative error.
                _close(np.array(accepted.position[:3])-data["q"][:3], expected_q[:3]-data["q"][:3], atol=WINDOW_ATOL, rtol=0)
                _close(accepted.position[3:], expected_q[3:], atol=WINDOW_ATOL)
                _close(accepted.velocity, expected_v, atol=WINDOW_ATOL)
                assert bool(accepted.history) == bool(delay and n)
        except Exception as error:
            failure = dict(type=type(error).__name__, message=str(error))
            raise
        finally:
            _save(directory, "floating-window", dict(input=data, mode=mode, h_s=H, steps=steps, delay=delay, kp=kp, kd=kd, cap=cap, rows=rows, failure=failure, retained=batch.snapshot(), stats=batch.stats()))
    return rows


@pytest.mark.parametrize("name", ("P01","P02_origin","P02_frame","P03","P03_external","P04"))
def test_p01_p04_time_jet_oracle_has_literal_physical_anchors(name):
    reference = _reference(); data = reference.case(name)
    _, _, actual = reference.advance(data, data["q"], data["v"], data["drive"], reference.external(data,data["q"]), H)
    wanted = {
        "P01": [0,0,-2,0,0,0,0], "P02_origin": [0,1,0,0,0,-.5],
        "P02_frame": [0,0,0,3,1,0], "P03": [-1.5,0,0,0,0,0,4.5],
        "P03_external": [0,0,0,0,0,0,3], "P04": [0,0,0,0,0,-2,6],
    }[name]
    _close(actual["acceleration"], wanted, atol=2e-12, rtol=0)
    host = floating_dynamics_terms(_model(data), FloatingRobotState(data["q"], data["v"]), FloatingRobotInputs(data["effort"]+data["drive"][:,2], *reference.external(data,data["q"])[1:]))
    _close(host.mass_matrix, actual["mass"], atol=2e-12, rtol=0)
    _close(host.bias, actual["bias"], atol=2e-12, rtol=0)
    _close(host.acceleration, wanted, atol=2e-12, rtol=0)


@pytest.mark.tyx
@pytest.mark.parametrize("name", ("P01","P02_origin","P02_frame","P03","P03_external","P04","P04_eccentric"))
def test_p01_p04_full_physical_windows(name, tyx_mode, tmp_path):
    data = _reference().case(name)
    rows = _window(data, tyx_mode, tmp_path)
    if name in ("P03", "P04"):
        # This is an observed discrete angular/linear momentum check on the
        # symmetric exact fixtures, not a claim for arbitrary changing inertia.
        for row in rows:
            observed = row["audit"]["actual_end_reference"]
            _close(observed["momentum"], np.zeros(3), atol=2e-6)
            if name == "P04":
                _close(observed["angular_momentum"], np.zeros(3), atol=2e-6)
    if name == "P01":
        for index, row in enumerate(rows, 1):
            masses = np.array([2.,1.])
            center = np.average(row["geometry"].com_position_m, axis=0, weights=masses)
            _close(center, [1/24,0,-H*H*index*(index+1)], atol=2e-5, rtol=0)


@pytest.mark.tyx
@pytest.mark.parametrize("name,placement,boost", (("P01","cycle",True),("P03","reverse",True),("P04_eccentric","cycle",False),("P05","large",True)))
def test_p03_p05_nonaxis_installation_boost_and_large_common_origin(name, placement, boost, tyx_mode, tmp_path):
    _window(_reference().case(name, n=3 if name=="P05" else None, placement=placement, boost=boost), tyx_mode, tmp_path)


@pytest.mark.tyx
@pytest.mark.parametrize("n", (0,1,9,16))
def test_p05_independent_full_root_and_branch_dimensions(n, tyx_mode, tmp_path):
    data = _reference().case("P05", n=n)
    rows = _window(data, tyx_mode, tmp_path)
    assert rows[0]["diagnostic"].mass.shape == (n+6,n+6)
    if n >= 9:
        # Ancestor cross blocks must be real, not independent per-joint surrogates.
        assert np.count_nonzero(np.abs(rows[0]["audit"]["same_entry_reference"]["mass"])-np.diag(np.abs(np.diag(rows[0]["audit"]["same_entry_reference"]["mass"])))) > n


def _tool_data():
    data = _reference().case("P02_origin")
    data.update(name="P05_tool", root_mass=3., root_com=np.array([1/12,0.,0.]), root_inertia=np.array([3/10,53/120,13/24]), force_point=np.array([1/4,0.,0.]))
    data["force"][0] = [0.,3.,0.]
    return data


@pytest.mark.tyx
def test_p05_real_folded_root_tool_mass_frame_and_load_window(tyx_mode, tmp_path):
    from pathlib import Path
    asset = ph.load_floating_urdf(Path(__file__).resolve().parents[1]/"fixtures/urdf/floating_fixed_composite.urdf", gravity=(0,0,0), collapse_fixed_joints=True)
    data = _tool_data()
    assert asset.model.body_count == 1 and asset.model.children is None
    _close(asset.model.root.inverse_mass, [1/3], atol=2e-12, rtol=0)
    _close(asset.model.root_com_m, data["root_com"], atol=2e-12, rtol=0)
    _close(asset.model.root.inertia_diagonal, data["root_inertia"][None], atol=2e-12, rtol=0)
    rows = _window(data, tyx_mode, tmp_path, model=asset.model)
    _close(rows[0]["diagnostic"].acceleration, [0,1,0,0,0,12/13])


@pytest.mark.tyx
@pytest.mark.parametrize("delay", (0,1))
@pytest.mark.parametrize("cap", (1/8,8.))
def test_p06_delayed_drive_external_immediate_and_bounded_full_response(delay, cap, tyx_mode, tmp_path):
    data = _reference().case("P05", n=3)
    data["drive"] = np.column_stack((np.full(3,1/8),np.full(3,-1/16),np.array([1.,-1.,.5])))
    kp,kd,caps = np.array([16.,8.,4.]),np.array([2.,1.,.5]),np.full(3,cap)
    rows = _window(data, tyx_mode, tmp_path, steps=4, delay=delay, kp=kp,kd=kd,cap=caps)
    for row in rows:
        assert np.all(np.abs(row["diagnostic"].drive) <= caps+5e-5)
        assert np.max(np.abs(row["diagnostic"].external)) > 0
    # Both the independent active-face KKT and actual full D residual are checked.
    if delay:
        _close(rows[0]["selected_drive"][:,2], np.zeros(3), atol=0,rtol=0)
        _close(rows[1]["selected_drive"], data["drive"], atol=0,rtol=0)