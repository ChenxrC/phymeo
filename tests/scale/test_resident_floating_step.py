"""C9 ordinary floating trees: frozen public tests, AUTHORING_UNEXECUTED.

Only old APIs are imported at collection. The first node establishes a physical
7x7 HOST premise before looking up the new resident public API. Execution belongs
to the parent; all literal masses, loads and tolerances are declared inputs.
"""

import gzip
import json
from functools import lru_cache
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


@lru_cache(maxsize=1)
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

def _run_worlds(items, mode, directory, *, replay):
    from dataclasses import replace

    rows, repeated, saved, failure = [], [], None, None
    with ph.TyxWorldBatch([(key, item["tree"]) for key,item in items.items()], mode, cache_root=directory) as batch:
        controls = {key:_packet(item["tree"],item["data"])[0] for key,item in items.items()}
        batch.set_controls(controls)
        try:
            for step in range(40):
                entry = batch.snapshot()
                assert batch.step(H) == dict.fromkeys(items, True)
                state,geometry,diagnostic,energy = batch.snapshot(),batch.tree_kinematics(),batch.tree_diagnostics(),batch.summary()
                audits, cache = {}, {}
                for key,item in items.items():
                    data = item["data"]; n = len(data["q"])-7
                    selected = np.column_stack((data["q"][7:],np.zeros(n),np.zeros(n))) if step == 0 else data["drive"]
                    external = _reference().external(data,data["q"])
                    signature = (n,tuple(entry[key].position),tuple(entry[key].velocity))
                    if signature not in cache:
                        audits[key] = _audit(data,entry[key],state[key],geometry[key],diagnostic[key],energy[key],selected,external)
                        cache[signature] = key
                    else:
                        other = cache[signature]
                        assert replace(entry[key],world=other) == entry[other]
                        assert replace(state[key],world=other) == state[other]
                        assert _json(geometry[key]) == _json(geometry[other])
                        assert _json(diagnostic[key]) == _json(diagnostic[other])
                        assert energy[key] == energy[other]
                        audits[key] = dict(exact_same_physics_as_world=other,identity_checked=key)
                    assert state[key].world == key and state[key].tick == step+1
                rows.append(dict(entry=entry,state=state,geometry=geometry,diagnostic=diagnostic,energy=energy,audits=audits,stats=batch.stats()))
                if step == 18:
                    saved = batch.snapshot()
            if replay:
                batch.restore(saved)
                assert batch.snapshot() == saved
                assert all(value is None for value in batch.tree_diagnostics().values())
                for index in range(19,40):
                    assert batch.step(H) == dict.fromkeys(items,True)
                    row = dict(state=batch.snapshot(),geometry=batch.tree_kinematics(),diagnostic=batch.tree_diagnostics(),energy=batch.summary(),stats=batch.stats())
                    repeated.append(row)
                    for field in ("state","geometry","diagnostic","energy"):
                        assert _json(row[field]) == _json(rows[index][field])
        except Exception as error:
            failure=dict(type=type(error).__name__,message=str(error)); raise
        finally:
            _save(directory,"floating-worlds",dict(mode=mode,h_s=H,inputs={k:i["data"] for k,i in items.items()},controls=controls,delay=1,rows=rows,checkpoint_19=saved,replay=repeated,failure=failure,retained=batch.snapshot(),stats=batch.stats()))
    return rows


@pytest.mark.tyx
def test_p05_w1_w4_w32_all_full_states_and_21_step_replay(tyx_mode,tmp_path):
    from dataclasses import replace

    keys = [503,91,1207,6]+[2003+17*i for i in range(28)]
    items = {}
    for i,key in enumerate(keys):
        data = _reference().case("P05",n=(0,1,9,16)[i%4])
        items[key] = dict(data=data,tree=_tree(data,delay=1))
    baseline = {key:_run_worlds({key:items[key]},tyx_mode,tmp_path/f"w1-{key}",replay=False) for key in keys[:4]}
    for count in (4,32):
        selected=dict(reversed(tuple(list(items.items())[:count])))
        rows = _run_worlds(selected,tyx_mode,tmp_path/f"w{count}",replay=True)
        for index,row in enumerate(rows):
            for ordinal,key in enumerate(keys[:count]):
                equivalent=keys[ordinal%4]; single=baseline[equivalent][index]
                # Preserve actual world labels in every saved record. Only this
                # assertion maps distinct declared copies to their W1 counterpart.
                assert replace(row["state"][key],world=equivalent) == single["state"][equivalent]
                for field in ("geometry","diagnostic","energy"):
                    assert _json(row[field][key]) == _json(single[field][equivalent])


def _mixed_four(directory):
    from test_resident_affine_contact_transactions import _ordinary
    from test_resident_affine_step import _ref as affine_reference, _load, _world, _action

    floating_data=_reference().case("P03")
    floating=_tree(floating_data,delay=1)
    ordinary,control=_ordinary()
    old=affine_reference().case("A01")
    old["motor"][:]=3.; old["effort"][:]=0.; old["force"][:]=0.; old["torque"][:]=0.
    asset=_load(directory/"affine",old)
    affine=_world(asset,old,delay=1)
    return [(503,floating),(91,ordinary),(1207,affine),(6,ph.FreeSphere(2,1/8,position=(0,0,2),gravity=(0,0,-8)))],{
        503:_packet(floating,floating_data)[0],91:control,1207:_action(affine,asset,old,ff=old["motor"]),6:(0,0,0),
    },floating_data,old


def _physical_snapshot(batch):
    return dict(state=batch.snapshot(),geometry=_json(batch.tree_kinematics()),energy=batch.summary(),epoch=batch.stats()["state_epoch"],bank=batch.stats()["current_buffer"])


def _floating_storage(batch,world=WORLD):
    return next(storage for specs,storage in batch._tree_storages() if world in specs and specs[world].kind in (9,10))


@pytest.mark.tyx
def test_p05_mixed_existing_fixed_affine_and_free_worlds_share_one_publish(tyx_mode,tmp_path):
    worlds,controls,data,affine=_mixed_four(tmp_path)
    rows=[]
    with ph.TyxWorldBatch(worlds,tyx_mode,cache_root=tmp_path) as batch:
        batch.set_controls(controls)
        for step in range(40):
            before=batch.snapshot(); epoch=batch.stats()["state_epoch"]
            assert batch.step(H) == dict.fromkeys(controls,True)
            after,geo,diag,energy=batch.snapshot(),batch.tree_kinematics(),batch.tree_diagnostics(),batch.summary()
            assert batch.stats()["state_epoch"] == epoch+1
            assert all(value.tick == step+1 for value in after.values())
            # Old ordinary stays unloaded; the affine delay is only its drive.
            _close(after[91].position,[0.]); _close(after[91].velocity,[0.])
            active_steps=step
            _close(after[1207].velocity,[active_steps*H/2])
            _close(after[1207].position,[active_steps*(active_steps+1)*H*H/4])
            _close(after[6].velocity,[0,0,-8*(step+1)*H])
            _close(after[6].position,[0,0,2-4*(step+1)*(step+2)*H*H])
            selected=np.zeros((1,3)) if step==0 else data["drive"]
            audit=_audit(data,before[503],after[503],geo[503],diag[503],energy[503],selected,_reference().external(data,data["q"]))
            rows.append(dict(entry=before,state=after,geometry=geo,diagnostic=diag,energy=energy,audit=audit,stats=batch.stats()))
        _save(tmp_path,"floating-existing-mixed",dict(input=data,affine_input=affine,controls=controls,h_s=H,rows=rows,stats=batch.stats()))


@pytest.mark.tyx
@pytest.mark.parametrize("phase",("upload","candidate","restore"))
@pytest.mark.parametrize("warm",(0,1))
def test_p06_real_late_failure_keeps_both_banks_and_same_request_retries(phase,warm,tyx_mode,tmp_path,monkeypatch):
    worlds,controls,data,old=_mixed_four(tmp_path)
    with ph.TyxWorldBatch(worlds,tyx_mode,cache_root=tmp_path/"attempt") as batch, ph.TyxWorldBatch(worlds,tyx_mode,cache_root=tmp_path/"reference") as fresh:
        batch.set_controls(controls); fresh.set_controls(controls)
        for _ in range(warm):
            assert batch.step(H) == fresh.step(H) == dict.fromkeys(controls,True)
        target=batch.snapshot()
        if phase=="restore":
            assert batch.step(H) == fresh.step(H) == dict.fromkeys(controls,True)
        else:
            # Dirty a real command on both paths so warm=1 still performs upload.
            changed=dict(controls); changed[6]=(1.,0.,0.)
            batch.set_controls(changed); fresh.set_controls(changed)
            storage=_floating_storage(batch)
            item=next(spec for key,spec in worlds if key==503)
            drive=data["drive"].copy(); drive[0,2]=2.
            changed[503]=item.inputs(drive,FloatingRobotInputs.zeros(1))
            batch.set_controls(changed); fresh.set_controls(changed)
        before=_physical_snapshot(batch); storage=_floating_storage(batch)
        method={"upload":"upload_controls","candidate":"check_advance","restore":"check_restore"}[phase]
        original=getattr(storage,method); called=[]
        def real_then_fail(*args,**kwargs):
            value=original(*args,**kwargs); called.append(True)
            raise RuntimeError("c9 after real "+phase)
        with monkeypatch.context() as patch:
            patch.setattr(storage,method,real_then_fail)
            with pytest.raises(RuntimeError,match="c9 after real"):
                batch.restore(target) if phase=="restore" else batch.step(H)
        assert called and _physical_snapshot(batch)==before
        assert all(value is None for value in batch.tree_diagnostics().values())
        if phase=="restore":
            batch.restore(target); fresh.restore(target)
        else:
            assert batch.step(H)==fresh.step(H)==dict.fromkeys(controls,True)
        assert batch.snapshot()==fresh.snapshot()
        assert _json(batch.tree_kinematics())==_json(fresh.tree_kinematics())
        assert _json(batch.tree_diagnostics())==_json(fresh.tree_diagnostics())
        _save(tmp_path,"floating-late-fault",dict(phase=phase,warm=warm,input=data,affine_input=old,before=before,restore_target=target,retried=batch.snapshot(),reference=fresh.snapshot(),geometry=batch.tree_kinematics(),diagnostic=batch.tree_diagnostics(),stats=batch.stats()))


@pytest.mark.tyx
@pytest.mark.parametrize("problem",("q_length","v_length","zero_quat","quat_norm","nan","local_position","joint_position","foreign_topology"))
def test_p06_invalid_restore_is_atomic_for_mixed_full_dimensions(problem,tyx_mode,tmp_path):
    from dataclasses import replace

    worlds,controls,data,old=_mixed_four(tmp_path)
    with ph.TyxWorldBatch(worlds,tyx_mode,cache_root=tmp_path) as batch:
        batch.set_controls(controls); assert batch.step(H)==dict.fromkeys(controls,True)
        before=_physical_snapshot(batch); state=before["state"][503]
        q=list(state.position); v=list(state.velocity)
        if problem=="q_length": q.pop()
        elif problem=="v_length": v.pop()
        elif problem=="zero_quat": q[3:7]=[0.]*4
        elif problem=="quat_norm": q[6]=1.01
        elif problem=="nan": v[0]=float("nan")
        elif problem=="local_position": q[0]=4097.
        elif problem=="joint_position": q[7]=4097.
        else:
            alien=_reference().case("P04")
            with ph.TyxWorldBatch([(503,_tree(alien))],tyx_mode,cache_root=tmp_path/"foreign") as other:
                state=replace(state,topology=other.snapshot()[503].topology)
        wrong=replace(state,position=tuple(q),velocity=tuple(v))
        with pytest.raises((ValueError,TypeError)):
            batch.restore({6:before["state"][6],503:wrong})
        assert _physical_snapshot(batch)==before
        _save(tmp_path,"floating-invalid-restore",dict(problem=problem,input=data,before=before,retained=batch.snapshot(),stats=batch.stats()))


@pytest.mark.tyx
@pytest.mark.parametrize("kind",("root_translation","joint_translation","rotation_vector"))
def test_p06_candidate_domain_failure_is_world_local_and_does_not_keep_old_diagnostic(kind,tyx_mode,tmp_path):
    from dataclasses import replace

    data=_reference().case("P03" if kind!="rotation_vector" else "P02_frame")
    data["drive"][:]=0; data["torque"][:]=0
    tree=_tree(data)
    with ph.TyxWorldBatch([(503,tree),(91,ph.FreeSphere(2,1/8,gravity=(0,0,0)))],tyx_mode,cache_root=tmp_path) as batch:
        assert batch.step(H)=={503:True,91:True}
        state=batch.snapshot()[503]; q=list(state.position); v=[0.]*len(state.velocity)
        if kind=="root_translation": q[0]=4096.; v[0]=1.
        elif kind=="joint_translation": q[7]=4096.; v[6]=1.
        else: v[5]=4096.
        boundary=replace(state,position=tuple(q),velocity=tuple(v))
        batch.restore({503:boundary}); before=batch.snapshot()
        assert batch.step(H)=={503:False,91:True}
        assert batch.snapshot()[503]==before[503]
        assert batch.snapshot()[91].tick==before[91].tick+1
        assert batch.tree_diagnostics()[503] is None
        batch.restore({503:state}); assert batch.step(H)=={503:True,91:True}
        _save(tmp_path,"floating-domain-retain",dict(kind=kind,input=data,before=before,recovered=batch.snapshot(),geometry=batch.tree_kinematics(),stats=batch.stats()))

@pytest.mark.tyx
def test_p06_full_cross_motor_mobility_low_budget_rejects_then_full_budget_agrees(tyx_mode,tmp_path):
    data=_reference().case("P05",n=2)
    data["root_com"][:]=0; data["root_inertia"][:]=.5; data["root_inertia_frame"][:]=[0,0,0,1]
    data["q"][:]=[0,0,0,0,0,0,1,0,0]; data["v"][:]=0; data["gravity"][:]=0
    c=data["children"]
    c.update(parents=np.array([-1,-1]),kinds=("prismatic","prismatic"),offsets=np.zeros((2,3)),orientations=np.tile([0.,0.,0.,1.],(2,1)),axes=np.tile([1.,0.,0.],(2,1)),mass=np.ones(2),com=np.zeros((2,3)),inertia=np.full((2,3),.25))
    data["drive"][:]=[[0,1/128,0],[0,-1/128,0]]
    data["effort"][:]=0; data["force"][:]=0; data["torque"][:]=0
    kp,kd,caps=np.zeros(2),np.full(2,1024.),np.full(2,64.)
    truth=_reference().mechanics(data,data["q"],data["v"])
    _close(truth["mass"][np.ix_([0,6,7],[0,6,7])],[[4,1,1],[1,1,0],[1,0,1]],atol=2e-12,rtol=0)
    _close(np.linalg.inv(truth["mass"])[6:,6:],[[1.5,.5],[.5,1.5]],atol=2e-12,rtol=0)
    wide=_tree(data,kp=kp,kd=kd,cap=caps)
    narrow=_tree(data,kp=kp,kd=kd,cap=caps,iterations=1)
    with ph.TyxWorldBatch([(503,narrow),(91,ph.FreeSphere(2,1/8,gravity=(0,0,0)))],tyx_mode,cache_root=tmp_path/"budget1") as batch:
        batch.set_controls({503:_packet(narrow,data)[0]}); before=batch.snapshot()
        assert batch.step(H)=={503:False,91:True}
        assert batch.snapshot()[503]==before[503] and batch.tree_diagnostics()[503] is None
        _save(tmp_path,"floating-low-budget",dict(input=data,before=before,retained=batch.snapshot(),stats=batch.stats()))
    rows=_window(data,tyx_mode,tmp_path/"budget512",steps=2,kp=kp,kd=kd,cap=caps)
    _close(rows[0]["diagnostic"].drive,[4,-4])
    _close(rows[0]["diagnostic"].acceleration,[0,0,0,0,0,0,4,-4])
    assert wide is not narrow


@pytest.mark.parametrize("problem",("n17","gain_width","negative_gain","delay2","bool_iterations","zero_iterations","nonunit_initial","load_effort_width","load_body_width"))
def test_p06_declared_input_dimensions_and_resources_reject_before_runtime(problem):
    data=_reference().case("P05",n=17 if problem=="n17" else 1)
    options={}
    if problem=="gain_width": options["kp"]=np.ones(2)
    elif problem=="negative_gain": options["kd"]=np.array([-1.])
    elif problem=="delay2": options["delay"]=2
    elif problem=="bool_iterations": options["iterations"]=True
    elif problem=="zero_iterations": options["iterations"]=0
    elif problem=="nonunit_initial": data["q"][3:7]=[0,0,0,1.01]
    if problem.startswith("load_"):
        tree=_tree(data)
        with pytest.raises((TypeError,ValueError)):
            external=FloatingRobotInputs(np.zeros(2 if problem=="load_effort_width" else 1),np.zeros((3 if problem=="load_body_width" else 2,3)),np.zeros((2,3)))
            tree.inputs(np.zeros((1,3)),external)
    else:
        with pytest.raises((TypeError,ValueError)):
            _tree(data,**options)


@pytest.mark.tyx
def test_p06_owned_outputs_controls_and_invalid_mixed_update_are_atomic(tyx_mode,tmp_path):
    data=_reference().case("P03")
    kp,kd,cap=np.zeros(1),np.zeros(1),np.array([8.])
    tree=_tree(data,kp=kp,kd=kd,cap=cap)
    drive=data["drive"].copy(); effort=np.zeros(1); force=np.zeros((2,3)); torque=np.zeros((2,3))
    external=FloatingRobotInputs(effort,force,torque); packet=tree.inputs(drive,external)
    drive[:]=99; effort[:]=99; force[:]=99; torque[:]=99
    kp[:]=99; kd[:]=99; cap[:]=0
    with ph.TyxWorldBatch([(503,tree),(91,_tree(data))],tyx_mode,cache_root=tmp_path) as batch:
        batch.set_controls({503:packet,91:packet})
        before=batch.snapshot()
        wrong_data=_reference().case("P05",n=2); other=_tree(wrong_data)
        wrong=_packet(other,wrong_data)[0]
        with pytest.raises((ValueError,TypeError)):
            batch.set_controls({503:tree.inputs(np.array([[0.,0.,2.]])),91:wrong})
        assert batch.snapshot()==before
        assert batch.step(H)=={503:True,91:True}
        state,geometry,diag=batch.snapshot(),batch.tree_kinematics(),batch.tree_diagnostics()
        _close(state[503].velocity,H*np.array([-1.5,0,0,0,0,0,4.5]))
        _close(state[503].velocity,state[91].velocity,atol=0,rtol=0)
        old_geometry,old_diag=_json(geometry),_json(diag)
        for value in (geometry[503].link_position_m,geometry[503].link_rotation,geometry[503].linear_velocity_m_s,diag[503].mass,diag[503].acceleration,diag[503].drive):
            assert not value.flags.writeable
            with pytest.raises(ValueError): value.setflags(write=True)
        assert batch.step(H)=={503:True,91:True}
        assert _json(geometry)==old_geometry and _json(diag)==old_diag
        with pytest.raises((ValueError,TypeError)): batch.kinematics([503])
        with pytest.raises((ValueError,TypeError)): batch.prepare_readback([503],kind="kinematics")
        _save(tmp_path,"floating-owned-update",dict(input=data,packet=packet,before=before,first=state,first_geometry=geometry,first_diagnostic=diag,retained=batch.snapshot(),stats=batch.stats()))