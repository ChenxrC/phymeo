"""C11 floating-affine public cases; AUTHORING_UNEXECUTED.

Existing HOST mechanics establish the literal premise before the first test
looks up the new public device declaration. No new product import at collection.
"""

import gzip
import hashlib
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph
from phymeo.dynamics.floating_robot import FloatingRobotState
from phymeo.solvers.floating_tree import FloatingRobotInputs, floating_dynamics_terms

H = 1 / 1024
ATOL, RTOL, WINDOW_ATOL = 5e-6, 5e-5, 2e-4
WORLD = 503
ASSETS = Path(__file__).resolve().parents[1] / "fixtures" / "urdf"
JAWS = ASSETS / "floating_mimic_jaws.urdf"
ELBOW = ASSETS / "floating_mimic_elbow.urdf"
RAW_LIMIT = 256 * 1024 * 1024
COMPRESSED_LIMIT = 64 * 1024 * 1024


def _json(value):
    if type(value) is object:
        # Process-local capability tokens remain runtime-only, not replay authority.
        return {
            "opaque_process_local_type": "builtins.object",
            "identity_serialized": False,
        }
    if isinstance(value, ph.DrivenRobotProgram):
        return _json(value.problem_signature())
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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def _save(directory, name, payload):
    """Complete lossless output; resource exhaustion is an IO failure, not physics."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name + ".json.gz")
    size = 0
    encoder = json.JSONEncoder(allow_nan=False, separators=(",", ":"))
    with gzip.open(path, "wb") as stream:
        for piece in encoder.iterencode(_json(payload)):
            data = piece.encode("utf-8")
            size += len(data)
            if size > RAW_LIMIT:
                raise RuntimeError("C11 complete payload exceeds declared 256 MiB raw limit")
            stream.write(data)
    if path.stat().st_size > COMPRESSED_LIMIT:
        raise RuntimeError("C11 complete payload exceeds declared 64 MiB gzip limit")


def _close(actual, expected, *, atol=ATOL, rtol=RTOL):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def _jaws(*, gravity=(0.0, 0.0, 0.0)):
    return ph.load_floating_mimic_urdf(JAWS, gravity=gravity)


def _jaws_state():
    return FloatingRobotState(np.array([0.0, 0, 0, 0, 0, 0, 1, 0]), np.zeros(7))


def _fixture_record(path):
    raw = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "utf8": raw.decode("utf-8"),
    }


@pytest.mark.tyx
def test_a01_public_affine_floating_root_reaction(tyx_mode, tmp_path):
    asset, initial = _jaws(), _jaws_state()
    full = asset.full_state(initial)
    external = FloatingRobotInputs.zeros(2)
    host = floating_dynamics_terms(asset.tree.model, full, external)
    literal_x_mass = np.array([[5.0, 1, 2], [1, 1, 0], [2, 0, 2]])
    literal_reduced_x_mass = np.array([[5.0, -1], [-1, 3]])
    literal_a = np.array([1 / 14, 0, 0, 0, 0, 0, 5 / 14, -5 / 14])
    literal_constraint = np.array([0.0, 0, 0, 0, 0, 0, -4 / 7, -4 / 7])
    assert asset.tree.model.body_count == 3
    assert full.q.shape == (9,) and full.v.shape == (8,)
    assert initial.q.shape == (8,) and initial.v.shape == (7,)
    assert host.mass_matrix.shape == (8, 8)
    _close(host.mass_matrix[np.ix_([0, 6, 7], [0, 6, 7])], literal_x_mass, atol=2e-12, rtol=0)
    _close(host.bias, np.zeros(8), atol=2e-12, rtol=0)
    _close(asset.matrix, [[1.0], [-1.0]], atol=0, rtol=0)
    _close(asset.offset_vector, [0, 0.004], atol=0, rtol=0)
    _close(asset.constraint_rows, [[1.0, 1.0]], atol=0, rtol=0)
    host_step = asset.step(initial, effort=np.array([1.0]), dt_s=H)
    _close(host_step.loads.acceleration_full, literal_a, atol=2e-12, rtol=0)
    _close(host_step.loads.constraint_effort_full, literal_constraint, atol=2e-12, rtol=0)
    assert host_step.loads.multiplier_force == pytest.approx(-4 / 7, abs=2e-12)
    assert host_step.loads.constraint_power_W == pytest.approx(0.0, abs=2e-12)
    # Independent momentum identity: 2 ar + (ar+u'') + 2(ar-u'') = 0.
    assert 5 * Fraction(1, 14) - Fraction(5, 14) == 0
    record = dict(
        case="A01",
        mode=tyx_mode,
        dt_s=H,
        asset=asset,
        fixture=_fixture_record(JAWS),
        initial=initial,
        full_initial=full,
        external=external,
        drive=[[0.0, 0.0, 1.0]],
        literal_x_mass=literal_x_mass,
        literal_reduced_x_mass=literal_reduced_x_mass,
        literal_acceleration=literal_a,
        literal_constraint=literal_constraint,
        host_premise=host,
        host_step=host_step,
        frames=[],
    )
    try:
        # This lookup is deliberately after the real old-API physical premises.
        tree = ph.FloatingAffineRobotTree(
            asset, initial, kp=np.zeros(1), kd=np.zeros(1), effort_limit=np.array([1.5])
        )
        packet = tree.inputs(np.array([[0.0, 0.0, 1.0]]), external=external)
        record["packet"] = packet
        with ph.TyxWorldBatch([(WORLD, tree)], tyx_mode, cache_root=tmp_path / "cache") as batch:
            batch.set_controls({WORLD: packet})
            entry = batch.snapshot()[WORLD]
            record["compiled"] = _compiled(batch)
            assert len(entry.position) == 8 and len(entry.velocity) == 7
            assert len(entry.full_position) == 9 and len(entry.full_velocity) == 8
            assert len(entry.control) == 5 and entry.control[0] == packet.graph_token
            assert batch.step(H) == {WORLD: True}
            accepted = batch.snapshot()[WORLD]
            geometry = batch.tree_kinematics()[WORLD]
            diagnostic = batch.tree_diagnostics()[WORLD]
            record["frames"].append(
                dict(
                    entry=entry,
                    accepted=accepted,
                    geometry=geometry,
                    diagnostic=diagnostic,
                    summary=batch.summary(),
                    stats=batch.stats(),
                )
            )
            assert accepted.tick == entry.tick + 1
            assert diagnostic.mass.shape == (8, 8)
            assert diagnostic.reduced_mass.shape == (7, 7)
            _close(diagnostic.mass[np.ix_([0, 6, 7], [0, 6, 7])], literal_x_mass, atol=2e-5)
            _close(
                diagnostic.reduced_mass[np.ix_([0, 6], [0, 6])], literal_reduced_x_mass, atol=2e-5
            )
            _close(diagnostic.full_acceleration, literal_a)
            _close(diagnostic.acceleration, literal_a[:7])
            _close(diagnostic.drive, [1.0])
            _close(diagnostic.constraint_effort_full, literal_constraint, atol=2e-5)
            _close(diagnostic.root_constraint_residual, np.zeros(6), atol=2e-5, rtol=0)
            _close(diagnostic.row_multipliers, [-4 / 7], atol=2e-5)
            assert diagnostic.constraint_power_w == pytest.approx(0.0, abs=2e-5)
            _close(accepted.velocity, H * literal_a[:7])
            _close(accepted.full_velocity, H * literal_a)
            _close(accepted.position[:3], [H * H / 14, 0, 0])
            _close(accepted.position[7:], [H * H * 5 / 14])
            _close(accepted.full_position[7:], [H * H * 5 / 14, 0.004 - H * H * 5 / 14])
            _close(accepted.applied_effort, [1.0])
            _close(accepted.full_applied_effort, [1.0, 0.0])
            _close(accepted.generalized_applied, [0, 0, 0, 0, 0, 0, 1])
            _close(accepted.full_generalized_applied, [0, 0, 0, 0, 0, 0, 1, 0])
            _close(geometry.position, accepted.full_position)
            _close(geometry.velocity, accepted.full_velocity)
            _close(geometry.independent_position, accepted.position)
            _close(geometry.independent_velocity, accepted.velocity)
            assert geometry.joint_names == ("driver",)
            assert geometry.full_joint_names == ("driver", "follower")
            v = np.asarray(accepted.full_velocity)
            _close((5 * v[0] + v[6] + 2 * v[7]) / 5, 0.0, atol=ATOL, rtol=0)
            for value in (
                diagnostic.mass,
                diagnostic.reduced_mass,
                diagnostic.full_acceleration,
                diagnostic.constraint_effort_full,
                geometry.position,
            ):
                assert not value.flags.writeable
            record["retained"] = batch.snapshot()
    except BaseException as exc:
        record["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        _save(tmp_path, "a01-public", record)


# Remaining frozen cases use a standalone independent reference, never product M/J/P.
@lru_cache(maxsize=1)
def _reference():
    import importlib.util

    path = (
        Path(__file__).resolve().parents[1]
        / "reference_newton/robot/resident_floating_affine_reference.py"
    )
    spec = importlib.util.spec_from_file_location("c11_full_component_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _asset(directory, data):
    from xml.etree import ElementTree as ET

    source = data["fixture"]
    if source in ("jaws", "elbow"):
        path = ELBOW if source == "elbow" else JAWS
    elif source == "graph":
        root = ET.fromstring(JAWS.read_text(encoding="utf-8"))
        root.append(
            ET.fromstring("""<link name="gauge"><inertial><mass value="0.5"/>
            <inertia ixx="0.02" iyy="0.03" izz="0.04" ixy="0" ixz="0" iyz="0"/>
            </inertial></link>""")
        )
        driver, multiplier = (
            ("follower", "-0.5") if data["variant"] == "chain" else ("driver", "0.5")
        )
        root.append(
            ET.fromstring(f'''<joint name="gauge_axis" type="prismatic">
            <parent link="base"/><child link="gauge"/><origin xyz="0.5 0 0"/><axis xyz="1 0 0"/>
            <limit lower="-0.003" upper="0.007" effort="0.125" velocity="0.03"/>
            <mimic joint="{driver}" multiplier="{multiplier}" offset="0.002"/></joint>''')
        )
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "declared-affine.urdf"
        path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    else:
        root = ET.Element("robot", name="c11_" + source)
        c = data["children"]
        masses = np.r_[data["root_mass"], c["mass"]]
        centers = np.vstack((data["root_com"], c["com"]))
        moments = np.vstack((data["root_inertia"], c["inertia"]))
        for i, (mass, center, inertia) in enumerate(zip(masses, centers, moments, strict=True)):
            link = ET.SubElement(root, "link", name="base" if i == 0 else f"body_{i - 1:02d}")
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "origin", xyz=" ".join(map(str, center)))
            ET.SubElement(inertial, "mass", value=str(mass))
            ET.SubElement(
                inertial,
                "inertia",
                ixx=str(inertia[0]),
                iyy=str(inertia[1]),
                izz=str(inertia[2]),
                ixy="0",
                ixz="0",
                iyz="0",
            )
        for i, parent in enumerate(c["parents"]):
            joint = ET.SubElement(root, "joint", name=f"axis_{i:02d}", type=c["kinds"][i])
            ET.SubElement(joint, "parent", link="base" if parent < 0 else f"body_{parent:02d}")
            ET.SubElement(joint, "child", link=f"body_{i:02d}")
            ET.SubElement(joint, "origin", xyz=" ".join(map(str, c["offsets"][i])))
            ET.SubElement(joint, "axis", xyz=" ".join(map(str, c["axes"][i])))
            ET.SubElement(joint, "limit", lower="-1", upper="1", velocity="1", effort="8")
            if i not in data["drivers"]:
                j = int(np.flatnonzero(data["P"][i])[0])
                ET.SubElement(
                    joint,
                    "mimic",
                    joint=f"axis_{data['drivers'][j]:02d}",
                    multiplier=str(data["P"][i, j]),
                    offset=str(data["offset"][i]),
                )
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "declared-affine.urdf"
        path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    asset = ph.load_floating_mimic_urdf(
        path, gravity=data["gravity"], collapse_fixed_joints=source == "elbow"
    )
    _close(asset.matrix, data["P"], atol=0, rtol=0)
    _close(asset.offset_vector, data["offset"], atol=2e-12, rtol=0)
    _close(asset.constraint_rows, data["C"][:, 6:], atol=0, rtol=0)
    return asset, _fixture_record(path)


def _item(directory, data, *, delay=0):
    import copy

    data = copy.deepcopy(data)
    data["delay"] = delay
    asset, source = _asset(directory, data)
    state = FloatingRobotState(data["q"], data["v"])
    tree = ph.FloatingAffineRobotTree(
        asset, state, kp=data["kp"], kd=data["kd"], effort_limit=data["caps"], delay_steps=delay
    )
    external = FloatingRobotInputs(np.zeros(len(data["effort"])), data["force"], data["torque"])
    command = tree.inputs(data["drive"], external=external)
    return dict(
        data=data,
        asset=asset,
        source=source,
        tree=tree,
        external=external,
        command=command,
        delay=delay,
        kind="floating_affine",
    )


def _selected(item, tick):
    data = item["data"]
    if tick < item["delay"]:
        held = np.zeros_like(data["drive"])
        held[:, 0] = np.asarray(data["q"][7:], np.float32)
        return held
    return data["drive"]


def _audit(item, entry, accepted, geometry, diagnostic, summary, *, h=None, wrench=None):
    ref, data = _reference(), item["data"]
    h = data["h"] if h is None else h
    selected = _selected(item, entry.tick)
    q, v, truth = ref.advance(
        data, np.asarray(entry.position), np.asarray(entry.velocity), selected, h, wrench=wrench
    )
    n, k = data["P"].shape
    assert (
        len(accepted.position),
        len(accepted.velocity),
        len(accepted.full_position),
        len(accepted.full_velocity),
    ) == (7 + k, 6 + k, 7 + n, 6 + n)
    assert (
        geometry.joint_count,
        geometry.body_count,
        geometry.velocity_count,
        geometry.configuration_count,
    ) == (n, n + 1, n + 6, n + 7)
    assert accepted.tick == entry.tick + 1 and accepted.control == entry.control
    assert (diagnostic.entry_tick, diagnostic.exit_tick, diagnostic.dt_s) == (
        entry.tick,
        accepted.tick,
        h,
    )
    _close(diagnostic.entry_position, entry.position)
    _close(diagnostic.entry_velocity, entry.velocity)
    _close(diagnostic.full_entry_position, entry.full_position)
    _close(diagnostic.full_entry_velocity, entry.full_velocity)
    for field in (
        "mass",
        "bias",
        "external",
        "linear_jacobian",
        "angular_jacobian",
        "reduced_mass",
        "reduced_bias",
        "reduced_external",
        "reduced_linear_jacobian",
        "reduced_angular_jacobian",
        "acceleration",
        "full_acceleration",
        "drive",
        "constraint_effort_full",
        "row_multipliers",
        "row_powers_w",
        "root_constraint_residual",
    ):
        _close(
            getattr(diagnostic, field),
            truth[field],
            atol=2e-5
            if field
            in (
                "mass",
                "reduced_mass",
                "constraint_effort_full",
                "row_multipliers",
                "row_powers_w",
                "root_constraint_residual",
            )
            else ATOL,
        )
    _close(accepted.position, q)
    _close(accepted.velocity, v)
    _close(accepted.full_position, truth["next_full_position"])
    _close(accepted.full_velocity, truth["next_full_velocity"])
    _close(accepted.applied_effort, truth["drive"])
    _close(accepted.full_applied_effort, (data["E"] @ truth["drive"])[6:])
    _close(accepted.generalized_applied, truth["reduced_applied"])
    _close(accepted.full_generalized_applied, truth["full_applied"])
    _close(geometry.position, accepted.full_position)
    _close(geometry.velocity, accepted.full_velocity)
    _close(geometry.independent_position, accepted.position)
    _close(geometry.independent_velocity, accepted.velocity)
    assert geometry.joint_names == item["asset"].joint_names
    assert geometry.full_joint_names == item["asset"].tree.joint_names
    actual_q, actual_v = np.asarray(accepted.full_position), np.asarray(accepted.full_velocity)
    end = ref.mechanics(data, actual_q, actual_v)
    for field, expected in (
        ("com_position_m", "com_position"),
        ("linear_velocity_m_s", "com_velocity"),
        ("angular_velocity_rad_s", "angular_velocity"),
    ):
        _close(getattr(geometry, field), end[expected])
    if data["parts"] is None:
        _close(geometry.link_position_m, end["position"])
        _close(geometry.link_rotation, end["rotation"])
    else:
        # Compare authored frames, not principal group axes with raw URDF axes.
        for group, name in enumerate(("base", "upper", "forearm")):
            site = item["asset"].tree.sites[name]
            actual_R = geometry.link_rotation[group] @ ref.base().quat_matrix(
                site.local_orientation_xyzw
            )
            actual_p = (
                geometry.link_position_m[group]
                + geometry.link_rotation[group] @ site.local_position_m
            )
            _close(actual_R, end["rotation"][group])
            _close(actual_p, end["position"][group])
    _close(summary["kinetic_energy_j"], end["energy"])
    assert diagnostic.control_residual <= 5e-5 and diagnostic.equation_residual <= 2e-5
    full_motor = data["E"] @ np.asarray(diagnostic.drive)
    lhs = diagnostic.mass @ diagnostic.full_acceleration + diagnostic.bias
    residual = lhs - diagnostic.external - full_motor
    scale = max(1.0, float(np.max(abs(lhs))), float(np.max(abs(diagnostic.external + full_motor))))
    _close(residual, diagnostic.constraint_effort_full, atol=2e-5 * scale, rtol=0)
    _close(residual[:6], np.zeros(6), atol=2e-5 * scale, rtol=0)
    _close(data["S"].T @ residual, np.zeros(k + 6), atol=2e-5 * scale, rtol=0)
    _close(residual, data["C"].T @ diagnostic.row_multipliers, atol=2e-5 * scale, rtol=0)
    powers = np.asarray(diagnostic.row_multipliers) * (data["C"] @ np.asarray(entry.full_velocity))
    _close(diagnostic.row_powers_w, powers, atol=2e-5, rtol=0)
    _close(diagnostic.constraint_power_w, np.sum(powers), atol=2e-5, rtol=0)
    power_scale = max(1.0, float(np.sum(abs(np.asarray(entry.full_velocity) * residual))))
    assert np.max(abs(powers), initial=0.0) <= 2e-5 * power_scale
    assert abs(diagnostic.constraint_power_w) <= 2e-5 * power_scale
    assert np.asarray(accepted.position[3:7]) @ np.asarray(entry.position[3:7]) >= 0
    for value in (
        diagnostic.mass,
        diagnostic.reduced_mass,
        diagnostic.full_acceleration,
        diagnostic.row_multipliers,
        geometry.independent_position,
    ):
        assert not value.flags.writeable
    return dict(
        same_entry=truth,
        next_q=q,
        next_v=v,
        actual_end=end,
        raw_full_residual=residual,
        original_row_power=powers,
        power_scale=power_scale,
    )


def _compiled(batch):
    # Immutable original model/graph plus actual FP32 compilation tables.
    return {world: spec for specs, _ in batch._tree_storages() for world, spec in specs.items()}


def _record(batch, plan=None):
    result = dict(
        state=batch.snapshot(),
        geometry=batch.tree_kinematics(),
        diagnostics=batch.tree_diagnostics(),
        summary=batch.summary(),
    )
    if plan is not None:
        result["receipts"] = dict(plan.last_accepted)
    return result


def _run(item, mode, directory, name, *, steps, replay=False):
    ref, data = _reference(), item["data"]
    q, v, saved = data["q"].copy(), data["v"].copy(), None
    output = dict(
        input=data,
        source=item["source"],
        asset=item["asset"],
        command=item["command"],
        mode=mode,
        rows=[],
        replay=[],
        saved=None,
    )
    with ph.TyxWorldBatch([(WORLD, item["tree"])], mode, cache_root=directory / "cache") as batch:
        batch.set_controls({WORLD: item["command"]})
        output["initial"] = _record(batch)
        output["compiled"] = _compiled(batch)
        try:
            for index in range(steps):
                entry = batch.snapshot()[WORLD]
                q, v, independent = ref.advance(data, q, v, _selected(item, entry.tick))
                assert batch.step(data["h"]) == {WORLD: True}
                row = _record(batch)
                audit = _audit(
                    item,
                    entry,
                    row["state"][WORLD],
                    row["geometry"][WORLD],
                    row["diagnostics"][WORLD],
                    row["summary"][WORLD],
                )
                row.update(
                    entry={WORLD: entry},
                    audit={WORLD: audit},
                    trajectory_reference=dict(q=q.copy(), v=v.copy(), mechanics=independent),
                )
                output["rows"].append(row)
                _close(row["state"][WORLD].position, q, atol=WINDOW_ATOL)
                _close(row["state"][WORLD].velocity, v, atol=WINDOW_ATOL)
                if index == 18:
                    saved = batch.snapshot()
                    output["saved"] = saved
            if data["name"] == "counts":
                output["body16_current_bounds"] = _count_bounds(batch, item)
            if replay:
                assert saved is not None and steps == 40
                batch.restore(saved)
                for index in range(19, 40):
                    assert batch.step(data["h"]) == {WORLD: True}
                    row = _record(batch)
                    output["replay"].append(row)
                    for field in ("state", "geometry", "diagnostics", "summary"):
                        assert _json(row[field]) == _json(output["rows"][index][field])
        except BaseException as exc:
            output["failure"] = dict(type=type(exc).__name__, message=str(exc))
            raise
        finally:
            output.update(retained=_record(batch), stats=batch.stats())
            _save(directory, name, output)
    return output


@pytest.mark.tyx
@pytest.mark.parametrize("placement", ("cycle",), ids=("cycle",))
def test_a01_cycle_translation_preserves_full_response(placement, tyx_mode, tmp_path):
    data = _reference().case("A01", placement=placement)
    _run(_item(tmp_path / "asset", data), tyx_mode, tmp_path, "a01-cycle", steps=4)


@pytest.mark.tyx
@pytest.mark.parametrize("variant", ("fan", "chain"), ids=("fan", "chain"))
def test_a03_original_graph_rows_and_offset(variant, tyx_mode, tmp_path):
    data = _reference().case("A03", variant=variant)
    item = _item(tmp_path / "asset", data)
    _close(item["asset"].lower, [-0.006 if variant == "chain" else -0.01], atol=2e-12)
    _close(item["asset"].upper, [0.014 if variant == "chain" else 0.01], atol=2e-12)
    _close(item["asset"].velocity_limit, [0.06], atol=2e-12)
    _run(item, tyx_mode, tmp_path, "a03-" + variant, steps=4)


@pytest.mark.tyx
@pytest.mark.parametrize(
    "variant", ("p0", "p1", "p2", "p1-cycle"), ids=("p0", "p1", "p2", "p1-cycle")
)
def test_a04_raw_parts_rotating_tool_response(variant, tyx_mode, tmp_path):
    data = _reference().case("A04", variant=variant)
    _run(_item(tmp_path / "asset", data), tyx_mode, tmp_path, "a04-" + variant, steps=1)


@pytest.mark.tyx
@pytest.mark.parametrize("variant", ("p1",), ids=("p1",))
def test_a04_rotating_full_reduced_checkpoint_replay(variant, tyx_mode, tmp_path):
    data = _reference().case("A04", variant=variant)
    _run(
        _item(tmp_path / "asset", data), tyx_mode, tmp_path, "a04-p1-window", steps=40, replay=True
    )


@pytest.mark.tyx
@pytest.mark.parametrize(
    "variant,delay,cap",
    (("d0-c1p5", 0, 1.5), ("d1-c1p5", 1, 1.5), ("d0-c0p25", 0, 0.25), ("d1-c0p25", 1, 0.25)),
    ids=("d0-c1p5", "d1-c1p5", "d0-c0p25", "d1-c0p25"),
)
def test_a01_pd_delay_and_current_com_load(variant, delay, cap, tyx_mode, tmp_path):
    data = _reference().case("A01")
    data.update(
        h=1 / 128,
        kp=np.array([100.0]),
        kd=np.array([2.0]),
        caps=np.array([cap]),
        drive=np.array([[0.01, 0.0, 0.0]]),
    )
    data["force"][0, 0] = 1.0
    item = _item(tmp_path / "asset", data, delay=delay)
    result = _run(item, tyx_mode, tmp_path, "pd-" + variant, steps=4)
    h, target = data["h"], (0.0 if delay else 0.01)
    eta = h * (100 * h + 2) / (14 / 5)
    literal = np.clip((100 * target - eta / 5) / (1 + eta), -cap, cap)
    _close(result["rows"][0]["diagnostics"][WORLD].drive, [literal])
    assert result["rows"][0]["state"][WORLD].control[0] == item["command"].graph_token


@pytest.mark.tyx
@pytest.mark.parametrize("variant", ("n16-k1", "n16-k15"), ids=("n16-k1", "n16-k15"))
def test_full_body_and_reduced_motor_counts(variant, tyx_mode, tmp_path):
    data = _reference().case("counts", variant=variant)
    item = _item(tmp_path / "asset", data)
    result = _run(item, tyx_mode, tmp_path, "counts-" + variant, steps=1)
    row = result["rows"][0]
    d = row["diagnostics"][WORLD]
    ids = [0, *range(6, 22)]
    literal = np.zeros((17, 17))
    literal[0, 0] = 10.0
    literal[0, 1:] = literal[1:, 0] = 0.5
    literal[1:, 1:] = np.eye(16) * 0.5
    _close(d.mass[np.ix_(ids, ids)], literal, atol=2e-5)
    assert row["geometry"][WORLD].body_count == 17
    assert d.full_acceleration.shape == (22,)


def _count_bounds(batch, item):
    checkpoint = batch.snapshot()[WORLD]
    key = ph.CollisionKey(WORLD, 116, 73)
    shape = ph.CollisionShape(key, "sphere", (901,), radius=1 / 32)
    local = np.zeros((1, 3))
    binding = ph.ResidentFloatingAffineTreeBinding(
        key, WORLD, 16, local, topology=checkpoint.topology, graph_token=item["command"].graph_token
    )
    before = batch.snapshot()
    with batch.prepare_collision(ph.CollisionModel((shape,)), (binding,)) as plan:
        result = plan.query()
        assert plan.require_current(result) is result
        actual = batch.tree_kinematics()[WORLD].link_position_m[16]
        bounds = result.bounds[result.shape_keys.index(key)]
        assert np.all(bounds[:3] <= actual - 1 / 32) and np.all(bounds[3:] >= actual + 1 / 32)
        independent = _reference().mechanics(
            item["data"], np.asarray(checkpoint.full_position), np.asarray(checkpoint.full_velocity)
        )["position"][16]
        _close(actual, independent)
        assert batch.snapshot() == before
        return dict(
            binding=binding,
            shape=shape,
            result=result,
            stored_center=actual,
            independent_center=independent,
        )
