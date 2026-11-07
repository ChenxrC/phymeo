"""F3 actual contact windows: full authored-root tensor, independent dynamics."""

import gzip
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

import phymeo as ph
from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.runtime._floating_robot_participant import FloatingRobotCommand
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.runtime.robot_scene import RobotCommand

FIXTURES = Path(__file__).parent / "fixtures" / "urdf"
CYCLE = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])
CENTER = np.array([1 / 8, -1 / 16, 1 / 32])
BODY_I = np.array([[3 / 8, -1 / 8, 0], [-1 / 8, 3 / 8, 0], [0, 0, 1 / 2]])
ROBOT, BALL, WORLD = 17, 91, 23


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return [{"key": _plain(k), "value": _plain(v)} for k, v in value.items()]
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [_plain(v) for v in sorted(value, key=repr)]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(type(value).__name__)


def _skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _rot(q):
    k = _skew(q[:3])
    return np.eye(3) + 2 * q[3] * k + 2 * k @ k


def build_case(directory, kind, *, transformed=False, reverse=False, gap=1 / 1024):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = "floating_fixed_composite.urdf" if kind == "root" else "floating_mimic_jaws.urdf"
    doc = ET.parse(FIXTURES / name)
    if kind == "root":
        doc.find(".//joint/origin").set("xyz", ".25 .125 0")

        def loader(path, **kw):
            return ph.load_floating_urdf(path, collapse_fixed_joints=True, **kw)

        center = np.array([0.25, 0.125, 0.0])
        count = 0
    else:
        inertia = doc.find(".//link[@name='base']/inertial")
        ET.SubElement(inertia, "origin", xyz=".125 -.0625 .03125")
        inertia.find("inertia").attrib.update(ixx=".375", iyy=".375", izz=".5", ixy="-.125")
        loader = ph.load_floating_mimic_urdf
        center = np.array([0.254, 0.0, 0.0])
        count = 1
    path = directory / name
    doc.write(path, encoding="unicode")
    rotation = CYCLE if transformed else np.eye(3)
    shift = np.array([0.5, -0.25, 0.75]) if transformed else np.zeros(3)
    quat = [0.5] * 4 if transformed else [0, 0, 0, 1]
    state = ph.FloatingRobotState(
        np.r_[shift, quat, np.zeros(count)], np.r_[rotation @ [0, 0.25, 0], np.zeros(3 + count)]
    )
    builder = ContactSceneBuilder(gravity=(0, 0, 0))

    def add_robot():
        builder.add_floating_urdf(
            ROBOT,
            path,
            robot_state=state,
            kp=np.zeros(count),
            kd=np.zeros(count),
            loader=loader,
            world=WORLD,
        )

    def add_ball():
        builder.add_sphere(
            BALL,
            1 / 16,
            mass=1,
            position=shift + rotation @ (center + [0, 1 / 8 + gap, 0]),
            world=WORLD,
        )

    for add in (add_ball, add_robot) if reverse else (add_robot, add_ball):
        add()
    builder.pair(ROBOT, BALL, RobotContactMaterial(2048, 0, 0), world=WORLD)
    command = FloatingRobotCommand(None if not count else RobotCommand([0], feedforward=[0]))
    return builder.build(backend="host"), command


def independent(kind, q, v):
    rotation = _rot(q[3:7])
    if kind == "root":
        offset = np.array([0.25, 0.125, 0.0])
        center = offset / 3
        tensor = np.diag([0.3, 0.4, 0.5]) + (2 / 3) * (
            offset @ offset * np.eye(3) - np.outer(offset, offset)
        )
        masses = [3.0]
        local = [center]
        inertias = [tensor]
    else:
        center = CENTER
        local = [center, [-0.25 + q[7], 0, 0], [0.25 + q[8], 0, 0]]
        masses = [2.0, 1.0, 2.0]
        inertias = [BODY_I, np.diag([1 / 16, 5 / 64, 3 / 32]), np.diag([3 / 32, 7 / 64, 1 / 8])]
    n = len(v)
    mass = np.zeros((n, n))
    bias = np.zeros(n)
    momentum = np.zeros(3)
    angular = np.zeros(3)
    energy = 0.0
    positions = []
    maps = []
    rotations = []
    for i, (m, point, tensor) in enumerate(zip(masses, local, inertias, strict=True)):
        arm = rotation @ (np.asarray(point) - center)
        j = np.zeros((3, n))
        j[:, :3] = np.eye(3)
        j[:, 3:6] = -_skew(arm)
        a = np.zeros((3, n))
        a[:, 3:6] = np.eye(3)
        relative_v = np.zeros(3)
        if i:
            j[:, 5 + i] = rotation[:, 0]
            relative_v = rotation[:, 0] * v[5 + i]
        world_i = rotation @ tensor @ rotation.T
        omega = v[3:6]
        vel = j @ v
        centrifugal = np.cross(omega, np.cross(omega, arm)) + 2 * np.cross(omega, relative_v)
        mass += m * j.T @ j + a.T @ world_i @ a
        bias += j.T @ (m * centrifugal) + a.T @ np.cross(omega, world_i @ omega)
        position = q[:3] + rotation @ point
        p = m * vel
        spin = world_i @ omega
        momentum += p
        angular += np.cross(position, p) + spin
        energy += (p @ vel + spin @ omega) / 2
        positions.append(position)
        maps.append((j, a))
        rotations.append(rotation)
    return mass, bias, np.asarray(positions), maps, momentum, angular, float(energy)


def run_window(directory, kind, *, h=1 / 2048, transformed=False, reverse=False):
    scene, command = build_case(directory, kind, transformed=transformed, reverse=reverse)
    frames = []
    positive = []
    begin = False
    ended = False
    try:
        with scene:
            ballrow = scene.rigid_rows[WORLD, BALL]
            first = scene.observe()
            row = first.robots[WORLD, ROBOT]
            values = independent(kind, row.full_q, row.full_v)
            initial_p = values[4] + first.rigid.velocity[ballrow]
            initial_l = values[5] + np.cross(
                first.rigid.position[ballrow], first.rigid.velocity[ballrow]
            )
            maximum_p = maximum_l = 0.0
            for step in range(1, round(0.25 / h) + 1):
                reading = scene.step(h, actions={ROBOT: command})
                row = reading.robots[WORLD, ROBOT]
                entryq = (
                    np.r_[row.entry_q[:7], [row.entry_q[7], -row.entry_q[7] + 0.004]]
                    if kind == "mimic"
                    else row.entry_q
                )
                entryv = (
                    np.r_[row.entry_v[:6], [row.entry_v[6], -row.entry_v[6]]]
                    if kind == "mimic"
                    else row.entry_v
                )
                mass, bias, positions, maps, *_ = independent(kind, entryq, entryv)
                wrenches = row.link_wrenches_world
                external = sum(
                    (j.T @ w[:3] + a.T @ w[3:] for (j, a), w in zip(maps, wrenches, strict=True)),
                    np.zeros(len(entryv)),
                )
                constraint = (
                    np.zeros_like(external)
                    if row.mimic_loads is None
                    else row.mimic_loads.constraint_effort_full
                )
                np.testing.assert_allclose(
                    mass @ row.acceleration_full + bias, external + constraint, atol=2e-9, rtol=0
                )
                force = reading.loads.inputs.rigid.force[ballrow]
                torque = reading.loads.inputs.rigid.torque[ballrow]
                np.testing.assert_allclose(
                    wrenches[:, :3].sum(axis=0) + force, 0, atol=1e-10, rtol=0
                )
                netmoment = np.cross(positions, wrenches[:, :3]).sum(axis=0) + wrenches[:, 3:].sum(
                    axis=0
                )
                netmoment += np.cross(reading.entry_rigid.position[ballrow], force) + torque
                np.testing.assert_allclose(netmoment, 0, atol=1e-10, rtol=0)
                end = independent(kind, row.full_q, row.full_v)
                p = end[4] + reading.rigid.velocity[ballrow]
                angular_momentum = end[5] + np.cross(
                    reading.rigid.position[ballrow], reading.rigid.velocity[ballrow]
                )
                angular_momentum += (
                    (2 / 5) * (1 / 16) ** 2 * reading.rigid.angular_velocity[ballrow]
                )
                maximum_p = max(maximum_p, float(np.linalg.norm(p - initial_p)))
                maximum_l = max(maximum_l, float(np.linalg.norm(angular_momentum - initial_l)))
                if any(load.force_n > 0 for load in reading.loads.contacts):
                    positive.append(reading.load_time_s)
                kinds = [event.kind for event in scene.contact_events.events]
                begin = begin or "begin" in kinds
                ended = ended or "end" in kinds
                frames.append(
                    dict(
                        reading=_plain(reading),
                        events=_plain(scene.contact_events),
                        geometry=_plain(scene.geometry()),
                        independent_mass=mass.tolist(),
                        independent_bias=bias.tolist(),
                        linear_momentum=p.tolist(),
                        angular_momentum=angular_momentum.tolist(),
                        robot_kinetic_energy_j=end[6],
                        force=force.tolist(),
                        torque=torque.tolist(),
                    )
                )
            final_unloaded = not any(load.force_n > 0 for load in reading.loads.contacts)
    except BaseException:
        with gzip.open(Path(directory) / "partial.json.gz", "wt", encoding="utf-8") as file:
            json.dump(dict(kind=kind, h=h, frames=frames), file, allow_nan=False)
        raise
    result = dict(
        kind=kind,
        h=h,
        transformed=transformed,
        reverse=reverse,
        frames=frames,
        max_linear_residual=maximum_p,
        max_angular_residual=maximum_l,
        first_load=None if not positive else positive[0],
        last_load=None if not positive else positive[-1],
        begin=begin,
        end=ended,
        final_unloaded=final_unloaded,
    )
    with gzip.open(Path(directory) / "full.json.gz", "wt", encoding="utf-8") as file:
        json.dump(result, file, allow_nan=False)
    assert len(frames) == round(0.25 / h)
    assert positive and 0 < positive[0] < positive[-1] < 0.24
    assert begin and ended and final_unloaded
    assert maximum_p <= 5e-4 and maximum_l <= 5e-4
    return result


@pytest.fixture(scope="module")
def windows(tmp_path_factory):
    return {
        (kind, variant): run_window(
            tmp_path_factory.mktemp(kind + variant),
            kind,
            h=1 / 4096 if variant == "half" else 1 / 2048,
            transformed=variant == "transform",
            reverse=variant == "transform",
        )
        for kind in ("root", "mimic")
        for variant in ("base", "half", "transform")
    }


def _robot(frame):
    return next(
        item["value"] for item in frame["reading"]["robots"] if item["key"] == [WORLD, ROBOT]
    )


@pytest.mark.parametrize("kind", ["root", "mimic"])
def test_real_full_tensor_load_unload_h_and_covariance(windows, kind):
    base = windows[kind, "base"]
    half = windows[kind, "half"]
    moved = windows[kind, "transform"]
    for a, b in zip(base["frames"], moved["frames"], strict=True):
        ra, rb = _robot(a), _robot(b)
        np.testing.assert_allclose(
            CYCLE @ np.asarray(ra["q"][:3]) + [0.5, -0.25, 0.75], rb["q"][:3], atol=2e-10, rtol=0
        )
        np.testing.assert_allclose(
            CYCLE @ _rot(np.asarray(ra["q"][3:7])),
            _rot(np.asarray(rb["q"][3:7])),
            atol=2e-10,
            rtol=0,
        )
        for sl in (slice(0, 3), slice(3, 6)):
            np.testing.assert_allclose(
                CYCLE @ np.asarray(ra["v"][sl]), rb["v"][sl], atol=2e-10, rtol=0
            )
        np.testing.assert_allclose(ra["q"][7:], rb["q"][7:], atol=2e-10, rtol=0)
        np.testing.assert_allclose(CYCLE @ np.asarray(a["force"]), b["force"], atol=2e-9, rtol=0)
    a, b = _robot(base["frames"][-1]), _robot(half["frames"][-1])
    assert np.max(np.abs(np.asarray(a["q"]) - b["q"])) < 5e-4
    assert np.max(np.abs(np.asarray(a["v"]) - b["v"])) < 5e-3


@pytest.mark.parametrize("kind", ["root", "mimic"])
def test_late_event_failure_retries_same_full_inertia_state(tmp_path, monkeypatch, kind):
    scene, command = build_case(tmp_path / "old", kind, gap=-1 / 1024)
    fresh, freshcommand = build_case(tmp_path / "fresh", kind, gap=-1 / 1024)
    with scene, fresh:
        old = _plain(scene.observe_scene())
        events = scene.contact_events

        def fail(*args, **kwargs):
            raise RuntimeError("F3 late event publication")

        with monkeypatch.context() as patch:
            patch.setattr(scene._event_adapter, "stage", fail)
            with pytest.raises(RuntimeError, match="F3 late"):
                scene.step(1 / 2048, actions={ROBOT: command})
        assert _plain(scene.observe_scene()) == old and scene.contact_events is events
        got = scene.step(1 / 2048, actions={ROBOT: command})
        expected = fresh.step(1 / 2048, actions={ROBOT: freshcommand})
        np.testing.assert_array_equal(
            got.robots[WORLD, ROBOT].full_q, expected.robots[WORLD, ROBOT].full_q
        )
        np.testing.assert_array_equal(
            got.robots[WORLD, ROBOT].full_v, expected.robots[WORLD, ROBOT].full_v
        )
        np.testing.assert_array_equal(got.rigid.velocity, expected.rigid.velocity)
        assert scene.contact_events == fresh.contact_events
