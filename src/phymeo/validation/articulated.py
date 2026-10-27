"""Independent SI audits for the frozen P5 constant-mass Y slider scenes.

These checks intentionally do not call the production contact or robot dynamics
operators. They do not claim general nonlinear articulated energy conservation.
"""

import numpy as np

from ..experiments.rigid_cloth import SceneRunFailure

EVENT_SHAPES = {
    "substep": (),
    "domain": (),
    "phase": (),
    "node": (),
    "sphere": (),
    "link": (),
    "world_point": (3,),
    "robot_local_point": (3,),
    "normal": (3,),
    "node_load": (3,),
    "robot_load": (3,),
    "robot_generalized_load": (1,),
    "robot_configuration_before": (1,),
    "robot_velocity_before": (1,),
    "robot_configuration_delta": (1,),
    "robot_velocity_delta": (1,),
    "node_position_before": (3,),
    "node_velocity_before": (3,),
    "node_position_delta": (3,),
    "node_velocity_delta": (3,),
    "h": (),
}
EVENT_INTEGER = {"substep", "domain", "phase", "node", "sphere", "link"}
AUDIT_FIELDS = (
    "penetration",
    "closing_speed",
    "load_pair_error",
    "robot_balance_error",
    "fixed_position_error",
    "fixed_velocity_error",
    "response_error",
    "min_det_f",
    "min_singular_value",
    "max_singular_value",
)
PRESCRIBED_AUDIT_FIELDS = tuple(
    "driver_trajectory_error" if name == "robot_balance_error" else name for name in AUDIT_FIELDS
)


def compact_event(event, domain, substep):
    row = {
        name: event[name] for name in EVENT_SHAPES if name in event and not name.startswith("node_")
    }
    row.update(substep=substep, domain=domain, phase=0 if event["phase"] == "position" else 1)
    for name in ("node_load",):
        row[name] = event[name]
    for target, source in (
        ("node_position_before", "node_configuration_before"),
        ("node_position_delta", "node_configuration_delta"),
        ("node_velocity_before", "node_velocity_before"),
        ("node_velocity_delta", "node_velocity_delta"),
    ):
        row[target] = event[source][event["node"]].copy()
    return row


def event_arrays(events):
    return {
        "event_" + name: np.asarray(
            [event[name] for event in events],
            dtype=np.int64 if name in EVENT_INTEGER else np.float64,
        ).reshape((-1, *shape))
        for name, shape in EVENT_SHAPES.items()
    }


def read_events(arrays, substep):
    return [
        {name: arrays["event_" + name][index] for name in EVENT_SHAPES}
        for index in np.flatnonzero(arrays["event_substep"] == substep)
    ]


def _max_error(left, right):
    return float(np.max(np.abs(np.asarray(left) - right), initial=0))


def audit_slider_substep(
    fixture, before, after, events, predicted_q, predicted_v, *, h, prescribed=None
):
    return _audit_slider_substep(
        fixture,
        before,
        after,
        events,
        predicted_q,
        predicted_v,
        h=h,
        prescribed=prescribed,
        limits=(2e-7, 2e-7, 1e-12, 1e-10, 1e-10),
        label="P5",
    )


def _audit_slider_substep(
    fixture,
    before,
    after,
    events,
    predicted_q,
    predicted_v,
    *,
    h,
    prescribed=None,
    limits,
    label,
):
    """Audit dynamics, or an explicitly supplied pair of external driver samples.

    The prescribed case has no robot domain and checks trajectory identity instead
    of inventing a momentum balance for the externally driven boundary.
    """
    names = fixture.node_names
    model = fixture.program.participants["robot"].model
    attachments = fixture.program.attachments
    mass = float(model.mass[0])
    before_robot, after_robot = (
        (before.domains["robot"], after.domains["robot"]) if prescribed is None else prescribed
    )
    expected_v = before_robot.v + h * fixture.inputs.domains["robot"].effort / mass
    expected_q = before_robot.q + h * expected_v
    response_error = max(_max_error(predicted_q, expected_q), _max_error(predicted_v, expected_v))
    position_generalized, velocity_generalized, actual_q_delta = 0.0, 0.0, 0.0
    pair_error, contact = 0.0, {name: 0 for name in names}
    for event in events:
        if set(event) != set(EVENT_SHAPES):
            raise SceneRunFailure("incomplete contact event")
        if any(
            np.asarray(event[name]).shape != shape or not np.isfinite(event[name]).all()
            for name, shape in EVENT_SHAPES.items()
        ):
            raise SceneRunFailure("invalid contact event shape or finite data")
        domain, phase = int(event["domain"]), int(event["phase"])
        if domain not in range(len(names)) or phase not in (0, 1) or event["h"] != h:
            raise SceneRunFailure("invalid contact event domain, phase or time")
        name = names[domain]
        node_model = fixture.program.participants[name].model
        node, sphere = int(event["node"]), int(event["sphere"])
        if not 0 <= node < node_model.particle_count or not 0 <= sphere < len(attachments.radii):
            raise SceneRunFailure("invalid recorded contact index")
        if event["link"] != 0 or not any(
            c.domain == name and np.any((c.node_indices == node) & (c.sphere_indices == sphere))
            for c in fixture.program.contacts
        ):
            raise SceneRunFailure("recorded event is not a bound contact pair")
        origin = model.joint_position[0].copy()
        origin[1] += event["robot_configuration_before"][0]
        center = origin + attachments.local_centers[sphere]
        point, load = event["world_point"], event["node_load"]
        offset = point - center
        normal = offset / np.linalg.norm(offset)
        response_error = max(
            response_error,
            _max_error(point, event["node_position_before"]),
            _max_error(point, origin + event["robot_local_point"]),
            _max_error(normal, event["normal"]),
            _max_error(load, normal * (normal @ load)),
        )
        scale = max(1.0, float(np.max(np.abs(load))))
        pair_error = max(pair_error, _max_error(load, -event["robot_load"]) / scale)
        generalized = float(event["robot_load"][1])
        response_error = max(
            response_error, _max_error(event["robot_generalized_load"], [generalized])
        )
        if prescribed is not None:
            response_error = max(
                response_error,
                _max_error(event["robot_configuration_before"], after_robot.q),
                _max_error(event["robot_velocity_before"], after_robot.v),
            )
        expected_robot_delta = generalized / mass if prescribed is None else 0.0
        if np.any(load != 0):
            contact[name] = 1
        if phase == 0:
            position_generalized += generalized
            actual_q_delta += event["robot_configuration_delta"][0]
            response_error = max(
                response_error,
                _max_error(event["robot_configuration_delta"], [expected_robot_delta]),
                _max_error(event["robot_velocity_delta"], [0]),
                _max_error(event["node_position_delta"], node_model.inverse_mass[node] * load),
                _max_error(event["node_velocity_delta"], np.zeros(3)),
            )
        else:
            velocity_generalized += generalized
            response_error = max(
                response_error,
                _max_error(event["robot_velocity_delta"], [expected_robot_delta]),
                _max_error(event["robot_configuration_delta"], [0]),
                _max_error(event["node_velocity_delta"], node_model.inverse_mass[node] * load),
                _max_error(event["node_position_delta"], np.zeros(3)),
            )
    balance = abs(
        mass * float(after_robot.v[0] - predicted_v[0])
        - position_generalized / h
        - velocity_generalized
    )
    if prescribed is not None:
        balance = max(
            _max_error(after_robot.q, predicted_q), _max_error(after_robot.v, predicted_v)
        )
    response_error = max(response_error, _max_error(after_robot.q, expected_q + actual_q_delta))
    penetration, closing, fixed_position, fixed_velocity = 0.0, 0.0, 0.0, 0.0
    min_det, min_sv, max_sv = 1.0, 1.0, 1.0
    for name in names:
        node_model = fixture.program.participants[name].model
        native, initial = after.domains[name], fixture.initial.domains[name]
        if not np.isfinite(native.position).all() or not np.isfinite(native.velocity).all():
            raise SceneRunFailure("non-finite node state")
        fixed = node_model.inverse_mass == 0
        fixed_position = max(
            fixed_position, _max_error(native.position[fixed], initial.position[fixed])
        )
        fixed_velocity = max(fixed_velocity, _max_error(native.velocity[fixed], 0))
        if name == "soft":
            cell = native.position[node_model.tetrahedra]
            edges = np.stack(
                (cell[:, 1] - cell[:, 0], cell[:, 2] - cell[:, 0], cell[:, 3] - cell[:, 0]), axis=2
            )
            gradient = edges @ node_model.rest_inverse
            singular = np.linalg.svd(gradient, compute_uv=False)
            min_det, min_sv, max_sv = (
                float(np.linalg.det(gradient).min()),
                float(singular.min()),
                float(singular.max()),
            )
    for pairs in fixture.program.contacts:
        nodes = after.domains[pairs.domain]
        center = model.joint_position[0] + attachments.local_centers[pairs.sphere_indices]
        center[:, 1] += after_robot.q[0]
        offset = nodes.position[pairs.node_indices] - center
        distance = np.linalg.norm(offset, axis=1)
        if np.any(distance == 0):
            raise SceneRunFailure("coincident center in accepted state")
        gap = distance - attachments.radii[pairs.sphere_indices]
        normal = offset / distance[:, None]
        speed = nodes.velocity[pairs.node_indices].copy()
        speed[:, 1] -= after_robot.v[0]
        normal_speed = np.sum(normal * speed, axis=1)
        active = gap <= fixture.settings.tolerance * fixture.settings.reference_length
        penetration = max(penetration, float(np.max(-gap, initial=0)))
        closing = max(closing, float(np.max(-normal_speed[active], initial=0)))
    values = dict(
        zip(
            AUDIT_FIELDS if prescribed is None else PRESCRIBED_AUDIT_FIELDS,
            (
                penetration,
                closing,
                pair_error,
                balance,
                fixed_position,
                fixed_velocity,
                response_error,
                min_det,
                min_sv,
                max_sv,
            ),
            strict=True,
        )
    )
    if (
        not all(np.isfinite(value) for value in values.values())
        or not np.isfinite(after_robot.q).all()
        or not np.isfinite(after_robot.v).all()
    ):
        raise SceneRunFailure("non-finite physical audit")
    if (
        penetration > limits[0]
        or closing > limits[1]
        or pair_error > limits[2]
        or balance > limits[3]
        or response_error > limits[4]
    ):
        raise SceneRunFailure(f"independent {label} contact/load audit failed: {values}")
    if fixed_position != 0 or fixed_velocity != 0 or min_det <= 0.2 or min_sv < 0.6 or max_sv > 1.4:
        raise SceneRunFailure(f"independent {label} fixed/material audit failed: {values}")
    return values, contact
