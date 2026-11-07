"""Pure tangent candidates for real single/multiple HOST robot scene owners.

The scene owns sampling, bounded controls, integration and atomic publication.
This adapter preserves the original normal law and every unselected load. It
replaces the selected smooth traction once using the existing coupled tangent
kernel, then audits actual entry-material/application motion before publication.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np

from ..constraints.robot_contact import RobotContactMaterial
from ..core.coupled import CoupledInputs
from ..core.rigid import RigidBodyInputs
from ..core.validation import finite_real
from ..solvers.collision_loads import SurfaceLoads
from ..solvers.tangent_history import TangentHistory, _motion_reference
from ._coupled_tangent_step import finish_tangent_step, prepare_tangent_step
from ._scene_coupled_tangent import SceneTangentConfig
from ._scene_tangent_ports import _array, build_ports, owner_poses, relative_response
from .material_stations import MaterialStationRegistry
from .robot_payload_scene import _frozen, _readonly


def configure(
    scene,
    registry,
    *,
    stiffness_n_per_m,
    motion_tolerance_m=1e-6,
    energy_tolerance_j=1e-8,
    max_frame_turn_rad=0.25,
    motion_reference="material",
):
    """Configure once under the scene lock; no sampled history is advanced here."""
    if not isinstance(registry, MaterialStationRegistry):
        raise TypeError("expected MaterialStationRegistry")
    if scene._time_s != 0 or getattr(scene, "_tangent", None) is not None:
        raise ValueError("configure tangent mode once before advancing the scene")
    values = tuple(
        finite_real(value, name)
        for value, name in (
            (stiffness_n_per_m, "stiffness"),
            (motion_tolerance_m, "motion tolerance"),
            (energy_tolerance_j, "energy tolerance"),
        )
    )
    if min(values) <= 0:
        raise ValueError("tangent stiffness and actual-motion/energy tolerances must be positive")
    shapes = {shape.key: shape for shape in scene._model.shapes}
    excluded = set(scene._model.excluded)
    for first, second in registry.pairs:
        if first not in shapes or second not in shapes or (first, second) in excluded:
            raise ValueError("station registry references an unknown or excluded scene shape pair")
        if not shapes[first].rigid or not shapes[second].rigid:
            raise ValueError("robot tangent mode requires rigid material shapes")
        law = scene._event_adapter._pair_law(first, second)
        if not isinstance(law, RobotContactMaterial):
            raise ValueError("robot tangent mode preserves RobotContactMaterial only")
    store = TangentHistory(128, max_frame_turn_rad=max_frame_turn_rad)
    config = SceneTangentConfig(registry, *values, store, _motion_reference(motion_reference))
    state = store.initial_state()
    scene._tangent, scene._tangent_state, scene._tangent_receipt = config, state, None


def _site_signature(load, site):
    """Exact current-sample locator, never a persistent history identity."""
    return (
        load.first,
        load.second,
        tuple(load.normal),
        load.gap_m,
        tuple(site.material.first_point),
        tuple(site.material.second_point),
        tuple(site.material.first_weights),
        tuple(site.material.second_weights),
        tuple(site.application_point_first_m),
        tuple(site.application_point_second_m),
        site.force_n,
        site.fraction,
        tuple(site.force_on_second_n),
    )


def _detail_signature(row):
    return (
        row["first"],
        row["second"],
        tuple(row["normal_world"]),
        row["gap_m"],
        tuple(row["first_surface_m"]),
        tuple(row["second_surface_m"]),
        tuple(row["first_weights"]),
        tuple(row["second_weights"]),
        tuple(row["application_point_first_m"]),
        tuple(row["application_point_second_m"]),
        row["normal_load_N"],
        row["fraction"],
        tuple(row["force_on_second_N"]),
    )


@dataclass(frozen=True)
class _Change:
    first_owner: object
    second_owner: object
    first_center: np.ndarray
    second_center: np.ndarray
    old: object
    new: object


def _details(rows, changes, *, require_all):
    output, matched = [], set()
    for original in rows:
        if not isinstance(original, Mapping):
            raise ValueError("contact details must retain their actual site mapping")
        key = _detail_signature(original)
        change = changes.get(key)
        if change is None:
            output.append(_frozen(dict(original)))
            continue
        if key in matched:
            raise ValueError("ambiguous duplicate selected contact detail")
        matched.add(key)
        for name, expected in (
            ("first_com_world_m", change.first_center),
            ("second_com_world_m", change.second_center),
        ):
            if not np.allclose(original[name], expected, atol=1e-12, rtol=1e-12):
                raise ValueError("contact detail COM differs from its physical material owner")
        site = change.new
        force = np.asarray(site.force_on_second_n)
        row = dict(original)
        row.update(
            force_on_first_N=tuple(-force),
            force_on_second_N=tuple(force),
            torque_on_first_Nm=tuple(
                np.cross(np.asarray(site.application_point_first_m) - change.first_center, -force)
            ),
            torque_on_second_Nm=tuple(
                np.cross(np.asarray(site.application_point_second_m) - change.second_center, force)
            ),
            power_W=site.power_w,
        )
        output.append(_frozen(row))
    if require_all and matched != set(changes):
        raise ValueError("contact details omit a selected material site")
    return tuple(output)


def _validate_measured(ports, measured):
    robots = {address for address, port in ports.items() if port.robot}
    if not isinstance(measured, Mapping) or set(measured) != robots:
        raise ValueError("measured contacts must cover exactly the prepared robot owners")
    for address in sorted(robots):
        value, port = measured[address], ports[address]
        contact = _array(value.joint_effort_full, (port.full_count,), "full contact effort")
        wrenches = _array(value.link_wrenches_world, (port.body_count, 6), "physical COM wrenches")
        root = _array(value.fixed_root_contact_wrench, (6,), "direct fixed root contact wrench")
        projected = np.einsum("nki,nk->i", port.linear, wrenches[:, :3]) + np.einsum(
            "nki,nk->i",
            port.angular,
            wrenches[:, 3:],
        )
        if not np.allclose(projected, contact, atol=1e-9, rtol=1e-10):
            raise ValueError("full contact effort differs from the actual COM wrenches")
        if port.full_count == port.body_count + 5 and np.any(root):
            raise ValueError("moving floating root cannot also carry a fixed root wrench")


def _rewrite(scene, ports, frames, events, measured, loads, details, replacements):
    """Scatter only current site force differences; preserve every other original row."""
    _validate_measured(ports, measured)
    owners = {
        (record.identity.first, record.identity.second): (record.first_owner, record.second_owner)
        for record in events.active
    }
    joint_delta = {
        address: np.zeros(port.full_count) for address, port in ports.items() if port.robot
    }
    link_delta = {
        address: np.zeros((port.body_count, 6)) for address, port in ports.items() if port.robot
    }
    root_delta = {address: np.zeros(6) for address, port in ports.items() if port.robot}
    rigid_force = np.zeros_like(loads.inputs.rigid.force)
    rigid_torque = np.zeros_like(loads.inputs.rigid.torque)
    changes, rewritten, changed_loads = {}, [], []
    used = set()
    for load_index, load in enumerate(loads.contacts):
        sites, changed = [], False
        local_force, local_torque = np.zeros_like(rigid_force), np.zeros_like(rigid_torque)
        for site_index, site in enumerate(load.sites):
            location = load_index, site_index
            if location not in replacements:
                sites.append(site)
                continue
            used.add(location)
            first, second = owners[load.first, load.second]
            force = _array(replacements[location], (3,), "selected site force")
            difference = force - np.asarray(site.force_on_second_n)
            relative = np.zeros(3)
            centers = []
            for owner, point, sign in (
                (first, site.application_point_first_m, -1.0),
                (second, site.application_point_second_m, 1.0),
            ):
                address = owner.world, owner.body
                port = ports[address]
                jacobian = port.jacobian(owner, point)
                center = port.center(owner, frames)
                centers.append(center)
                relative += sign * (jacobian @ port.full_velocity)
                signed = sign * difference
                moment = np.cross(np.asarray(point) - center, signed)
                if port.robot:
                    joint_delta[address] += jacobian.T @ signed
                    index = port.index(owner)
                    if index == -1:
                        root_delta[address] += np.r_[signed, moment]
                    else:
                        link_delta[address][index] += np.r_[signed, moment]
                else:
                    row = scene.rigid_rows[address]
                    rigid_force[row] += signed
                    rigid_torque[row] += moment
                    local_force[row] += signed
                    local_torque[row] += moment
            updated = replace(site, force_on_second_n=tuple(force), power_w=float(force @ relative))
            signature = _site_signature(load, site)
            if signature in changes:
                raise ValueError("ambiguous duplicate selected site in current load components")
            changes[signature] = _Change(first, second, centers[0], centers[1], site, updated)
            sites.append(updated)
            changed = True
        revised = (
            replace(load, sites=tuple(sites), power_w=sum(site.power_w for site in sites))
            if changed
            else load
        )
        rewritten.append(revised)
        if changed:
            changed_loads.append((load, revised, local_force, local_torque))
    if used != set(replacements):
        raise ValueError("selected site index differs from its current load component")
    revised_loads = SurfaceLoads(
        CoupledInputs(
            loads.inputs.particles,
            RigidBodyInputs(
                loads.inputs.rigid.force + rigid_force,
                loads.inputs.rigid.torque + rigid_torque,
            ),
        ),
        tuple(rewritten),
    )
    revised_measured = {}
    for address, value in measured.items():
        local_contacts = []
        local_force = np.zeros_like(value.loads.inputs.rigid.force)
        local_torque = np.zeros_like(value.loads.inputs.rigid.torque)
        for load in value.loads.contacts:
            matches = [row for row in changed_loads if row[0] == load]
            if len(matches) > 1:
                raise ValueError("current owner load cannot identify a unique changed component")
            if matches:
                _, revised, force, torque = matches[0]
                local_contacts.append(revised)
                local_force += force
                local_torque += torque
            else:
                local_contacts.append(load)
        local_loads = SurfaceLoads(
            CoupledInputs(
                value.loads.inputs.particles,
                RigidBodyInputs(
                    value.loads.inputs.rigid.force + local_force,
                    value.loads.inputs.rigid.torque + local_torque,
                ),
            ),
            tuple(local_contacts),
        )
        revised_measured[address] = replace(
            value,
            loads=local_loads,
            joint_effort_full=_readonly(value.joint_effort_full + joint_delta[address]),
            link_wrenches_world=_readonly(value.link_wrenches_world + link_delta[address]),
            fixed_root_contact_wrench=_readonly(
                value.fixed_root_contact_wrench + root_delta[address]
            ),
            details=_details(value.details, changes, require_all=False),
        )
    result = MappingProxyType(revised_measured)
    _validate_measured(ports, result)
    return result, revised_loads, _details(details, changes, require_all=True)


@dataclass(frozen=True, eq=False)
class SceneTangentAttempt:
    """One unpublished candidate, bound to the scene's exact accepted history base."""

    measured: Mapping
    loads: SurfaceLoads
    details: tuple
    _scene: object
    _config: SceneTangentConfig
    _history: object
    _selection: object
    _trial: object
    _sample_time_s: float

    def finish(self, candidate_body, end_geometry_by_address):
        if (
            self._scene._tangent is not self._config
            or self._scene._tangent_state is not self._history
            or self._scene._time_s != self._sample_time_s
        ):
            raise ValueError("stale tangent attempt or changed scene history base")
        frames = owner_poses(self._scene, candidate_body, end_geometry_by_address)
        material, application = self._selection.measure_motion(frames)
        return finish_tangent_step(
            self._config.store,
            self._history,
            self._trial,
            material,
            application,
            motion_tolerance_m=self._config.motion_tolerance_m,
            energy_tolerance_j=self._config.energy_tolerance_j,
        )


def prepare(scene, prepared, measured, loads, details, external, h):
    """Consume one real entry sample and return loads for each original integrator."""
    config = scene._tangent
    if not isinstance(config, SceneTangentConfig) or not isinstance(loads, SurfaceLoads):
        raise TypeError("expected configured scene and original SurfaceLoads")
    h = finite_real(h, "tangent physical step")
    if h <= 0:
        raise ValueError("tangent physical step must be positive")
    history, sample_time = scene._tangent_state, scene._time_s
    geometries = {address: value.geometry for address, value in prepared.items()}
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            ports = build_ports(scene, prepared, external)
            frames = owner_poses(scene, scene._body, geometries)
            events = scene._event_adapter.stage(
                scene._contact_events,
                loads,
                geo=geometries,
                rigid=scene._body,
                sample_time_s=sample_time,
                commit_time_s=sample_time + h,
            )
            selection = config.registry.bind(events, frames, sample_time_s=sample_time)
            locations, friction = [], []
            for binding in selection.bindings:
                indices = [
                    i for i, load in enumerate(loads.contacts) if load == binding.component.load
                ]
                if len(indices) != 1:
                    raise ValueError("material station requires a unique current load component")
                location = indices[0], binding.site_index
                if location in locations:
                    raise ValueError("duplicate selected current site")
                locations.append(location)
                if not isinstance(binding.component.law, RobotContactMaterial):
                    raise ValueError("selected robot material must retain its original law")
                friction.append(binding.component.law.friction)
            selected = set(config.registry.pairs)
            normal = {
                (i, j): site.force_n * np.asarray(load.normal)
                for i, load in enumerate(loads.contacts)
                if (load.first, load.second) in selected
                for j, site in enumerate(load.sites)
                if site.force_n > 0
            }
            if set(locations) != set(normal):
                raise ValueError("every selected positive normal site needs one material station")
            normal_measured, normal_loads, normal_details = _rewrite(
                scene,
                ports,
                frames,
                events,
                measured,
                loads,
                details,
                normal,
            )
            world, free = relative_response(
                ports, selection.bindings, normal_measured, normal_loads, scene.rigid_rows, h
            )
            trial = prepare_tangent_step(
                config.store,
                history,
                selection.samples,
                world,
                free,
                sample_time_s=sample_time,
                dt_s=h,
                stiffness_n_per_m=np.full(len(locations), config.stiffness_n_per_m),
                friction=np.asarray(friction, dtype=float),
                motion_reference=config.motion_reference,
            )
            replacement = {
                location: normal[location] + force
                for location, force in zip(locations, trial.force_on_second_n, strict=True)
            }
            revised, revised_loads, revised_details = _rewrite(
                scene,
                ports,
                frames,
                events,
                normal_measured,
                normal_loads,
                normal_details,
                replacement,
            )
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise ValueError(f"scene tangent response is not representable: {exc}") from exc
    return SceneTangentAttempt(
        revised,
        revised_loads,
        revised_details,
        scene,
        config,
        history,
        selection,
        trial,
        sample_time,
    )
