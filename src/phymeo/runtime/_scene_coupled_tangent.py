"""Opt-in HOST scene bridge: one sampled load, one integration, one publication.

Explicit material registration is required. Default scene friction is untouched.
The tangent balance is a constitutive work audit, not a total scene energy claim.
"""

from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np

from ..core.coupled import CoupledInputs
from ..core.rigid import RigidBodyInputs
from ..dynamics.rotation import quaternion_matrix
from ..solvers.collision_loads import SurfaceLoads
from ..solvers.tangent_history import TangentHistory, _motion_reference
from ._coupled_tangent_step import finish_tangent_step, prepare_tangent_step
from .contact_events import (
    ContactComponent,
    ContactOwner,
    SceneContactEvents,
    SceneContactRecord,
    _ordinary_owner,
)
from .contact_history import ContactIdentity, _material
from .material_stations import MaterialStationRegistry, OwnerPose


@dataclass(frozen=True)
class SceneTangentConfig:
    registry: MaterialStationRegistry
    stiffness_n_per_m: float
    motion_tolerance_m: float
    energy_tolerance_j: float
    store: TangentHistory
    motion_reference: str = "material"


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
    if not isinstance(registry, MaterialStationRegistry):
        raise TypeError("expected MaterialStationRegistry")
    if scene._time_s != 0 or getattr(scene, "_tangent", None) is not None:
        raise ValueError("configure tangent mode once before advancing the scene")
    values = np.asarray([stiffness_n_per_m, motion_tolerance_m, energy_tolerance_j], float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("stiffness and actual-motion/energy tolerances must be positive")
    if hasattr(scene, "_models"):
        if scene._backend != "host" or scene._models[0].particle_count:
            raise ValueError("scene tangent mode currently requires HOST rigid participants")
    elif scene._mechanism is not None or any(
        kind != "prismatic" for kind in scene._tree.model.joint_types
    ):
        raise ValueError("scene tangent mode currently requires ordinary prismatic joints")
    store = TangentHistory(128, max_frame_turn_rad=max_frame_turn_rad)
    config = SceneTangentConfig(
        registry, *map(float, values), store, _motion_reference(motion_reference)
    )
    state = store.initial_state()
    scene._tangent, scene._tangent_state, scene._tangent_receipt = config, state, None


def _skew(r):
    x, y, z = r
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def ordinary_events(scene, loads, h):
    rotations = quaternion_matrix(scene._state.rigid.orientation)
    records = []
    for load in loads.contacts:
        if load.force_n <= 0:
            continue
        first, second = load.first, load.second
        component = ContactComponent(
            load,
            _material(load, scene._state, scene._system.bindings, rotations),
            "normal_layer",
            scene._system.layers[first, second],
        )
        records.append(
            SceneContactRecord(
                ContactIdentity(first, second, 1),
                _ordinary_owner(first, scene._system.bindings),
                _ordinary_owner(second, scene._system.bindings),
                (component,),
            )
        )
    return SceneContactEvents(
        sample_time_s=scene._time_s, commit_time_s=scene._time_s + h, active=tuple(records)
    )


def owner_poses(scene, body, geo=None):
    rotations = quaternion_matrix(body.orientation)
    out = {}
    for (world, identifier), row in scene.rigid_rows.items():
        owner = ContactOwner(world, identifier, "rigid", "rigid_com")
        out[owner] = OwnerPose(owner, body.position[row], rotations[row])
    if geo is not None:
        for metadata in scene._event_adapter._frames.values():
            owner = metadata.owner
            if owner.kind == "robot_link":
                out[owner] = OwnerPose(
                    owner, geo.position[owner.link_index], geo.rotation[owner.link_index]
                )
            elif owner.kind == "robot_root":
                out[owner] = OwnerPose(
                    owner, scene._event_adapter._root_position, scene._event_adapter._root_rotation
                )
    return MappingProxyType(out)


class SceneTangentStep:
    """Private per-attempt staging; never mutates the scene's accepted state."""

    def __init__(
        self,
        scene,
        body,
        model,
        h,
        external,
        *,
        geo=None,
        linear=None,
        mass=None,
        effort=None,
        bias=None,
        measured=None,
    ):
        self.scene, self.body, self.h = scene, body, h
        self.config = scene._tangent
        self.geo, self.linear = geo, linear
        self.nq = 0 if geo is None else len(scene._arm.q)
        self.dofs = self.nq + 6 * model.body_count
        self.poses = owner_poses(scene, body, geo)
        self.mobility = np.zeros((self.dofs, self.dofs))
        self.velocity = np.zeros(self.dofs)
        self.acceleration = np.zeros(self.dofs)
        self.measured = measured
        if self.nq:
            self.mobility[: self.nq, : self.nq] = np.linalg.solve(mass, np.eye(self.nq))
            self.velocity[: self.nq] = scene._arm.v
            self.acceleration[: self.nq] = np.linalg.solve(
                mass, effort + measured.joint_effort_full - bias
            )
        rotations = quaternion_matrix(body.orientation)
        for row, inverse_mass in enumerate(model.inverse_mass):
            start = self.nq + 6 * row
            if inverse_mass == 0:
                continue
            R = rotations[row]
            inertia = model.inertia_diagonal[row]
            omega_body = R.T @ body.angular_velocity[row]
            inverse_inertia = (R * (1 / inertia)) @ R.T
            self.mobility[start : start + 3, start : start + 3] = np.eye(3) * inverse_mass
            self.mobility[start + 3 : start + 6, start + 3 : start + 6] = inverse_inertia
            self.velocity[start : start + 3] = body.velocity[row]
            self.velocity[start + 3 : start + 6] = body.angular_velocity[row]
            self.acceleration[start : start + 3] = (
                model.gravity + inverse_mass * external.rigid.force[row]
            )
            self.acceleration[start + 3 : start + 6] = inverse_inertia @ external.rigid.torque[row]
            self.acceleration[start + 3 : start + 6] -= R @ (
                np.cross(omega_body, inertia * omega_body) / inertia
            )

    def point_jacobian(self, owner, point):
        J = np.zeros((3, self.dofs))
        if owner.kind == "rigid":
            row = self.scene.rigid_rows[owner.world, owner.body]
            start = self.nq + row * 6
            J[:, start : start + 3] = np.eye(3)
            J[:, start + 3 : start + 6] = -_skew(np.asarray(point) - self.body.position[row])
        elif owner.kind == "robot_link":
            # Configuration gate restricts the entire ordinary tree to translations.
            J[:, : self.nq] = self.linear[owner.link_index]
        elif owner.kind != "robot_root":
            raise ValueError("unsupported tangent physical owner")
        return J

    def _rows(self, events):
        rows = {}
        for record in events.active:
            for component in record.components:
                load = component.load
                for index, site in enumerate(load.sites):
                    rows[(load.first, load.second, id(load), index)] = (
                        record,
                        component,
                        site,
                        self.point_jacobian(record.second_owner, site.application_point_second_m)
                        - self.point_jacobian(record.first_owner, site.application_point_first_m),
                    )
        return rows

    def _replace(self, loads, events, forces):
        """Replace exact current sites and accumulate the same bilateral load difference."""
        delta = np.zeros(self.dofs)
        contacts = []
        link_delta = None if not self.nq else np.zeros_like(self.measured.link_wrenches_world)
        root_delta = np.zeros(6)
        owners = {
            (r.identity.first, r.identity.second): (r.first_owner, r.second_owner)
            for r in events.active
        }
        for load_index, load in enumerate(loads.contacts):
            sites = []
            for index, site in enumerate(load.sites):
                key = load_index, index
                if key not in forces:
                    sites.append(site)
                    continue
                a, b = owners[load.first, load.second]
                force = np.asarray(forces[key])
                difference = force - site.force_on_second_n
                Ja = self.point_jacobian(a, site.application_point_first_m)
                Jb = self.point_jacobian(b, site.application_point_second_m)
                delta += (Jb - Ja).T @ difference
                power = float(force @ ((Jb - Ja) @ self.velocity))
                sites.append(replace(site, force_on_second_n=tuple(force), power_w=power))
                for owner, point, signed in (
                    (a, site.application_point_first_m, -difference),
                    (b, site.application_point_second_m, difference),
                ):
                    if owner.kind == "robot_link":
                        link = owner.link_index
                        # COM Jacobian frame and public wrenches use link COM.
                        com = (
                            self.geo.position[link]
                            + self.geo.rotation[link] @ self.scene._tree.model.com[link]
                        )
                        link_delta[link] += np.r_[signed, np.cross(np.asarray(point) - com, signed)]
                    elif owner.kind == "robot_root":
                        root_delta += np.r_[
                            signed,
                            np.cross(np.asarray(point) - self.poses[owner].position_m, signed),
                        ]
            contacts.append(
                replace(load, sites=tuple(sites), power_w=sum(s.power_w for s in sites))
            )
        rigid_delta = delta[self.nq :].reshape(-1, 6)
        inputs = CoupledInputs(
            loads.inputs.particles,
            RigidBodyInputs(
                loads.inputs.rigid.force + rigid_delta[:, :3],
                loads.inputs.rigid.torque + rigid_delta[:, 3:],
            ),
        )
        return SurfaceLoads(inputs, tuple(contacts)), delta, link_delta, root_delta

    def prepare(self, loads, events):
        selected = set(self.config.registry.pairs)
        normal = {}
        for i, load in enumerate(loads.contacts):
            if (load.first, load.second) in selected and load.force_n > 0:
                for j, site in enumerate(load.sites):
                    normal[i, j] = np.asarray(load.normal) * site.force_n
        normal_loads, removed, link_removed, root_removed = self._replace(loads, events, normal)
        # Binding uses the original real sample and its normal magnitudes; only its traction changes.
        selection = self.config.registry.bind(events, self.poses, sample_time_s=self.scene._time_s)
        matrices, locations, friction = [], [], []
        for binding in selection.bindings:
            matches = [i for i, c in enumerate(loads.contacts) if c == binding.component.load]
            if len(matches) != 1:
                raise ValueError("material station requires a unique current load component")
            location = matches[0], binding.site_index
            if location in locations:
                raise ValueError("duplicate selected load site")
            locations.append(location)
            matrices.append(
                self.point_jacobian(binding.second_owner, binding.application_point_second_m)
                - self.point_jacobian(binding.first_owner, binding.application_point_first_m)
            )
            friction.append(binding.component.law.friction)
        J = np.asarray(matrices).reshape(-1, self.dofs)
        internal = np.zeros(self.dofs)
        # Robot initial acceleration already includes the complete sampled robot load.
        internal[self.nq :] = np.c_[loads.inputs.rigid.force, loads.inputs.rigid.torque].ravel()
        acceleration = self.acceleration + self.mobility @ (internal + removed)
        free_velocity = self.velocity + self.h * acceleration
        trial = prepare_tangent_step(
            self.config.store,
            self.scene._tangent_state,
            selection.samples,
            J @ self.mobility @ J.T,
            (J @ free_velocity).reshape(-1, 3),
            sample_time_s=self.scene._time_s,
            dt_s=self.h,
            stiffness_n_per_m=np.full(len(selection.samples), self.config.stiffness_n_per_m),
            friction=np.asarray(friction),
            motion_reference=self.config.motion_reference,
        )
        replacement = {
            location: normal[location] + force
            for location, force in zip(locations, trial.force_on_second_n, strict=True)
        }
        revised, added, link_added, root_added = self._replace(normal_loads, events, replacement)
        self.selection, self.trial = selection, trial
        self.delta = removed + added
        self.link_delta = None if not self.nq else link_removed + link_added
        self.root_delta = root_removed + root_added
        return revised

    def finish(self, body, geo=None):
        material, application = self.selection.measure_motion(owner_poses(self.scene, body, geo))
        return finish_tangent_step(
            self.config.store,
            self.scene._tangent_state,
            self.trial,
            material,
            application,
            motion_tolerance_m=self.config.motion_tolerance_m,
            energy_tolerance_j=self.config.energy_tolerance_j,
        )


def ordinary_context(scene, external, h):
    return SceneTangentStep(scene, scene._state.rigid, scene._models[1], h, external)


def robot_prepare(scene, measured, geo, linear, mass, bias, effort, external, h):
    ctx = SceneTangentStep(
        scene,
        scene._body,
        scene._rigid_model,
        h,
        external,
        geo=geo,
        linear=linear,
        mass=mass,
        bias=bias,
        effort=effort,
        measured=measured,
    )
    events = scene._event_adapter.stage(
        scene._contact_events,
        measured.loads,
        geo=geo,
        rigid=scene._body,
        sample_time_s=scene._time_s,
        commit_time_s=scene._time_s + h,
    )
    loads = ctx.prepare(measured.loads, events)
    details = []
    for detail, contact in zip(measured.details, loads.contacts, strict=True):
        row = dict(detail)
        if len(contact.sites) != 1:
            raise ValueError("robot details require one source site per component")
        site = contact.sites[0]
        force = np.asarray(site.force_on_second_n)
        row.update(
            force_on_first_N=tuple(-force),
            force_on_second_N=tuple(force),
            torque_on_first_Nm=tuple(
                np.cross(
                    np.asarray(site.application_point_first_m) - row["first_com_world_m"], -force
                )
            ),
            torque_on_second_Nm=tuple(
                np.cross(
                    np.asarray(site.application_point_second_m) - row["second_com_world_m"], force
                )
            ),
            power_W=site.power_w,
        )
        details.append(row)
    result = replace(
        measured,
        loads=loads,
        joint_effort_full=measured.joint_effort_full + ctx.delta[: ctx.nq],
        link_wrenches_world=measured.link_wrenches_world + ctx.link_delta,
        fixed_root_contact_wrench=measured.fixed_root_contact_wrench + ctx.root_delta,
        details=tuple(details),
    )
    return result, ctx
