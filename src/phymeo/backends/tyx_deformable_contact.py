"""Entry-sampled calibrated normal response between real nodes and a free rigid body.

This owner composes original D1 ordered material and resident floating mechanics.
It is a discrete explicit response, not CCD, an implicit contact solve, or a
claim of complete cloth/self-contact support.
"""

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType

import numpy as np

from ..core.state import StepInputs
from ..geometry._triangle_distance_dispatch import BoundaryDistanceLimits
from ..solvers.collision_loads import NormalLayer
from ._deformable_boundary_source import DeformableBoundaryModel, DeformableBoundarySurface
from ._deformable_response_source import DeformableResponseBoundaryAdapter
from ._resident_boundary_query import ResidentBoundaryQueryPlan, _digest
from ._resident_boundary_source import freeze, owned
from ._resident_participant_layout import ResidentParticipant
from ._world_step_layout import as_fp32
from .mujoco_tyx._resources import DeviceResources
from .resident_scene import ResidentSceneBatch
from .tyx_deformable_batch import TyxDeformableBatch


@dataclass(frozen=True, eq=False)
class DeformableContactState:
    world: int
    node: object
    rigid: object


@dataclass(frozen=True, eq=False)
class DeformableContactReceipt:
    schema: str
    binding_sha256: str
    source_revision: int
    destination_revision: int
    query: object
    node_receipt: object
    rigid_publication: Mapping
    worlds: Mapping
    states: Mapping


@dataclass(frozen=True, eq=False)
class DeformableContactCheckpoint:
    schema: str
    owner: object
    binding_sha256: str
    revision: int
    nodes: object
    rigid: Mapping
    last_step: object
    content_sha256: str


class _ResponseEngine:
    def __init__(self, scene, node_counts, capacity):
        from ._deformable_runtime_lease import ResidentDeformableRuntimeLease

        self.lease = ResidentDeformableRuntimeLease(scene)
        self.mode = self.lease.mode
        try:
            self.runtime = self.lease.claim(self.mode)
        except BaseException as primary:
            try:
                self.lease.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "D3 lease claim and cleanup failed", [primary, cleanup]
                ) from primary
            raise
        self.tx = scene._service._batch._tx
        self.resources = DeviceResources()
        self.resources.set_runtime(self.lease)
        self.counts = dict(
            launches=0, uploads=0, uploaded_bytes=0, downloads=0, downloaded_bytes=0, syncs=0
        )
        self.worlds = len(node_counts)
        self.offsets = np.asarray([0, *np.cumsum(node_counts)], np.int32)
        p, n, w = max(1, capacity), int(self.offsets[-1]), self.worlds
        self.capacity = capacity
        self.arrays = {}
        try:
            with self.tx.use_runtime(self.runtime):
                for name, shape, dtype in (
                    ("rows", (p, 13), np.float32),
                    ("law", (p, 5), np.float32),
                    ("ids", (p, 5), np.int32),
                    ("bary", (p, 3), np.float32),
                    ("lever", (p, 3), np.float32),
                    ("nodes", (n, 3), np.float32),
                    ("wrench", (w, 6), np.float32),
                    ("status", (w,), np.int32),
                    ("offsets", (w + 1,), np.int32),
                ):
                    array = (
                        np.empty(shape, dtype)
                        if self.mode == "interpret"
                        else self.tx.cuda_empty(shape, dtype)
                    )
                    if self.mode == "cuda":
                        self.resources.add_buffer(array)
                    self.arrays[name] = array
                self.upload("offsets", self.offsets)
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "D3 force construction and cleanup failed", [primary, cleanup]
                ) from primary
            raise

    @property
    def native_bytes(self):
        return sum(a.nbytes for a in self.arrays.values())

    def upload(self, name, value):
        array = self.arrays[name]
        if value.shape != array.shape or value.dtype != array.dtype:
            raise ValueError("D3 force upload shape/dtype mismatch")
        if self.mode == "cuda":
            array.copy_from(value)
            self.counts["uploads"] += 1
            self.counts["uploaded_bytes"] += value.nbytes
        else:
            np.copyto(array, value)

    def read(self, name):
        array = self.arrays[name]
        value = array.copy() if self.mode == "interpret" else array.to_numpy()
        self.counts["downloads"] += 1
        self.counts["downloaded_bytes"] += value.nbytes
        return value

    def evaluate(self, rows, ids, bary, lever):
        from ._tyx_deformable_contact import transpose_response
        from ._tyx_normal_layer import normal_layer_rows

        count = len(rows)
        if count > self.capacity:
            raise ValueError("D3 contact capacity exceeded")
        with self.tx.use_runtime(self.runtime):
            for name, values in (("rows", rows), ("ids", ids), ("bary", bary), ("lever", lever)):
                template = np.zeros(self.arrays[name].shape, self.arrays[name].dtype)
                template[:count] = values
                self.upload(name, template)
            a = self.arrays
            normal_layer_rows(count, a["rows"], a["law"], mode=self.mode)
            transpose_response(
                self.worlds,
                count,
                a["offsets"],
                a["ids"],
                a["bary"],
                a["lever"],
                a["law"],
                a["nodes"],
                a["wrench"],
                a["status"],
                mode=self.mode,
            )
            self.counts["launches"] += 2
            self.runtime.sync()
            self.counts["syncs"] += 1
            law, nodes, wrench, status = (
                self.read(n) for n in ("law", "nodes", "wrench", "status")
            )
        if np.any(status != 1) or not all(
            np.isfinite(a).all() for a in (law[:count], nodes, wrench)
        ):
            raise FloatingPointError("D3 nonfinite contact transpose")
        return law[:count], nodes, wrench

    def close(self):
        self.resources.close()


class TyxDeformableContactBatch:
    """Exclusive common owner; every selected logical world publishes atomically.

    One ordinary six-DOF floating body and one fixed-topology D1 node model per
    world. Every surface facet is an explicitly declared discrete layer spring.
    The class does not promise invariance under arbitrary surface refinement.
    """

    def __init__(
        self,
        node_worlds,
        participants,
        model,
        bindings,
        materials_by_pair,
        *,
        mode="cuda",
        settings=None,
        limits=None,
        contact_capacity=128,
        memory_limit_bytes=256 * 1024 * 1024,
        node_memory_limit_bytes=256 * 1024 * 1024,
        cache_root=None,
    ):
        from ._deformable_runtime_lease import ResidentDeformableRuntimeLease

        self._lock = RLock()
        self._closed = False
        self._busy = False
        self._poisoned = False
        self._owner = object()
        self._revision = 0
        self.last_step = None
        self.last_attempt = None
        self._context = None
        self._prepared_attempt = self._prepared_expected = None
        self._scene = self._nodes = self._query = self._engine = None
        self._public_query = None
        node_worlds, participants, bindings = (
            tuple(node_worlds),
            tuple(participants),
            tuple(bindings),
        )
        if type(model) is not DeformableBoundaryModel:
            raise TypeError("D3 needs DeformableBoundaryModel")
        if type(contact_capacity) is not int or not 0 <= contact_capacity <= 128:
            raise ValueError("D3 contact capacity must be in [0,128]")
        if type(memory_limit_bytes) is not int or memory_limit_bytes <= 0:
            raise ValueError("D3 native memory limit must be positive")
        if (
            not participants
            or len(participants) > 8
            or any(type(p) is not ResidentParticipant for p in participants)
        ):
            raise ValueError("D3 requires 1..8 registered ordinary floating participants")
        worlds = tuple(sorted(p.key.world for p in participants))
        if len(set(worlds)) != len(worlds) or any(
            p.family != "floating" or p.body_count != 1 for p in participants
        ):
            raise ValueError("D3 requires exactly one ordinary free body in every world")
        if tuple(sorted(w.key for w in node_worlds)) != worlds:
            raise ValueError("D3 node and rigid worlds differ")
        # Both owners use the same stable world order, independently of the
        # caller's interleaving of node and rigid declarations.
        node_worlds = tuple(sorted(node_worlds, key=lambda item: item.key))
        participants = tuple(sorted(participants, key=lambda item: item.key.world))
        if any(len(p.compiled.topology) == 0 for p in participants):
            raise ValueError("D3 needs complete physical topology")
        preflight_counts = [len(w.initial_state.position) for w in node_worlds]
        preflight_bytes = (
            116 * max(1, contact_capacity)
            + 12 * sum(preflight_counts)
            + 28 * len(worlds)
            + 4 * (len(worlds) + 1)
        )
        if preflight_bytes >= memory_limit_bytes:
            raise MemoryError("D3 response buffers leave no boundary query budget")
        shapes = {s.key: s for s in model.shapes}
        surfaces = [s for s in model.shapes if type(s) is DeformableBoundarySurface]
        expected = {
            tuple(sorted((s.key, r.key)))
            for s in surfaces
            for r in model.shapes
            if type(r) is not DeformableBoundarySurface
            and s.key.world == r.key.world
            and s.key.body != r.key.body
            and s.group & r.mask
            and r.group & s.mask
            and tuple(sorted((s.key, r.key))) not in model.excluded
        }
        if not isinstance(materials_by_pair, Mapping) or set(materials_by_pair) != expected:
            raise ValueError("D3 requires the complete eligible pair material table")
        if any(
            law is not None and (type(law) is not NormalLayer or law.friction != 0)
            for law in materials_by_pair.values()
        ):
            raise ValueError("D3 supports calibrated normal layers with mu=0")
        self._laws = MappingProxyType(dict(materials_by_pair))
        self._shapes = shapes
        self._world_keys = worlds
        self._participants = {p.key.world: p.key for p in participants}
        self._capacity = contact_capacity
        self._native_limit = memory_limit_bytes
        self._expected = None
        try:
            self._scene = ResidentSceneBatch(participants, mode=mode, cache_root=cache_root)
            lease = ResidentDeformableRuntimeLease(self._scene)
            try:
                self._nodes = TyxDeformableBatch(
                    node_worlds,
                    mode=mode,
                    settings=settings,
                    trace=True,
                    memory_limit_bytes=node_memory_limit_bytes,
                    cache_root=cache_root,
                    _runtime_lease=lease,
                )
            except BaseException as primary:
                # Once claimed, the node DeviceResources ledger alone owns
                # release, including sticky UNKNOWN_RELEASE dependencies.
                if not lease._claimed:
                    try:
                        lease.close()
                    except BaseException as cleanup:
                        raise BaseExceptionGroup(
                            "D3 node construction and lease cleanup failed", [primary, cleanup]
                        ) from primary
                raise
            if self._nodes.world_keys != self._scene.world_keys:
                raise ValueError("D3 sources have different stable world order")
            node_counts = [self._nodes._models[w].n for w in worlds]
            required = (
                116 * max(1, contact_capacity)
                + 12 * sum(node_counts)
                + 28 * len(worlds)
                + 4 * (len(worlds) + 1)
            )
            if required >= memory_limit_bytes:
                raise MemoryError("D3 response buffers leave no boundary query budget")
            self._adapter = DeformableResponseBoundaryAdapter(self._nodes, self._scene)
            self._query = ResidentBoundaryQueryPlan(
                self._nodes,
                model,
                bindings,
                limits=BoundaryDistanceLimits() if limits is None else limits,
                memory_limit_bytes=memory_limit_bytes - required,
                source_adapter=self._adapter,
            )
            self._engine = _ResponseEngine(self._scene, node_counts, contact_capacity)
            if self._engine.native_bytes != required:
                raise RuntimeError("D3 native buffer inventory differs")
            self.binding_sha256 = _digest(
                (
                    "tyx-deformable-contact-v1",
                    self._nodes.binding_sha256,
                    self._scene.layout.fingerprint,
                    model,
                    bindings,
                    self._laws,
                    contact_capacity,
                    mode,
                    self._query._limits,
                )
            )
            self._expected = (self._nodes.revision, self._scene.state_epoch)
            self._initial_nodes = self._nodes.checkpoint()
            self._initial_rigid = self._scene.snapshot()
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                error = BaseExceptionGroup("D3 construction and cleanup failed", [primary, cleanup])
                error.contact_owner = self
                raise error from primary
            raise

    @property
    def revision(self):
        return self._revision

    @property
    def world_keys(self):
        return self._world_keys

    def _check_open(self):
        if self._closed or self._poisoned:
            raise RuntimeError("D3 owner is closed or requires recovery")
        if self._expected != (self._nodes.revision, self._scene.state_epoch):
            raise ValueError("D3 underlying owner was independently mutated")

    def _ready(self, expected_revision=None):
        self._check_open()
        if self._busy:
            raise RuntimeError("D3 owner cannot be reentered")
        if expected_revision is not None and (
            type(expected_revision) is not int or expected_revision != self._revision
        ):
            raise ValueError("stale D3 revision")

    @contextmanager
    def _locked(self):
        with self._lock:
            if self._scene is None or self._nodes is None:
                raise RuntimeError("D3 owner construction is incomplete")
            with self._scene._service.locked(), self._nodes._lock:
                yield

    def _snapshot(self):
        nodes, rigid = self._nodes.snapshot(), self._scene.snapshot()
        if any(nodes[w].tick != rigid[w].tick for w in self.world_keys):
            raise RuntimeError("D3 accepted node and rigid ticks differ")
        return MappingProxyType(
            {w: DeformableContactState(w, nodes[w], rigid[w]) for w in self.world_keys}
        )

    def snapshot(self):
        with self._locked():
            self._ready()
            return self._snapshot()

    def set_controls(self, controls):
        with self._locked():
            self._ready()
            self._scene.set_controls(controls)
            self._revision += 1

    def query(self, *, expected_revision):
        """Query the current accepted joint state without advancing physics.

        The publication is bound to the common revision, including control
        changes. A step's entry query is evidence for that step, not a current
        accepted-state query after publication.
        """
        with self._locked():
            if type(expected_revision) is not int:
                raise TypeError("query requires an integer expected_revision")
            self._ready(expected_revision)
            result = self._query_entry()
            self._public_query = (self._revision, result)
            return result

    def require_query_current(self, result, *, expected_revision):
        """Validate this owner's latest explicit query and both state sources."""
        with self._locked():
            if type(expected_revision) is not int:
                raise TypeError("query requires an integer expected_revision")
            self._ready(expected_revision)
            if (
                self._public_query is None
                or self._public_query[0] != self._revision
                or self._public_query[1] is not result
            ):
                raise ValueError("query is not current for this D3 revision")
            return self._query.require_current(result)

    def _query_entry(self):
        margin = max((law.width_m for law in self._laws.values() if law is not None), default=0.0)
        return self._query.query(margin=margin)

    def step(self, inputs, h, *, expected_revision):
        with self._locked():
            self._ready(expected_revision)
            if not isinstance(inputs, Mapping) or set(inputs) != set(self.world_keys):
                raise ValueError("D3 needs complete nodal external-force inputs")
            inputs = {
                w: StepInputs(
                    as_fp32(
                        inputs[w].force, "D3 external nodal force", (self._nodes._models[w].n, 3)
                    )
                )
                for w in self.world_keys
            }
            dt = float(as_fp32(h, "D3 timestep", ()))
            if dt <= 0:
                raise ValueError("D3 timestep must be positive")
            entry = self._snapshot()
            self._context = dict(
                phase="entry-query",
                error=None,
                entry=entry,
                query=None,
                node_receipt=None,
                rigid_publication=None,
                inputs=inputs,
                h=dt,
            )
            self._busy = True
            try:
                self._context["query"] = self._query_entry()
                with self._nodes._transaction(self._nodes.revision):
                    receipt = self._scene._service.step(dt, self, deformable=self._nodes)
                # Shared service already performed the indivisible physical commit.
                self._expected = self._prepared_expected
                self._revision = receipt.destination_revision
                self.last_step = receipt
                self.last_attempt = self._prepared_attempt
                return receipt
            except BaseException as error:
                self._context["error"] = error
                self._context["boundary_attempt"] = self._query.last_attempt
                self.last_attempt = freeze(dict(self._context))
                raise
            finally:
                self._context = None
                self._busy = False

    def _law_rows(self, entry):
        context = self._context
        query = context["query"]
        if (
            query.state_epoch != self._nodes.revision
            or query.source["rigid_epoch"] != self._scene.state_epoch
        ):
            raise ValueError("D3 entry query no longer binds both owners")
        sites = []
        indices, bary, levers, rows = [], [], [], []
        slots = {w: i for i, w in enumerate(self.world_keys)}
        offsets = self._engine.offsets
        for group in query.groups:
            surface = group.surface
            world = surface.world
            model = self._nodes._models[world]
            source = query.source["surfaces"][surface]
            facet_ids = source["facet_ids"]
            for row in group.packet.rows:
                round_key = group.round_keys[row.segment_id]
                law = self._laws[tuple(sorted((surface, round_key)))]
                if law is None:
                    continue
                if row.status != "RESOLVED" or row.normal is None:
                    raise RuntimeError("D3 cannot load an unresolved finite-distance candidate")
                if row.gap >= law.width_m:
                    continue
                if len(sites) >= self._capacity:
                    raise ValueError("D3 active contact capacity exceeded")
                participant = self._participants[world]
                packet = entry[participant]
                geometry = packet["geometry"]
                if packet["state"].tick != context["entry"][world].node.tick:
                    raise RuntimeError("D3 mechanical entry and query tick differ")
                normal = np.asarray(row.normal, np.float64)
                beta = np.asarray(row.barycentric, np.float64)
                if abs(np.linalg.norm(normal) - 1.0) > 2e-6:
                    raise ValueError("D3 normal violates the original calibrated-layer unit gate")
                node_indices = tuple(
                    int(np.flatnonzero(model.point_ids == point)[0]) for point in row.vertex_ids
                )
                if tuple(source["oriented_point_ids"][row.facet_id]) != row.vertex_ids:
                    raise ValueError(
                        "D3 facet ordinal does not bind its original oriented point IDs"
                    )
                # Public rows already include their pair frame; only raw kernel
                # witnesses still need pair_origins_relative_m.
                origin = group.packet.origin_m
                axis_point = origin + np.asarray(row.axis_point_relative_m, np.float64)
                node_point = origin + np.asarray(row.triangle_point_relative_m, np.float64)
                rigid_point = axis_point + self._shapes[round_key].radius * normal
                com = geometry.com_position_m[0]
                lever = rigid_point - com
                vn = beta @ context["entry"][world].node.velocity[np.asarray(node_indices)]
                vr = geometry.linear_velocity_m_s[0] + np.cross(
                    geometry.angular_velocity_rad_s[0], lever
                )
                relative = vn - vr
                law_row = np.r_[
                    normal,
                    relative,
                    law.width_m,
                    row.gap,
                    law.stiffness_n_per_m,
                    1.0,
                    law.damping_s_per_m,
                    0.0,
                    law.slip_speed_m_s,
                ]
                law_row = as_fp32(law_row, "D3 calibrated layer row", (13,))
                local_point = geometry.link_rotation[0].T @ (
                    rigid_point - geometry.link_position_m[0]
                )
                local_axis = geometry.link_rotation[0].T @ (
                    axis_point - geometry.link_position_m[0]
                )
                site = dict(
                    world=world,
                    surface=surface,
                    round=round_key,
                    facet_id=int(facet_ids[row.facet_id]),
                    facet_ordinal=int(row.facet_id),
                    point_ids=tuple(row.vertex_ids),
                    node_indices=node_indices,
                    barycentric=owned(beta),
                    axis_parameter=row.axis_parameter,
                    normal=owned(normal),
                    gap=row.gap,
                    node_point=owned(node_point),
                    rigid_point=owned(rigid_point),
                    rigid_com=owned(com),
                    relative_velocity=owned(relative),
                    law_input=owned(law_row),
                    rigid_point_local=owned(local_point),
                    rigid_axis_local=owned(local_axis),
                    radius=self._shapes[round_key].radius,
                )
                sites.append(site)
                index = slots[world]
                indices.append((index, *(int(offsets[index]) + n for n in node_indices), index))
                bary.append(beta)
                levers.append(lever)
                rows.append(law_row)
        return (
            sites,
            np.asarray(rows, np.float32).reshape(-1, 13),
            np.asarray(indices, np.int32).reshape(-1, 5),
            np.asarray(bary, np.float32).reshape(-1, 3),
            np.asarray(levers, np.float32).reshape(-1, 3),
        )

    def sample(self, entry, h):
        self._check_open()
        if self._context is None or not self._busy:
            raise RuntimeError("D3 sample requires its common attempt")
        self._context["phase"] = "entry-force"
        sites, rows, indices, bary, levers = self._law_rows(entry)
        law, node_force, wrench = self._engine.evaluate(rows, indices, bary, levers)
        for i, site in enumerate(sites):
            f = law[i, 2:5].astype(np.float64)
            site.update(
                law_output=owned(law[i]),
                node_forces=owned(bary[i].astype(np.float64)[:, None] * f),
                rigid_force=owned(-f),
                rigid_torque=owned(np.cross(levers[i].astype(np.float64), -f)),
            )
        inputs, world_contact, wrenches = {}, {}, {}
        for i, world in enumerate(self.world_keys):
            start, end = map(int, self._engine.offsets[i : i + 2])
            contact = node_force[start:end]
            total = as_fp32(
                self._context["inputs"][world].force + contact.astype(np.float64),
                "D3 complete uploaded node force",
                contact.shape,
            )
            inputs[world] = StepInputs(total)
            world_contact[world] = owned(contact)
            wrenches[self._participants[world]] = owned(wrench[i : i + 1])
        self._context.update(
            sites=freeze(sites),
            node_contact_force=freeze(world_contact),
            wrenches=freeze(wrenches),
            complete_node_inputs=freeze(inputs),
            mechanical_entry=entry,
            phase="node-candidate",
        )
        receipt = self._nodes._stage_step(inputs, h)
        self._context["node_receipt"] = receipt
        self._context["phase"] = "rigid-candidate"
        return dict(
            wrenches=MappingProxyType(wrenches),
            rejected=MappingProxyType(dict.fromkeys(self.world_keys, False)),
            payload=self._context,
        )

    def prepare_publication(
        self, entry, candidate, diagnostics, projected, accepted, status, payload, h
    ):
        return self._prepare_response_publication(
            entry, candidate, diagnostics, projected, accepted, status, payload, h
        )

    def _prepare_response_publication(
        self, entry, candidate, diagnostics, projected, accepted, status, payload, h
    ):
        self._context["phase"] = "candidate-audit"
        if (
            payload is not self._context
            or set(accepted) != set(self.world_keys)
            or not all(accepted.values())
        ):
            raise RuntimeError("D3 all worlds must accept both physical domains")
        node_receipt = payload["node_receipt"]
        if node_receipt is None or node_receipt.source_revision != self._nodes.revision:
            raise RuntimeError("D3 node candidate does not bind this attempt")
        rigid_states = self._scene._capsules({key: row["state"] for key, row in candidate.items()})
        worlds, states = {}, {}
        for world in self.world_keys:
            participant = self._participants[world]
            before, after = entry[participant], candidate[participant]
            state = node_receipt.states[world]
            if (
                after["state"].tick != state.tick
                or state.tick != payload["entry"][world].node.tick + 1
            ):
                raise RuntimeError("D3 candidates do not share one actual successor tick")
            if participant not in diagnostics or participant not in projected:
                raise RuntimeError("D3 rigid candidate lacks complete mechanics diagnostics")
            model = self._nodes._models[world]
            material = np.zeros((model.n, 3), np.float64)
            row = node_receipt.worlds[world]
            if len(row.trace) != row.sweeps:
                raise RuntimeError("D3 fixed-node reaction requires every actual ordered sweep")
            for trace in row.trace:
                for k, (i, j) in enumerate(model.edges):
                    impulse = trace[k, 14:17].astype(np.float64) * float(trace[k, 17]) / h
                    material[int(i)] += impulse
                    material[int(j)] -= impulse
            fixed = np.zeros_like(material)
            mask = model.weights == 0
            fixed[mask] = -h * row.force[mask].astype(np.float64) - material[mask]
            selected = tuple(site for site in payload["sites"] if site["world"] == world)
            work = before_potential = after_potential = dissipation = 0.0
            for site in selected:
                beta = site["barycentric"]
                nodes = np.asarray(site["node_indices"], np.int32)
                node_after = beta @ state.position[nodes].astype(np.float64)
                geometry = after["geometry"]
                rigid_after = (
                    geometry.link_position_m[0]
                    + geometry.link_rotation[0] @ site["rigid_point_local"]
                )
                axis_after = (
                    geometry.link_position_m[0]
                    + geometry.link_rotation[0] @ site["rigid_axis_local"]
                )
                force = site["law_output"][2:5].astype(np.float64)
                work += float(
                    force @ (node_after - site["node_point"])
                    - force @ (rigid_after - site["rigid_point"])
                )
                law = site["law_input"]
                stiffness, width = float(law[8]), float(law[6])
                compression = max(0.0, width - float(law[7]))
                before_potential += 0.5 * stiffness * compression**2
                gap_after = float(np.linalg.norm(axis_after - node_after)) - site["radius"]
                after_potential += 0.5 * stiffness * max(0.0, width - gap_after) ** 2
                rate, magnitude = map(float, site["law_output"][:2])
                dissipation += (stiffness * compression - magnitude) * rate
            if not np.isfinite([work, before_potential, after_potential, dissipation]).all():
                raise FloatingPointError("D3 material work audit is nonfinite")
            # This is the true sampled damping power, not an energy credit used
            # by either physical solver. Finite-step work balance is reported.
            world_record = dict(
                entry=payload["entry"][world],
                sites=selected,
                node_contact_force=payload["node_contact_force"][world],
                node_total_force=row.force,
                rigid_wrench=payload["wrenches"][participant],
                uploaded_rigid_wrench=owned(
                    np.asarray(payload["wrenches"][participant], np.float32)
                ),
                material_impulse=owned(material),
                fixed_reaction_impulse=owned(fixed),
                contact_work=work,
                entry_dissipation_rate=dissipation,
                layer_potential_before=before_potential,
                layer_potential_after=after_potential,
                work_balance=work + after_potential - before_potential + h * dissipation,
                potential_semantics="advected entry material sites; not endpoint closest feature",
                node_external_work=float(
                    np.sum(
                        payload["inputs"][world].force
                        * (state.position - payload["entry"][world].node.position)
                    )
                ),
                rigid_external_entry_power=float(
                    before["external"] @ np.asarray(before["state"].velocity)
                ),
            )
            worlds[world] = freeze(world_record)
            states[world] = DeformableContactState(world, state, rigid_states[world])
        rigid_publication = freeze(
            dict(
                entry=entry,
                candidate=candidate,
                diagnostics=diagnostics,
                projected=projected,
                accepted=accepted,
                status=status,
            )
        )
        self._context["rigid_publication"] = rigid_publication
        receipt = DeformableContactReceipt(
            "tyx-deformable-contact-step-v1",
            self.binding_sha256,
            self._revision,
            self._revision + 1,
            payload["query"],
            node_receipt,
            rigid_publication,
            MappingProxyType(worlds),
            MappingProxyType(states),
        )
        self._prepared_attempt = freeze(dict(self._context, phase="accepted"))
        self._prepared_expected = (node_receipt.destination_revision, self._scene.state_epoch + 1)
        return receipt

    def checkpoint(self):
        with self._locked():
            self._ready()
            nodes, rigid = self._nodes.checkpoint(), self._scene.snapshot()
            digest = _digest((self.binding_sha256, nodes, rigid, self.last_step))
            return DeformableContactCheckpoint(
                "tyx-deformable-contact-checkpoint-v1",
                self._owner,
                self.binding_sha256,
                self._revision,
                nodes,
                rigid,
                self.last_step,
                digest,
            )

    def _selected(self, worlds):
        values = self.world_keys if worlds is None else tuple(worlds)
        if len(set(values)) != len(values) or any(
            type(w) is not int or w not in self.world_keys for w in values
        ):
            raise ValueError("D3 unknown or duplicate selected world")
        return tuple(w for w in self.world_keys if w in values)

    def _replace(self, node_states, rigid_states, selected, *, last_step, node_last_step=None):
        # The shared service checks the node pending replacement before either
        # bank changes. No sequential public reset/restore or fake rollback.
        expected = dict(self._nodes._states)
        expected.update({w: node_states[w] for w in selected})
        flat = {}
        for world in selected:
            packet = rigid_states[world]
            keys = self._scene.layout.worlds[world].participant_keys
            if (
                packet.world != world
                or packet.owner is not self._scene._owner
                or packet.registry_fingerprint != self._scene.layout.fingerprint
                or set(packet.participants) != set(keys)
                or packet.tick != expected[world].tick
                or any(
                    p.world != world or p.tick != packet.tick for p in packet.participants.values()
                )
            ):
                raise ValueError("D3 replacement rigid world/owner/tick binding differs")
            flat.update(packet.participants)
        if not selected:
            return
        prepared_expected = (self._nodes.revision + 1, self._scene.state_epoch + 1)
        prepared_revision = self._revision + 1
        self._busy = True
        try:
            with self._nodes._transaction(self._nodes.revision):
                self._nodes._stage_replace(selected, expected, last_step=node_last_step)
                self._scene._service.restore(
                    flat,
                    expected_epoch=self._scene.state_epoch,
                    deformable=self._nodes,
                )
            self._expected = prepared_expected
            self._revision = prepared_revision
            self.last_step = last_step
        finally:
            self._busy = False

    def restore(self, checkpoint, *, expected_revision, worlds=None):
        with self._locked():
            self._ready(expected_revision)
            if type(checkpoint) is not DeformableContactCheckpoint or (
                checkpoint.schema != "tyx-deformable-contact-checkpoint-v1"
                or checkpoint.owner is not self._owner
                or checkpoint.binding_sha256 != self.binding_sha256
                or tuple(checkpoint.rigid) != self.world_keys
                or checkpoint.content_sha256
                != _digest(
                    (self.binding_sha256, checkpoint.nodes, checkpoint.rigid, checkpoint.last_step)
                )
            ):
                raise ValueError("D3 checkpoint owner/model/layout/content differs")
            states = self._nodes._validate_checkpoint(checkpoint.nodes)
            selected = self._selected(worlds)
            self._replace(
                states,
                checkpoint.rigid,
                selected,
                last_step=checkpoint.last_step if selected == self.world_keys else None,
                node_last_step=checkpoint.nodes.last_step if selected == self.world_keys else None,
            )

    def reset(self, worlds=None, *, expected_revision):
        with self._locked():
            self._ready(expected_revision)
            selected = self._selected(worlds)
            states = self._nodes._validate_checkpoint(self._initial_nodes)
            self._replace(states, self._initial_rigid, selected, last_step=None)

    def stats(self):
        with self._locked():
            return freeze(
                dict(
                    revision=self._revision,
                    world_count=len(self.world_keys),
                    nodes=self._nodes.stats(),
                    scene=self._scene.stats(),
                    query=self._query.stats(),
                    response=self._engine.counts,
                    response_native_bytes=self._engine.native_bytes,
                    native_query_response_limit=self._native_limit,
                    node_memory_limit_scope="D1 device plus reserved HOST arrays, separate",
                    contact_sampling="discrete_entry",
                    friction=False,
                    ccd=False,
                )
            )

    def close(self):
        with self._lock:
            if self._busy:
                raise RuntimeError("cannot close D3 during its common transaction")
            errors = []
            for owner in (self._query, self._engine, self._nodes, self._scene):
                if owner is not None:
                    try:
                        owner.close()
                    except BaseException as error:
                        errors.append(error)
            if errors:
                self._poisoned = True
                raise BaseExceptionGroup("D3 close retains unresolved owners", errors)
            self._closed = True

    def __enter__(self):
        self._ready()
        return self

    def __exit__(self, exc_type, error, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if error is not None:
                raise BaseExceptionGroup("D3 body and cleanup failed", [error, cleanup]) from error
            raise
