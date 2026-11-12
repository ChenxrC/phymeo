"""D3 dual accepted source: actual node banks plus actual floating-body FK.

Only source assembly is new. BVH, pair packing, and finite distance kernels are
the accepted M1/D2 pipeline. Body-local registration is never current HOST FK.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from ..constraints.collision_shapes import CollisionMotion, CollisionShape
from ._deformable_boundary_source import (
    DeformableBoundaryAdapter, DeformableBoundaryModel, _Accepted,
)
from ._resident_boundary_source import ResidentBoundarySourceView, freeze, owned, physical_sources, stored_interval
from ._resident_collision_plan import ResidentStaticBinding
from ._resident_floating_binding import ResidentFloatingTreeBinding
from ._resident_participant_layout import ResidentParticipantBinding


@dataclass
class _DualLease:
    loan: object
    scene: object
    rigid: object
    mechanical: object
    active: bool = True

    def require(self):
        if not self.active:
            raise RuntimeError("D3 source lease expired")
        self.loan.revision
        batch = self.scene._service._batch
        if (
            self.rigid.owner is not batch._collision_owner
            or self.rigid.state_epoch != batch._state_epoch
            or self.rigid.active != batch._active
            or self.rigid.runtime is not self.loan.runtime
            or self.rigid.mode != self.loan.mode
        ):
            raise ValueError("D3 rigid source is stale or belongs to another Runtime")
        self.scene.layout.require_current()


class DeformableResponseBoundaryAdapter(DeformableBoundaryAdapter):
    source_kind = "deformable-floating-entry-device-v1"

    def __init__(self, batch, scene):
        super().__init__(batch)
        self.scene = scene
        self._moving = {}
        self._world_rigid = {}
        self._registration = None

    def compile_registration(self, plan, model, bindings):
        if type(model) is not DeformableBoundaryModel:
            raise TypeError("D3 requires the explicit open deformable surface model")
        if self.scene.world_keys != self.batch.world_keys:
            raise ValueError("D3 node and rigid world order differ")
        wrappers = tuple(bindings)
        rounds = {s.key: s for s in model.shapes if type(s) is CollisionShape}
        if len(wrappers) != len(rounds):
            raise ValueError("D3 requires exactly one moving binding for every round")
        by_key = {}
        locals_only = []
        with self.scene._service.borrow_boundary_source(
            self.scene._owner, self.scene.layout.fingerprint
        ) as (accepted, mechanical):
            if accepted.runtime is not plan._runtime or accepted.mode != plan._mode:
                raise ValueError("D3 sources require the same real Runtime")
            sources = physical_sources(accepted)
            self._world_rigid = {
                p.key.world: (p.key, accepted.slots[mechanical[p.key]])
                for p in self.scene.layout.participants
            }
            for wrapper in wrappers:
                if type(wrapper) is not ResidentParticipantBinding:
                    raise TypeError("D3 moving rounds require ResidentParticipantBinding")
                binding = wrapper.binding
                key = binding.key
                if key not in rounds or key in by_key:
                    raise ValueError("D3 moving shape membership differs")
                participant = wrapper.participant
                entry = self.scene.layout.entries.get(participant)
                if entry is None or entry.family != "floating" or entry.body_count != 1:
                    raise ValueError("D3 supports one ordinary floating body per world")
                if type(binding) is not ResidentFloatingTreeBinding or binding.group_index != 0:
                    raise TypeError("D3 requires a direct root-body floating binding")
                if getattr(binding, "_source_shape", None) is not None:
                    raise ValueError("D3 asset lowering is a later composition")
                if self.scene.layout.body_owners.get((key.world, key.body)) != (participant, 0):
                    raise ValueError("D3 rigid material body ownership differs")
                native = mechanical[participant]
                source = sources[2]
                spec = source.specs[native]
                if binding.topology != spec.topology:
                    raise ValueError("D3 direct binding topology differs")
                points = np.asarray(binding.local_points_m, np.float64)
                if points.shape != (len(rounds[key].vertex_ids), 3):
                    raise ValueError("D3 round material point arity differs")
                if (rounds[key].kind == "sphere" and len(points) != 1) or (
                    rounds[key].kind == "capsule" and len(points) != 2
                ):
                    raise ValueError("D3 supports native one-point sphere/two-point capsule")
                slot = accepted.slots[native]
                by_key[key] = wrapper
                self._moving[key] = dict(
                    participant=participant, native=native, slot=slot,
                    row=int(source.rows[native]), group=0, topology=spec.topology,
                    model_token=spec.model_token, anchor=owned(spec.anchor, np.float64),
                    local=owned(points, np.float64),
                )
                # Reuse only D2's bounded material/topology registration. These
                # body-local constants are replaced by explicit moving metadata
                # before any device allocation or query; never a static pose.
                locals_only.append(ResidentStaticBinding(key, CollisionMotion(points)))
        super().compile_registration(plan, model, locals_only)
        self._registration = plan._registration
        meta = plan._constants["shape_meta"].copy()
        rigid_meta = np.full(meta.shape, -1, np.int32)
        shifts = np.zeros((len(meta), 9), np.float32)
        exact = np.zeros((len(meta), 3), np.float64)
        definitions = []
        for i, shape in enumerate(plan._model.shapes):
            row = dict(plan._source_base["bindings"][i])
            if shape.key in self._moving:
                item = self._moving[shape.key]
                rigid_meta[i] = (2, item["row"], 0, item["slot"], meta[i, 4], meta[i, 5])
                meta[i, 0] = -1  # D1 gather must not overwrite moving-round output.
                value, lower, upper = stored_interval(item["anchor"])
                shifts[i] = np.r_[value, lower, upper]
                exact[i] = item["anchor"]
                row.update(
                    frame="floating_body_local", participant=item["participant"],
                    mechanical_world=item["native"], native_slot=item["slot"],
                    native_row=item["row"], native_group=0, topology=item["topology"],
                    model_token=item["model_token"],
                )
            definitions.append(row)
        plan._constants = dict(
            plan._constants, shape_meta=owned(meta, np.int32),
            rigid_meta=owned(rigid_meta, np.int32), anchor_shifts=owned(shifts),
        )
        plan._bindings = tuple(
            by_key[s.key] if s.key in by_key else plan._bindings[i]
            for i, s in enumerate(plan._model.shapes)
        )
        plan._source_base.update(
            bindings=tuple(definitions), moving=freeze(self._moving),
            rigid_owner=self.scene._owner,
            rigid_world_layouts=("floating",) * len(plan._world_keys),
            rigid_registry_fingerprint=self.scene.layout.fingerprint,
            anchor_shifts_exact=owned(exact),
            static_flags=owned(np.zeros(len(meta), bool)),
            uploaded_constants=plan._constants,
        )
        plan._counts["borrowed_device_bytes"] = (
            12 * sum(self.batch._models[w].n for w in plan._world_keys)
            + 4 * len(plan._world_keys) + self._borrowed_rigid_bytes()
        )
        self._signature = (model, wrappers)

    def _borrowed_rigid_bytes(self):
        with self.scene._service.borrow_boundary_source(
            self.scene._owner, self.scene.layout.fingerprint
        ) as (accepted, _):
            source = physical_sources(accepted)[2]
            # Logical referenced arrays, unique identities; not another allocation.
            return sum(a.nbytes for a in (source.meta, source.fk, accepted.ticks))

    @contextmanager
    def borrow_source(self, plan):
        with self.batch.borrow_vertices(expected_revision=self.batch.revision) as loan:
            with self.scene._service.borrow_boundary_source(
                self.scene._owner, self.scene.layout.fingerprint
            ) as (rigid, mechanical):
                lease = _DualLease(loan, self.scene, rigid, mechanical)
                accepted = _Accepted(
                    plan._owner, plan._topology_generation, loan.revision,
                    loan.active_bank, loan.world_keys, loan.runtime, loan.mode,
                )
                try:
                    lease.require()
                    yield ResidentBoundarySourceView(
                        accepted, plan._model, plan._bindings, plan._owner,
                        plan._registration, plan._world_keys, plan._shape_keys,
                        plan._shape_revisions, plan._world_layouts, plan._shape_rows,
                        plan._shape_groups, plan._shape_layouts,
                        tuple(loan.position(w) for w in plan._world_keys), lease,
                    )
                finally:
                    lease.active = False

    def validate_source(self, plan, view):
        super().validate_source(plan, view)
        lease = view._lease
        lease.require()
        if (
            plan._registration is not self._registration
            or self.scene._owner is not plan._source_base["rigid_owner"]
            or self.scene.layout.fingerprint != plan._source_base["rigid_registry_fingerprint"]
        ):
            raise ValueError("D3 rigid registration changed")
        source = physical_sources(lease.rigid)[2]
        for item in self._moving.values():
            native = lease.mechanical[item["participant"]]
            if (
                native != item["native"]
                or lease.rigid.slots[native] != item["slot"]
                or source.rows[native] != item["row"]
                or source.specs[native].topology != item["topology"]
                or source.specs[native].model_token != item["model_token"]
                or not np.array_equal(source.specs[native].anchor, item["anchor"])
            ):
                raise ValueError("D3 moving material source differs")
        return view

    def source_prepass_bytes(self, plan):
        return super().source_prepass_bytes(plan) + 96 * len(plan._shape_keys)

    def empty_source(self, plan, view):
        outputs, raw, ticks = super().empty_source(plan, view)
        rigid = view._lease.rigid
        rigid_ticks = plan._download(rigid.ticks, "source/rigid_ticks")
        for i, world in enumerate(plan._world_keys):
            if rigid_ticks[self._world_rigid[world][1]] != ticks[i]:
                raise RuntimeError("D3 empty source node/rigid tick mismatch")
        fk = plan._download(physical_sources(rigid)[2].fk, "source/actual_rigid_fk")
        raw.update(rigid_ticks=rigid_ticks, actual_rigid_fk=fk,
                   poses=np.empty((0, 12), np.float32), native_poses=np.empty((0, 12), np.float32))
        plan._counts["actual_bank_host_readback_bytes"] = fk.nbytes
        return outputs, raw, ticks

    def gather(self, plan, view, ledger):
        from ._tyx_boundary_source import gather_boundary_points, translate_boundary_frame
        from ._tyx_deformable_boundary import gather_deformable_points
        from ._tyx_deformable_contact import clear_response_poses

        self.validate_source(plan, view)
        s, n, w = len(plan._shape_keys), int(plan._offsets[-1]), len(plan._world_keys)
        outputs = {
            name: plan._allocate(ledger, shape=shape, dtype=dtype)
            for name, shape, dtype in (
                ("points", (n, 3), np.float32), ("point_bounds", (n, 6), np.float32),
                ("stamps", (s, 4), np.int32), ("ticks", (w,), np.int32),
                ("poses", (s, 12), np.float32), ("native_poses", (s, 12), np.float32),
            )
        }
        plan._launch("source/clear_poses", clear_response_poses, s, outputs["poses"])
        loan = view._lease.loan
        for slot, world in enumerate(plan._world_keys):
            source = loan.position(world)
            plan._launch(
                "source/node:" + str(world), gather_deformable_points, s, slot,
                int(source.shape[0]), plan._buffers["shape_meta"],
                plan._buffers["node_indices"], plan._buffers["local"],
                plan._buffers["local_bounds"], source, loan.tick(world),
                outputs["points"], outputs["point_bounds"], outputs["stamps"], outputs["ticks"],
            )
        rigid = view._lease.rigid
        source = physical_sources(rigid)[2]
        plan._launch(
            "source/floating", gather_boundary_points, s, 2, 1,
            plan._buffers["rigid_meta"], plan._buffers["local"],
            plan._buffers["local_bounds"], source.meta, source.fk, rigid.ticks,
            outputs["poses"], outputs["stamps"], outputs["points"], outputs["point_bounds"],
        )
        plan._launch(
            "source/floating_world", translate_boundary_frame, s,
            plan._buffers["rigid_meta"], plan._buffers["anchor_shifts"], outputs["poses"],
            outputs["native_poses"], outputs["stamps"], outputs["points"], outputs["point_bounds"],
        )
        raw = {name: plan._download(a, "source/" + name) for name, a in outputs.items()}
        ticks = raw.pop("ticks")
        self.logical_ticks(plan, view, ticks)
        rigid_ticks = plan._download(rigid.ticks, "source/rigid_ticks")
        fk = plan._download(source.fk, "source/actual_rigid_fk")
        for i, world in enumerate(plan._world_keys):
            if rigid_ticks[self._world_rigid[world][1]] != ticks[i]:
                raise RuntimeError("D3 node and rigid world ticks differ")
        actual = {world: plan._download(loan.position(world), "source/actual_node:" + str(world))
                  for world in plan._world_keys}
        for name, shape, dtype in (
            ("points", (n, 3), np.float32), ("point_bounds", (n, 6), np.float32),
            ("stamps", (s, 4), np.int32), ("poses", (s, 12), np.float32),
            ("native_poses", (s, 12), np.float32),
        ):
            a = raw[name]
            if a.shape != shape or a.dtype != dtype or not np.isfinite(a).all():
                raise RuntimeError("D3 gathered " + name + " is malformed")
        if rigid_ticks.dtype != np.int32 or fk.dtype != np.float32 or not np.isfinite(fk).all():
            raise RuntimeError("D3 actual rigid source readback is malformed")
        if np.any(raw["point_bounds"][:, :3] > raw["points"]) or np.any(
            raw["points"] > raw["point_bounds"][:, 3:]
        ):
            raise RuntimeError("D3 bounds omit their material points")
        deltas = np.empty((n, 3), np.float64)
        slots = {world: i for i, world in enumerate(plan._world_keys)}
        for i, shape in enumerate(plan._model.shapes):
            begin, end = map(int, plan._offsets[i:i + 2])
            world = shape.key.world
            stamp = raw["stamps"][i]
            if shape.key in self._moving:
                item = self._moving[shape.key]
                if rigid_ticks[item["slot"]] != ticks[slots[world]]:
                    raise RuntimeError("D3 node and rigid accepted ticks differ")
                expected_stamp = (1, ticks[slots[world]], item["slot"], item["row"])
                native = fk[item["row"], 0, :12]
                if raw["native_poses"][i].tobytes() != native.tobytes():
                    raise RuntimeError("D3 gathered rigid pose is not its actual FK bank")
                expected = native[:3].astype(np.float64) + item["local"] @ native[3:].astype(
                    np.float64
                ).reshape(3, 3).T + item["anchor"]
            else:
                slot = slots[world]
                expected_stamp = (1, ticks[slot], slot, slot)
                bank = actual[world]
                if bank.shape != loan.position(world).shape or bank.dtype != np.float32:
                    raise RuntimeError("D3 actual node source shape differs")
                expected = bank[plan._constants["node_indices"][begin:end]]
                if raw["points"][begin:end].tobytes() != expected.tobytes():
                    raise RuntimeError("D3 gathered nodes differ from the accepted bank")
                if raw["point_bounds"][begin:end, :3].tobytes() != expected.tobytes() or (
                    raw["point_bounds"][begin:end, 3:].tobytes() != expected.tobytes()
                ):
                    raise RuntimeError("D3 gathered node bounds differ")
            if tuple(map(int, stamp)) != tuple(map(int, expected_stamp)):
                raise RuntimeError("D3 source stamp mismatch")
            deltas[begin:end] = raw["points"][begin:end].astype(np.float64) - expected
        raw.update(
            material_deltas=deltas, actual_bank_positions=actual,
            actual_rigid_fk=fk, rigid_ticks=rigid_ticks,
        )
        plan._counts["actual_bank_host_readback_bytes"] = sum(a.nbytes for a in actual.values()) + fk.nbytes
        plan._raw_prefix["source/material_reconstruction_delta"] = owned(deltas)
        self.validate_source(plan, view)
        return outputs, raw, ticks

    def source_record(self, plan, raw, ticks, view):
        base = dict(super().source_record(plan, raw, ticks, view))
        base.update(
            rigid_epoch=view._lease.rigid.state_epoch,
            rigid_active_bank=view._lease.rigid.active,
            rigid_ticks=raw["rigid_ticks"], actual_rigid_fk=raw["actual_rigid_fk"],
            current_rigid_poses=raw["poses"], native_rigid_poses=raw["native_poses"],
            semantics="dual-accepted-entry; no response has yet committed",
        )
        return freeze(base)
