"""One-original-snapshot S2-L replay coordinator and transactional history."""

import numpy as np

from ..constraints.cloth_self_response import (
    PhaseCapacities,
    PhaseFreeData,
    make_self_contact_load_plan,
    propose_self_contact_phase,
    response_matrix,
)
from ..core.capabilities import SolverFailure
from ..core.scene import SceneState, StageMetrics, fingerprint
from ..coupling.cloth_self_contact import SelfContactBinding
from ..coupling.ports import ParticlePointPort
from ..geometry.cloth_self_anchors import (
    DENOMINATOR,
    SCHEMA,
    make_self_contact_anchors,
    proof_data,
    q,
    require_area_path,
    require_fixed_support,
)
from ..geometry.cloth_self_candidates import candidate_self_pairs
from ..geometry.prism_prism_manifold import PrismPairIdentity, prism_prism_manifold
from ..geometry.prism_prism_sweep import prism_prism_sweep, verify_prism_prism_sweep
from ..validation.cloth_self_response import SelfContactResponseLedger, phase_residual, upper
from ._cloth_self_contact_replay import SelfContactReplay, require_selected_pre


class ClothSelfContactWorkspace:
    def __init__(self, program, state, inputs, h, settings, context):
        self.program, self.h, self.settings, self.context = program, h, settings, context
        self.replay = SelfContactReplay(program, state, inputs, h, settings, context)
        self.initial, self.inputs = self.replay.initial, self.replay.inputs
        self.domain, self.model = program.domain, program.model
        self.closed = self.predicted = self.prepared = self.published = False
        self.final = self.receipt = None
        self.history = {
            f.name.split("/")[-1]: self.initial.history[f.name] for f in program.own_fields
        }
        self.history_owners = self._validate_history()
        self.anchors = self._anchors(self.initial, initial_only=True)
        self.held = np.zeros((len(self.anchors.owner_tokens), 3), np.float64)
        self.terminal = self.held.copy()
        self.rebuilds = 0

    def _open(self):
        if self.closed:
            raise RuntimeError("self-contact workspace closed")

    def _validate_history(self):
        data = self.history
        active = data["active"]
        if np.any(data["window_positive"] < 0) or int(data["epoch"]) < 0:
            raise SolverFailure("invalid noncancelling history counters")
        result = {"next_generation": int(data["epoch"]) + 1}
        for name in (
            "face_ids",
            "node_ids",
            "counts",
            "offsets",
            "offset_bits",
            "generations",
            "phase_normals",
            "phase_world",
        ):
            if np.any(data[name][~active] != 0):
                raise SolverFailure("dirty inactive contact history")
        for row in np.flatnonzero(active):
            ids, faces = data["node_ids"][row], data["face_ids"][row]
            if (
                np.any(faces < 0)
                or np.any(faces >= len(self.program.topology.triangles))
                or faces[0] == faces[1]
            ):
                raise SolverFailure("history face identity out of range")
            endpoints = []
            for side in range(2):
                expected = tuple(map(int, self.program.topology.triangles[faces[side]]))
                actual = tuple(map(int, ids[side]))
                if actual not in (
                    expected,
                    expected[1:] + expected[:1],
                    expected[2:] + expected[:2],
                ):
                    raise SolverFailure("history reversed or changed oriented triangle")
                counts = tuple(map(int, data["counts"][row, side]))
                if min(counts) < 0 or sum(counts) != DENOMINATOR:
                    raise SolverFailure("history material address invalid")
                offset = data["offsets"][row, side]
                bits = int(np.array(offset).view(np.int64))
                if (
                    bits != data["offset_bits"][row, side]
                    or abs(offset) > self.program.topology.half_thickness_m[faces[side]]
                ):
                    raise SolverFailure("history thickness/bit identity invalid")
                start = min(range(3), key=lambda i: actual[i:] + actual[:i])
                endpoints.append(
                    (
                        int(faces[side]),
                        actual[start:] + actual[:start],
                        counts[start:] + counts[:start],
                        bits,
                    )
                )
            key = (SCHEMA, self.domain, tuple(sorted(endpoints)))
            generation = int(data["generations"][row])
            if key in result or not 0 <= generation <= int(data["epoch"]):
                raise SolverFailure("duplicate or invalid history generation")
            result[key] = generation
        return result

    def _identities(self, state):
        x0 = self.initial.domains[self.domain].position
        x1 = state.domains[self.domain].position
        pairs = candidate_self_pairs(self.program.topology, x0, x1)
        if len(pairs) > self.program.limits.max_contacts:
            raise SolverFailure("candidate pair resource exhausted")
        return tuple(
            PrismPairIdentity(
                self.domain, np.array(pair, np.int64), self.program.topology.triangles[list(pair)]
            )
            for pair in pairs
        )

    def _manifold(self, state, identity):
        x = state.domains[self.domain].position
        return prism_prism_manifold(
            x[identity.node_ids[0]],
            x[identity.node_ids[1]],
            identity=identity,
            half_thickness_m=self.program.topology.half_thickness_m[identity.face_ids],
            proof_limits=self.program.limits.geometry_limits,
        )

    def _anchors(self, state, *, initial_only=False):
        identities = self._identities(state)
        manifolds = []
        for identity in identities:
            first = self._manifold(self.initial, identity)
            if first.exact_gap_bounds is None:
                raise SolverFailure("initial candidate geometry is unresolved")
            if first.exact_gap_bounds[0] < -q(self.program.budgets.penetration_m):
                raise SolverFailure("initial state already exceeds prism penetration budget")
            if initial_only:
                manifolds.append(first)
                continue
            current = self._manifold(state, identity)
            pair_key = tuple(sorted(map(int, identity.face_ids)))
            charged = any(
                np.any(self.held[row] != 0)
                for row, pair in enumerate(self.anchors.pair_index)
                if tuple(sorted(map(int, self.anchors.identities[int(pair)].face_ids))) == pair_key
            )
            penetrating_trial = (
                current.exact_gap_bounds is not None and current.exact_gap_bounds[1] < 0
            )
            if first.exact_gap_bounds[1] <= 0 and (charged or penetrating_trial):
                manifolds.append(first)
            else:
                manifolds.append(current)
        return make_self_contact_anchors(
            manifolds,
            identities=identities,
            history=self.history_owners,
            limits=self.program.limits,
            budgets=self.program.budgets,
        )

    def _binding(self, state):
        base = ParticlePointPort(
            self.model,
            state.domains[self.domain],
            np.arange(self.model.particle_count, dtype=np.int64),
            domain=self.domain,
        )
        a = self.anchors
        return SelfContactBinding(base, a.triangles, a.weights, a.offsets, a.pairs)

    def _closed_mask(self):
        values = []
        for pair in self.anchors.pair_index:
            identity = self.anchors.identities[int(pair)]
            bounds = self._manifold(self.initial, identity).exact_gap_bounds
            values.append(bounds is not None and bounds[1] <= 0)
        return np.array(values, dtype=bool)

    def _data(self, state, world, *, phase, initial_binding, represented):
        binding = self._binding(state)
        normals = self.anchors.normals
        k = len(world)
        points = binding.world_points[binding.pairs]
        gaps = np.sum((points[:, 0] - points[:, 1]) * normals, axis=1)
        w = response_matrix(initial_binding)
        scales = np.array(
            [normals[i] @ w[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] @ normals[i] for i in range(k)],
            np.float64,
        )
        held_normal = np.sum(self.held * normals, axis=1)
        return dict(
            world=world,
            normals=normals,
            velocity=binding.operator.relative_velocities(),
            gaps=gaps,
            scale=scales,
            initial_closed=self._closed_mask(),
            held_normal=held_normal,
            represented_nodes=represented,
            points_initial=initial_binding.world_points[initial_binding.pairs],
            points_final=points,
        )

    def _geometry(self, state):
        x0, x1 = (s.domains[self.domain].position for s in (self.initial, state))
        for face in self.program.topology.triangles:
            require_area_path(x0[face], x1[face])
        items = []
        for identity in self._identities(state):
            a, b = (np.stack((x0[ids], x1[ids])) for ids in identity.node_ids)
            thickness = self.program.topology.half_thickness_m[identity.face_ids]
            args = dict(
                identity=identity,
                half_thickness_m=thickness,
                h=self.h,
                penetration_budget_m=self.program.budgets.penetration_m,
                proof_limits=self.program.limits.geometry_limits,
            )
            certificate = prism_prism_sweep(a, b, **args)
            if not verify_prism_prism_sweep(certificate, a, b, **args):
                raise SolverFailure("actual affine path has no complete two-normal certificate")
            items.append(
                dict(
                    identity=identity,
                    thickness=thickness,
                    path_a=a,
                    path_b=b,
                    certificate=certificate,
                )
            )
        return tuple(items)

    def _propose(self, binding, data, phase):
        caps = PhaseCapacities(
            data["initial_closed"],
            np.ones(len(self.held), bool),
            np.maximum(data["held_normal"], 0.0),
            self.anchors.owner_tokens,
        )
        free = PhaseFreeData(
            data["velocity"], data["gaps"], data["world"], self.h, self.program.budgets.velocity_mps
        )
        proposal = propose_self_contact_phase(
            binding,
            self.anchors.normals,
            free,
            phase=phase,
            capacities=caps,
            law=self.program.law,
            limits=self.program.limits,
        )
        self.replay.record(dict(kind="proposal", phase=phase, proof=proposal))
        return proposal

    def predict(self):
        self._open()
        if self.predicted:
            raise RuntimeError("self-contact prediction already complete")
        try:
            self._solve()
            self.predicted = True
        except BaseException as error:
            self.replay.record(
                dict(
                    kind="rejected-attempt",
                    reason=str(error),
                    original_state=self.initial,
                    original_inputs=self.inputs,
                )
            )
            raise

    def _solve(self):
        program = self.program
        binding0 = self._binding(self.initial)
        held_plan = make_self_contact_load_plan(binding0, self.held, phase="held")
        selected = self.replay.evaluate(held_plan)
        for iteration in range(program.limits.max_outer_replays):
            pre = selected["terminal"]["pre_impulse_state"]
            proposed_anchors = self._anchors(pre)
            if proposed_anchors.owner_tokens != self.anchors.owner_tokens:
                if np.any(self.held != 0):
                    raise SolverFailure("charged complete clipping owners changed")
                if self.rebuilds >= program.limits.max_rebuilds:
                    raise SolverFailure("material-anchor rebuild budget exhausted")
                self.rebuilds += 1
                self.anchors = proposed_anchors
                self.held = np.zeros((len(self.anchors.owner_tokens), 3), np.float64)
                self.terminal = self.held.copy()
                binding0 = self._binding(self.initial)
                held_plan = make_self_contact_load_plan(binding0, self.held, phase="held")
                selected = self.replay.evaluate(held_plan)
                continue
            if not np.any(self.held != 0):
                # An unloaded owner may keep its material key while the exact
                # terminal ray/geometry changes. Refresh its complete evidence.
                self.anchors = proposed_anchors
                binding0 = self._binding(self.initial)
                held_plan = make_self_contact_load_plan(binding0, self.held, phase="held")
            data = self._data(
                pre,
                self.held,
                phase="held",
                initial_binding=binding0,
                represented=selected["held_receipt"]["represented_nodes"],
            )
            score, _ = phase_residual(
                data, phase="held", law=program.law, h=self.h, budgets=program.budgets
            )
            geometry = None
            if score <= program.budgets.velocity_mps:
                try:
                    geometry = self._geometry(pre)
                    require_fixed_support(
                        self.anchors,
                        self.initial.domains[self.domain].position,
                        pre.domains[self.domain].position,
                        np.any(self.held != 0, axis=1),
                    )
                except SolverFailure:
                    geometry = None
            if geometry is not None:
                break
            proposal = self._propose(binding0, data, "held")
            old = self.held.copy()
            accepted = False
            for halving in range(program.limits.max_line_search_halvings + 1):
                candidate = old + (proposal.world_impulses - old) / (2**halving)
                if np.array_equal(candidate, old):
                    continue
                plan = make_self_contact_load_plan(binding0, candidate, phase="held")
                trial = self.replay.evaluate(plan)
                current = trial["terminal"]["pre_impulse_state"]
                trial_data = self._data(
                    current,
                    candidate,
                    phase="held",
                    initial_binding=binding0,
                    represented=trial["held_receipt"]["represented_nodes"],
                )
                new_score, _ = phase_residual(
                    trial_data, phase="held", law=program.law, h=self.h, budgets=program.budgets
                )
                if new_score < score or new_score <= program.budgets.velocity_mps / 64:
                    self.held, held_plan, selected = candidate, plan, trial
                    accepted = True
                    break
            if not accepted:
                raise SolverFailure(
                    "held replay line search exhausted with unresolved geometry/law"
                )
        else:
            raise SolverFailure("held continuation replay resource exhausted")
        self.selected_pre = pre
        binding1 = self._binding(pre)
        self.terminal = np.zeros_like(self.held)
        terminal_data = self._data(
            pre,
            self.terminal,
            phase="terminal",
            initial_binding=binding1,
            represented=np.zeros_like(held_plan.node_impulses),
        )
        proposal = self._propose(binding1, terminal_data, "terminal")
        self.terminal = proposal.world_impulses
        terminal_plan = make_self_contact_load_plan(binding1, self.terminal, phase="terminal")
        final = self.replay.evaluate(held_plan, terminal_plan)
        require_selected_pre(final, selected)
        final_state = final["trial"].result.state
        held_data = self._data(
            pre,
            self.held,
            phase="held",
            initial_binding=binding0,
            represented=final["held_receipt"]["represented_nodes"],
        )
        terminal_data = self._data(
            final_state,
            self.terminal,
            phase="terminal",
            initial_binding=binding1,
            represented=final["terminal_receipt"]["represented_nodes"],
        )
        held_data.update(
            plan=held_plan,
            actual_receipt=final["held_receipt"],
            witness=final["held_mapping"]["capacity_witness"],
        )
        terminal_data.update(
            plan=terminal_plan,
            actual_receipt=final["terminal_receipt"],
            witness=final["terminal_mapping"]["capacity_witness"],
        )
        ledger = SelfContactResponseLedger(
            self.model,
            program.topology,
            law=program.law,
            material=program.materials,
            proof_limits=program.limits.geometry_limits,
            inner_settings=program.inner_settings,
            context=self.context,
        )
        physical = ledger.require(
            self.initial,
            pre,
            final_state,
            original_inputs=self.inputs,
            h=self.h,
            geometry=geometry,
            anchors=self.anchors,
            held=held_data,
            terminal=terminal_data,
            material_evidence=final["material_path"],
            budgets=program.budgets,
        )
        history = self._history_after(final_state, physical)
        self.final = SceneState(final_state.domains, history)
        self.receipt = dict(
            scope="S2-L-nonincident-unique-ray-constant-normal-held-v1",
            backend=self.context.backend,
            h=self.h,
            original_state=self.initial,
            original_inputs=self.inputs,
            selected_pre=pre,
            final_state=self.final,
            held_plan=held_plan,
            terminal_plan=terminal_plan,
            anchors=proof_data(self.anchors),
            physical=physical,
            replay=final,
            geometry=tuple(proof_data(item) for item in geometry),
            replays=self.replay.count,
            rebuilds=self.rebuilds,
            native_fallback=False,
        )
        self.replay.record(dict(kind="accepted-candidate", **self.receipt))

    def _history_after(self, final, physical):
        program, a = self.program, self.anchors
        result = program.initial_history()
        data = {f.name.split("/")[-1]: result[f.name] for f in program.own_fields}
        epoch = int(self.history["epoch"]) + 1
        if epoch >= np.iinfo(np.int64).max:
            raise SolverFailure("contact history generation exhausted")
        data["epoch"][...] = epoch
        for i, pair in enumerate(a.pair_index):
            identity = a.identities[int(pair)]
            data["active"][i] = True
            data["face_ids"][i] = identity.face_ids
            data["node_ids"][i] = identity.node_ids
            data["counts"][i] = a.counts[a.pairs[i]]
            data["offsets"][i] = a.offsets[a.pairs[i]]
            data["offset_bits"][i] = a.offsets[a.pairs[i]].view(np.int64)
            data["generations"][i] = a.generations[i]
            data["phase_normals"][i] = (a.normals[i], a.normals[i])
            data["phase_world"][i] = (self.held[i], self.terminal[i])
        for i, (key, budget) in enumerate(
            (
                ("positive_energy_upper_j", program.budgets.energy_j),
                ("angular_absolute_nms", program.budgets.angular_impulse_nms),
            )
        ):
            total = q(self.history["window_positive"][i]) + q(physical[key])
            if total > q(budget):
                raise SolverFailure("noncancelling whole-window physical budget exceeded")
            data["window_positive"][i] = upper(total)
        result.update({f.name: final.history[f.name] for f in program.base.history_fields})
        return result

    def position_sweep(self):
        self._open()
        if not self.predicted or self.prepared:
            raise RuntimeError("invalid self-contact position stage")

    def position_metrics(self):
        self.position_sweep()
        return StageMetrics()

    def prepare_velocity(self):
        self._open()
        if not self.predicted or self.prepared:
            raise RuntimeError("invalid self-contact prepare stage")
        self.prepared = True

    def velocity_sweep(self):
        self._open()
        if not self.prepared:
            raise RuntimeError("self-contact velocity requires prepare")
        self.published = True

    def velocity_metrics(self):
        self._open()
        if not self.published:
            raise RuntimeError("velocity sweep required")
        return StageMetrics()

    def result(self):
        self._open()
        if not self.published:
            raise RuntimeError("self-contact result not ready")
        return self.final

    def diagnostics(self):
        self._open()
        if not self.published:
            raise RuntimeError("self-contact diagnostics not ready")
        return {"self_contact": self.receipt}

    def close(self):
        if not self.closed:
            self.closed = True
            if fingerprint((self.initial, self.inputs)) != self.replay.original_token:
                raise SolverFailure("self-contact close observed original snapshot mutation")
