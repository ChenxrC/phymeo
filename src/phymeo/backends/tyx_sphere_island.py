"""Bounded world-local, frictionless contact islands of translating finite spheres."""

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from numbers import Integral
from threading import RLock
from types import MappingProxyType

import numpy as np

from ._sphere_island_device import SphereIslandDevice
from ._world_step_layout import FreeSphere, SpherePlaneWorld, as_fp32, world_key

_REASONS = {
    1: "contact_capacity",
    2: "undefined_normal",
    3: "intermediate_crossing",
    4: "position_residual",
    5: "velocity_residual",
    6: "nonfinite_candidate",
    7: "separated_contact_chart",
    8: "contact_energy_gain",
    9: "momentum_residual",
}
_POS_TOL = 1e-5
_VEL_TOL = 2e-5


@dataclass(frozen=True)
class SphereIslandWorld:
    """One to eight positive-mass finite spheres. Body IDs are local to this world.

    The optional common plane is horizontal in world coordinates (+Z up). All
    pairs use the same normal restitution and have zero friction and no spin.
    """

    bodies: tuple
    plane_height: float | None = None
    restitution: float = 0.0

    def __post_init__(self):
        entries = []
        ids = set()
        for key, sphere in self.bodies:
            if len(entries) >= 8:
                raise ValueError("an island requires 1 to 8 spheres")
            key = world_key(key)
            if key in ids:
                raise ValueError("duplicate local sphere key")
            if type(sphere) is not FreeSphere:
                raise TypeError(
                    "each body must be a FreeSphere; configure the shared plane on the island"
                )
            ids.add(key)
            entries.append((key, sphere))
        if not entries:
            raise ValueError("an island requires 1 to 8 spheres")
        e = float(as_fp32(self.restitution, "restitution", ()))
        if not 0 <= float(self.restitution) <= 1:
            raise ValueError("restitution must lie in [0,1]")
        object.__setattr__(self, "bodies", tuple(sorted(entries)))
        object.__setattr__(self, "restitution", e)
        if self.plane_height is not None:
            object.__setattr__(
                self, "plane_height", float(as_fp32(self.plane_height, "plane height", ()))
            )


@dataclass(frozen=True)
class SphereIslandBody:
    position: tuple
    velocity: tuple


@dataclass(frozen=True)
class SphereIslandState:
    world: int
    tick: int
    bodies: Mapping


@dataclass(frozen=True)
class SphereIslandContact:
    """A->B normal at the accepted pose. A missing first ID is the fixed floor."""

    world: int
    first: int | None
    second: int
    normal: tuple
    point_first: tuple
    point_second: tuple
    initial_gap_m: float
    free_gap_m: float
    gap_m: float
    normal_impulse_n_s: float
    target_speed_m_s: float


@dataclass(frozen=True)
class SphereIslandIssue:
    world: int
    first: int
    second: int
    reason: str
    initial_gap_m: float
    free_gap_m: float
    closest_free_fraction: float
    closest_free_gap_m: float


@dataclass(frozen=True)
class SphereIslandEnergy:
    body_ids: tuple
    normal_work_j: float
    relative_kinetic_change_j: float
    work_allowance_j: float
    momentum_work_error_j: float


@dataclass(frozen=True)
class SphereIslandResult:
    accepted: bool
    reason: str | None
    required_contacts: int | None
    state: SphereIslandState
    contacts: tuple
    issues: tuple
    position_residual_m: float | None
    velocity_residual_m_s: float | None
    contact_kinetic_change_j: float | None
    momentum_residual_n_s: float | None
    energy_components: tuple[SphereIslandEnergy, ...] = ()


def _vector(a):
    return tuple(map(float, a))


def _issue(world, left, right, code, old, free, radii, ids):
    start = old[right, :3] - old[left, :3]
    finish = free[right] - free[left]
    travel = finish - start
    size = radii[right] + radii[left]
    speed2 = float(travel @ travel)
    t = float(np.clip(-(start @ travel) / speed2, 0, 1)) if speed2 else 0.0
    g0, g1 = np.linalg.norm(start) - size, np.linalg.norm(finish) - size
    middle = np.linalg.norm(start + t * travel) - size
    if not code:
        if np.linalg.norm(start) == 0 or np.linalg.norm(finish) == 0:
            code = 2
        elif 0 < t < 1 and middle <= 0 and (g1 > 0 or start @ finish <= 0):
            code = 3
    if code not in (2, 3):
        return None
    return SphereIslandIssue(
        world, ids[left], ids[right], _REASONS[code], float(g0), float(g1), t, float(middle)
    )


def _component_energy(ids, mass, old_v, new_v, contacts, residual_vectors):
    """A distant body or a common tangential boost cannot pay for a contact."""
    groups = [{body} for body in ids]
    loaded = [c for c in contacts if c.normal_impulse_n_s > 0]
    for c in loaded:
        if c.first is None:
            continue
        first = next(group for group in groups if c.first in group)
        second = next(group for group in groups if c.second in group)
        if first is not second:
            first.update(second)
            groups.remove(second)
    slots = dict(zip(ids, range(len(ids)), strict=True))
    reports = []
    for group in groups:
        rows = [c for c in loaded if c.second in group]
        if not rows:
            continue
        members = sorted(group)
        indices = np.array([slots[body] for body in members])
        reference = np.average(old_v[indices], axis=0, weights=mass[indices])
        if any(c.first is None for c in rows):
            # A static horizontal plane can supply a normal impulse, but no tangential impulse.
            reference[2] = 0
        local_mid = (old_v[indices] - reference) / 2 + (new_v[indices] - reference) / 2
        local_delta = new_v[indices] - old_v[indices]
        kinetic = float(np.sum(mass[indices, None] * local_delta * local_mid))
        work, normal_scale, impulse_sum = 0.0, 0.0, 0.0
        for c in rows:
            b = slots[c.second]
            incoming, outgoing = old_v[b], new_v[b]
            if c.first is not None:
                a = slots[c.first]
                incoming, outgoing = incoming - old_v[a], outgoing - new_v[a]
            n = np.asarray(c.normal)
            u0, u1 = float(incoming @ n), float(outgoing @ n)
            impulse_sum += c.normal_impulse_n_s
            work += c.normal_impulse_n_s * (u0 + u1) / 2
            normal_scale += c.normal_impulse_n_s * (abs(u0) + abs(u1))
        allowance = float(8 * _VEL_TOL * impulse_sum + 64 * np.finfo(np.float32).eps * normal_scale)
        # The separate, per-body momentum gate caps this actually observed error.
        binding = float(np.sum(abs(residual_vectors[indices]) * abs(local_mid)))
        reports.append(SphereIslandEnergy(tuple(members), work, kinetic, allowance, binding))
    return tuple(reports)


def _result(key, world, state, old, forces, h, staged, rows, status, errors, issues, pairs):
    code, needed = map(int, status)
    if code != 0 and code not in _REASONS:
        raise RuntimeError("invalid TYX sphere island status")
    if not 0 <= needed <= len(pairs):
        raise RuntimeError("invalid TYX sphere island contact count")
    ids = tuple(b for b, _ in world.bodies)
    mass = np.array([s.mass for _, s in world.bodies])
    radii = np.array([s.radius for _, s in world.bodies])
    gravity = np.array([s.gravity for _, s in world.bodies])
    old = old.astype(float)
    free_v = old[:, 3:] + h * (gravity + forces / mass[:, None])
    free_x = old[:, :3] + h * free_v
    pos, vel = map(float, errors)
    found = []
    for i, (left, right) in enumerate(pairs):
        if left >= 0:
            problem = _issue(key, left, right, int(issues[i]), old, free_x, radii, ids)
            if problem:
                found.append(problem)
                if not code:
                    code = 2 if problem.reason == "undefined_normal" else 3

    def refused():
        return SphereIslandResult(
            False,
            _REASONS[code],
            None if code in (2, 3, 6) else needed,
            state,
            (),
            tuple(found),
            None if code in (2, 3, 6) or not np.isfinite(pos) else pos,
            vel if code == 5 and np.isfinite(vel) else None,
            None,
            None,
        )

    if code:
        return refused()
    if (
        not np.isfinite(staged).all()
        or not np.isfinite(rows).all()
        or not np.isfinite(errors).all()
    ):
        code = 6
        return refused()
    x, v = staged[:, :3].astype(float), staged[:, 3:].astype(float)
    ledger = np.zeros_like(v)
    contacts = []
    for (left, right), row in zip(pairs, rows, strict=True):
        if left < 0:
            n = np.array([0.0, 0.0, 1.0])
            gap = x[right, 2] - radii[right] - world.plane_height
            first_point = (*x[right, :2], world.plane_height)
            relative = v[right]
            reciprocal = 1 / mass[right]
        else:
            direction = x[right] - x[left]
            distance = np.linalg.norm(direction)
            if not distance:
                code = 2
                return refused()
            n = direction / distance
            gap = distance - radii[left] - radii[right]
            first_point = _vector(x[left] + radii[left] * n)
            relative = v[right] - v[left]
            reciprocal = 1 / mass[right] + 1 / mass[left]
        pos = max(pos, -float(gap))
        if row[4] not in (0.0, 1.0):
            raise RuntimeError("invalid TYX active sphere pair")
        if not row[4]:
            if gap <= 0:
                vel = max(vel, -float(relative @ n))
            continue
        if gap > _POS_TOL or np.linalg.norm(n - row[:3]) > 1e-5:
            code = 7
            return refused()
        lam = float(row[5])
        target = float(row[6])
        speed = float(relative @ n)
        vel = max(
            vel,
            abs(lam - max(0.0, lam + (target - speed) / reciprocal)) * reciprocal,
            -lam * reciprocal,
        )
        impulse = lam * n
        ledger[right] += impulse
        if left >= 0:
            ledger[left] -= impulse
        contacts.append(
            SphereIslandContact(
                key,
                None if left < 0 else ids[left],
                ids[right],
                _vector(n),
                first_point,
                _vector(x[right] - radii[right] * n),
                float(row[8]),
                float(row[7]),
                float(gap),
                lam,
                target,
            )
        )
    if len(contacts) != needed:
        raise RuntimeError("incomplete TYX sphere island contact set")
    if pos > _POS_TOL:
        code = 4
    elif vel > _VEL_TOL:
        code = 5
    residual_vectors = mass[:, None] * (v - free_v) - ledger
    error = float(np.max(np.linalg.norm(residual_vectors, axis=1)))
    change = float(0.5 * np.sum(mass[:, None] * (v - free_v) * (v + free_v)))
    scale = mass[:, None] * np.maximum(abs(free_v), abs(v)) + abs(ledger)
    allowance = 64 * np.finfo(np.float32).eps * scale + mass[:, None] * _VEL_TOL
    energy = _component_energy(ids, mass, free_v, v, contacts, residual_vectors)
    if not code and np.any(abs(residual_vectors) > allowance):
        code = 9
    if not code and any(
        item.normal_work_j > item.work_allowance_j
        or item.relative_kinetic_change_j > item.work_allowance_j + item.momentum_work_error_j
        for item in energy
    ):
        code = 8
    if code:
        return SphereIslandResult(
            False, _REASONS[code], needed, state, (), tuple(found), pos, vel, change, error, energy
        )
    updated = SphereIslandState(
        key,
        state.tick + 1,
        MappingProxyType(
            {b: SphereIslandBody(_vector(x[i]), _vector(v[i])) for i, b in enumerate(ids)}
        ),
    )
    return SphereIslandResult(
        True, None, needed, updated, tuple(contacts), (), pos, vel, change, error, energy
    )


class TyxSphereIslandBatch:
    """Advance bounded interacting worlds using complete device pair tables.

    HOST supplies the current checkpoints and optional one-step COM forces, then
    independently audits all final surface gaps, impulses, complementarity and
    kinetic energy. A rejected physical world stays at its previous checkpoint;
    a device failure leaves every world and its previous accepted report intact.
    """

    def __init__(self, worlds, mode="cuda", *, cache_root=None):
        if mode not in ("cuda", "interpret"):
            raise ValueError("sphere island batch requires cuda or interpret")
        self._lock, self._closed = RLock(), False
        self._mode, self._cache, self._engine = mode, cache_root, None
        self._counts = dict(
            completed_kernels=0,
            synchronizations=0,
            device_allocations=0,
            static_uploaded_bytes=0,
            uploaded_bytes=0,
            downloaded_bytes=0,
        )
        self._worlds = {}
        self._ranges = {}
        self._local_pairs = {}
        self._initial = {}
        starts = [0]
        offsets = [0]
        props = []
        sides = []
        restitution = []
        for raw, spec in worlds:
            key = world_key(raw)
            if key in self._worlds:
                raise ValueError("duplicate island world")
            if len(self._worlds) >= 128:
                raise ValueError("at most 128 island worlds per batch")
            if not isinstance(spec, SphereIslandWorld):
                raise TypeError("expected SphereIslandWorld")
            self._worlds[key] = spec
            start = len(props)
            rowstart = len(sides)
            initial = {}
            for body, s in spec.bodies:
                stop = 0.0
                if spec.plane_height is not None:
                    stop = SpherePlaneWorld(
                        s.mass, s.radius, plane_height=spec.plane_height
                    ).center_stop()
                props.append((1 / s.mass, s.mass, s.radius, *s.gravity, stop))
                initial[body] = SphereIslandBody(s.position, s.velocity)
            local = list(range(len(spec.bodies)))
            pairs = [(-1, b) for b in local] if spec.plane_height is not None else []
            pairs.extend(combinations(local, 2))
            self._local_pairs[key] = tuple(pairs)
            sides.extend((start + a if a >= 0 else -1, start + b) for a, b in pairs)
            self._ranges[key] = (slice(start, len(props)), slice(rowstart, len(sides)))
            self._initial[key] = SphereIslandState(key, 0, MappingProxyType(initial))
            starts.append(len(props))
            offsets.append(len(sides))
            restitution.append(spec.restitution)
        if not self._worlds:
            raise ValueError("at least one island world required")
        self._static = (
            as_fp32(props, "sphere properties", (len(props), 7)),
            np.asarray(starts, np.int32),
            np.asarray(offsets, np.int32),
            np.asarray(sides or [(-1, 0)], np.int32),
            np.asarray(restitution, np.float32),
        )
        self._states = dict(self._initial)
        self._last = dict.fromkeys(self._worlds)
        self._batches = 0

    def _open(self):
        if self._closed:
            raise RuntimeError("sphere island batch is closed")

    def _keys(self, keys):
        selected = tuple(self._worlds) if keys is None else tuple(world_key(k) for k in keys)
        if len(set(selected)) != len(selected):
            raise ValueError("duplicate selected world")
        for k in selected:
            if k not in self._worlds:
                raise KeyError(k)
        return selected

    def snapshot(self, keys=None):
        with self._lock:
            self._open()
            return {k: self._states[k] for k in self._keys(keys)}

    def last_result(self, key):
        with self._lock:
            self._open()
            return self._last[world_key(key)]

    def reset(self, keys=None):
        with self._lock:
            self._open()
            for k in self._keys(keys):
                self._states[k], self._last[k] = self._initial[k], None

    def _retire(self):
        if self._engine is not None:
            engine, self._engine = self._engine, None
            for k, v in engine.counts.items():
                self._counts[k] += v
            engine.close()

    def _inputs(self, h, forces, contact_capacity, iterations):
        h = float(as_fp32(h, "island substep", ()))
        if h <= 0:
            raise ValueError("positive island substep required")
        if (
            isinstance(iterations, (bool, np.bool_))
            or not isinstance(iterations, Integral)
            or not 1 <= iterations <= 128
        ):
            raise ValueError("iterations must be an integer in [1,128]")
        forces = {} if forces is None else forces
        caps = {} if contact_capacity is None else contact_capacity
        if not isinstance(forces, Mapping) or not isinstance(caps, Mapping):
            raise TypeError("expected world-keyed force and capacity mappings")
        self._keys(forces)
        self._keys(caps)
        initial = np.empty((len(self._static[0]), 6), np.float32)
        external = np.zeros((len(initial), 3), np.float32)
        capacity = []
        for key in self._worlds:
            bs, _ = self._ranges[key]
            packet = forces.get(key, {})
            if not isinstance(packet, Mapping):
                raise TypeError("expected local sphere force mapping")
            ids = self._states[key].bodies
            for local in packet:
                if world_key(local) not in ids:
                    raise KeyError((key, local))
            for i, (local, state) in enumerate(ids.items(), bs.start):
                initial[i] = (*state.position, *state.velocity)
                if local in packet:
                    external[i] = as_fp32(packet[local], "sphere COM force", (3,))
            limit = caps.get(key, len(self._local_pairs[key]))
            if (
                isinstance(limit, (bool, np.bool_))
                or not isinstance(limit, Integral)
                or not 0 <= limit <= np.iinfo(np.int32).max
            ):
                raise ValueError("invalid per-world contact capacity")
            capacity.append(limit)
        return h, initial, external, np.asarray(capacity, np.int32)

    def _trial(self, states, initial, external, capacity, h, iterations):
        """Compute candidate states only; no physical world is published here."""
        if self._engine is None:
            self._engine = SphereIslandDevice(self._mode, self._cache, *self._static)
        try:
            state, rows, status, errors, issues = self._engine.step(
                initial,
                external,
                capacity,
                h,
                int(iterations),
                _POS_TOL,
                _VEL_TOL,
            )
            result = {}
            for i, (k, spec) in enumerate(self._worlds.items()):
                if k not in states:
                    continue
                bs, ps = self._ranges[k]
                result[k] = _result(
                    k,
                    spec,
                    states[k],
                    initial[bs],
                    external[bs],
                    h,
                    state[bs],
                    rows[ps],
                    status[i],
                    errors[i],
                    issues[ps],
                    self._local_pairs[k],
                )
        except BaseException:
            self._retire()
            raise
        return result

    def _publish(self, result):
        states = self._states | {k: r.state for k, r in result.items()}
        last = self._last | {k: r for k, r in result.items() if r.accepted}
        self._states, self._last = states, last
        self._batches += 1
        return MappingProxyType(result)

    def step(self, h, *, forces=None, contact_capacity=None, iterations=96):
        """Advance all worlds; force packets (N) address world then local sphere ID."""
        with self._lock:
            self._open()
            h, initial, external, capacity = self._inputs(h, forces, contact_capacity, iterations)
            result = self._trial(self._states, initial, external, capacity, h, iterations)
            return self._publish(result)

    def step_substepped(
        self, h, *, substeps=8, keys=None, forces=None, contact_capacity=None, iterations=96
    ):
        """Advance one public tick through 1–64 dyadic substeps, publishing once per world.

        A closing, exactly touching initial configuration may first receive an impulse at t=0.
        Duration-positive intervals use the same TYX kernels and audits as step(). Contact rows
        belong to their sampled stage, with times relative to the input checkpoint. Supply keys
        to retry only selected worlds without advancing successful neighbors.
        """
        from .sphere_island_substeps import advance

        with self._lock:
            self._open()
            return advance(self, h, substeps, keys, forces, contact_capacity, iterations)

    def stats(self):
        with self._lock:
            live = {} if self._engine is None else self._engine.counts
            return dict(
                mode=self._mode,
                completed_batches=self._batches,
                worlds=len(self._worlds),
                pair_slots=int(self._static[2][-1]),
                **{k: v + live.get(k, 0) for k, v in self._counts.items()},
            )

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._retire()

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *args):
        self.close()
