"""Caller-fed, per-pair guards for one real withdrawal episode.

This module neither simulates nor verifies a geometry extractor. The caller must
supply the complete, fixed native-geometry and declared-material pair catalogue,
including initially clear pairs. Native source geometry and a pressure material
are distinct keys and retain their own penetration definitions. Maximum depth,
minimum separating normal speed and the noncancelling normal-load sum are per
key; aggregation across keys cannot excuse a worsening contact.

Forces belong to the accepted step on its input geometry, possibly evaluated with
an implicit velocity. Input/output normal speeds instead belong to the actual
input/output q,v. These are discrete samples, not swept/continuous-clearance
proofs. The original table supports are explicitly exempt from release progress.

Seed with the preceding accepted step, not invented forces at its endpoint. Old
area-face evidence may be unknown (None); the first real input binds that missing
baseline explicitly. No such omission is accepted in subsequent step evidence.
Trial assessment does not mutate history; the execution owner must use its real
checkpoint/restore protocol for unselected physical trials. A valid but unsafe
accepted step is retained, never described as a physics rollback.
"""

from dataclasses import dataclass
from math import isfinite
from numbers import Real

_FORCE_PHASE = "accepted-step-input-geometry"
_GEOMETRY_PHASE = "input-and-output-configuration"


@dataclass(frozen=True, slots=True)
class ReleasePairKey:
    kind: str
    identity: tuple[str, ...]
    body_ids: tuple[int, int]

    def __post_init__(self):
        object.__setattr__(self, "identity", tuple(self.identity))
        object.__setattr__(self, "body_ids", tuple(self.body_ids))


@dataclass(frozen=True, slots=True)
class ReleasePairEvidence:
    key: ReleasePairKey
    input_depth_m: float | None
    output_depth_m: float | None
    normal_load_n: float | None
    input_normal_speed_m_s: float | None
    output_normal_speed_m_s: float | None
    input_contact: bool | None
    output_contact: bool | None
    output_strictly_separated: bool | None
    force_on_a_world: tuple[float, float, float] | None
    force_on_b_world: tuple[float, float, float] | None

    def __post_init__(self):
        for name in ("force_on_a_world", "force_on_b_world"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(value))


@dataclass(frozen=True, slots=True)
class ReleaseStep:
    world: int
    input_tick: int
    output_tick: int
    input_time: float
    output_time: float
    input_q: tuple[float, ...]
    input_v: tuple[float, ...]
    output_q: tuple[float, ...]
    output_v: tuple[float, ...]
    pairs: tuple[ReleasePairEvidence, ...]
    publication: str = "accepted"
    complete_pair_catalog: bool = True
    force_phase: str = _FORCE_PHASE
    geometry_phase: str = _GEOMETRY_PHASE

    def __post_init__(self):
        for name in ("input_q", "input_v", "output_q", "output_v", "pairs"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class ReleasePolicy:
    max_steps: int
    depth_tolerance_m: float
    load_tolerance_n: float
    inward_tolerance_m_s: float
    balance_tolerance_n: float


@dataclass(frozen=True, slots=True)
class ReleaseViolation:
    code: str
    key: ReleasePairKey


@dataclass(frozen=True, slots=True)
class ReleaseDecision:
    status: str
    admissible: bool
    release_complete: bool
    regular_control_authorized: bool
    committed: bool
    violations: tuple[ReleaseViolation, ...]
    newly_bound_keys: tuple[ReleasePairKey, ...]
    step: ReleaseStep
    sampled_only: bool = True


@dataclass(frozen=True, slots=True)
class _State:
    initial: ReleaseStep | None = None
    last: ReleaseStep | None = None
    contacts: tuple = ()
    history: tuple[ReleaseDecision, ...] = ()
    status: str = "not_started"


def _require(ok, detail):
    if not ok:
        raise ValueError("release monitor: " + detail)


def _number(value, name, *, nonnegative=False):
    _require(not isinstance(value, bool) and isinstance(value, Real), name + " must be real")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("release monitor: " + name + " must be finite") from error
    _require(isfinite(result) and (not nonnegative or result >= 0), name + " is invalid")
    return result


def _key(key):
    _require(isinstance(key, ReleasePairKey), "pair key type")
    _require(key.kind in ("native-geometry", "area-material"), "pair geometry/material kind")
    _require(
        1 <= len(key.identity) <= 16 and all(type(x) is str and x for x in key.identity),
        "complete pair identity",
    )
    _require(
        len(key.body_ids) == 2
        and all(type(x) is int and x >= 0 for x in key.body_ids)
        and key.body_ids[0] != key.body_ids[1],
        "two distinct body identities",
    )


def _vector(value, name):
    _require(len(value) > 0, name + " must not be empty")
    for item in value:
        _number(item, name)


def _pair(row, policy, *, seed):
    _require(isinstance(row, ReleasePairEvidence), "pair evidence type")
    _key(row.key)
    partial = seed and row.key.kind == "area-material"
    for name in ("input_depth_m", "output_depth_m", "normal_load_n"):
        value = getattr(row, name)
        if value is None:
            _require(partial, "missing " + name)
        else:
            _number(value, name, nonnegative=True)
    for name in ("input_contact", "output_contact", "output_strictly_separated"):
        value = getattr(row, name)
        _require(type(value) is bool or (partial and value is None), "missing/invalid " + name)
    for phase in ("input", "output"):
        speed = getattr(row, phase + "_normal_speed_m_s")
        touching = getattr(row, phase + "_contact")
        depth = getattr(row, phase + "_depth_m")
        if speed is None:
            _require(partial or touching is False, "missing contact normal speed")
        else:
            _number(speed, phase + " normal speed")
        _require(touching is not False or depth in (None, 0), "depth/contact contradiction")
    _require(
        row.input_contact is not False or row.normal_load_n in (None, 0),
        "load/contact contradiction",
    )
    _require(
        row.output_strictly_separated is not True
        or (row.output_contact is False and row.output_depth_m == 0),
        "strict separation contradicts output contact/depth",
    )
    for force in (row.force_on_a_world, row.force_on_b_world):
        if force is None:
            _require(partial, "missing actual bilateral force")
        else:
            _require(len(force) == 3, "world force must have three components")
            _vector(force, "world force")
    if row.force_on_a_world is not None and row.force_on_b_world is not None:
        _require(
            all(
                abs(a + b) <= policy.balance_tolerance_n
                for a, b in zip(row.force_on_a_world, row.force_on_b_world, strict=True)
            ),
            "bilateral action/reaction mismatch",
        )


def _frame(frame, policy, *, seed=False):
    _require(isinstance(frame, ReleaseStep), "step evidence type")
    _require(type(frame.world) is int and frame.world >= 0, "world identity")
    _require(
        type(frame.input_tick) is int
        and type(frame.output_tick) is int
        and frame.input_tick >= 0
        and frame.output_tick == frame.input_tick + 1,
        "one real input/output tick required",
    )
    _require(frame.complete_pair_catalog is True, "complete pair catalogue required")
    _require(
        frame.force_phase == _FORCE_PHASE and frame.geometry_phase == _GEOMETRY_PHASE,
        "force/geometry phase mismatch",
    )
    _require(frame.publication in ("accepted", "trial"), "publication kind")
    a = _number(frame.input_time, "input time", nonnegative=True)
    b = _number(frame.output_time, "output time", nonnegative=True)
    _require(b > a, "positive real step time")
    for name in ("input_q", "input_v", "output_q", "output_v"):
        _vector(getattr(frame, name), name)
    _require(
        len(frame.input_q) == len(frame.output_q) and len(frame.input_v) == len(frame.output_v),
        "input/output state dimensions",
    )
    _require(1 <= len(frame.pairs) <= 256, "bounded nonempty pair catalogue")
    for row in frame.pairs:
        _pair(row, policy, seed=seed)
    _require(len({row.key for row in frame.pairs}) == len(frame.pairs), "duplicate pair identity")


def _initial_contact(row):
    if row.input_contact is True or row.output_contact is True or (row.normal_load_n or 0) > 0:
        return True
    if row.input_contact is None or row.output_contact is None or row.normal_load_n is None:
        return None
    return False


class ReleaseMonitor:
    """One sequential owner; selected physical history and trials stay distinct.

    ``start`` consumes the last accepted step before withdrawal. ``assess`` takes
    a trial from that same endpoint and never advances this monitor. ``observe``
    takes a selected accepted step, including an unsafe one, and retains it.
    Invalid/incomplete evidence raises without changing monitor state.

    ``released`` stays guarded until the same withdrawal path ends: any renewed
    contact, positive load or lost strict separation is unsafe. It never grants
    regular-control permission; a separate ordinary-path audit is still needed.
    No tolerance is silently added: all five defaults are exact nonincrease or
    non-inward comparisons. The step cap is cumulative across the whole episode.
    """

    __slots__ = ("_policy", "_support_keys", "_state")

    def __init__(
        self,
        *,
        support_keys=(),
        max_steps=256,
        depth_tolerance_m=0.0,
        load_tolerance_n=0.0,
        inward_tolerance_m_s=0.0,
        balance_tolerance_n=0.0,
    ):
        _require(not hasattr(self, "_state"), "an initialized monitor cannot restart")
        _require(type(max_steps) is int and 1 <= max_steps <= 100000, "bounded max_steps")
        values = tuple(
            _number(value, name, nonnegative=True)
            for name, value in (
                ("depth tolerance", depth_tolerance_m),
                ("load tolerance", load_tolerance_n),
                ("inward tolerance", inward_tolerance_m_s),
                ("balance tolerance", balance_tolerance_n),
            )
        )
        supports = tuple(support_keys)
        for key in supports:
            _key(key)
            _require(key.kind == "native-geometry", "only original native support can be exempt")
        _require(len(set(supports)) == len(supports), "duplicate support identity")
        object.__setattr__(self, "_policy", ReleasePolicy(max_steps, *values))
        object.__setattr__(self, "_support_keys", supports)
        object.__setattr__(self, "_state", _State())

    def __setattr__(self, name, value):
        raise AttributeError("release monitor state is read-only")

    def __delattr__(self, name):
        raise AttributeError("release monitor state is read-only")

    @property
    def policy(self):
        return self._policy

    @property
    def started(self):
        return self._state.initial is not None

    @property
    def status(self):
        return self._state.status

    @property
    def history(self):
        return self._state.history

    @property
    def last_step(self):
        return self._state.last

    @property
    def terminal(self):
        return (
            self.status in ("unsafe", "step_budget") or len(self.history) >= self.policy.max_steps
        )

    def start(self, step):
        _require(not self.started, "start is accepted only once")
        _require(step.publication == "accepted", "seed publication must be accepted")
        _frame(step, self.policy, seed=True)
        contacts = tuple((row.key, _initial_contact(row)) for row in step.pairs)
        allowed = dict(contacts)
        _require(
            all(allowed.get(key) is True for key in self._support_keys),
            "support exemption requires actual initial support",
        )
        _require(
            any(value is True and key not in self._support_keys for key, value in contacts),
            "initial non-support release contact required",
        )
        object.__setattr__(self, "_state", _State(step, step, contacts, (), "continue"))

    def _assess(self, step, *, committed):
        _require(self.started, "start is required")
        _require(not self.terminal, "terminal monitor cannot continue")
        _require(
            step.publication == ("accepted" if committed else "trial"),
            "publication differs from assessment/observation",
        )
        _frame(step, self.policy)
        previous, policy = self._state.last, self.policy
        _require(
            step.world == previous.world
            and step.input_tick == previous.output_tick
            and step.input_time == previous.output_time
            and step.input_q == previous.output_q
            and step.input_v == previous.output_v,
            "continuous accepted endpoint required",
        )
        prior = {row.key: row for row in previous.pairs}
        _require({row.key for row in step.pairs} == set(prior), "pair catalogue/identity changed")
        allowed = dict(self._state.contacts)
        violations, newly_bound, release_pairs = [], [], []

        def fail(code, key):
            item = ReleaseViolation(code, key)
            if item not in violations:
                violations.append(item)

        for row in step.pairs:
            key, old = row.key, prior[row.key]
            prior_depth = (
                old.output_depth_m if old.output_depth_m is not None else old.input_depth_m
            )
            if allowed[key] is None or old.normal_load_n is None or prior_depth is None:
                newly_bound.append(key)
            if allowed[key] is None:
                # Missing old area-face evidence binds at the new INPUT only;
                # a contact first appearing at its output is still a new collision.
                allowed[key] = row.input_contact or row.normal_load_n > 0
            if not allowed[key]:
                if row.input_contact or row.output_contact or row.normal_load_n > 0:
                    fail("new_contact", key)
                continue
            if key in self._support_keys:
                continue
            release_pairs.append(row)
            if (
                old.normal_load_n is not None
                and row.normal_load_n > old.normal_load_n + policy.load_tolerance_n
            ):
                fail("normal_load_increased", key)
            if (
                prior_depth is not None
                and row.input_depth_m > prior_depth + policy.depth_tolerance_m
                or row.output_depth_m > row.input_depth_m + policy.depth_tolerance_m
            ):
                fail("penetration_increased", key)
            for speed in (row.input_normal_speed_m_s, row.output_normal_speed_m_s):
                if speed is not None and speed < -policy.inward_tolerance_m_s:
                    fail("inward_motion", key)
            if self.status == "released":
                if row.input_contact or row.output_contact:
                    fail("contact_reappeared", key)
                if not row.output_strictly_separated:
                    fail("separation_lost", key)
                if row.normal_load_n > 0:
                    fail("normal_load_reappeared", key)
        clear = bool(release_pairs) and all(
            row.normal_load_n == 0 and row.output_strictly_separated for row in release_pairs
        )
        count = len(self.history) + 1
        status = (
            "unsafe"
            if violations
            else "released"
            if clear
            else "step_budget"
            if count >= policy.max_steps
            else "continue"
        )
        decision = ReleaseDecision(
            status,
            status in ("continue", "released"),
            status == "released",
            False,
            committed,
            tuple(violations),
            tuple(newly_bound),
            step,
        )
        return decision, tuple(allowed.items())

    def assess(self, step):
        return self._assess(step, committed=False)[0]

    def observe(self, step):
        decision, contacts = self._assess(step, committed=True)
        object.__setattr__(
            self,
            "_state",
            _State(
                self._state.initial, step, contacts, self.history + (decision,), decision.status
            ),
        )
        return decision
