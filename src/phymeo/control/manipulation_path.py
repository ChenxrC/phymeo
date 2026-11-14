"""Fixed-grid native joint-path queries, not continuous collision certificates.

Only scratch kinematics and collision detection are evaluated. Original contact
filters and compiled model data are preserved. Allowed overlap is a reference
query policy, not a physical penetration tolerance or manipulation success.
"""

import json
from dataclasses import dataclass
from hashlib import sha256
from math import ceil, isfinite
from numbers import Integral, Real

import numpy as np

from .manipulation_diagnostic import _configuration, _joints, _readonly, _require

MAX_MODEL_BYTES = 8 * 1024 * 1024
MAX_CONTACT_RECORDS = 65536


def _number(value, name):
    _require(isinstance(value, Real) and not isinstance(value, bool), name + " must be real")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(name + " must be finite") from error
    _require(isfinite(result), name + " must be finite")
    return result


def _items(values, name, minimum, maximum, rank=1):
    _require(
        type(values) in (tuple, list)
        or (isinstance(values, np.ndarray) and values.ndim == rank),
        name + " requires an explicit bounded sequence",
    )
    _require(minimum <= len(values) <= maximum, name + " exceeds its item budget")
    return tuple(values)


def _model_bytes(mj, model):
    size = mj.mj_sizeModel(model)
    _require(0 < size <= MAX_MODEL_BYTES, "model binding exceeds 8 MiB")
    binary = np.empty(size, dtype=np.uint8)
    mj.mj_saveModel(model, None, binary)
    return binary.tobytes()


@dataclass(frozen=True)
class NativeJointPathContact:
    geometry_ids: tuple[int, int]
    geometry_names: tuple[str | None, str | None]
    distance_m: float
    position_world: tuple[float, float, float]
    allowed_depth_m: float | None
    sampled_clear: bool


@dataclass(frozen=True)
class NativeJointPathSample:
    time_s: float
    q: np.ndarray
    contacts: tuple[NativeJointPathContact, ...]
    sampled_clear: bool


@dataclass(frozen=True)
class NativeJointPathAudit:
    status: str
    sampled_clear: bool
    sample_count: int
    samples: tuple[NativeJointPathSample, ...]
    input_binding: str
    worst_disallowed_depth_m: float


def audit_native_joint_path(
    model,
    times,
    q_knots,
    joint_names,
    *,
    allowed_contacts=(),
    max_time_step=0.001,
    penetration_tolerance_m=1e-4,
    max_samples=4097,
):
    """Query every fixed-grid sample of the selected scalar-joint reference.

    Times contain 2..32 strictly increasing seconds, starting at zero and ending
    by 4 seconds. Each segment has ceil(duration/max_time_step) intervals; shared
    endpoints occur once. Position uses cubic 3*u**2-2*u**3 with zero endpoint
    velocity: the joined path is C1, not generally C2. No sample is adapted away
    and a collision does not stop collection of later samples.

    Full q knots are validated in native FP64 representation. Every unselected
    coordinate, including FREE/BALL state and signed zeros, must match by bytes.
    Only named HINGE/SLIDE coordinates move, within original native joint bounds.
    Each output q is an independent immutable byte-backed copy.

    allowed_contacts is an explicit tuple of (geom_name1, geom_name2, depth_m).
    Unordered pairs must be unique; each declared depth is in [0, .05] metres.
    These allowances cover intended reference-command overlap only. Callers
    must explicitly name expected pairs; no purpose or geometry name is inferred.
    Undeclared pairs use penetration_tolerance_m, restricted to [0, 1e-4].
    All detected contacts, including allowed ones, retain native order, signed
    distance and position. Unnamed geoms retain None as their native name.

    worst_disallowed_depth_m is the maximum EXCESS beyond the applicable pair
    allowance, not raw penetration. sampled_clear refers only to these samples;
    it provides no CCD, continuous-path, force, or task-success certificate.
    input_binding hashes the full original MJB and all canonical input values.
    At most 65536 native contacts may be recorded across the whole query, including
    allowed contacts. Resource overflow raises ValueError without a classification.
    """
    import mujoco as mj

    _require(isinstance(model, mj.MjModel), "an original MuJoCo model is required")
    _require(
        0 < model.nq <= 45
        and 0 < model.nv <= 35
        and 1 <= model.nbody <= 128
        and 0 <= model.ngeom <= 128,
        "model exceeds joint-path audit dimensions",
    )
    stamps = tuple(_number(value, "time") for value in _items(times, "times", 2, 32))
    _require(
        stamps[0] == 0.0
        and stamps[-1] <= 4.0
        and all(left < right for left, right in zip(stamps[:-1], stamps[1:], strict=True)),
        "times must increase from zero and end by 4 seconds",
    )
    step = _number(max_time_step, "max_time_step")
    tolerance = _number(penetration_tolerance_m, "penetration_tolerance_m")
    _require(step > 0 and 0 <= tolerance <= 1e-4, "invalid sampling step or penetration tolerance")
    _require(
        isinstance(max_samples, Integral)
        and not isinstance(max_samples, bool)
        and 1 <= max_samples <= 4097,
        "max_samples must be an integer in 1..4097",
    )
    budget = int(max_samples)
    names = _items(joint_names, "joint_names", 1, 35)
    names, _, qa, _, lower, upper = _joints(mj, model, names)
    raw_knots = _items(q_knots, "q_knots", len(stamps), len(stamps), rank=2)
    knots = tuple(
        _configuration(
            mj,
            model,
            [_number(value, "q knot") for value in _items(row, "full q knot", model.nq, model.nq)],
        )
        for row in raw_knots
    )
    selected = list(qa)
    unselected = np.ones(model.nq, dtype=np.bool_)
    unselected[selected] = False
    stationary = knots[0][unselected].tobytes()
    for knot in knots:
        _require(knot[unselected].tobytes() == stationary, "unselected coordinates differ by bytes")
        _require(
            np.all(knot[selected] >= lower) and np.all(knot[selected] <= upper),
            "selected knot exceeds original joint limits",
        )

    _require(
        type(allowed_contacts) is tuple
        and len(allowed_contacts) <= model.ngeom * (model.ngeom - 1) // 2,
        "allowed_contacts requires an explicit bounded tuple",
    )
    allowances, declarations = {}, []
    for declaration in allowed_contacts:
        _require(type(declaration) is tuple and len(declaration) == 3, "invalid allowed contact")
        first, second, raw_depth = declaration
        _require(
            type(first) is str and first and type(second) is str and second and first != second,
            "allowed contact needs two distinct geom names",
        )
        depth = _number(raw_depth, "allowed depth")
        _require(0 <= depth <= 0.05, "allowed reference depth must lie in 0..0.05 metres")
        pair = tuple(
            sorted(int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)) for name in (first, second))
        )
        _require(pair[0] >= 0 and pair not in allowances, "unknown or duplicate allowed geom pair")
        allowances[pair] = depth
        declarations.append((*sorted((first, second)), depth))

    # Plan the entire bounded grid before any scratch kinematics/collision call.
    counts, count = [], 1
    for left, right in zip(stamps[:-1], stamps[1:], strict=True):
        intervals = (right - left) / step
        _require(isfinite(intervals) and intervals <= budget - 1, "sample budget exceeded")
        intervals = max(1, ceil(intervals))
        count += intervals
        _require(count <= budget, "sample budget exceeded")
        counts.append(intervals)
    grid = [(stamps[0], knots[0].copy())]
    for segment, intervals in enumerate(counts):
        left, right = stamps[segment : segment + 2]
        start, end = knots[segment : segment + 2]
        for index in range(1, intervals + 1):
            if index == intervals:
                stamp, q = right, end.copy()
            else:
                u = index / intervals
                smooth = u * u * (3.0 - 2.0 * u)
                stamp, q = left + (right - left) * u, start.copy()
                with np.errstate(over="ignore", invalid="ignore"):
                    q[selected] = start[selected] + smooth * (end[selected] - start[selected])
            _require(stamp > grid[-1][0], "sample times are not representable in increasing order")
            _require(
                np.isfinite(q).all()
                and np.all(q[selected] >= lower)
                and np.all(q[selected] <= upper)
                and q[unselected].tobytes() == stationary,
                "interpolated q is invalid",
            )
            grid.append((stamp, q))

    binary = _model_bytes(mj, model)
    request = json.dumps(
        dict(
            schema="native-joint-path-audit-v1",
            interpolation="cubic-smoothstep-C1",
            times=stamps,
            q_knots_fp64=[knot.tobytes().hex() for knot in knots],
            joint_names=names,
            allowed_contacts=sorted(declarations),
            max_time_step=step,
            penetration_tolerance_m=tolerance,
            max_samples=budget,
        ),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    binding = sha256(len(binary).to_bytes(8, "little") + binary + request).hexdigest()
    data, samples, worst = mj.MjData(model), [], 0.0
    total_contacts = 0
    for stamp, q in grid:
        data.qpos[:], data.time = q, stamp
        mj.mj_kinematics(model, data)
        _require(
            np.isfinite(data.geom_xpos).all() and np.isfinite(data.geom_xmat).all(),
            "nonfinite native path geometry",
        )
        mj.mj_collision(model, data)
        _require(not any(int(item.number) for item in data.warning), "native collision warning")
        _require(data.qpos.tobytes() == q.tobytes(), "native query changed candidate q")
        total_contacts += int(data.ncon)
        _require(total_contacts <= MAX_CONTACT_RECORDS, "total contact record budget exceeded")
        contacts = []
        for native in data.contact[: data.ncon]:
            pair = tuple(map(int, native.geom))
            _require(
                len(pair) == 2 and all(0 <= geom < model.ngeom for geom in pair),
                "native contact does not identify two model geoms",
            )
            distance = float(native.dist)
            point = tuple(map(float, native.pos))
            _require(isfinite(distance) and all(map(isfinite, point)), "nonfinite native contact")
            allowance = allowances.get(tuple(sorted(pair)))
            limit = tolerance if allowance is None else allowance
            excess = max(0.0, -distance - limit)
            worst = max(worst, excess)
            contacts.append(
                NativeJointPathContact(
                    pair,
                    tuple(mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom) for geom in pair),
                    distance,
                    point,
                    allowance,
                    excess == 0.0,
                )
            )
        clear = all(contact.sampled_clear for contact in contacts)
        samples.append(NativeJointPathSample(stamp, _readonly(q), tuple(contacts), clear))
    _require(_model_bytes(mj, model) == binary, "native model changed during path audit")
    clear = all(sample.sampled_clear for sample in samples)
    return NativeJointPathAudit(
        "sampled_clear" if clear else "sampled_collision",
        clear,
        len(samples),
        tuple(samples),
        binding,
        worst,
    )
