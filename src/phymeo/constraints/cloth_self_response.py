"""Owned S2-L laws and proposal-only full shared-base response algebra.

No function in this module advances a particle state. Actual replay is the
acceptance oracle; the host matrix is a preconditioner, including cross terms.
"""

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from ..core.capabilities import SolverFailure
from ..core.scene import fingerprint
from ..geometry._prism_pair_exact import PrismPairProofLimits
from .friction import solve_friction_disk


def owned(value, shape=None, dtype=np.float64, name="array"):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype):
        raise TypeError(f"{name} requires {np.dtype(dtype)} ndarray storage")
    if (shape is not None and value.shape != shape) or not np.isfinite(value).all():
        raise ValueError(f"invalid {name} shape or finite values")
    return np.frombuffer(value.tobytes(order="C"), dtype=dtype).reshape(value.shape)


def positive(value, name, *, zero=False):
    if type(value) is not float or not np.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f"{name} must be a finite {'nonnegative' if zero else 'positive'} float")
    return value


@dataclass(frozen=True)
class SelfContactResponseLaw:
    mu_static: float
    mu_dynamic: float
    normal_domain: str = field(kw_only=True)
    phase_capacity: str = field(default="owned_split_v1", kw_only=True)
    restitution: float = field(default=0.0, kw_only=True)

    def __post_init__(self):
        positive(self.mu_static, "mu_static", zero=True)
        positive(self.mu_dynamic, "mu_dynamic", zero=True)
        if self.mu_static != self.mu_dynamic:
            raise ValueError("S2-L requires equal static and dynamic coefficients")
        if self.normal_domain != "unique_exposed_ray_v1" or self.phase_capacity != "owned_split_v1":
            raise ValueError("unsupported self-contact law")
        if type(self.restitution) is not float or self.restitution != 0.0:
            raise ValueError("only zero restitution is admitted")


@dataclass(frozen=True)
class SelfContactResponseLimits:
    max_contacts: int = 128
    max_anchors_per_pair: int = 64
    max_rebuilds: int = 8
    max_outer_replays: int = 512
    max_line_search_halvings: int = 16
    max_normal_iterations: int = 512
    max_tangent_iterations: int = 96
    geometry_limits: PrismPairProofLimits = field(default_factory=PrismPairProofLimits)

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name == "geometry_limits":
                if not isinstance(value, PrismPairProofLimits):
                    raise TypeError("geometry_limits requires PrismPairProofLimits")
            elif type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive builtin integer")


@dataclass(frozen=True)
class SelfContactBudgets:
    penetration_m: float
    anchor_error_m: float
    velocity_mps: float
    impulse_ns: float
    energy_j: float
    angular_impulse_nms: float

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            positive(getattr(self, name), name, zero=name == "penetration_m")


@dataclass(frozen=True)
class PhaseFreeData:
    """Actual current relative velocity and endpoint gap, with current proposal.

    h [s], gaps [m], velocity [m/s], current_world [N*s]. Held iterations
    replace the normal component by gap/h; this is not a trajectory proof.
    """
    velocity: np.ndarray
    gaps: np.ndarray
    current_world: np.ndarray
    h: float
    tolerance_mps: float

    def __post_init__(self):
        k = len(self.velocity)
        for name, shape in (("velocity", (k, 3)), ("gaps", (k,)), ("current_world", (k, 3))):
            object.__setattr__(self, name, owned(getattr(self, name), shape, name=name))
        positive(self.h, "h")
        positive(self.tolerance_mps, "tolerance_mps")


@dataclass(frozen=True)
class PhaseCapacities:
    """This step's owner mask and separately projected held normal impulses."""
    initial_closed: np.ndarray
    active: np.ndarray
    held_normal_ns: np.ndarray
    owners: tuple

    def __post_init__(self):
        k = len(self.owners)
        if len(set(self.owners)) != k:
            raise ValueError("duplicate friction capacity owner")
        for name in ("initial_closed", "active"):
            object.__setattr__(self, name, owned(getattr(self, name), (k,), bool, name))
        p = owned(self.held_normal_ns, (k,), name="held normal")
        if np.any(p < 0):
            raise ValueError("negative held normal capacity")
        object.__setattr__(self, "held_normal_ns", p)
        object.__setattr__(self, "owners", tuple(self.owners))


@dataclass(frozen=True)
class PhaseProposal:
    phase: str
    world_impulses: np.ndarray
    normal_ns: np.ndarray
    tangent_ns: np.ndarray
    capacity_ns: np.ndarray
    response: np.ndarray
    residual_mps: float
    iterations: int
    rank_receipt: tuple

    def __post_init__(self):
        k = len(self.normal_ns)
        for name, shape in (("world_impulses", (k, 3)), ("normal_ns", (k,)),
                            ("tangent_ns", (k, 3)), ("capacity_ns", (k,)),
                            ("response", (3*k, 3*k))):
            object.__setattr__(self, name, owned(getattr(self, name), shape, name=name))


def tangent_frame(n):
    axis = np.eye(3)[int(np.argmin(np.abs(n)))]
    t = np.cross(n, axis)
    t /= np.linalg.norm(t)
    return np.column_stack((n, t, np.cross(n, t)))


def response_matrix(binding):
    """Probe the PUBLIC full same-domain operator; never apply its candidate."""
    k = len(binding.pairs)
    result = np.empty((3*k, 3*k), np.float64)
    for j in range(3*k):
        unit = np.zeros((k, 3), np.float64)
        unit.flat[j] = 1.0
        result[:, j] = np.asarray(binding.operator.response(unit)).reshape(-1)
    if not np.isfinite(result).all():
        raise SolverFailure("nonfinite full shared-base response")
    return result


def rank_witness(matrix):
    """Exact LDL of stored response and an inverse-norm conditioning bound.

    A nonsymmetric stored response is enclosed by the symmetric part, with its
    explicit skew norm. No eigenvalue clipping or diagonal regularization.
    """
    n = len(matrix)
    if not n:
        return ((), (0, 1), (0, 1))
    q = Fraction.from_float
    a = [[(q(float(matrix[i, j])) + q(float(matrix[j, i]))) / 2 for j in range(n)] for i in range(n)]
    l = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    d = []
    for i in range(n):
        pivot = a[i][i] - sum(l[i][k]**2*d[k] for k in range(i))
        if pivot <= 0:
            raise SolverFailure("full pressure response rank is not admitted")
        d.append(pivot)
        for j in range(i + 1, n):
            l[j][i] = (a[j][i] - sum(l[j][k]*l[i][k]*d[k] for k in range(i))) / pivot
    inverse = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    for col in range(n):
        y = [Fraction(0)]*n
        for i in range(n):
            y[i] = Fraction(int(i == col)) - sum(l[i][j]*y[j] for j in range(i))
        x = [Fraction(0)]*n
        for i in reversed(range(n)):
            x[i] = y[i]/d[i] - sum(l[j][i]*x[j] for j in range(i + 1, n))
        for i in range(n):
            inverse[i][col] = x[i]
    inv_norm = max(sum(map(abs, row)) for row in inverse)
    norm = max(sum(map(abs, row)) for row in a)
    skew = max(sum(abs(q(float(matrix[i, j])) - a[i][j]) for j in range(n)) for i in range(n))
    if skew * inv_norm >= 1:
        raise SolverFailure("stored response asymmetry exhausts positive-response witness")
    asymmetry = max(abs(q(float(matrix[i, j]))-q(float(matrix[j, i]))) for i in range(n) for j in range(n))
    raw_scale = max(abs(q(float(v))) for row in matrix for v in row)
    if asymmetry > Fraction(1, 10**12)*raw_scale:
        raise SolverFailure("full response exceeds relative asymmetry ceiling")
    condition = norm*inv_norm
    if condition >= 10**12:
        raise SolverFailure("full response condition number is not strictly below 1e12")
    token = lambda v: (v.numerator, v.denominator)
    return (tuple(map(token, d)), token(condition), token(skew*inv_norm))


def propose_self_contact_phase(binding, normal_rays, free_data, *, phase, capacities, law, limits):
    if phase not in ("held", "terminal"):
        raise ValueError("unknown phase")
    if not isinstance(law, SelfContactResponseLaw) or not isinstance(limits, SelfContactResponseLimits):
        raise TypeError("explicit law and limits required")
    if not isinstance(free_data, PhaseFreeData) or not isinstance(capacities, PhaseCapacities):
        raise TypeError("typed free data and phase capacity required")
    k = len(binding.pairs)
    n = owned(normal_rays, (k, 3), name="normal rays")
    if len(capacities.owners) != k or len(free_data.gaps) != k or np.any(abs(np.sum(n*n, axis=1)-1) > 1e-12):
        raise ValueError("normal/capacity binding mismatch")
    w = response_matrix(binding)
    witness = rank_witness(w)
    frames = [tangent_frame(row) for row in n]
    transform = np.zeros((3*k, 3*k))
    for i, frame in enumerate(frames):
        transform[3*i:3*i+3, 3*i:3*i+3] = frame
    proposal_w = w*0.5 + w.T*0.5
    local_w = transform.T @ proposal_w @ transform
    velocity = free_data.velocity.copy()
    if phase == "held":
        velocity += n*(free_data.gaps/free_data.h - np.sum(n*velocity, axis=1))[:, None]
    free = transform.T @ (velocity.reshape(-1) - w @ free_data.current_world.reshape(-1))
    impulse = transform.T @ free_data.current_world.reshape(-1)
    radii = np.zeros(k)
    residual = 0.0
    for iteration in range(1, limits.max_normal_iterations + 1):
        normal_ids = np.arange(0, 3*k, 3)
        without_normal = impulse.copy()
        without_normal[normal_ids] = 0.0
        normal_free = free[normal_ids] + local_w[normal_ids] @ without_normal
        charged = capacities.active & ((normal_free < 0) | (impulse[normal_ids] > 0))
        active_ids = normal_ids[charged]
        impulse[normal_ids] = 0.0
        if len(active_ids):
            # Full normal active block is only a proposal; exact geometry and
            # actual replay independently decide whether it can be accepted.
            solution = np.linalg.solve(local_w[np.ix_(active_ids, active_ids)], -normal_free[charged])
            impulse[active_ids] = np.maximum(solution, 0.0)
        for i in range(k):
            block = slice(3*i, 3*i + 3)
            if not capacities.active[i]:
                impulse[block] = 0.0
                continue
            u = free[block] + local_w[block] @ impulse
            impulse[3*i] = max(0.0, impulse[3*i] - u[0]/local_w[3*i, 3*i])
            cap = impulse[3*i]
            if phase == "terminal" and not capacities.initial_closed[i]:
                cap += capacities.held_normal_ns[i]
            if phase == "held" and not capacities.initial_closed[i]:
                cap = 0.0
            radii[i] = law.mu_dynamic*cap
            ts = slice(3*i + 1, 3*i + 3)
            u = free[ts] + local_w[ts] @ impulse - local_w[ts, ts] @ impulse[ts]
            disk = solve_friction_disk(local_w[ts, ts], u, float(cap),
                                      mu_static=law.mu_static, mu_dynamic=law.mu_dynamic,
                                      tolerance=free_data.tolerance_mps/64,
                                      max_iterations=limits.max_tangent_iterations)
            impulse[ts] = disk.impulse
        actual = free + local_w @ impulse
        residual = 0.0
        for i in range(k):
            if not capacities.active[i]:
                continue
            u, p = actual[3*i:3*i+3], impulse[3*i:3*i+3]
            scale = local_w[3*i, 3*i]
            residual = max(residual, -u[0], abs(min(scale*p[0], u[0])))
            tnorm, speed = np.linalg.norm(p[1:]), np.linalg.norm(u[1:])
            residual = max(residual, max(0.0, tnorm-radii[i])*scale)
            if radii[i] > 0 and speed > free_data.tolerance_mps/64:
                residual = max(residual, np.linalg.norm(p[1:] + radii[i]*u[1:]/speed)*scale)
        if residual <= free_data.tolerance_mps/64:
            break
    else:
        raise SolverFailure("conditional full-response solve exhausted; unknown")
    world = (transform @ impulse).reshape(k, 3)
    normals = np.sum(world*n, axis=1)
    return PhaseProposal(phase, world, normals, world-n*normals[:, None], radii, w,
                         float(residual), iteration, witness)


@dataclass(frozen=True)
class SelfContactLoadPlan:
    phase: str
    domain: str
    triangles: np.ndarray
    weights: np.ndarray
    offsets: np.ndarray
    pairs: np.ndarray
    jacobians: np.ndarray
    world_points: np.ndarray
    world_impulses: np.ndarray
    node_impulses: np.ndarray
    raw_response: np.ndarray
    binding_token: str

    def __post_init__(self):
        if self.phase not in ("held", "terminal"):
            raise ValueError("invalid load phase")
        for name in ("triangles", "weights", "offsets", "pairs", "jacobians", "world_points", "world_impulses", "node_impulses", "raw_response"):
            dtype = np.int64 if name in ("triangles", "pairs") else np.float64
            object.__setattr__(self, name, owned(getattr(self, name), dtype=dtype, name=name))


def make_self_contact_load_plan(binding, world_impulses, *, phase):
    world = owned(world_impulses, (len(binding.pairs), 3), name="world impulses")
    domains = tuple(binding.operator.domains)
    if len(domains) != 1:
        raise ValueError("one physical base is required")
    nodes = binding.node_impulses(world)
    arrays = dict(triangles=binding.triangles, weights=binding.weights, offsets=binding.offsets,
                  pairs=binding.pairs, jacobians=binding.jacobians, world_points=binding.world_points,
                  raw_response=response_matrix(binding))
    token = fingerprint(dict(phase=phase, domain=domains[0], **arrays, world_impulses=world, node_impulses=nodes))
    return SelfContactLoadPlan(phase, domains[0], **arrays, world_impulses=world,
                               node_impulses=nodes, binding_token=token)


def rational_response_inverse(matrix):
    """R1/R2 exact symmetric admission, returning a certification inverse only."""
    a = [[value if isinstance(value, Fraction) else Fraction(value) for value in row] for row in matrix]
    n = len(a)
    if any(len(row) != n for row in a) or any(a[i][j] != a[j][i] for i in range(n) for j in range(n)):
        raise SolverFailure("exact response must be square and symmetric")
    if not n:
        return (), ((), (0, 1))
    l = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    pivots = []
    for i in range(n):
        pivot = a[i][i]-sum(l[i][k]**2*pivots[k] for k in range(i))
        if pivot <= 0:
            raise SolverFailure("exact capacity response lacks positive LDL pivots")
        pivots.append(pivot)
        for j in range(i+1, n):
            l[j][i] = (a[j][i]-sum(l[j][k]*l[i][k]*pivots[k] for k in range(i)))/pivot
    inverse = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    for column in range(n):
        y, x = [Fraction(0)]*n, [Fraction(0)]*n
        for i in range(n):
            y[i] = Fraction(int(i == column))-sum(l[i][j]*y[j] for j in range(i))
        for i in reversed(range(n)):
            x[i] = y[i]/pivots[i]-sum(l[j][i]*x[j] for j in range(i+1, n))
            inverse[i][column] = x[i]
    condition = max(sum(map(abs, row)) for row in a)*max(sum(map(abs, row)) for row in inverse)
    if condition >= 10**12:
        raise SolverFailure("exact capacity response condition must be strictly below 1e12")
    token = lambda value: (value.numerator, value.denominator)
    return tuple(tuple(row) for row in inverse), (tuple(map(token, pivots)), token(condition))
