"""Pure HOST staging and actual-motion acceptance for coupled tangent history.

The scene supplies a current normal sample, full world mobility and complete free
velocity. This adapter does not detect contacts, integrate participants or publish
state. The original smooth-friction sampling is owned by the scene seam.
"""

from dataclasses import dataclass, field, fields
from math import fsum, isclose

import numpy as np

from ..core.validation import finite_real
from ..solvers.coupled_tangent import solve_coupled_tangent
from ..solvers.tangent_history import StationKey, StationSample, TangentHistory, TangentUpdate


def _array(value, shape, name):
    a = np.asarray(value)
    if a.shape != shape or a.dtype != np.float64 or not np.isfinite(a).all():
        raise ValueError(f"{name} requires finite FP64 shape {shape}")
    return np.array(a, copy=True)


def _owned(value):
    a = np.asarray(value, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError("tangent step candidate contains a non-finite value")
    return np.frombuffer(a.tobytes(), dtype=np.float64).reshape(a.shape)


def _maximum_norm(value):
    norms = np.hypot.reduce(value, axis=-1)
    if not np.isfinite(norms).all():
        raise ValueError("tangent displacement norm is not representable")
    return float(np.max(norms, initial=0.0))


@dataclass(frozen=True, eq=False)
class _ArraySeal:
    value: np.ndarray
    shape: tuple
    strides: tuple
    dtype: str
    contents: bytes

    def intact(self, value):
        return (
            value is self.value
            and value.shape == self.shape
            and value.strides == self.strides
            and value.dtype.str == self.dtype
            and not value.flags.writeable
            and value.tobytes() == self.contents
        )


def _seal(value):
    if isinstance(value, np.ndarray):
        return _ArraySeal(value, value.shape, value.strides, value.dtype.str, value.tobytes())
    return value


def _intact(value, bound):
    if isinstance(bound, _ArraySeal):
        return bound.intact(value)
    if isinstance(bound, (float, int, str, bool, type(None))):
        return type(value) is type(bound) and value == bound
    return value is bound


def _object_seal(value):
    return (
        ()
        if value is None
        else tuple((f.name, _seal(getattr(value, f.name))) for f in fields(value))
    )


def _object_intact(value, bound):
    return all(_intact(getattr(value, name), sealed) for name, sealed in bound)


@dataclass(frozen=True, eq=False)
class _PreparedBinding:
    store: TangentHistory
    history: object
    values: tuple
    base_values: tuple
    history_values: tuple
    result_values: tuple
    old_stored_energy_j: float


@dataclass(frozen=True, eq=False)
class TangentStepTrial:
    """Owned candidate: keys/force rows retain the complete canonical input order.

    history_trial/result contain only positive-normal rows; result is None for an
    empty active set. No force is silently dropped when zero-normal rows intervene.
    """

    _binding: _PreparedBinding = field(repr=False)
    keys: tuple
    history_trial: object
    result: object
    active_indices: tuple[int, ...]
    frames_world: np.ndarray
    stiffness_n_per_m: np.ndarray
    force_on_second_n: np.ndarray
    predicted_displacement_m: np.ndarray


@dataclass(frozen=True, eq=False)
class TangentStepReceipt:
    """Successful physical audit; all work/energy is in J, motion errors in m.

    energy_residual_j is the total actual work + stored-energy change + losses.
    The per-station residual is also retained so cancellation cannot hide failure.
    accepted does not mean the caller has published either physical or history state.
    """

    accepted: bool
    energy_residual_j: float
    material_motion_error_m: float
    application_motion_error_m: float
    material_application_error_m: float
    actual_work_j: float
    application_work_j: np.ndarray
    material_work_j: np.ndarray
    per_station_energy_residual_j: np.ndarray
    released_energy_j: float
    actual_material_displacement_m: np.ndarray
    actual_application_displacement_m: np.ndarray

    def __post_init__(self):
        for name in (
            "application_work_j",
            "material_work_j",
            "per_station_energy_residual_j",
            "actual_material_displacement_m",
            "actual_application_displacement_m",
        ):
            object.__setattr__(self, name, _owned(getattr(self, name)))


def _validate_world_mobility(W):
    if not W.size:
        return W
    scale = float(np.max(np.abs(W)))
    if scale and np.max(np.abs(W / scale - W.T / scale)) > 1e-12:
        raise ValueError("world mobility must be symmetric to relative 1e-12")
    W = 0.5 * W + 0.5 * W.T
    if scale and np.linalg.eigvalsh(W / scale)[0] < -32 * np.finfo(float).eps * len(W):
        raise ValueError("world mobility must be positive semidefinite within roundoff")
    return W  # Do not clip physical eigenvalues.


def prepare_tangent_step(
    store,
    history,
    samples,
    relative_mobility_per_kg,
    free_relative_velocity_m_s,
    *,
    sample_time_s,
    dt_s,
    stiffness_n_per_m,
    friction,
):
    """Project a full 3n world response and complete free velocity, then stage A+B.

    Samples must already be in strictly increasing StationKey order. Arrays use
    FP64: W (3n,3n) kg^-1, v_free (n,3) m/s, kt (n,) N/m, mu (n,) dimensionless.
    n may be zero and is bounded by both the store capacity and 128 stations.
    Stiffness cannot change within a retained spring episode without a reset.
    """
    if not isinstance(store, TangentHistory):
        raise TypeError("store must be TangentHistory")
    current = []
    for i, s in enumerate(samples):
        if i >= min(store.capacity, 128):
            raise ValueError("tangent step station capacity exceeded (maximum 128)")
        if not isinstance(s, StationSample) or not isinstance(s.key, StationKey):
            raise TypeError("samples require StationSample and StationKey")
        current.append(s)
    keys = tuple(s.key for s in current)
    if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
        raise ValueError("samples require distinct canonical StationKey order")
    n = len(current)
    W = _array(relative_mobility_per_kg, (3 * n, 3 * n), "world mobility")
    free = _array(free_relative_velocity_m_s, (n, 3), "free relative velocity")
    kt = _array(stiffness_n_per_m, (n,), "tangent stiffness")
    mu = _array(friction, (n,), "friction")
    if np.any(kt <= 0) or np.any(mu < 0):
        raise ValueError("require positive stiffness and nonnegative friction")
    h = finite_real(dt_s, "dt_s")
    staged = store.prepare(history, current, sample_time_s=sample_time_s, dt_s=h)
    # B has now validated every sample, including zero-normal rows, and owns all
    # active addresses. Do not keep caller-owned frames or StationKey instances.
    keys = tuple(StationKey(int(s.key.world), int(s.key.token)) for s in current)
    frames = np.asarray([s.frame_world for s in current], dtype=float).reshape(n, 3, 3)
    index = {key: i for i, key in enumerate(keys)}
    active = tuple(index[key] for key in staged.keys)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            W = _validate_world_mobility(W)
            for p, i in zip(staged.stations, active, strict=True):
                prior = p.prior
                if prior is not None:
                    expected = (
                        0.5
                        * kt[i]
                        * float(np.dot(prior.elastic_displacement_m, prior.elastic_displacement_m))
                    )
                    if (
                        prior.stiffness_n_per_m is not None and prior.stiffness_n_per_m != kt[i]
                    ) or not isclose(
                        prior.stored_energy_j,
                        expected,
                        rel_tol=1e-10,
                        abs_tol=1e-300,
                    ):
                        raise ValueError(
                            "retained stiffness/energy cannot change without an episode reset"
                        )
            force = np.zeros((n, 3))
            result = None
            if active:
                T = np.zeros((3 * n, 2 * len(active)))
                for a, i in enumerate(active):
                    T[3 * i : 3 * i + 3, 2 * a : 2 * a + 2] = staged.frames_world[a, :, :2]
                result = solve_coupled_tangent(
                    staged.elastic_tangent_m,
                    (h * (T.T @ free.ravel())).reshape(-1, 2),
                    T.T @ W @ T,
                    h,
                    staged.normal_force_n,
                    stiffness_n_per_m=kt[list(active)],
                    friction=mu[list(active)],
                )
                force = (T @ result.force_n.ravel()).reshape(n, 3)
            predicted = h * free + (h * h * (W @ force.ravel())).reshape(n, 3)
            owned = (_owned(frames), _owned(kt), _owned(force), _owned(predicted))
            old_energy = finite_real(
                fsum(p.update.stored_energy_j for p in history.active), "old spring energy"
            )
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise ValueError(f"tangent step preparation is not representable: {exc}") from exc
    values = (keys, staged, result, active, *owned)
    binding = _PreparedBinding(
        store,
        history,
        tuple(_seal(v) for v in values),
        _object_seal(history),
        _object_seal(staged),
        _object_seal(result),
        old_energy,
    )
    return TangentStepTrial(binding, *values)


def _checked_trial(store, history, trial):
    if not isinstance(trial, TangentStepTrial) or not isinstance(trial._binding, _PreparedBinding):
        raise ValueError("expected a prepared tangent trial")
    bound = trial._binding
    if bound.store is not store or bound.history is not history:
        raise ValueError("stale tangent trial or different history owner/base")
    values = tuple(getattr(trial, f.name) for f in fields(trial) if f.name != "_binding")
    try:
        intact = (
            len(values) == len(bound.values)
            and all(_intact(v, b) for v, b in zip(values, bound.values, strict=True))
            and _object_intact(history, bound.base_values)
            and _object_intact(trial.history_trial, bound.history_values)
            and _object_intact(trial.result, bound.result_values)
        )
    except (AttributeError, TypeError, ValueError):
        intact = False
    if not intact:
        raise ValueError("tangent trial binding differs from its prepared values")
    return bound


def finish_tangent_step(
    store,
    history,
    trial,
    actual_material_displacement_m,
    actual_application_displacement_m,
    *,
    motion_tolerance_m,
    energy_tolerance_j,
):
    """Audit actual entry-anchor motion/work, then return a pure history candidate.

    Both relative displacement arrays are (n,3) m in the same canonical input
    order. They must track the original material/application points through the
    actual candidate, not newly selected closest points. Bounds are explicit,
    finite positive absolute m/J. Failure raises ValueError and publishes nothing.
    """
    bound = _checked_trial(store, history, trial)
    mt = finite_real(motion_tolerance_m, "motion_tolerance_m")
    et = finite_real(energy_tolerance_j, "energy_tolerance_j")
    if mt <= 0 or et <= 0:
        raise ValueError("motion and energy tolerances must be positive")
    n = len(trial.keys)
    material = _array(actual_material_displacement_m, (n, 3), "actual material displacement")
    application = _array(
        actual_application_displacement_m, (n, 3), "actual application displacement"
    )
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            material_error = _maximum_norm(material - trial.predicted_displacement_m)
            application_error = _maximum_norm(application - trial.predicted_displacement_m)
            pair_error = _maximum_norm(material - application)
            if max(material_error, application_error, pair_error) > mt:
                raise ValueError("actual material/application displacement failed the motion gate")
            app_work = np.einsum("ij,ij->i", trial.force_on_second_n, application)
            mat_work = np.einsum("ij,ij->i", trial.force_on_second_n, material)
            residual = np.zeros(n)
            release = trial.history_trial.dropped_spring_energy_j
            stored = losses = 0.0
            updates = {}
            result = trial.result
            if result is not None:
                active = list(trial.active_indices)
                actual_tangent = np.einsum(
                    "nki,nk->ni", trial.history_trial.frames_world[:, :, :2], material[active]
                )
                check = result.verify_motion(actual_tangent, tolerance_m=mt)
                if not check.accepted:
                    raise ValueError("actual tangent displacement failed the motion gate")
                residual[active] = check.energy_residual_j + app_work[active] - check.work_j
                if np.max(np.abs(check.energy_residual_j), initial=0.0) > et:
                    raise ValueError("actual material work failed the spring energy gate")
                release += fsum(result.released_energy_j)
                stored = fsum(result.stored_energy_j)
                losses = fsum((*result.plastic_dissipation_j, *result.implicit_dissipation_j))
                for a, (p, i) in enumerate(zip(trial.history_trial.stations, active, strict=True)):
                    prior = p.prior
                    elastic = (
                        trial.history_trial.frames_world[a, :, :2]
                        @ result.elastic_displacement_m[a]
                    )
                    updates[p.identity.key] = TangentUpdate(
                        tuple(map(float, elastic)),
                        float(result.stored_energy_j[a]),
                        (0.0 if prior is None else prior.plastic_slip_m)
                        + float(np.hypot.reduce(result.plastic_increment_m[a])),
                        (0.0 if prior is None else prior.tangent_work_j) + float(app_work[i]),
                        result.regime[a],
                        float(trial.stiffness_n_per_m[i]),
                    )
            actual_work = fsum(app_work)
            balance = fsum((actual_work, stored, -bound.old_stored_energy_j, losses, release))
            if (
                not np.isfinite((actual_work, balance, release)).all()
                or not np.isfinite(residual).all()
            ):
                raise ValueError("actual tangent work/energy is not representable")
            if (
                abs(balance) > et
                or np.max(np.abs(residual), initial=0.0) > et
                or np.max(np.abs(app_work - mat_work), initial=0.0) > et
            ):
                raise ValueError("actual application work failed the spring energy gate")
            receipt = TangentStepReceipt(
                True,
                balance,
                material_error,
                application_error,
                pair_error,
                actual_work,
                app_work,
                mat_work,
                residual,
                float(release),
                material,
                application,
            )
            candidate = store.commit(history, trial.history_trial, updates)
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise ValueError(f"tangent step acceptance is not representable: {exc}") from exc
    return candidate, receipt
