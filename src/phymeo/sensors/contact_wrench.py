"""World-isolated contact-force and moment measurements from recorded physical loads."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..constraints.collision_shapes import CollisionKey
from ..core.capabilities import SolverFailure
from ..dynamics.rotation import quaternion_matrix
from ..runtime.program import SceneStepResult
from ..solvers.collision_loads import CollisionLoadStep, CollisionLoadSystem, SurfaceLoads
from .robot_frame import RobotSite


def _index(value):
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) and value >= 0


@dataclass(frozen=True)
class ContactSelector:
    """Collision world/body/optional shape, or a structural scene's named body.

    For cloth-link records, body is the cloth domain name or (robot domain,
    link index). An optional shape selects a face key or a link capsule key.
    """

    world: int
    body: int | str | tuple[str, int]
    shape: int | str | None = None

    def __post_init__(self):
        if not _index(self.world):
            raise ValueError("contact world must be a nonnegative integer")
        b = self.body
        if not (
            _index(b)
            or (isinstance(b, str) and b)
            or (
                isinstance(b, tuple)
                and len(b) == 2
                and isinstance(b[0], str)
                and b[0]
                and _index(b[1])
            )
        ):
            raise ValueError("invalid contact body")
        if self.shape is not None and not (
            _index(self.shape) or isinstance(self.shape, str) and self.shape
        ):
            raise ValueError("invalid contact shape")

    def matches(self, side):
        return (
            self.world == side.world
            and self.body == side.body
            and (self.shape is None or self.shape == side.shape)
        )


@dataclass(frozen=True, eq=False)
class ContactWrenchReading:
    """Force [N], tangential force [N] and moment [N m], all expressed in sensor axes."""

    force_n: np.ndarray
    friction_force_n: np.ndarray
    torque_nm: np.ndarray
    reference_world_m: tuple
    sensor_to_world_xyzw: tuple
    sensing: ContactSelector
    counterparts: tuple[ContactSelector, ...] | None
    contributing_pairs: tuple[tuple[ContactSelector, ContactSelector], ...]
    site_sides: int
    source: str
    phase: str
    substep_index: int | None
    dt_s: float | None


def _selector(value):
    if isinstance(value, CollisionKey):
        return ContactSelector(value.world, value.body, value.shape)
    if not isinstance(value, ContactSelector):
        raise TypeError("expected a ContactSelector or CollisionKey")
    return value


def _vector(value, name):
    a = np.asarray(value)
    if a.shape != (3,) or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise ValueError(f"invalid recorded {name}")
    return np.asarray(a, dtype=float)


def _surface_rows(loads):
    for pair in loads.contacts:
        yield (
            _selector(pair.first),
            _selector(pair.second),
            pair.normal,
            (
                (
                    s.application_point_first_m,
                    s.application_point_second_m,
                    -np.asarray(s.force_on_second_n),
                    s.force_on_second_n,
                )
                for s in pair.sites
            ),
        )


def _reading(
    rows,
    sensing,
    counterparts,
    reference,
    orientation,
    source,
    phase,
    catalog=None,
    index=None,
    dt=None,
):
    sensing = _selector(sensing)
    if counterparts is not None:
        if isinstance(counterparts, (ContactSelector, CollisionKey)):
            counterparts = (counterparts,)
        counterparts = tuple(dict.fromkeys(_selector(v) for v in counterparts))
        if any(v.world != sensing.world for v in counterparts):
            raise ValueError("contact counterpart must be in the sensing world")
    if catalog is not None:
        for selection in (sensing, *(counterparts or ())):
            if not any(selection.matches(side) for side in catalog):
                raise ValueError("contact selector is not registered in this world")
    frame = RobotSite(None, reference, orientation)
    ref = np.asarray(frame.local_position_m)
    world_to_sensor = quaternion_matrix(np.array([frame.local_orientation_xyzw]))[0].T
    force, friction, moment = np.zeros((3, 3))
    pairs = {}
    count = 0
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for first, second, normal, sites in rows:
                if first.world != second.world:
                    raise ValueError("invalid recorded contact across worlds")
                sides = [
                    (i, a, b)
                    for i, (a, b) in enumerate(((first, second), (second, first)))
                    if sensing.matches(a)
                    and (counterparts is None or any(s.matches(b) for s in counterparts))
                ]
                if not sides:
                    continue
                n = _vector(normal, "world normal")
                magnitude = np.hypot.reduce(n)
                if magnitude == 0:
                    raise ValueError("recorded contact normal is zero")
                n = n / magnitude
                for pa, pb, fa, fb in sites:
                    for i, a, b in sides:
                        pos = _vector(pb if i else pa, "application point")
                        f = _vector(fb if i else fa, "force")
                        force += f
                        friction += f - n * (f @ n)
                        moment += np.cross(pos - ref, f)
                        count += 1
                        pairs[a, b] = None
            outputs = [world_to_sensor @ v for v in (force, friction, moment)]
            if not all(np.isfinite(v).all() for v in outputs):
                raise FloatingPointError
    except (FloatingPointError, OverflowError) as exc:
        raise SolverFailure("contact wrench exceeds FP64 range") from exc
    for v in outputs:
        v.flags.writeable = False
    return ContactWrenchReading(
        *outputs,
        frame.local_position_m,
        frame.local_orientation_xyzw,
        sensing,
        counterparts,
        tuple(pairs),
        count,
        source,
        phase,
        index,
        dt,
    )


def measure_contact_wrench(
    source,
    sensing,
    *,
    counterparts=None,
    reference_world_m=(0.0, 0.0, 0.0),
    sensor_to_world_xyzw=(0.0, 0.0, 0.0, 1.0),
    published_by=None,
):
    """Read an accepted CollisionLoadSystem entry frame or an explicit SurfaceLoads sample.

    Passing a CollisionLoadStep directly requires its current publishing system
    so that a staged frame from an outer transaction cannot mimic publication.
    SurfaceLoads is an explicit observation at its sampled state; its source
    label does not imply an accepted step. No time-step conversion is applied.
    """
    catalog = None
    if isinstance(source, (CollisionLoadSystem, CollisionLoadStep)):
        if isinstance(source, CollisionLoadSystem):
            if published_by is not None:
                raise ValueError("the source system already identifies its publisher")
            published_by = source
            # Capture one committed frame; a concurrent later publication cannot change this sample.
            frame = source.last_frame
            if frame is None:
                raise ValueError("no published collision load frame")
        else:
            frame = source
            if (
                not isinstance(published_by, CollisionLoadSystem)
                or published_by.last_frame is not frame
            ):
                raise ValueError("expected the currently published collision load frame")
        loads = frame.loads
        catalog = tuple(_selector(s.key) for s in published_by.model.shapes)
        source_name, phase = "CollisionLoadStep", "substep-entry"
    elif isinstance(source, SurfaceLoads):
        if published_by is not None:
            raise ValueError("a standalone SurfaceLoads sample has no publisher")
        loads = source
        source_name, phase = "SurfaceLoads", "sampled-state"
    else:
        raise TypeError("expected CollisionLoadSystem, published CollisionLoadStep or SurfaceLoads")
    return _reading(
        _surface_rows(loads),
        sensing,
        counterparts,
        reference_world_m,
        sensor_to_world_xyzw,
        source_name,
        phase,
        catalog,
    )


def measure_cloth_link_wrench(
    result,
    program,
    sensing,
    *,
    counterparts=None,
    reference_world_m=(0.0, 0.0, 0.0),
    sensor_to_world_xyzw=(0.0, 0.0, 0.0, 1.0),
    world=0,
    substep=-1,
):
    """Read one accepted structural cloth-link midpoint from its producing scene.

    The scalar scene has no built-in multi-world ID. world binds this result
    to the caller's world. A rejected outer result publishes no readings,
    including any substeps which had provisionally succeeded.
    """
    from ..solvers.compliant_cloth_contact import CompliantClothContactProgram

    if not isinstance(result, SceneStepResult) or not isinstance(
        program, CompliantClothContactProgram
    ):
        raise TypeError("expected the scene step result and its producing cloth program")
    if not result.report.accepted:
        raise ValueError("cloth wrench requires an accepted outer step")
    if not _index(world) or _selector(sensing).world != world:
        raise ValueError("sensing world does not match this cloth scene")
    if (
        not isinstance(substep, Integral)
        or isinstance(substep, (bool, np.bool_))
        or not -len(result.report.substeps) <= substep < len(result.report.substeps)
    ):
        raise ValueError("invalid substep index for this report")
    sample = result.report.substeps[substep]
    structure = program.cloth_links
    body = sample.diagnostics.get("robot_contact")
    if not sample.accepted:
        raise ValueError("cloth wrench requires an accepted substep")
    if (
        structure is None
        or sample.diagnostics.get("force_sample") != "substep-midpoint"
        or body is None
        or "cloth_links" not in body
    ):
        raise ValueError("no structural cloth-link midpoint loads in this report")
    links = {
        (link.domain, link.key): ContactSelector(world, (link.domain, link.link_index), link.key)
        for link in structure.links
    }
    faces = {
        name: ContactSelector(world, structure.cloth_domain, name) for name in structure.face_names
    }

    def rows():
        for record in body["cloth_links"]:
            link = links.get((record["robot"], record["link"]))
            face = faces.get(record["face"])
            if link is None or face is None:
                raise ValueError("recorded cloth-link pair does not match this program")
            yield (
                face,
                link,
                record["normal"],
                (
                    (s["point_m"], s["point_m"], s["force_on_cloth_n"], s["force_on_link_n"])
                    for s in record["sites"]
                ),
            )

    return _reading(
        rows(),
        sensing,
        counterparts,
        reference_world_m,
        sensor_to_world_xyzw,
        "cloth_links",
        "substep-midpoint",
        (*links.values(), *faces.values()),
        sample.index,
        sample.dt,
    )
