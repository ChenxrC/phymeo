"""World-frame contact totals and counterpart-resolved observations from one physical sample."""

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import chain

import numpy as np

from ..constraints.collision_shapes import CollisionKey
from ..core.capabilities import SolverFailure
from .contact_wrench import (
    ContactSelector,
    _cloth_observation,
    _selector,
    _surface_observation,
    _vector,
)


@dataclass(frozen=True, eq=False)
class ContactForceMatrix:
    sensing: tuple[ContactSelector, ...]
    counterparts: tuple[ContactSelector, ...]
    total_force_n: np.ndarray
    total_friction_n: np.ndarray
    force_n: np.ndarray
    friction_n: np.ndarray
    position_m: np.ndarray
    weight_n: np.ndarray
    source: str
    phase: str


def _slots(selection, catalog):
    if isinstance(selection, (CollisionKey, ContactSelector)):
        selection = (selection,)
    selectors = tuple(_selector(v) for v in selection)
    index = {}
    bodies = set()
    for i, sel in enumerate(selectors):
        body = (sel.world, sel.body)
        full = (*body, sel.shape)
        if full in index or (*body, None) in index or (sel.shape is None and body in bodies):
            raise ValueError("contact matrix selectors overlap on one axis")
        if catalog is not None and not any(sel.matches(key) for key in catalog):
            raise ValueError("contact selector is not registered in this world")
        index[full] = i
        bodies.add(body)
    return selectors, index


def _lookup(key, index):
    return index.get((key.world, key.body, key.shape), index.get((key.world, key.body, None)))


def measure_contact_matrix(source, sensing, counterparts=(), *, published_by=None):
    """Read one immutable-by-convention sample into dense, world-frame HOST arrays.

    Rows and columns follow caller order. Totals include all contact partners;
    pair arrays include only requested counterparts. Empty or cross-world slots
    contain zeros. Positions are effective surface midpoints weighted by the
    individual force magnitudes, with weight_n exposing empty or force-free slots.
    Either axis may select distinct shapes or whole bodies in multiple worlds.
    """
    loads, catalog, source_name, phase = _surface_observation(source, published_by)

    def observations():
        for pair in loads.contacts:
            sites = (
                (s.force_on_second_n, s.material.first_point, s.material.second_point)
                for s in pair.sites
            )
            yield pair.first, pair.second, pair.normal, sites

    return _reduce_matrix(observations(), catalog, source_name, phase, sensing, counterparts)


def _reduce_matrix(observations, catalog, source_name, phase, sensing, counterparts):
    rows, rindex = _slots(sensing, catalog)
    cols, cindex = _slots(counterparts, catalog)
    if not rows:
        raise ValueError("contact matrix requires at least one sensing row")
    shape = (len(rows), len(cols))
    total = np.zeros((len(rows), 3))
    total_t = np.zeros_like(total)
    pair_f = np.zeros((*shape, 3))
    pair_t = np.zeros_like(pair_f)
    pos = np.zeros_like(pair_f)
    weight = np.zeros(shape)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for a, b, normal, sites in observations:
                if a.world != b.world:
                    raise ValueError("invalid contact across worlds")
                sides = [
                    (r, _lookup(other, cindex), sign)
                    for key, other, sign in ((a, b, -1), (b, a, 1))
                    if (r := _lookup(key, rindex)) is not None
                ]
                if not sides:
                    continue
                n = _vector(normal, "normal")
                length = np.hypot.reduce(n)
                if not length:
                    raise ValueError("contact normal is zero")
                n = n / length
                for force, pa, pb in sites:
                    f = _vector(force, "force on second")
                    tangent = f - n * (f @ n)
                    mag = np.hypot.reduce(f)
                    midpoint = 0.5 * _vector(pa, "first surface point") + 0.5 * _vector(
                        pb, "second surface point"
                    )
                    for r, c, sign in sides:
                        total[r] += sign * f
                        total_t[r] += sign * tangent
                        if c is not None:
                            pair_f[r, c] += sign * f
                            pair_t[r, c] += sign * tangent
                            pos[r, c] += mag * midpoint
                            weight[r, c] += mag
            divisor = weight[:, :, None]
            np.divide(pos, divisor, out=pos, where=divisor > 0)
    except FloatingPointError as error:
        raise SolverFailure("contact force matrix exceeds FP64 range") from error
    for array in (total, total_t, pair_f, pair_t, pos, weight):
        array.flags.writeable = False
    return ContactForceMatrix(
        rows, cols, total, total_t, pair_f, pair_t, pos, weight, source_name, phase
    )


def _cloth_matrix_rows(body, links, faces, radius):
    for record in body["cloth_links"]:
        link = links.get((record["robot"], record["link"]))
        face = faces.get(record["face"])
        if link is None or face is None:
            raise ValueError("recorded cloth-link pair does not match this program")
        n = _vector(record["normal"], "cloth-link normal")
        a, b = np.asarray(record["axis_ends_m"])
        r = radius[record["robot"], record["link"]]
        # Reconstruct material surface points from measured midpoint geometry.
        # The equal/opposite physical forces were applied at the cloth middle point.
        sites = (
            (
                s["force_on_cloth_n"],
                a + s["axis_parameter"] * (b - a) + r * n,
                np.asarray(s["point_m"]) - record["half_thickness_m"] * n,
            )
            for s in record["sites"]
        )
        yield link, face, n, sites


def measure_cloth_link_matrix(results, programs, sensing, counterparts=(), *, substep=-1):
    """Read midpoint observations from a world-keyed group of accepted cloth-link scenes.

    A rejected world invalidates the requested group snapshot. To observe unaffected
    worlds, pass the same explicit subset of keys in both mappings.
    """
    if (
        not isinstance(results, Mapping)
        or not isinstance(programs, Mapping)
        or not results
        or set(results) != set(programs)
    ):
        raise ValueError("require results and programs for the same nonempty set of worlds")
    catalog, streams = [], []
    time_slot = None
    for world, result in results.items():
        program = programs[world]
        sample, body, links, faces = _cloth_observation(result, program, world, substep)
        slot = (sample.index, sample.dt)
        if time_slot is not None and slot != time_slot:
            raise ValueError("inconsistent substep in cloth-link group")
        time_slot = slot
        catalog.extend(links.values())
        catalog.extend(faces.values())
        radius = {(link.domain, link.key): link.radius_m for link in program.cloth_links.links}
        streams.append(_cloth_matrix_rows(body, links, faces, radius))
    return _reduce_matrix(
        chain.from_iterable(streams),
        catalog,
        "cloth_links",
        "substep-midpoint",
        sensing,
        counterparts,
    )
