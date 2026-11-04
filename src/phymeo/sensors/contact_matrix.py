"""World-frame contact totals and counterpart-resolved observations from one physical sample."""

from dataclasses import dataclass

import numpy as np

from ..constraints.collision_shapes import CollisionKey
from ..core.capabilities import SolverFailure
from .contact_wrench import ContactSelector, _selector, _surface_observation, _vector


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
            for contact in loads.contacts:
                a, b = contact.first, contact.second
                if a.world != b.world:
                    raise ValueError("invalid contact across worlds")
                sides = [
                    (r, _lookup(other, cindex), sign)
                    for key, other, sign in ((a, b, -1), (b, a, 1))
                    if (r := _lookup(key, rindex)) is not None
                ]
                if not sides:
                    continue
                n = _vector(contact.normal, "normal")
                length = np.hypot.reduce(n)
                if not length:
                    raise ValueError("contact normal is zero")
                n = n / length
                for site in contact.sites:
                    f = _vector(site.force_on_second_n, "force on second")
                    tangent = f - n * (f @ n)
                    mag = np.hypot.reduce(f)
                    midpoint = 0.5 * _vector(
                        site.material.first_point, "first surface point"
                    ) + 0.5 * _vector(site.material.second_point, "second surface point")
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
