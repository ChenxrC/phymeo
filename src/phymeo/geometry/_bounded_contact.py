"""Bounded time search with a pointwise distance enclosure and a path speed bound."""

from fractions import Fraction as F

from ..core.capabilities import SolverFailure


def first_bounded_contact(intervals, distance, speed, radius):
    # DFS in time order. All cells before the earliest pending leaf have a
    # separating lower bound. A positive result needs an occupied sample.
    stack = list(reversed(intervals))
    pending = hit = None
    for _ in range(4096):
        if not stack:
            if pending is not None:
                raise SolverFailure("rotation_contact_unresolved")
            return None
        lo, hi = stack.pop()
        lower = lo if pending is None else pending
        if hit is not None and hit - lower <= F(1, 1 << 27):
            return lower, hit
        if pending is not None and lo - pending > F(1, 1 << 27):
            raise SolverFailure("rotation_contact_unresolved")
        mid = (lo + hi) / 2
        d2, error = distance(mid)
        if d2 > (radius + error + speed * (hi - lo) / 2) ** 2:
            continue
        if radius >= error and d2 <= (radius - error) ** 2:
            hit = mid if hit is None else min(hit, mid)
            if hit - lower <= F(1, 1 << 27):
                return lower, hit
            stack.append((lo, mid))
        elif hi - lo <= F(1, 1 << 36):
            if pending is None:
                pending = lo
        else:
            stack.extend(((mid, hi), (lo, mid)))
    raise SolverFailure("rotation_cell_capacity")
