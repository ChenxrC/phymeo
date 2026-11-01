"""Use force derivatives to resolve between-node force modes on a common motion path."""

from fractions import Fraction as Q

import numpy as np
from numpy.polynomial import chebyshev as ch
from p65_common_motion import Forces, need


def additional_modes(nodes, card, values, unit_slopes):
    """Seven new modes vanish at all nodes. All value and derivative arrays share one basis."""
    degree = len(nodes) - 1
    need(
        degree == 13 and values.shape == unit_slopes.shape and values.shape[0] == 14,
        "slope basis schema",
    )
    z = 2 * nodes - 1
    exact = [Q(2 ** (2 * len(nodes) - 1))]
    for s in nodes:
        factor = Q(1, 2) - Q(float(s))
        row = [Q(0)] * (len(exact) + 1)
        for j, c in enumerate(exact):
            row[j] += factor * c
            row[j + 1] += c / 4
            row[abs(j - 1)] += c / 4
        exact = row
    omega = np.array(exact, float)
    deriv = 2 * ch.chebval(z, ch.chebder(omega))
    need(
        np.min(abs(deriv)) > 1 and np.max(abs(ch.chebval(z, omega))) < 1e-11, "bad node polynomial"
    )
    low = card @ values
    ds_low = ch.chebvander(z, degree - 1) @ (2 * ch.chebder(low, axis=0))
    # The degree-six coefficients use all fourteen physical time slopes.
    q = (card @ ((unit_slopes - ds_low) / deriv[:, None]))[:7]
    high = np.zeros((21, values.shape[1]))
    for j, ow in enumerate(omega):
        for k, qk in enumerate(q):
            high[j + k] += ow * qk / 2
            high[abs(j - k)] += ow * qk / 2
    need(np.max(abs(ch.chebvander(z, 20) @ high)) < 1e-10, "nodal value changed by added modes")
    return high


def primitives(coeff, nodes, h):
    z = 2 * nodes - 1
    result = []
    for order in (1, 2):
        ant = ch.chebint(coeff, m=order, lbnd=-1, scl=0.5, axis=0)
        vals = (h**order) * (ch.chebvander(z, len(ant) - 1) @ ant)
        vals[0] = 0.0
        result.append(vals)
    return result


def packed(f):
    return np.concatenate((f.material, f.pressures, f.rows.ravel()))


def unpacked(coeff, n, owners):
    return Forces(
        coeff[:, :n],
        coeff[:, n + owners :].reshape(len(coeff), owners, n),
        coeff[:, n : n + owners],
    )


class SlopeModes:
    def __init__(self):
        self.coeff = None

    def acceleration(self, coeff, model):
        fields = unpacked(coeff, len(model.mass), len(model.owners))
        return (fields.material + fields.rows.sum(axis=1)) / model.mass

    def motion(self, nodes, model, h):
        return primitives(self.acceleration(self.coeff, model), nodes, h)

    def update(self, model, h, nodes, card, V, P, jets, frames, matrix, a, da, v, dx, du):
        values = np.stack([packed(f) for f in frames])
        slopes = []
        for jet, vel, acc in zip(jets, v + du, a + da, strict=True):
            slopes.append(
                packed(
                    Forces(jet.mq @ vel, jet.rq @ vel + jet.rv @ acc, jet.pq @ vel + jet.pv @ acc)
                )
            )
        high = additional_modes(nodes, card, values, h * np.stack(slopes))
        delta_high = high if self.coeff is None else high - self.coeff
        extra_v, extra_x = primitives(self.acceleration(delta_high, model), nodes, h)
        rhs = np.stack(
            [
                ((jet.mq + jet.rq.sum(axis=0)) @ x + jet.rv.sum(axis=0) @ u) / model.mass
                for jet, x, u in zip(jets, extra_x, extra_v, strict=True)
            ]
        )
        d = len(model.mass)
        feedback = np.zeros_like(da)
        feedback[1:] = np.linalg.solve(matrix[d:, d:], rhs[1:].ravel()).reshape(len(nodes) - 1, d)
        need(
            np.max(abs(matrix @ feedback.ravel() - rhs.ravel())) < 1e-10,
            "slope feedback linear solve",
        )
        new_dx = dx + extra_x + h * h * (P @ feedback)
        new_du = du + extra_v + h * (V @ feedback)
        revised = [
            jet.at_linear_increment(x, u) for jet, x, u in zip(jets, new_dx, new_du, strict=True)
        ]
        linear_acc = np.stack(
            [(model.external + f.material + f.rows.sum(axis=0)) / model.mass for f in revised]
        )
        error = float(np.max(abs(linear_acc - (a + da + feedback))))
        need(error < 1e-10, "slope feedback did not preserve the shared motion")
        self.coeff = high
        raw = dict(
            coefficients=high.copy(),
            feedback_acceleration=feedback,
            feedback_error=error,
            integrated_velocity=extra_v,
            integrated_position=extra_x,
            normalized_force_slopes=h * np.stack(slopes),
        )
        return revised, da + feedback, raw

    def fields(self, n, owners):
        need(self.coeff is not None, "missing slope modes")
        return unpacked(self.coeff, n, owners)
