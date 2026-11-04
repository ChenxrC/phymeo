"""Geometry-driven, bounded rendezvous for two planar hands before measured load transfer."""

import numpy as np

from .grasp_transfer import bounded_approach


class PlanarPresentation:
    """Share the horizontal material-to-hand displacement while keeping the donor at its held height.

    The intercept initially splits the observed horizontal reach equally. The receiving hand
    continues to measure the actual cloth or exterior waypoint at every control interval.
    An address acquisition or a lost donor load changes the donor's goal to a measured hold.
    """

    def __init__(self):
        self.goal = None
        self.stopped = None

    def update(
        self,
        *,
        donor,
        donor_velocity,
        receiver,
        receiver_velocity,
        material,
        waypoint,
        waypoint_velocity,
        held_height,
        loaded,
        closing,
        dt,
        peak_speed,
        peak_acceleration,
    ):
        vectors = tuple(
            np.asarray(v, dtype=float)
            for v in (
                donor,
                donor_velocity,
                receiver,
                receiver_velocity,
                material,
                waypoint,
                waypoint_velocity,
            )
        )
        if any(v.shape != (2,) or not np.isfinite(v).all() for v in vectors) or not np.isfinite(
            held_height
        ):
            raise ValueError("expected measured finite x/z tool and material coordinates")
        a, va, b, vb, p, goal, rate = vectors
        if self.goal is None:
            if not loaded or closing:
                return None
            self.goal = np.array([a[0] + 0.5 * (b[0] - p[0]), held_height])
            self.goal.flags.writeable = False
        if (closing or not loaded) and self.stopped is None:
            self.stopped = a.copy()
            self.stopped.flags.writeable = False
        commanded = self.goal if self.stopped is None else self.stopped
        at, av = bounded_approach(
            a,
            va,
            commanded,
            np.zeros(2),
            dt,
            peak_speed=peak_speed / 2,
            peak_acceleration=peak_acceleration,
        )
        projected = goal.copy()
        projected_rate = rate.copy()
        if self.stopped is None:
            projected[0] += commanded[0] - a[0]
            projected_rate[0] -= va[0]
        bt, bv = bounded_approach(
            b,
            vb,
            projected,
            projected_rate,
            dt,
            peak_speed=peak_speed,
            peak_acceleration=peak_acceleration,
        )
        return (
            at,
            av,
            bt,
            bv,
            dict(
                action="present" if self.stopped is None else "hold",
                donor_goal_xz_m=commanded.tolist(),
                receiver_waypoint_xz_m=projected.tolist(),
                remaining_donor_x_m=float(commanded[0] - a[0]),
                vertical_command_axis="receiver",
            ),
        )
