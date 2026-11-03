"""Matched donor-reference perturbations and measurements for a finite receiver aperture."""

from dataclasses import dataclass

import numpy as np

from ..control.material_grasp import ParallelJawWindow
from ..core.validation import finite_real


@dataclass(frozen=True)
class DonorPulse:
    direction: tuple[float, float]
    duration: float
    acceleration: float

    @classmethod
    def toward(cls, origin, target, *, duration, peak_speed, peak_acceleration):
        origin, target = (np.asarray(x, dtype=float) for x in (origin, target))
        if origin.shape != (2,) or target.shape != (2,) or not np.isfinite([origin, target]).all():
            raise ValueError("expected two finite planar points")
        duration = finite_real(duration, "pulse duration")
        v = finite_real(peak_speed, "speed")
        a = finite_real(peak_acceleration, "acceleration")
        if min(duration, v, a) <= 0:
            raise ValueError("duration and motion bounds must be positive")
        diff = target - origin
        norm = np.linalg.norm(diff)
        if norm == 0 or not np.isfinite(norm):
            raise ValueError("pulse needs a measured approach direction")
        return cls(tuple(diff / norm), duration, min(a, 2 * v / duration))

    @property
    def distance(self):
        return self.acceleration * self.duration**2 / 4

    def sample(self, time, sign):
        """Translation and velocity relative to the same parked donor command."""
        if sign not in (-1, 0, 1):
            raise ValueError("expected a negative, neutral or positive pulse")
        t = finite_real(time, "sample time")
        if t < 0:
            raise ValueError("sample time must be nonnegative")
        t = min(t, self.duration)
        if t <= self.duration / 2:
            x = 0.5 * self.acceleration * t * t
            v = self.acceleration * t
        else:
            remain = self.duration - t
            x = self.distance - 0.5 * self.acceleration * remain**2
            v = self.acceleration * remain
        axis = sign * np.asarray(self.direction)
        return axis * x, axis * v


def pad_coordinates(first, second, point, clearance):
    """Signed footprint and jaw-axis entry margins, with positive values inside both."""
    window = ParallelJawWindow(first, second)
    p = np.asarray(point, dtype=float)
    reserve = finite_real(clearance, "pad clearance")
    if p.shape != (3,) or not np.isfinite(p).all() or reserve < 0:
        raise ValueError("a finite material point and nonnegative clearance are required")
    sides = []
    for polygon in (np.asarray(first, dtype=float), np.asarray(second, dtype=float)):
        edge = np.roll(polygon, -1, axis=0) - polygon
        normal = np.cross(edge[0], polygon[-1] - polygon[0])
        normal /= np.linalg.norm(normal)
        inward = np.cross(normal, edge) / np.linalg.norm(edge, axis=1)[:, None]
        sides.extend(np.einsum("ij,ij->i", p - polygon, inward))
    axial = min(
        window.axis @ (p - np.mean(first, axis=0)), window.axis @ (np.mean(second, axis=0) - p)
    )
    return dict(
        center=window.center.copy(),
        axis=window.axis.copy(),
        side_margin_m=float(min(sides) - reserve),
        axial_margin_m=float(axial - reserve),
    )


def qualify_response(
    neutral, plus, minus, normal, entrance, *, width, weight, dt, baseline_contact_index
):
    """Apply the frozen paired-response, load and subsequent physical-path conditions."""
    if not (len(neutral) == len(plus) == len(minus)) or not 0 <= baseline_contact_index < len(
        neutral
    ):
        raise ValueError("aligned physical windows and a baseline event index are required")
    normal, entrance = (np.asarray(v, dtype=float) for v in (normal, entrance))
    if any(
        v.shape != (3,) or not np.isclose(np.linalg.norm(v), 1, rtol=0, atol=1e-10)
        for v in (normal, entrance)
    ):
        raise ValueError("unit evaluation directions are required")
    threshold = finite_real(width, "layer width") / 2
    if min(threshold, finite_real(weight, "weight"), finite_real(dt, "dt")) <= 0:
        raise ValueError("positive durations, layer and weight are required")
    series = {1: [], -1: []}
    authority = []
    qualifying = []
    branches = {1: plus, -1: minus}
    for base, positive, negative in zip(neutral, plus, minus, strict=True):
        if max(abs(base["end_s"] - r["end_s"]) for r in (positive, negative)) > 1e-10:
            raise ValueError("response frames are not simultaneous")
        for sign, row in ((1, positive), (-1, negative)):
            dp = np.subtract(row["material_m"], base["material_m"])
            dcenter = np.subtract(row["jaw_center_m"], base["jaw_center_m"])
            series[sign].append(
                dict(
                    end_s=row["end_s"],
                    hazard_projection_m=float(normal @ dp),
                    entrance_projection_m=float(entrance @ (dp - dcenter)),
                    displacement_m=dp.tolist(),
                    support=row["min_A_weaker_pad_n"] > 0.20
                    and row["mean_A_up_n"] > 0.25 * weight
                    and row.get("within_motion_budget", True),
                    minimum_structure_change_m=row["min_structure_m"] - base["min_structure_m"],
                )
            )
    base_event = neutral[baseline_contact_index]["first_structure_force_s"]
    candidate_checks = {}
    for sign, rows in branches.items():
        response = series[sign]
        opposite = series[-sign]
        intervals = {}
        for key in ("hazard_projection_m", "entrance_projection_m"):
            intervals[key] = [
                i
                for i in range(1, len(rows))
                if all(
                    response[j]["support"]
                    and opposite[j]["support"]
                    and response[j][key] >= threshold
                    and opposite[j][key] <= 0
                    for j in (i - 1, i)
                )
            ]
        has_authority = any(intervals.values())
        if has_authority:
            authority.append(sign)
        event = next(
            (
                r["first_structure_force_s"]
                for r in rows
                if r["first_structure_force_s"] is not None
            ),
            None,
        )
        delay = (
            None
            if base_event is None
            else (rows[-1]["end_s"] if event is None else event) - base_event
        )
        baseline_row = rows[baseline_contact_index]
        bearing = response[baseline_contact_index]["support"]
        gain = baseline_row["min_structure_m"] - neutral[baseline_contact_index]["min_structure_m"]
        shared = any(x["shared"] and y["shared"] for x, y in zip(rows, rows[1:]))
        downstream = shared or (
            bearing and (gain >= threshold or (delay is not None and delay >= dt - 1e-12))
        )
        if has_authority and downstream:
            qualifying.append(sign)
        candidate_checks[sign] = dict(
            two_frame_authority_end_indices=intervals,
            baseline_event_gap_gain_m=gain,
            first_structure_force_s=event,
            delay_or_lower_bound_s=delay,
            censored=event is None,
            actual_shared_two_frames=shared,
            downstream=downstream,
        )
    return dict(
        go=bool(qualifying),
        authority_signs=authority,
        qualifying_signs=qualifying,
        threshold_m=threshold,
        response=series,
        candidates=candidate_checks,
    )
