"""Bounded native reference routes between proposed push contacts.

No dynamics or state writes: all FK/IK/collision work uses scratch native data.
Each returned phase is sampled; neither proposal nor route is a CCD certificate.
"""

from dataclasses import dataclass
from math import atan2, ceil, cos, pi, sin

import numpy as np

from .manipulation_diagnostic import (
    _configuration,
    _joints,
    _readonly,
    _require,
    solve_native_point_ik,
)
from .manipulation_path import audit_native_joint_path


@dataclass(frozen=True)
class NativePushRoute:
    status: str
    proposal: object
    source_q: np.ndarray
    nodes: tuple
    attempts: tuple
    ik_calls: int
    path_calls: int
    initial_reference_projection: tuple


def orbit_waypoints(center_xy, start_xy, end_xy, radius_m, clockwise):
    """A declared circular detour, never a collision-free guarantee."""
    c, a, b = (np.asarray(x, dtype=float) for x in (center_xy, start_xy, end_xy))
    _require(all(x.shape == (2,) and np.isfinite(x).all() for x in (c, a, b)), "orbit XY vectors")
    _require(type(clockwise) is bool and np.isfinite(radius_m) and radius_m > 0, "orbit policy")
    _require(
        np.linalg.norm(a - c) > 1e-12 and np.linalg.norm(b - c) > 1e-12, "orbit radial direction"
    )
    left, right = atan2(*(a - c)[::-1]), atan2(*(b - c)[::-1])
    delta = (right - left) % (2 * pi)
    if clockwise and delta != 0:
        delta -= 2 * pi
    count = max(1, ceil(abs(delta) / (pi / 4)))
    _require(count <= 8, "orbit waypoint budget")
    # At most eight intermediate points plus the final point at the end angle.
    return tuple(
        tuple(
            c + radius_m * np.array((cos(left + delta * k / count), sin(left + delta * k / count)))
        )
        for k in range(count + 1)
    )


class NativePushRoutePlanner:
    """Serial bounded planner; cumulative attempts are never reset per candidate."""

    def __init__(
        self,
        model,
        joint_names,
        *,
        object_geom_names,
        pusher_geom_name,
        max_candidates=12,
        max_ik_per_segment=64,
        max_ik=512,
        max_paths=128,
    ):
        import mujoco as mj

        self.model = model
        self.names, _, self.qa, _, self.lower, self.upper = _joints(mj, model, joint_names)
        self.object_names = tuple(object_geom_names)
        _require(
            1 <= len(self.object_names) <= 2
            and len(set(self.object_names)) == len(self.object_names),
            "object names",
        )
        self.pusher_name = pusher_geom_name
        self.pusher = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, pusher_geom_name))
        gids = [int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, n)) for n in self.object_names]
        _require(
            self.pusher >= 0 and all(g >= 0 and g != self.pusher for g in gids), "route geometry"
        )
        self.body_name = model.body(int(model.geom_bodyid[self.pusher])).name
        self.local_point = _readonly(model.geom_pos[self.pusher])
        _require(self.body_name is not None, "named pusher body")
        for value, cap in (
            (max_candidates, 12),
            (max_ik_per_segment, 64),
            (max_ik, 512),
            (max_paths, 128),
        ):
            _require(type(value) is int and 1 <= value <= cap, "route budget")
        self.max_candidates, self.max_ik_per_segment, self.max_ik, self.max_paths = (
            max_candidates,
            max_ik_per_segment,
            max_ik,
            max_paths,
        )
        self.ik_calls = self.path_calls = 0

    def plan(self, q, proposals, *, previous_direction=None):
        import mujoco as mj

        source = _configuration(mj, self.model, q)
        _require(type(proposals) is tuple and 1 <= len(proposals) <= 384, "bounded proposal tuple")
        qa = list(self.qa)
        start = source.copy()
        start[qa] = np.clip(start[qa], self.lower, self.upper)
        projection = tuple(start[qa] - source[qa])
        attempts = []
        initial_ik = self.ik_calls
        if previous_direction is not None:
            direction = np.asarray(previous_direction, dtype=float)
            _require(
                direction.shape == (3,)
                and np.isfinite(direction).all()
                and abs(np.linalg.norm(direction) - 1) < 2e-6,
                "previous unit direction",
            )
        else:
            direction = None

        def finish(status, proposal=None, nodes=()):
            return NativePushRoute(
                status,
                proposal,
                _readonly(source),
                tuple(nodes),
                tuple(attempts),
                self.ik_calls,
                self.path_calls,
                projection,
            )

        def solve(target, seed, record, label):
            if (
                self.ik_calls >= self.max_ik
                or self.ik_calls - initial_ik >= self.max_ik_per_segment
            ):
                raise RuntimeError("ik_budget")
            self.ik_calls += 1
            value = solve_native_point_ik(
                self.model,
                target,
                seed,
                self.names,
                body_name=self.body_name,
                local_point=self.local_point,
            )
            record.setdefault("ik", []).append((label, value))
            return value.configuration if value.converged else None

        def phase(knots, label, allowance, record):
            if self.path_calls >= self.max_paths:
                raise RuntimeError("path_budget")
            self.path_calls += 1
            pairs = (
                ()
                if allowance is None
                else tuple((self.pusher_name, name, allowance) for name in self.object_names)
            )
            times = (
                tuple(0.25 * i for i in range(len(knots)))
                if label in ("approach", "withdraw")
                else ((0.0, 0.3) if label == "entry" else (0.0, 0.7, 0.95))
            )
            audit = audit_native_joint_path(
                self.model, times, tuple(knots), self.names, allowed_contacts=pairs
            )
            record.setdefault("paths", []).append((label, audit))
            if not audit.sampled_clear:
                return None
            return tuple((t, label, _readonly(k[qa])) for t, k in zip(times, knots, strict=True))

        def combine(parts):
            nodes = []
            elapsed = 0.0
            for part in parts:
                for j, (stamp, label, robotq) in enumerate(part):
                    if nodes and j == 0:
                        continue
                    nodes.append((elapsed + stamp, label, robotq))
                elapsed += part[-1][0]
            return tuple(nodes)

        scratch = mj.MjData(self.model)
        scratch.qpos[:] = start
        mj.mj_kinematics(self.model, scratch)
        center = scratch.geom_xpos[self.pusher].copy()
        base_parts = []
        approach_start = start
        try:
            if direction is not None:
                release = {"kind": "withdraw"}
                attempts.append(release)
                freed = solve(center - 0.025 * direction, start, release, "withdraw")
                if freed is None:
                    release["reason"] = "withdraw_ik"
                    return finish("withdraw_rejected")
                withdrawal = phase((start, freed), "withdraw", 0.001, release)
                if withdrawal is None:
                    release["reason"] = "withdraw_path"
                    return finish("withdraw_rejected")
                base_parts.append(withdrawal)
                approach_start = freed
                scratch.qpos[:] = freed
                mj.mj_kinematics(self.model, scratch)
                center = scratch.geom_xpos[self.pusher].copy()
            groups = {}
            for p in proposals:
                _require(p.status == "PROPOSAL_ONLY", "proposal status")
                if p.reference_usable and p.predicted_improvement > 0:
                    groups.setdefault(p.index, []).append(p)
            for index, choices in list(groups.items())[: self.max_candidates]:
                candidate = {
                    "kind": "candidate",
                    "index": index,
                    "proposal_bindings": tuple(p.input_binding for p in choices),
                }
                attempts.append(candidate)
                first = choices[0]
                pre = solve(first.precontact_center_world, approach_start, candidate, "precontact")
                if pre is None:
                    candidate["reason"] = "precontact_ik"
                    continue
                contact = solve(first.contact_center_world, pre, candidate, "contact")
                if contact is None:
                    candidate["reason"] = "contact_ik"
                    continue
                approach = phase((approach_start, pre), "approach", None, candidate)
                if approach is None:
                    for clockwise in (False, True):
                        detour = {"kind": "orbit", "clockwise": clockwise, "index": index}
                        candidate.setdefault("detours", []).append(detour)
                        radius = (
                            first.mobility_radius_m
                            + float(self.model.geom_size[self.pusher, 0])
                            + 0.025
                        )
                        points = orbit_waypoints(
                            first.object_com_world[:2],
                            center[:2],
                            first.precontact_center_world[:2],
                            radius,
                            clockwise,
                        )
                        seed = approach_start
                        knots = [seed]
                        for xy in points:
                            seed = solve(
                                (*xy, first.precontact_center_world[2]), seed, detour, "orbit"
                            )
                            if seed is None:
                                break
                            knots.append(seed)
                        if seed is None:
                            detour["reason"] = "orbit_ik"
                            continue
                        final_pre = solve(first.precontact_center_world, seed, detour, "orbit_pre")
                        if final_pre is None:
                            detour["reason"] = "orbit_pre_ik"
                            continue
                        # The orbit branch can change IK solution. Contact must share that branch.
                        final_contact = solve(
                            first.contact_center_world, final_pre, detour, "orbit_contact"
                        )
                        if final_contact is None:
                            detour["reason"] = "orbit_contact_ik"
                            continue
                        knots.append(final_pre)
                        approach = phase(knots, "approach", None, detour)
                        if approach is not None:
                            pre, contact = final_pre, final_contact
                            detour["reason"] = "sampled_clear"
                            break
                        detour["reason"] = "orbit_path"
                if approach is None:
                    candidate["reason"] = "approach_path"
                    continue
                entry = phase((pre, contact), "entry", 0.0005, candidate)
                if entry is None:
                    candidate["reason"] = "entry_path"
                    continue
                for proposal in choices:
                    attempt = {"kind": "push", "distance": proposal.distance_m}
                    candidate.setdefault("pushes", []).append(attempt)
                    pushed = solve(proposal.push_center_world, contact, attempt, "push")
                    if pushed is None:
                        attempt["reason"] = "push_ik"
                        continue
                    pushing = phase(
                        (contact, pushed, pushed), "push", proposal.distance_m + 0.001, attempt
                    )
                    if pushing is None:
                        attempt["reason"] = "push_path"
                        continue
                    attempt["reason"] = candidate["reason"] = "selected"
                    return finish(
                        "sampled_reference_ready",
                        proposal,
                        combine((*base_parts, approach, entry, pushing)),
                    )
            return finish("no_qualified_route")
        except RuntimeError as error:
            if str(error) not in ("ik_budget", "path_budget"):
                raise
            attempts.append({"kind": "budget", "reason": str(error)})
            return finish(str(error))
