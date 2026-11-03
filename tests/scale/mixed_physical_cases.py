"""The four-world mixed G2/G3 unit, its original steppers and its checked public frame."""

import json
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256

import numpy as np
from test_resident_step import slider
from test_world_contact_loads import fixture

import phymeo as ph

H = 1 / 256
WINDOW = 4


@dataclass
class Population:
    worlds: list
    roles: dict
    inputs: list


def population(count):
    if count not in (4, 16):
        raise ValueError("one or four physical units required")
    worlds = []
    roles = {}
    frames = [({}, {}) for _ in range(WINDOW)]
    for unit in range(count // 4):
        hit, ball, clear, rail = (1009 * (unit + 1) + i for i in (31, 3, 67, 11))
        hit_setup = fixture()
        clear_setup = fixture(lift=0.5)
        drive, initial = slider(delay=1)
        worlds.extend(
            (
                (hit, ph.ClothLinkWorld(*hit_setup[:2])),
                (ball, ph.FreeSphere(2.0, 0.125, position=(0, 0, 2), gravity=(0, 0, -8))),
                (clear, ph.ClothLinkWorld(*clear_setup[:2])),
                (rail, ph.PrismaticSlider(drive, initial)),
            )
        )
        roles.update([(hit, "contact"), (ball, "ball"), (clear, "separated"), (rail, "slider")])
        for index, (controls, loads) in enumerate(frames):
            sign = 1 if index < 2 else -1
            controls[ball] = (sign * unit / 16, 0, -unit / 32)
            controls[rail] = drive.inputs(
                np.array([sign * (1 + unit / 8)]),
                np.array([sign * 0.25]),
                feedforward=np.array([unit / 16]),
            )
            for key in (hit, clear):
                load = ph.SceneInputs(
                    {"cloth": ph.StepInputs.zeros(3), "arm": ph.RobotInputs.zeros(1)}
                )
                factor = sign * unit if key == hit else -sign * unit
                load.domains["cloth"].force[unit % 3] = [factor / 16, 0, factor / 32]
                load.domains["arm"].effort[0] = factor / 8
                loads[key] = load
    return Population(worlds, roles, frames)


@dataclass(frozen=True)
class PhysicalFrame:
    flags: tuple
    material: tuple
    checkpoints: tuple


def signature(frames):
    data = json.dumps(
        [asdict(x) for x in frames], sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return sha256(data.encode("utf8")).hexdigest()


class MixedRun:
    def __init__(self, case, grouped, mode, cache, stack):
        self.case = case
        self.keys = tuple(k for k, _ in case.worlds)
        self.mode = mode
        self.group = (
            stack.enter_context(ph.TyxBatchedScene(case.worlds, mode, cache_root=cache))
            if grouped
            else None
        )
        self.singles = {}
        if not grouped:
            for key, spec in case.worlds:
                factory = (
                    ph.TyxWorldContactBatch
                    if isinstance(spec, ph.ClothLinkWorld)
                    else ph.TyxWorldBatch
                )
                self.singles[key] = stack.enter_context(
                    factory([(key, spec)], mode, cache_root=cache)
                )

    def snapshot(self):
        if self.group is not None:
            return self.group.snapshot()
        return {k: work.snapshot()[k] for k, work in self.singles.items()}

    def reset(self, keys=None):
        if self.group is not None:
            self.group.reset(keys)
        else:
            for k in self.keys if keys is None else keys:
                self.singles[k].reset()

    def advance(self, index, capacity=None):
        controls, loads = self.case.inputs[index]
        if self.group is not None:
            self.group.set_controls(controls)
            out = self.group.step(H, loads, site_capacity=capacity)
            flags, material = out.accepted, out.contact_results
        else:
            flags, material = {}, {}
            for k, work in self.singles.items():
                if k in loads:
                    cap = {k: capacity[k]} if capacity is not None and k in capacity else None
                    result = work.step(H, {k: loads[k]}, site_capacity=cap)[k]
                    flags[k] = result.accepted
                    material[k] = result
                else:
                    work.set_controls({k: controls[k]})
                    flags[k] = work.step(H)[k]
        checkpoints = self.snapshot()
        return PhysicalFrame(
            tuple(flags.items()), tuple(material.items()), tuple(checkpoints.items())
        )

    def counts(self):
        if self.group is None:
            sources = [
                (
                    "material" if self.case.roles[k] in ("contact", "separated") else "resident",
                    op.stats(),
                )
                for k, op in self.singles.items()
            ]
        else:
            group = self.group.stats()
            sources = [(kind, group[kind]) for kind in ("material", "resident")]
        select = {
            "resident": (
                "submitted_kernels",
                "synchronizations",
                "status_downloaded_bytes",
                "snapshot_downloaded_bytes",
                "control_uploaded_bytes",
                "device_allocations",
                "accepted_world_steps",
                "rejected_world_steps",
            ),
            "material": (
                "completed_kernels",
                "synchronizations",
                "uploaded_bytes",
                "static_uploaded_bytes",
                "downloaded_bytes",
                "device_allocations",
                "engine_builds",
            ),
        }
        result = Counter()
        for kind, stats in sources:
            for name in select[kind]:
                result[kind + "_" + name] += stats.get(name, 0)
        if self.mode == "cuda":
            import tyx

            native = tyx.runtime_stats()
            for k in ("cuda_kernel_launches", "cuda_h2d_bytes", "cuda_d2h_bytes", "cuda_d2d_bytes"):
                result[k] = getattr(native, k)
        return dict(result)


class HostReferences:
    """Original single-system CPU physics, separately from the same-output TYX comparison."""

    def __init__(self, case, stack):
        self.case = case
        self.ops = {}
        self.states = {}
        self.models = {}
        setting = ph.StepSettings(substeps=1, position_iterations=1, velocity_iterations=1)
        for k, spec in case.worlds:
            role = case.roles[k]
            if role == "ball":
                m = spec.mass
                self.models[k] = ph.RigidBodyModel(
                    np.array([1 / m]),
                    np.full((1, 3), 0.4 * m * spec.radius**2),
                    np.array(spec.gravity),
                )
                self.ops[k] = ph.RigidSemiImplicitSolver()
                self.states[k] = ph.RigidBodyState(
                    np.array([spec.position]),
                    np.array([[0.0, 0.0, 0.0, 1.0]]),
                    np.array([spec.velocity]),
                    np.zeros((1, 3)),
                )
            else:
                program = spec.program
                self.ops[k] = stack.enter_context(program.stepper(setting))
                self.states[k] = spec.state if role == "slider" else spec.initial
        self.max_diffs = {}
        self.max_work_residual = 0.0
        self.max_momentum = 0.0

    def compare(self, a, b, atol=8e-7, field="other"):
        a, b = np.asarray(a), np.asarray(b)
        self.max_diffs[field] = max(self.max_diffs.get(field, 0.0), float(np.max(np.abs(a - b))))
        np.testing.assert_allclose(a, b, rtol=2e-6, atol=atol)

    def advance_and_check(self, index, frame):
        controls, loads = self.case.inputs[index]
        mats = dict(frame.material)
        assert list(dict(frame.flags)) == list(self.case.roles) and all(dict(frame.flags).values())
        for k, point in frame.checkpoints:
            role = self.case.roles[k]
            assert point.tick == index + 1
            if role == "ball":
                inputs = ph.RigidBodyInputs(np.array([controls[k]]), np.zeros((1, 3)))
                out = self.ops[k].step(self.models[k], self.states[k], inputs, H)
                assert out.report.accepted
                self.compare(point.position, out.state.position[0], field="position_m")
                self.compare(point.velocity, out.state.velocity[0], field="velocity_m_s")
            elif role == "slider":
                out = self.ops[k].step(self.states[k], controls[k], H)
                assert out.report.accepted
                p = out.state.domains["robot"]
                self.compare(point.position, p.q, field="position_m")
                self.compare(point.velocity, p.v, field="velocity_m_s")
                self.compare(
                    point.applied_effort,
                    out.report.substeps[0].diagnostics["actuator_effort"],
                    field="force_n",
                )
                self.compare(point.history, out.state.history["robot/drive_queue"].ravel())
            else:
                out = self.ops[k].step(self.states[k], loads[k], H)
                assert out.report.accepted and mats[k].accepted
                cloth = out.state.domains["cloth"]
                arm = out.state.domains["arm"]
                self.compare(point.position, cloth.position, field="position_m")
                self.compare(point.velocity, cloth.velocity, field="velocity_m_s")
                self.compare(point.q, arm.q[0], field="position_m")
                self.compare(point.qd, arm.v[0], field="velocity_m_s")
                audit = mats[k].audit
                assert audit == point.last_audit
                if role == "contact":
                    assert 1 <= len(audit.sample.sites) <= 2 and audit.sample.normal_force_n > 0
                    if index == 0:
                        assert (
                            len(audit.sample.sites) == 2
                            and abs(audit.sample.normal_force_n - 2) < 1e-6
                        )
                else:
                    assert audit.sample.sites == () and audit.sample.normal_force_n == 0
                report = out.report.substeps[0].diagnostics
                assert report["force_sample"] == "substep-midpoint"
                body = report["robot_contact"]
                self.compare(
                    audit.sample.joint_effort_n,
                    body["cloth_link_efforts"]["arm"][0],
                    field="force_n",
                )
                original_nodes = np.zeros((3, 3))
                if role == "contact":
                    (record,) = body["cloth_links"]
                    self.compare(audit.sample.gap_m, record["gap_m"], field="position_m")
                    self.compare(
                        audit.sample.normal_force_n, record["normal_force_n"], field="force_n"
                    )
                    self.compare(audit.sample.power_w, record["power_w"], field="power_w")
                    native_sites = sorted(record["sites"], key=lambda s: s["axis_parameter"])
                    grouped_sites = sorted(audit.sample.sites, key=lambda s: s.axis_parameter)
                    for s, native in zip(grouped_sites, native_sites, strict=True):
                        self.compare(s.axis_parameter, native["axis_parameter"], field="parameters")
                        self.compare(s.normal_force_n, native["normal_force_n"], field="force_n")
                        for node, weight in native["face_weights"]:
                            original_nodes[node] += weight * np.asarray(native["force_on_cloth_n"])
                else:
                    assert body["cloth_links"] == [] or body["cloth_links"] == ()
                self.compare(audit.sample.node_forces_n, original_nodes, field="force_n")
                old = self.states[k].domains
                work = np.sum(
                    (cloth.position - old["cloth"].position) * loads[k].domains["cloth"].force
                ) + float((arm.q - old["arm"].q) @ loads[k].domains["arm"].effort)
                self.compare(audit.external_work_j, work, atol=2e-8, field="work_j")
                nodes = np.asarray(audit.sample.node_forces_n)
                self.compare(nodes.sum(axis=0) + audit.sample.link_force_n, np.zeros(3))
                gap = abs(
                    audit.kinetic_change_j
                    - audit.external_work_j
                    - audit.contact_work_j
                    - audit.elastic_work_j
                )
                impulse = np.max(
                    np.abs(
                        np.array(audit.momentum_change_ns)
                        - audit.external_impulse_ns
                        - audit.support_impulse_ns
                    )
                )
                self.max_work_residual = max(self.max_work_residual, gap)
                self.max_momentum = max(self.max_momentum, float(impulse))
                assert gap < 3e-6 and impulse < 3e-6
            self.states[k] = out.state
