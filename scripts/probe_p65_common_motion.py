"""One opt-in public HOST SceneProgram for a research common motion path."""

import argparse
import gzip
import json
from fractions import Fraction as Q
from hashlib import sha256
from pathlib import Path

import numpy as np
from audit_p65_gravity_six_owner import normal_rate
from check_p65_common_motion import Rejected, verify_curve
from p65_common_motion import DEGREE, GRID, Forces, checkpoint, endpoint, fit, need, unpack
from probe_p65_gravity_six_owner import local_force, offset
from probe_p65_joint_readback import PLAN as FIXTURE
from probe_p65_joint_readback import fixture, save
from verify_p65_contact_layer_screen import sha

import phymeo as ph
from phymeo.constraints.bending import bending_angle_gradient
from phymeo.core.capabilities import SolverFailure
from phymeo.core.scene import HistoryField, StageMetrics, fingerprint
from phymeo.runtime.program import ProgramCapabilities
from phymeo.runtime.scene_stepper import SceneStepper

PLAN = FIXTURE.with_name("p65-common-motion-protocol.json")
PLAN_SHA = "2e4fe64e768055c07333889e7189bf8e35ef1cfb2529c1831f9a71ef43410ce8"


def thaw(item):
    if isinstance(item, dict):
        if set(item) == {"dtype", "shape", "values"}:
            x = np.asarray(item["values"], dtype=item["dtype"])
            need(list(x.shape) == item["shape"], "stored array shape")
            return x
        return {k: thaw(v) for k, v in item.items()}
    if isinstance(item, list):
        return [thaw(x) for x in item]
    return item


def saved(path):
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    return thaw(json.loads(data))


def joint_state(q, v, packet):
    pos = np.asarray(q, float)
    vel = np.asarray(v, float)
    return ph.SceneState(
        {
            "cloth": ph.ParticleState(pos[:12].reshape(4, 3), vel[:12].reshape(4, 3)),
            "robot": ph.RobotState(pos[12:], vel[12:]),
        },
        {"motion": packet},
    )


def coordinates(state):
    c = state.domains["cloth"]
    r = state.domains["robot"]
    return np.r_[c.position.ravel(), r.q], np.r_[c.velocity.ravel(), r.v]


class ClothPlatform:
    """Six original material ports share the same returned force state and domain masses."""

    def __init__(self, parts):
        _, _, _, _, self.material, self.robot, self.cloth, self.surfaces, patch = parts
        self.parts = parts
        self.patch_vertices = patch.local_vertices
        self.rho = self.surfaces[0].half_thickness_m
        self.rest = Q(float(self.material.bending.rest_angle[0]))
        self.faces = np.array([s.node_indices for s in self.surfaces], int)
        self.owners = tuple(
            s.key + "/" + str(node) + "/lower" for s in self.surfaces for node in s.node_indices
        )
        self.mass = np.r_[np.repeat(1 / self.cloth.inverse_mass, 3), self.robot.mass[0]]
        self.external = np.r_[np.tile(self.cloth.gravity, 4) * self.mass[:12], 0.0]
        self.signature = fingerprint(
            dict(
                law="paired SC-L1 common motion v1",
                scene=parts[0].identity,
                material=self.material,
                surfaces=self.surfaces,
                patch=patch,
                owners=self.owners,
                area=1 / 6,
                stiffness=500.0,
                damping=1000.0,
                width=0.002,
            )
        )

    def maps(self, q, v):
        state = ph.ParticleState(q[:12].reshape(4, 3), v[:12].reshape(4, 3))
        port = offset(self.cloth, state, self.surfaces)
        D = np.zeros((len(self.owners), len(q)))
        for owner, ids in enumerate(np.repeat(self.faces, 3, axis=0)):
            for i, node in enumerate(ids):
                D[owner, 3 * node : 3 * node + 3] += port.jacobians[owner, i, 2]
        D[:, -1] = -1.0
        return (
            D,
            port.positions()[:, 2] - q[-1] - self.patch_vertices[0, 2],
            port.velocities()[:, 2] - v[-1],
        )

    def forces(self, q, v):
        state = np.r_[q[:12], v[:12], q[-1], v[-1]]
        _, field = local_force(state, self.parts)
        D, _, _ = self.maps(q, v)
        loads = D * field["force"][:, None]
        need(
            np.max(abs(loads[:, :12].sum(axis=0) - field["contact_force"].ravel())) < 1e-12,
            "full node map",
        )
        return Forces(np.r_[field["material_force"].ravel(), 0.0], loads, field["force"])

    def linearization(self, q, v, first):
        D, gap, rate = self.maps(q, v)
        area = 1 / 6
        active = (gap < 0.002) & (rate < 0.001)
        dq = -D.T @ ((area * 500 * np.maximum(1 - 1000 * rate, 0) * active)[:, None] * D)
        dv = -D.T @ ((area * 500 * 1000 * np.maximum(0.002 - gap, 0) * active)[:, None] * D)
        x = q[:12].reshape(4, 3)
        for (a, b), alpha in zip(
            self.material.distance.indices, self.material.distance.compliance, strict=True
        ):
            u = x[a] - x[b]
            u /= np.linalg.norm(u)
            row = np.zeros(len(q))
            row[3 * a : 3 * a + 3] = u
            row[3 * b : 3 * b + 3] = -u
            dq -= np.outer(row, row) / alpha
        for ids, alpha in zip(
            self.material.bending.indices, self.material.bending.compliance, strict=True
        ):
            _, grad = bending_angle_gradient(x[ids])
            row = np.zeros(len(q))
            for node, d in zip(ids, grad, strict=True):
                row[3 * node : 3 * node + 3] = d
            dq -= np.outer(row, row) / alpha
        return dq, dv

    def endpoint_rate_error(self, q, v, expected):
        x = np.asarray(q[:12], float).reshape(4, 3)
        vel = np.asarray(v[:12], float).reshape(4, 3)
        rates = []
        for face in self.faces:
            _, dn = normal_rate(x, vel, face)
            rates.extend(vel[face, 2] - float(v[-1]) - self.rho * dn[2])
        return float(np.max(np.abs(np.array(rates) - expected)))


class PathProgram:
    capabilities = ProgramCapabilities(
        "numpy", {"float64"}, {"particles", "robot"}, {"common_motion"}, history="explicit"
    )
    history_fields = (HistoryField("motion", (256,), np.int32),)

    def __init__(self, model, reference=None, fault_curve=None, fault=None):
        self.model = model
        self.reference = reference
        self.fault_curve = fault_curve
        self.fault = fault
        self.attempt = {}

    def problem_signature(self):
        return dict(model=self.model.signature, degree=DEGREE, grid=GRID, protocol=PLAN_SHA)

    def create_workspace(self, scene, state, inputs, h, settings, context):
        if scene.identity != self.model.parts[0].identity or context.backend != "numpy":
            raise SolverFailure("unsupported original scene")
        for inp in inputs.domains.values():
            if any(np.any(a) for a in vars(inp).values()):
                raise SolverFailure("unsupported nonzero original nongravity inputs in this chart")
        return Workspace(self, state, inputs, h)

    def close(self):
        pass


class Workspace:
    def __init__(self, program, state, inputs, h):
        self.program = program
        self.model = program.model
        self.state = state
        self.h = h
        self.packet = state.history["motion"]
        self.binding = fingerprint((program.problem_signature(), state, inputs, h))
        self.receipt = None

    def predict(self):
        try:
            q, v, t = unpack(self.packet, self.model.signature)
            qread, vread = coordinates(self.state)
            need(
                np.array_equal(qread, np.asarray(q, float))
                and np.array_equal(vread, np.asarray(v, float)),
                "registered input disagrees with checkpoint",
            )
            claim = self.program.fault_curve
            if claim is None:
                claim = fit(self.model, q, v, self.h, self.binding)
                claim.update(
                    incoming_position=np.asarray(q, float), incoming_velocity=np.asarray(v, float)
                )
            self.claim = claim
            q1, v1 = endpoint(claim, q, v)
            packet = checkpoint(q1, v1, t + Q(self.h), self.model.signature)
            self.out = joint_state(q1, v1, packet)
            self.program.attempt = dict(curve=claim, candidate=self.out, inlet=self.state)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            raise SolverFailure(str(exc)) from exc

    def position_sweep(self):
        try:
            if self.receipt is None:
                self.receipt = verify_curve(
                    self.claim,
                    self.packet,
                    self.out.history["motion"],
                    self.model,
                    self.binding,
                    self.program.reference,
                )
                self.program.attempt["certificate"] = self.receipt
                if not self.receipt["accepted"]:
                    failed = [k for k, val in self.receipt["checks"].items() if not val]
                    raise Rejected("independent common path checks: " + ", ".join(failed))
        except (ValueError, ArithmeticError) as exc:
            self.program.attempt["rejected_reason"] = str(exc)
            raise SolverFailure(str(exc)) from exc

    def position_metrics(self):
        return StageMetrics(
            {"position_force_defect": self.receipt["node_force_position_defect_upper_m"]}
        )

    def prepare_velocity(self):
        if not self.receipt["accepted"]:
            raise SolverFailure("unverified common path")

    def velocity_sweep(self):
        # Position and velocity are defined by a single integral. No second update.
        return None

    def velocity_metrics(self):
        actual = self.claim["iterations"][-1]["residual"] * self.h
        return StageMetrics({"collocation_change": float(actual)})

    def result(self):
        return self.out

    def diagnostics(self):
        return dict(common_motion=self.program.attempt)

    def close(self):
        self.receipt = None


def verified_config():
    spec = json.loads(PLAN.read_bytes())
    previous = PLAN.with_name(spec["previous"])
    f = json.loads(FIXTURE.read_bytes())
    need(
        sha(PLAN, normalize=True) == PLAN_SHA
        and sha(previous, normalize=True) == spec["previous_sha256_lf"]
        and sha(FIXTURE, normalize=True) == spec["fixture_sha256_lf"],
        "protocol binding",
    )
    need(
        Path(ph.__file__).resolve().parent == (PLAN.parents[2] / "src/phymeo").resolve(),
        "actual project source",
    )
    return f, json.loads(previous.read_bytes())["coupled_reference"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("first", "next", "faults"), required=True)
    args = parser.parse_args()
    root = args.root
    f, reference = verified_config()
    parts = fixture(f)
    scene, initial, inputs, base = parts[:4]
    model = ClothPlatform(parts)
    pos, vel = coordinates(initial)
    inlet = checkpoint(pos, vel, Q(0), model.signature)
    if args.phase == "next":
        audit = saved(root / "first-audit.json")
        data = (root / "first-output.bin").read_bytes()
        need(
            audit["accepted"] and sha256(data).hexdigest() == audit["output_sha256"],
            "uncertified disk successor",
        )
        inlet = np.frombuffer(data, dtype="<i4").copy()
    q, v, _ = unpack(inlet, model.signature)
    state = joint_state(q, v, inlet)
    before = fingerprint((state, inputs))
    cases = ("inlet", "owner", "bubble") if args.phase == "faults" else (args.phase,)
    for case in cases:
        fault_curve = None
        if args.phase == "faults":
            fault_curve = saved(root / "first-attempt.json.gz")["curve"]
            if case == "inlet":
                fault_curve["incoming_velocity"][0] += 0.125
            elif case == "owner":
                fault_curve["owners"] = fault_curve["owners"][1:]
            else:
                # a(t)=1-6s+6s² has both temporal moments zero. Equal/opposite internal load.
                for node, sgn in ((0, 1), (1, -1)):
                    fault_curve["material"][0, 3 * node] += sgn * GRID // 64
                    fault_curve["material"][2, 3 * node] += sgn * 3 * GRID // 64
        program = PathProgram(
            model, reference if args.phase == "first" else None, fault_curve, case
        )
        with SceneStepper(
            scene,
            program,
            ph.StepSettings(
                substeps=1,
                position_iterations=1,
                velocity_iterations=1,
                tolerance=1e-8,
                record_substeps=True,
            ),
            ph.BackendContext("numpy"),
            base.registry,
        ) as runtime:
            result = runtime.step(state, inputs, float(Q(f["h_s"])))
        save(root / f"{case}-attempt.json.gz", program.attempt)
        save(root / f"{case}-result.json.gz", result)
        accepted = result.report.accepted
        reason = result.report.reason
        need(fingerprint((state, inputs)) == before, "caller state or inputs mutated")
        print(case, "accepted", accepted, "reason", reason)
        if not accepted:
            need(
                fingerprint(result.state) == fingerprint(state) and not result.substep_states,
                "failed real frame leaked state",
            )
            if args.phase != "faults":
                return 2
            if case == "bubble":
                need(
                    fingerprint(program.attempt["candidate"])
                    == fingerprint(saved_state(root, "first")),
                    "bubble changed endpoint",
                )
                need(
                    "force_position_defect" in str(reason)
                    or "force_velocity_defect" in str(reason),
                    "bubble not distinguished by physical force",
                )
            continue
        need(args.phase != "faults", "forged path accepted")
        with (root / f"{case}-output.bin").open("xb") as out:
            out.write(result.state.history["motion"].tobytes())
        rec = program.attempt["certificate"]
        print("hard/law/energy", rec["checks"], "J error", max(rec["j_defect_upper_ns"]))
    base.close()
    return 0


def saved_state(root, case):
    result = saved(root / f"{case}-result.json.gz")["state"]
    p = result["history"]["motion"]
    cloth = result["domains"]["cloth"]
    robot = result["domains"]["robot"]
    return ph.SceneState(
        {
            "cloth": ph.ParticleState(cloth["position"], cloth["velocity"]),
            "robot": ph.RobotState(robot["q"], robot["v"]),
        },
        {"motion": p},
    )


if __name__ == "__main__":
    raise SystemExit(main())
