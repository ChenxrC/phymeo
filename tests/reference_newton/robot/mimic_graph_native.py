"""Native Newton fan-out momentum and actual import outcome for a follower that drives another."""

import argparse
import json
from pathlib import Path

import numpy as np
from mimic_graph_oracle import H, fan, kkt

ROOT = Path(__file__).parent


def make(case):
    import newton

    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    b.default_shape_cfg.density = 0
    b.add_urdf(
        str(ROOT / f"mimic_{case}.urdf"),
        floating=False,
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        ignore_inertial_definitions=False,
        joint_damping_mode="passive",
        up_axis=newton.Axis.Z,
    )
    return b


def real_fan(device, ref):
    import newton

    b = make("fan")
    labels = [str(n).rsplit("/", 1)[-1] for n in b.joint_label]
    joints = [labels.index(n) for n in ref["names"]]
    iq = [b.joint_q_start[i] for i in joints]
    iv = [b.joint_qd_start[i] for i in joints]
    for j in iv:
        for key in (
            "joint_armature",
            "joint_damping",
            "joint_target_ke",
            "joint_target_kd",
            "joint_limit_ke",
            "joint_limit_kd",
        ):
            getattr(b, key)[j] = 0.0
    model = b.finalize(device=device)
    state, after, ctrl = model.state(), model.state(), model.control()
    q = state.joint_q.numpy()
    v = state.joint_qd.numpy()
    q[iq] = ref["q"]
    v[iv] = ref["v"]
    state.joint_q.assign(q)
    state.joint_qd.assign(v)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    child = model.joint_child.numpy()[joints]
    offset = int(model.joint_qd_start.numpy()[int(model.articulation_start.numpy()[0])])
    cols = np.asarray(iv) - offset
    mass = newton.eval_mass_matrix(model, state).numpy()[0][np.ix_(cols, cols)].astype(float)
    f = np.zeros_like(state.body_f.numpy())
    f[child, :3] = ref["forces"]
    state.clear_forces()
    state.body_f.assign(f)
    drive = np.zeros_like(ctrl.joint_f.numpy())
    drive[iv] = ref["motor"]
    ctrl.joint_f.assign(drive)
    solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    solver.step(state, after, ctrl, None, H)
    v0 = state.joint_qd.numpy()[iv].astype(float)
    v1 = after.joint_qd.numpy()[iv].astype(float)
    acc = (v1 - v0) / H
    balance = mass @ acc - ref["external"] - ref["motor"]
    leaders = model.joint_mimic_joint.numpy()
    coeff = model.joint_mimic_coeffs.numpy()
    return dict(
        names=ref["names"],
        native_leaders=[labels[int(leaders[i])] for i in joints[2:]],
        native_coefficients=coeff[joints[2:]].tolist(),
        v_before=v0.tolist(),
        v_after=v1.tolist(),
        q_after=after.joint_q.numpy()[iq].tolist(),
        acceleration=acc.tolist(),
        raw_mass=mass.tolist(),
        native_projected_mass=(ref["P"].T @ mass @ ref["P"]).tolist(),
        applied_motor_full=drive[iv].tolist(),
        body_forces=f[child, :3].tolist(),
        reconstructed_multipliers=balance[2:].tolist(),
        reconstructed_reaction_full=balance.tolist(),
    )


def main():
    import newton
    import warp as wp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--load", choices=("zero-row", "two-rows"), default="zero-row")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    wp.init()
    chain = None
    if args.load == "zero-row":
        try:
            b = make("chain")
        except ValueError as exc:
            chain = dict(accepted=False, reason=str(exc))
        else:
            chain = dict(accepted=True, joints=b.joint_count)
    ref = fan(both_loaded=args.load == "two-rows")
    row = real_fan(args.device, ref)
    result = dict(
        device=str(wp.get_device(args.device)),
        version=newton.__version__,
        load=args.load,
        chain=chain,
        fan=row,
    )
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    if chain is not None:
        assert not chain["accepted"] and "mimic" in chain["reason"].lower()
    assert row["native_leaders"] == ["screw", "screw"]
    np.testing.assert_allclose(row["native_coefficients"], [[0.004, -1.0], [0.002, 0.5]], atol=2e-7)
    a, lam = kkt(ref["mass"], ref["bias"], ref["external"], ref["motor"], ref["C"])
    np.testing.assert_allclose(row["raw_mass"], ref["mass"], atol=2e-6, rtol=0)
    np.testing.assert_allclose(row["acceleration"], a, atol=3e-4, rtol=0)
    np.testing.assert_allclose(lam, ref["expected_multipliers"], atol=3e-14, rtol=0)
    np.testing.assert_allclose(row["reconstructed_multipliers"], lam, atol=3e-4, rtol=0)
    np.testing.assert_allclose(row["q_after"], ref["q"] + H * (ref["v"] + H * a), atol=2e-7, rtol=0)
    print(
        "native",
        args.device,
        "fan qdd",
        row["acceleration"],
        "row forces",
        row["reconstructed_multipliers"],
        "chain",
        chain,
    )


if __name__ == "__main__":
    main()
