# Finite parallel-jaw module (G1)

Status: CODE_READY_TESTS_PENDING. The user requested parallel code development and serial tests. No test, numerical run, build or native validation has been executed for this new module.

The public submodule phymeo.builders.parallel_jaw_gripper creates a real 3-link coordinate-prismatic RobotModel: massive +Z carriage, +Y left finger, -Y right finger, two finite RobotContactPatch rectangles. All three efforts are finite requests. Scalar finger mass broadcasts to both fingers; a two-vector supports asymmetric masses. Configuration/evidence inputs are owned; the returned initial RobotState is intentionally mutable and independent.

Positive finger q closes the jaws. opening_m(q)=2*half_gap-q1-q2 is a signed diagnostic, not a stop constraint. gripper_inputs rejects out-of-limit effort; negative finger effort opens. Gravity/contact remain runtime computations. No control prediction is published as q/v; no cloth is pinned, welded or moved by this builder.

Authored tests: independent shared-carriage FK/Jacobian, finite patch dimensions, symmetric/asymmetric masses, bounded signed effort requests, ownership, invalid mass/inertia/shape/precision and opening diagnostics. These tests remain UNEXECUTED pending the parent execution queue. Syntax/lint checks also have not run.

G1 is not load-bearing acceptance. G2 must assemble public material/contact runtime, finite closing/lift control and actual dynamic cloth, then demonstrate independently audited high-mu hold and low-mu/insufficient-force slip with durations, halved steps, changes of geometry/mass, actual backends and rollback. Full folded-cloth acceptance additionally requires bending and self-contact integration. General articulated thickness support remains a separate required capability.

Contract: ../superpowers/specs/2026-10-30-p65-parallel-cloth-modules-design.md
