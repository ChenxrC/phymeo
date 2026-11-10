# MuJoCo-TYX T3 normal-contact source handoff

Status: **AUTHORING ONLY / UNEXECUTED**. This report is about the three new device modules, not a passing test or an accepted physics domain. Parent owns static admission, shared runtime/receipts, independent tests, NOTICE and all execution. No import, collection, numeric probe, test, lint, formatter, compiler, build or native run was performed by this author.

## Scope and reuse

The frozen T3 scope is six native primitive pairs: plane-sphere, plane-capsule, plane-box, sphere-sphere, sphere-capsule and sphere-box, with condim=1. It uses a fixed-root T1 base or a single-free-root T2 base, including the complete current native M, COM Jacobians, root/joint coupling, bias, passive and actuator fields. Independent multiple free roots, friction, equality, native active joint limits, meshes and CCD remain outside this batch.

Only these new product modules belong to this lane:

- src/phymeo/backends/mujoco_tyx/_normal_geometry.py
- src/phymeo/backends/mujoco_tyx/_normal_solver.py
- src/phymeo/backends/mujoco_tyx/_normal_device.py

Original T0/T1/T2 source is unchanged in this worktree. The device calls the existing tree/floating geometry, mechanics and solve functions. It performs five kernels and one synchronization per requested step: original geometry, original mechanics, current contact geometry/J, joint normal QP, original Euler solve/integration. No CPU/Warp step or host dynamics is introduced.

The final original solve receives total_applied = original_applied + J.T f. Its old constraint and rows buffers are disposable; separate normal_constraint and normal_rows retain the actual new solve. The original applied receipt remains the non-contact applied force. qacc is the full physical M solve, while smooth_acceleration is stored before contact. integration_acc remains the original (M+hD) solve, so damping is neither omitted nor doubled.

## Frozen static interface

Parent TyxNormalModel wraps .base, with .normal_contact=True and attribute forwarding. Base kind is 2 or 3. The new device constructor keeps the existing argument contract: (plan, mode, cache_root, q, v, time, nbody, h, capacities).

- geom_ids[G,2]: native body ID and native geometry type.
- geom_data[G,15]: local position3, local rotation9 row-major, native size3.
- pair_ids[P,4]: ordered geom1, geom2, stable slot start, slot count.
- pair_law[P,16]: margin, gap, native summed body_invweight0, solref2, solimp5, mjMINVAL, mjMINIMP, mjMAXIMP, three reserved entries.
- pair_friction[P,5]: parent retains the native mixed friction metadata; no tangential force is applied for condim1.
- slot_ids[C,4]: pair index, native feature index, geom1, geom2.
- solver_iterations: source 1..512, interpreted as a total face-solve budget for the explicitly named new algorithm.
- solver_tolerance: stored FP32 max(native tolerance, 2e-5), used by both device gate and published metadata.

Hard ceilings are G128/P16/C32. A pair reserves 1/2/8/1/1/1 slots respectively; plane-box keeps original corner IDs, despite at most four present contacts. Actual present and active counts are independent of reserved slot count. Equality to caller capacity is accepted; only greater counts fail. No candidate truncation or overflow-as-success path exists.

## Geometry and native differences

The formulas follow the fixed native MuJoCo 3.3.7 sources and the local MuJoCo-Warp 0.0.2 contact/constraint layouts. These deliberate fixed-version choices differ from Warp 0.0.2 in some degenerate/deep cases:

- Plane-box traverses corners 0..7 and accepts only local normal projection <=0 and world distance <=margin, stopping after four. It does not sort the corners and does not emit all eight deep-penetration corners.
- Sphere-capsule uses direct projection on its unit axis and clamps to the native half length; it does not add 1e-6 to a segment denominator.
- Coincident sphere centers use the normalized cross of the two world geometry Z axes; only a degenerate cross falls back to +X.
- Sphere-box computes the exact clamped closest local point. An inside sphere uses the first nearest face in -X,+X,-Y,+Y,-Z,+Z order, preserving strict-comparison ties.
- Contact inclusion uses distance <=margin. Constraint activation separately uses distance-(margin-gap)<0.
- Native frame construction retains the capsule-axis tangent seed for plane-capsule, including native degenerate normalization behavior. Other pairs start with a zero tangent seed.
- Zero numeric normal J is retained. The parent handles native structural filtering; a zero projected row is not a reason to drop its regularization or force.

For each current contact position x, each body's point column is Jv_COM + Jomega cross (x-COM). The row is normal dot (Jpoint_body2-Jpoint_body1), retaining both bodies and all generalized coordinates. Current Jv is recorded even for detected contacts outside the active constraint layer.

The native regularizer uses the compile-time native body_invweight0 sum, not the instantaneous diagonal J M^-1 J.T. Solref/solimp, margin/gap, default refsafe behavior and positive/direct solref branches follow the admitted T0 normal law. The CPU midpoint rule for zero solimp width or equal endpoint impedance is retained.

## Joint solve and finite failure semantics

The new numerical solver is a cold-start bounded dual active set, explicitly not native Newton, CG, PGS, line search or warm start.

It factors physical M once and forms a0=M^-1(applied+passive-bias), Z=M^-1 J.T and A=J Z+diag(R). It solves min over f>=0 of 0.5 f.T A f + (J a0-aref).T f. All active rows participate in the same matrix, including duplicate or mutually coupled contacts.

Starting at f=0, an improving inactive coordinate enters the working face. A Cholesky solve obtains the unconstrained minimizer on that face. A blocking negative component causes a feasible line segment update and removal of the blocking face. Every face solve, including blocked candidates, consumes the same finite budget. No per-point physical update or extra hidden budget occurs.

The final gate is checked again after the last permitted face solve. For each active row, the projected gradient violation is abs(gradient) if force>0, otherwise max(-gradient,0). The common scale is max(1, max rows of abs(linear_i)+sum_j abs(A_ij f_j)). normalized_kkt is max violation divided by that scale. Negative force, nonfinite intermediates, nonpositive factor pivots or unmet final tolerance reject publication.

Status codes retain 1 capacity, 2 nonfinite, 3 joint limits, 4 nonpositive physical matrix; T3 uses 5 for nonconvergence/no improving face despite unmet stationarity and 6 for nonpositive normal face matrix. Parent decodes these statuses and owns batch rejection/publication.

This is an engineering FP32 solve with a declared scaled KKT gate. It is not a claim of native per-iteration identity, bitwise force equality, exact complementarity or indefinite-budget convergence.

## Download and row schema

Existing download keys remain. Additional keys are normal_rows, normal_jacobian, dual_matrix and solver_diagnostics; smooth_acceleration and constraint are replaced with the independent true normal-stage buffers.

normal_rows[W,C,32]:

| Columns | Meaning |
|---|---|
| 0,1 | present contact, active normal constraint |
| 2,3 | signed distance, solved normal force |
| 4,5 | R, D=1/R for active rows |
| 6,7,8 | aref, current Jv, impedance |
| 9:12 | same current contact position |
| 12:21 | native frame row-major, normal first |
| 21,22,23 | include margin, margin, gap |
| 24 | J times smooth acceleration |
| 25,26 | final dual gradient, force times gradient |
| 27:32 | reserved zero |

normal_jacobian[W,C,nvpad] retains stable slots. dual_matrix[W,C,C] contains the shared response for detected rows, with regularization only on active constraint rows; unpopulated slots remain zero. Solver diagnostics[W,8] are face-solve count, normalized KKT, absolute KKT, minimum force across reserved slots, present count, active count, converged flag and budget. No-active-row success has zero solves and converged=1. All scratch is reset each request, so reset/clone needs no hidden warm-start state.

## Fixed primary source map and attribution

- MuJoCo 3.3.7 engine_collision_primitive.c: plane/sphere/capsule rules and plane-box manifold.
- MuJoCo 3.3.7 engine_collision_box.c: sphere-box, authored upstream by Svetoslav Kolev.
- MuJoCo 3.3.7 engine_util_spatial.c and engine_util_blas.c: contact-frame and degenerate normalization semantics.
- Local MuJoCo-Warp 0.0.2 collision_primitive_core.py, collision_primitive.py and constraint.py: porting reference for primitive formulas, native field roles, material mixing and normal efc quantities.
- Local MuJoCo-Warp 0.0.2 solver.py: native objective/context reference; its Newton/CG implementation is not represented as ported.

Fixed primary URLs:
https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_primitive.c
https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_box.c
https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_util_spatial.c
https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_util_blas.c

Geometry headers retain DeepMind 2021, Svetoslav Kolev 2016, Newton Developers 2025 and Phymeo 2026 attribution. Parent updates package NOTICE. Source links and manual review are not numerical validation.

## Required parent validation

The independent author supplies the frozen multi-case tests and CPU same-input comparisons. Parent should validate actual geometry membership, native current J/M/bias, all active R/aref/forces and coupled QP, physical and integration acceleration, consecutive motion, two-sided force/moment, present-but-inactive rows, zero projected J, capacity equality/overflow, finite-budget failure, reset/clone next step, all-or-nothing publication and real CUDA launches. Parent also performs focused static checks and package/source-outside-wheel validation.

All functionality in this report remains UNEXECUTED until the parent records those results against the final tested source. No old green group or old T0/T1/T2 evidence was rerun by this lane.
