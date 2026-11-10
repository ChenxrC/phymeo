# MuJoCo-TYX T3a fixed-source implementation review

Status: source research and manual algebra only, 2026-11-10. No import, probe, collection, test, formatter, build or numerical execution. This note supports the [T3a implementation plan](../superpowers/plans/2026-11-10-mujoco-tyx-coupled-normal.md); it does not certify an implementation or repeat the accepted T1/T2 review.

## Fixed sources and decision

The references below target **MuJoCo 3.3.7**. The comparison source is the already supplied **mujoco_warp 0.0.2**, under parent `artifacts/p65-unified-scene-20261110-01/optional-deps/mujoco_warp/_src/`. CPU source semantics take precedence where that Warp version differs.

| Concern | Fixed primary source | Local Warp comparison |
|---|---|---|
| Row construction, impedance, reference acceleration | [engine_core_constraint.c](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_core_constraint.c) functions `mj_instantiateContact`, `mj_diagApprox`, `getsolparam`, `getimpedance`, `mj_makeImpedance`, `mj_referenceConstraint` | `constraint.py:58-124`, `1255-1380`, `1425-1548` |
| Dual publication and native Newton stopping | [engine_solver.c](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_solver.c) lines 62-70, 1760-1817 | `solver.py:1114-1165`, `1981-2012`, `2070+` |
| Smooth dynamics, warmstart, Euler damping | [engine_forward.c](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_forward.c) lines 523+, 655-657, 768-810 | `forward.py` Euler section |
| Pair filtering, parameter mixing, inactive contacts | [engine_collision_driver.c](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_driver.c) lines 143-159, 1169-1284, 1319-1355, 1477-1478 | `collision_primitive.py:430-564` |
| Plane and round primitives | [engine_collision_primitive.c](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_primitive.c) lines 23-78, 178-277 | `collision_primitive_core.py:106-178`, `228-278`, `312-360` |
| Sphere-box inside/outside convention | [engine_collision_box.c](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_box.c#L33-L83) | sphere-box primitive |
| Degenerate normalization and complete frame | [engine_util_blas.c](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_util_blas.c#L104-L120), [engine_util_spatial.c](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_util_spatial.c#L470-L496) | frame helpers |

A bounded coupled active-set solver is an appropriate explicit implementation of the frictionless dual QP. Name it `normal-dual-active-set-v1`. Do not call its FP32 result exact arithmetic, or claim native Newton/PGS iterations, stopping statistics or warmstart reproduction. This is sufficient to proceed with this batch's implementation and independent tests.

## Coupled normal problem

Notation: physical mass matrix `M` includes armature; `rhs = applied + passive - bias`, with passive already containing joint damping. The current world contact midpoint is `x`, normal `n` points from geom 1 to geom 2, and both point Jacobians are evaluated at that same `x`:

```text
J_i = n_i^T (Jpoint_body2(x_i) - Jpoint_body1(x_i))
s_i = dist_i - includemargin_i
vel_i = J_i v
aref_i = -B_i vel_i - K_i d_i s_i
R_i = max(MINVAL, (1-d_i) diagApprox_i / d_i)
diagApprox_i = body_invweight0[body1,0] + body_invweight0[body2,0]
Dconstraint_i = 1/R_i
```

Here `d_i` is impedance, not joint damping. `diagApprox` is compiled approximate inverse weight, **not** the actual diagonal of `J M^-1 J^T`. Native reference acceleration adds no separate `Jdot*v` term. Contact rows with a numerically zero normal Jacobian are retained; the structural no-affected-DOFs branch is different. These details follow the named constraint functions above.

The following equivalence is a manual derivation for the admitted frictionless domain:

```text
a0 = M^-1 rhs
W = J M^-1 J^T
A = W + diag(R)
b = J a0 - aref
minimize over f >= 0:  0.5 f^T A f + b^T f
KKT: f >= 0, g = A f + b >= 0, f_i g_i = 0
qfrc_constraint = J^T f
qacc = a0 + M^-1 J^T f
```

Eliminating `f` gives the primal objective `0.5(a-a0)^T M(a-a0) + 0.5 sum_i Dconstraint_i min(0,(Ja-aref)_i)^2`. Its stationarity is `M(a-a0)=J^T f`, with `f=-Dconstraint*min(Ja-aref,0)`. Positive `R` makes the dual matrix positive definite even for redundant rows. Keep every admitted row and every cross-contact term; independent scalar clamping solves a different problem. FP32 pivot failure must reject, not silently add a new regularizer.

Native Euler then uses `(M + h Djoint) a_int = rhs + J^T f`, updates velocity and integrates pose with the updated velocity. The contact QP uses physical `M`; it must not use `M+hDjoint`. Preserve smooth `a0`, constrained `qacc`, and `a_int` as distinct observations. Forces are not impulses. The world body forces are `-f*n` and `+f*n`; their COM torques are `(x-xipos_body) cross force`. This yields the same generalized force as `J^T f`.

## Solver parameters and failure boundary

Use the native impedance clamping and direct/standard solref cases. In particular:

- Clamp `dmin` and `dmax` independently; do not sort descending values. Width at or below `MINVAL`, or equal endpoints, uses their **average**. Warp 0.0.2's width clamp does not reproduce that CPU edge case.
- Otherwise `x=abs(s)/width` gives endpoint impedances outside `[0,1]`; the middle uses the native piecewise power curve and clamped midpoint/power.
- Standard `K=1/max(MINVAL,dmax^2*timeconst^2*dampratio^2)`, `B=2/max(MINVAL,dmax*timeconst)`; direct `K=-ref0/max(MINVAL,dmax^2)`, `B=-ref1/max(MINVAL,dmax)`. Ref-safe timeconst clamping applies only to standard format.
- Mixed positive/nonpositive solref components trigger native default replacement. Explicit rejection is preferable for T3a to an unimplemented silent replacement.

Native Newton has cost-improvement/gradient tests scaled by model inertia and DOF count, plus line-search options; `alpha==0` also terminates. Native warmstart compares candidate accelerations. Thus this batch's solve-budget mapping and scaled KKT stopping are public choices, not renamed native statistics. The parent plan maps iterations 1..512 to total active-set linear solves and uses `max(source tolerance,2e-5)` as the normalized KKT gate. The residual normalization must be stated and stored with the actual budget/solve count. Budget exhaustion, nonfinite arithmetic, bad pivot, or failed KKT must reject the whole candidate publication.

A conventional bounded implementation starts at zero force, admits a violating inactive gradient with a deterministic tie rule, solves the principal free system, and takes a feasible segment to a blocking zero force when necessary. Re-solving/removing rows consumes the same budget. Exhaustive active-set enumeration is suitable only as an independent tiny-case test oracle, not a scaling strategy for the device solver.

## Three contact sets and native geometry boundaries

Keep model possible pairs, detected contacts, and constraint rows separate. Detection accepts the primitive's `dist <= margin` boundary. Row inclusion requires `dist < margin-gap`; detected inactive contacts retain identity and zero force. An included row may still have optimal force zero. `mj_instantiateContact` additionally excludes a structurally empty affected-DOF chain; `mj_addConstraint` does not discard a contact merely because its normal Jacobian values vanish.

Implicit collision masks use the two directed mask tests joined by OR. Same-weld pairs are excluded; parent-child welded pairs are excluded except when a weld body is world. Margin and gap are independent maxima of the geoms. Priority selects the winning contact parameters; with equal priority, positive solref uses solmix weighting, direct-format mixing uses componentwise minima, and solimp uses weighting. Geom type ordering determines normal direction. T3a explicitly rejects explicit pairs/excludes, unsupported option flags and mixed condim other than 1 rather than overlooking them.

| Pair | CPU 3.3.7 selection and edge rule | Consequence for this batch |
|---|---|---|
| Plane-sphere | Plane world Z; one signed surface distance; midpoint halfway across separation | Use world midpoint for both Jacobians, not sphere center or separate witnesses |
| Plane-capsule | Positive axis endpoint then negative endpoint, each sphere-plane test; at most two | Preserve endpoint identity/order and initial tangent equal to capsule axis |
| Plane-box | Scan corner bits 0..7; keep `dist<=margin` and center-relative normal offset `<=0`; return after four | Do not sort lowest four. Warp 0.0.2 returns eight; its wrapper also loops eight |
| Sphere-sphere | For coincident centers use cross of the two geom world Z axes; normalize fallback is +X | Always choosing +X differs for rotated coincident geoms |
| Sphere-capsule | Project onto unit capsule axis and clamp to +/- half-length; then sphere-sphere | CPU has no `+1e-6` denominator; do not inherit the Warp segment helper bias |
| Sphere-box | Exterior uses clamped local center. Interior/boundary chooses nearest face, strict tie order `-x,+x,-y,+y,-z,+z` | At exact cubic center choose -X face, inward normal +X, signed distance `-halfsize-radius` |

For sphere-box, let local sphere center be `c`, closest face distance be `ell`, and inward unit vector `e`. Interior midpoint is `c + e*(radius-ell)/2`. Exterior: let clamped point be `p`, `n=(p-c)/norm(p-c)`; midpoint is `(p+c+radius*n)/2`, distance `norm(p-c)-radius`. Transform point and normal by the box pose. The initial normalization fallback must not replace this interior branch or reverse the force sign.

CPU `mju_normalize3` returns the original length; a vector shorter than `MINVAL` becomes +X. `mju_makeFrame` normalizes the normal, chooses Y or Z if the initial tangent is too short, removes its normal component, normalizes, and crosses for the final axis. Plane-capsule supplies the axis tangent; the other listed primitives start with zero tangent. Reconstructing every frame only from its normal loses this distinction. Condim1 dynamics use only the normal, but public frame comparison must identify any retained degeneracy rather than silently claim exact CPU parity.

## Implementation and independent validation checklist

These are discriminating cases for the plan's existing test matrix, not a new gate requiring further theory before coding:

- One slider with two simultaneous normals and nonzero off-diagonal `W`; compare with an independent small dual active-set oracle. Include eligible-but-zero-force and duplicate normal rows.
- Rotated offset free body with a four-corner plane manifold; check generalized contact force and both world COM wrenches. Exercise a moving sibling body on each side, not just world versus root.
- Plane-box at a tilt and at a corner-order tie; margin equality, margin-gap equality, and changing detection inventory; capsule endpoint order and zero/near-zero segment projection.
- Sphere-box center/interior/face/edge/corner and equal closest-face distances; coincident spheres with different geom orientations. Independent native reference must compare the CPU-selected manifold, not Warp's differing one.
- Nonzero joint damping distinguishes physical constraint acceleration from integration acceleration. An asymmetric mass/coupling case detects use of approximate or diagonal mass in `W`.
- Predeclare normalized KKT calculation, contact counts/capacity, failure codes and solve budget. Exercise exhausted budget, invalid pivot/nonfinite candidate and late-world fault; verify state/force/history publish together or none publish.
- Preserve detected inactive contacts and zero-force rows in saved observation. Verify selected reset, clone's real next step, and independent W4 world state. Do not substitute CPU/Warp numerical work inside a device step.

The current T3a plan is bounded to the six listed pairs and condim1. Friction, additional finite shape pairs, broad general native options and G1 support remain separate follow-ups. Research/source agreement here is not an execution pass.
