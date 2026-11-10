# MuJoCo-TYX T4: fixed-source mixed constraints and pyramidal friction

Status: SOURCE_RESEARCH_ONLY. Baseline `73a85f8affccde5ebf21a4629845d06af6949479`, MuJoCo 3.3.7, installed mujoco_warp 0.0.2. This document is a source-derived implementation contract recommendation, not an executed result. No project imports, probes, tests, formatting, compilation, installation, or native runs were performed in this lane.

## 1. Decision and limits

Implement one explicitly named `pyramidal-box-active-set-v1` coupled solver for contact condim 1/3/4/6, scalar hinge/slide soft limits, and DOF dry friction. Preserve the source cone. Reject unsupported elliptic models rather than silently turning an ellipse into a pyramid. Keep the already accepted T3 normal-only path available. Equalities, tendon constraints, ball-joint limits, plugins, and noslip postprocessing require separate declared support; they must not disappear when the new route is selected.

The parent reports its new compile-only inventory of the original primitive G1: pyramidal cone, 34 geoms, 502 possible pairs, 942 reserved geometry slots, 29 limited joints, collision priority 1 with condim 6. These are parent observations, not measurements by this research lane. The original mesh-based G1 scene and the primitive asset are distinct inputs; do not transfer the primitive admission result to mesh collision, actuator force clamps, or the mesh scene's dry-friction defaults.

## 2. Fixed primary sources and reuse

CPU reference: [MuJoCo 3.3.7 constraint assembly](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_core_constraint.c), [solver](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_solver.c), [forward dynamics](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_forward.c), [contact parameter mixing](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_driver.c), and [pyramid decoding](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_util_misc.c). Line references below refer to these fixed raw source files, not GitHub page line numbers.

Local comparison source is the parent's `artifacts/p65-unified-scene-20261110-01/optional-deps/mujoco_warp/_src/{constraint,solver,forward}.py`; adjacent distribution METADATA declares version 0.0.2 and Apache-2.0. Both projects use Apache-2.0; preserve source notices and attribution for translated code. [MuJoCo's fixed license](https://github.com/google-deepmind/mujoco/blob/3.3.7/LICENSE) and the installed Warp distribution's `licenses/LICENSE` were read. Warp is useful for device decomposition, but the identified CPU/Warp differences below prevent blind translation.

Raw CPU text for this research is cached under ignored `artifacts/p65-mujoco-tyx-t4-research-20261110-01/upstream-3.3.7`; older collision/util files are in the existing T3b research cache. No cached text is imported or executed.

## 3. Spatial rows and physical force

Let contact frame rows be normal n, tangent t1, tangent t2. Let Jp be the full point-velocity Jacobian difference body2 minus body1 and Jr the full relative angular Jacobian. Retain every articulated/free-root column, both bodies, and their common ancestors. Contact assembly in CPU lines 915–1010 produces:

| condim | Physical coordinates | Pyramid edge count |
| --- | --- | --- |
| 1 | n dot Jp | 1 |
| 3 | n,t1,t2 dot Jp | 4 |
| 4 | preceding rows, n dot Jr | 6 |
| 6 | preceding rows, t1,t2 dot Jr | 10 |

For d>1 and k=1..d-1, append `Jnormal + mu[k-1] Jk`, then `Jnormal - mu[k-1] Jk`. Each edge force lambda is nonnegative. All edges retain contact normal distance and inclusion margin as their position inputs; tangential rows are not separately zero-position unilateral constraints.

CPU `mj_assignFriction` lines 97–107 clamps all five mixed coefficients to at least `mjMINMU`. Implicit geom mixing expands slide/spin/roll into [slide, slide, spin, roll, roll] (driver lines 1347–1352). Preserve priority, material mixing, contact frame orientation, and the selected condim. A zero coefficient in a frictional source contact is still clamped; condim 1 is the strict frictionless path.

Decode exactly as `mju_decodePyramid` lines 850–869:
`f_normal=sum(lambda)`; `f_k=mu[k-1]*(lambda[2k-2]-lambda[2k-1])`.
Publish six physical contact-frame components, unused components zero, plus original edge forces and row identities. Spin/roll components are pure torques: world moment is lever-arm cross world linear force plus rotated contact torque. Do not apply the lever arm to torque twice.

The represented physical cone is `sum_k |f_k|/mu[k-1] <= f_normal`, a weighted L1 cone. It is neither independent tangential box bounds nor the elliptic L2 cone. Sliding coefficients are dimensionless; spin/roll coefficients have length units. Redundant edge coordinates retain their native regularization cost: an arbitrary wrench-only regularizer is a different problem.

## 4. R, D, aref, and parameter order

The physical M includes the admitted armature and all articulated coupling. Set `a0=M^-1 rhs`, where rhs includes external/passive forces and subtracts bias. Use this M in the constraint problem. Euler later integrates with its separately assembled `M+h*joint_damping`; it must not replace physical M in J M^-1 J^T. Publish physical qacc and the integration acceleration distinctly.

CPU `mj_diagApprox` lines 1098–1108 uses dof_invweight0 for DOF friction and the joint's first DOF for a scalar joint limit. Contact initial approximations (1164–1187) use `tran=body_invweight0[b1,0]+body_invweight0[b2,0]` and corresponding rotational `rot`. Normal-only initial dA=tran. Pyramid edge pair k has initial `tran+mu[k]^2*(tran if k<2 else rot)`.

CPU parameter processing (1195–1365) selects source contact/joint/DOF solref and solimp, applies REFSAFE timeconst>=2h when enabled, clamps both impedance endpoints individually without sorting them, uses the average endpoint impedance when width<=mjMINVAL, and follows the fixed piecewise-power impedance function otherwise. Preserve direct negative solref semantics. Existing explicit rejection of mixed-sign solref is acceptable as an admission boundary, not a claimed native result.

Let imp be current impedance and impmax the clamped second endpoint. Standard format:
`K=1/max(MINVAL,impmax^2*ref0^2*ref1^2)`,
`B=2/max(MINVAL,impmax*ref0)`.
Direct format:
`K=-ref0/max(MINVAL,impmax^2)`,
`B=-ref1/max(MINVAL,impmax)`.
DOF friction has K=0. Pyramid edges use the normal K/B, not friction solref and not K=0.

For a frictional contact, CPU lines 1369–1482 first computes
`Rbase=max(MINVAL,(1-imp)/imp * tran*(1+mu[0]^2))`,
then `R1=Rbase/max(MINVAL,impratio)`,
`mu_scaled=mu[0]*sqrt(R1/Rbase)`,
and **all** edge rows receive `Rpy=2*mu_scaled^2*Rbase`.
This final common R does not gain a factor d-1 for condim 4/6. Initial spin/roll dA values do not survive this overwrite. Preserve staged clamp ordering; there is no additional final R clamp in this CPU branch. Nonpositive/nonfinite stored R is an explicit device failure, never repaired.

Finally CPU sets `D=1/R` and rewrites published `diagApprox=R*imp/(1-imp)`. Compare against this final value, not the initial geometric approximation. CPU reference acceleration (2217–2228) is
`aref=-B*(J*v)-K*imp*(pos-margin)`.
There is no extra hand-added Jdot*v term. Every pyramid edge uses its own Jedge*v and the same normal position term.

Warp 0.0.2 `_efc_contact_pyramidal` multiplies the first-edge approximation by `2*mu[0]^2/impratio` before its shared helper clamps R. CPU clamps Rbase first. These differ near the floor. Warp's helper also clamps width instead of using CPU's flat-width average; retain CPU semantics for declared CPU equivalence.

## 5. Limits and dry friction join the same problem

CPU assembly order is equality, DOF/tendon friction, limits, contacts (constraint source 1912–1915). This batch can retain the no-equality/no-tendon domain while preserving the remaining order and stable identifiers.

Scalar limit assembly (696–744) loops lower then upper for each joint. Lower: `dist=q-lo,J=+e_dof`; upper: `dist=hi-q,J=-e_dof`. Include each row iff `dist<jnt_margin`, with pos=dist, margin=jnt_margin, lower force bound 0 and no finite upper force bound. Both sides may be active if margins overlap. Warp 0.0.2 chooses only the nearest side; this is not CPU-equivalent in that domain. Scalar qpos and DOF addresses differ after a free root: use native addresses. A soft limit is not a blanket rejection of qpos outside range; route admission and reset need to agree.

DOF friction assembly (641–672) adds `J=e_dof` whenever F=dof_frictionloss>0, with pos=margin=0 and bounds [-F,+F]. Use dof solref/solimp, K=0, and `aref=-B*qvel[dof]`. It is a bounded constraint coupled to all other rows, not `-F*sign(v)` and not another viscous damper. Free-root DOFs are not structurally excluded if the source model actually assigns them frictionloss.

Reject or explicitly implement source disable flags, override flags, noslip iterations, equality/tendon rows, and unsupported joints. An option that changes these equations must not be accepted merely because its value happens to leave a particular fixture unchanged.

## 6. One convex box problem and bounded solve

Collect rows into J, positive diagonal R, aref, and bounds l/u. Normal, limit, and pyramid variables have [0,+infinity); dry-friction variables have [-F,+F]. With `b=J*a0-aref`, solve

`min_{l<=f<=u} 0.5*f^T*(J*M^-1*J^T+R)*f + b^T*f`.

R>0 makes this strictly convex even for duplicate contacts or zero numerical J rows. Do not discard a structurally valid row because J is numerically zero. KKT gradient is `g=J*a-aref+R*f`, with `a=M^-1*(rhs+J^T*f)`: interior g=0; lower-bound g>=0; upper-bound g<=0, plus force feasibility.

The equivalent primal cost is `0.5*(a-a0)^T*M*(a-a0)+sum psi(J*a-aref)`. Unilateral psi is `0.5*D*min(z,0)^2`; dry-friction psi is `0.5*D*z^2` for |z|<=RF and `F*|z|-0.5*R*F^2` otherwise. CPU constraint update lines 2236–2316 implement these force/state formulas. Elliptic contacts use a different coupled cone cost/Hessian (2320–2421), so this scalar box solver must reject them.

A dual face solve need not factor an E-by-E matrix. For free rows F and bound rows B with fixed f_B, solve

`H*a = rhs + J_F^T*D_F*aref_F + J_B^T*f_B`,
`H = M + J_F^T*D_F*J_F`,
`f_F = D_F*(aref_F-J_F*a)`.

This is exact algebraic elimination of the same face KKT equations. The matrix dimension is nv, not the number of friction edges. Retain all row contributions and all physical M cross terms. Accumulate a symmetric matrix in a declared order; reject nonfinite or nonpositive pivots rather than adding a hidden regularizer.

Suggested finite algorithm: feasible cold start f=0; solve the current free face; move only to the first bound hit when a candidate leaves the box, binding all simultaneous hits. Once at a feasible face minimizer, release all KKT-violating bound rows as a block. If the expanded Newton direction initially points outside a newly freed zero-bound coordinate, alpha=0 binds those blocking coordinates and the internal face repair continues before another release phase. Count every face linear solve. Detect repeated working set with unchanged force as stall. This avoids a mandatory one-coordinate-per-iteration startup for a 320-edge foot manifold, but is not a proof that every model finishes within 100 iterations.

Keep native source iterations as the explicit face-solve budget (current admission 1..512), never enlarge G1's 100 behind its back. A failed/stalled/nonfinite/budget-exhausted world rejects the whole publication transaction. No CPU fallback, contact truncation, R repair, or stale-force reuse.

CPU Newton/CG stops on scaled cost improvement or gradient, or its source iteration/line-search limits (solver 1896–1957). Native scale without islands is `1/(meaninertia*max(1,nv))`; native warmstart compares its prior acceleration and smooth costs (forward 564–631). This custom method solves the same assembled convex problem when verified, but does not reproduce native iterations, search directions, warmstart, or intermediate forces. Report custom face solves and KKT residual separately from native statistics.

## 7. Assembly and solver ABI recommendation

The parent's current T4 proposal separates geometry reservation from live solve capacity: at most 128 geoms, 1024 possible pairs, 2048 reserved geometry slots, 128 detected contacts, and 512 actual EFC rows. These are new declared T4 caps, not retroactive changes to accepted T3. Preserve all candidate descriptors and their status. Geometry must visit every admitted pair; a stable device compaction then maps detected contacts and active EFC rows. Exceeding a detected-contact or EFC cap fails, without truncation or taking the first rows.

The earlier 32-contact domain needed up to 32*10+2*29+35=413 EFC rows. The G1 inventory has 9420 potential edges, so preallocating or rejecting by potential-edge count is wrong for the new route. Reserve descriptors for the full model; allocate dense solve arrays only up to actual-cap 512. Detected-but-inactive contacts (distance >= inclusion margin) keep geometry diagnostics while contributing no EFC row. Count candidate slots, detected contacts, active contacts, and EFC rows independently.

Proposed solver inputs:
- physical `M[W,V,V]`, smooth `rhs[W,V]`; V=nv<=35, never nq.
- `J[W,E,V]`, `R/aref/lower/upper[W,E]`, finite-bound masks, actual `nefc[W]`.
- immutable row identity: kind, source joint/DOF or detected contact, contact edge/component/sign, source candidate slot, model/body identities.
- source iteration budget and declared KKT threshold; use the parent-selected `max(source_tolerance,2e-5)` only as the explicit custom normalized stopping contract.
- upstream status, destination transaction bank; no reads from a previous successful step's row count/forces.

Proposed outputs:
- `f/g/working_set[W,E]`, physical acceleration, generalized constraint force `J^T*f`.
- actual face-solve count, feasibility/KKT maxima and their row scales, mass-equation residual, terminal status.
- decoded contact-frame wrench remains a separate assembly/publication operation with exact row mapping.
- original J/R/D/aref, final diagApprox, input q/v and full model binding remain available for independent checking.

Do not encode infinity as a large finite force cap; use bound masks. Clear unused storage deterministically and prevent stale valid flags. The driver can compact stable present rows once; free/bound state arrays suffice inside each face. Cost per face is O(|F|*V^2+V^3+E*V), scratch O(V^2+E), beyond retained J. No E^2 matrix is required. This is a dimensional cost argument, not a performance measurement.

The KKT scale must be explicit. A practical existing-style row scale is `max(1,abs(b_i)+sum_j abs(A_ij*f_j))`, but forming A defeats the proposed memory reduction. An implementable conservative equation scale is `max(1,abs(aref_i)+sum_v abs(J_iv*a_v)+abs(R_i*f_i))`; publish it with the unscaled residual and name the new normalization. Mixed translational/rotational rows have different physical units; this numeric normalization is a stopping convention, not a universal dimensional error guarantee. Parent must freeze one formula before tests, including feasibility and the actual stored FP32 residual comparison.

## 8. Minimum oracle matrix before G1 claims

| Group | Required differentiating cases | Checks |
| --- | --- | --- |
| Contact rows | condim 1/3/4/6, offset COM, nonidentity body/frame, both bodies dynamic | Full current J, body/DOF identity, contact frame, all edge order and decoded six-wrench components |
| Sliding | loaded static support; subcritical push; breakaway; signed t1/t2 and diagonal slip; unload/recontact | Diamond/L1 behavior versus an incorrect tangent box or ellipse; force, velocity, persistent state |
| Rotational friction | condim4 pure spin; condim6 both roll axes; zero COM translation with nonzero angular velocity | Jr rows, spin/roll units, pure torque and correct lever-arm contribution |
| Coupling | simultaneous feet/contact patches on a floating articulated tree; moving branch against moving branch | Full M and off-diagonal constraint coupling; physical qacc versus Euler acceleration |
| Parameters | impratio, direct/standard solref, REFSAFE, flat-width solimp, friction floor, weak/strong normal regularization | Native final R/D/diagApprox/aref; CPU/Warp clamp-order boundary |
| Limits | lower/upper, exact activation boundary, soft penetration, overlapping margins, nq!=nv | CPU both-side rows, native addresses, force signs and simultaneous contact/limit solve |
| Dry friction | positive/negative/zero velocity, interior and saturated force; contact plus dry friction | Signed bounds, K=0, Huber state, no sign-force substitute |
| Solver | tiny coupled independently enumerated box QPs, redundant/zero J, near-bound KKT, insufficient budget | Same unique solution, explicit status, full mass/force residual, no silent partial publication |
| Runtime | W=1 and multiworld, distinct controls, clone/reset then genuine next step; late-world failure | Continuous input/output binding, atomic failure, row compaction and no stale-force leakage |
| Capacity | exact declared live caps and first overflow; many candidates but few active contacts | No truncation, candidate and EFC counts separated, deterministic mapping |

Use fixed CPU 3.3.7 to compare assembled equations, then its converged force/acceleration within declared FP32 tolerances; finite native iteration mismatch is not automatically a different mathematical problem. Also use independent tiny box-QP solutions so matching one CPU run is not the sole solver oracle. Retain all failed attempts. Interpret and CUDA tests share authored inputs and independent native references; no product re-execution merely to obtain diagnostics.

G1 is the subsequent integration case with its original condim6/priority, real actuators, initial velocity, contacts, limits, and source options. A successful primitive model compile is not a dance result. Passing G1 alone is insufficient without the distinguishing row/solver/runtime matrix above.

## 9. Immediate handoff

This research is complete enough to author a general mixed-row solver now. Parent owns final static admission, capacities, common row layout, routing, and transaction checks. The row author owns spatial friction/limits/dry-friction assembly; the solver can be a new isolated module behind the agreed ABI. Preserve accepted T3 behavior and do not spend another batch proving unrestricted convergence. The explicit failure boundary and source-matched equations make this work reviewable and testable.
