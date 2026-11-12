# D6-A: objective axial material dissipation, 2026-11-13

Status: AUTHORING_GRANTED / UNEXECUTED. Bounded follow-up to the actual D4 steady-support failure, inside the user's functional-completion window. No promise that the old N1 windows pass. Original D4 inputs, outcomes and limits remain unchanged. Freeze new functionality at 06:15; root owns every execution and shared seam.

## Decision and public interface

Use research option B: one ordered implicit axial velocity pass on the unpublished elastic candidate. Read [research](../../development/p65-deformable-dissipation-followup-20261113.md); primary constitutive context is [XPBD section 5](https://mmacklin.com/xpbd.pdf) and [Newton v1.7 spring damping](https://github.com/newton-physics/newton/blob/v1.7.0/newton/_src/solvers/xpbd/kernels.py). This is our explicit first-order split, not a claim of an identical Newton algorithm or a global network BE solve.

NEW public DistanceDissipation(ordered_point_ids, coefficients_ns_per_m):
- owned immutable bytes-backed int64 (E,2) IDs and float64 (E,) finite nonnegative coefficients, max E=1536.
- exact declared edge order and orientation must equal point_ids[constraints.indices]; no sorting, omission or inferred parameter.
- c must round to finite normal-or-zero FP32 without nonzero underflow. Static inputs keep old validation.
- DeformableNodeWorld gets keyword-only dissipation=None. None and explicit all-zero retain original numerical path, allocations and launches. Explicit zero still binds the declaration into source/checkpoint identity.
- Static fixed-node velocities remain exactly zero. No moving prescribed supports in this slice.

Nonzero parameters are optional new material semantics. In D1: original predictor/projection/reconstruct -> preserve v_reconstructed -> damping on standby v -> validate complete candidate -> original atomic publish. D3 shares this candidate; its entry-sampled contact force is unchanged. New accepted velocity is used by the next real step. Never mutate an accepted bank after commit.

## Device operation and fixed trace

For each edge in declared order, use candidate x and current standby velocities.
Normalize d=x_i-x_j with scaled norm. For c=0 skip without inventing a direction.
For c>0 reject zero/nonfinite direction; no old world-axis fallback.
For finite stored n define g2=dot(n,n), A=w_i+w_j, s=dot(n,v_i-v_j),
b=h*c, denom=1+b*A*g2, j=-b*s/denom.
Update v_i += w_i*j*n and v_j -= w_j*j*n using explicit existing FP32 primitive helpers. Fixed endpoints unchanged. Both-fixed edges with zero velocities are legal zero impulse.
Using g2 preserves the same discrete identity for a stored nonunit n; ideal normalized n recovers the research formula.
Each edge executes once per physical h, independent of elastic sweep count. No extra damping sweeps, global drag, lambda or parameter search.

NEW kernel signature:
dissipate_distances(count, h, edges, coefficients, weights, x, v, trace, status).
trace shape (P,32), P=max(1,E), float32; status(1,) int32; caller starts trace/status zero.
Columns: 0:3 n; 3 s_before; 4 j; 5 actual s_after; 6:9 vi_before; 9:12 vj_before;
12:15 vi_after; 15:18 vj_after; 18 h; 19 c; 20 wi; 21 wj; 22 length;
23 denominator; 24:27 requested impulse_i=j*n; 27 g2; 28:32 zero reserved.
Skipped c=0 rows save endpoint pre/post and parameters, zero n/rates/j/impulse/length, denom=1/g2=0.
Nonfinite intermediates/status reject the whole candidate before any publication.

## Phase receipt and bookkeeping

NEW immutable DistanceDissipationStep:
schema='distance-dissipation-step-v1', ordered_point_ids, coefficients_ns_per_m (original float64),
reconstructed_velocity and dissipated_velocity (float32 N,3), trace (float32 E,32),
node_impulse (float64 N,3, accumulated requested device j*n including fixed endpoints),
node_velocity_impulse (float64 N,3, actual delta-v / original inverse mass; fixed entries zero),
kinetic_change (float64 E), ideal_work (float64 E), rounding_work (float64 E).
make_dissipation_step(parameters, native_edges, weights, reconstructed_velocity, dissipated_velocity, trace) is a host observer only, never generates device velocity. Own every array. Validate shape/finite/chain endpoints and final velocity against trace; each row's actual pre/post velocity must connect exactly in original order.
For each edge ideal_work=j*s_before+.5*A*g2*j²; actual kinetic change is from stored endpoint velocities (positive-mass endpoints only); rounding_work=actual-ideal. This observer is not an independent oracle.
No energy credit or cancellation allowance. Tests separately check raw impulse/rates/objectivity and one-edge nonpositive work with declared FP32 update error bounds.

Root defines DeformableDissipatedWorldStep subclass of original DeformableWorldStep with dissipation field. Original absent/zero row remains exact old type/fields. Checkpoint hash/validation explicitly include the subtype and all phase arrays. Do not globally relax type validation.

D3 retains material_impulse as original ordered elastic impulse; adds damping_impulse only when enabled. Fixed reaction includes -(h*force+elastic_impulse+damping_impulse). Save both phases and original contact forces. Existing strict elastic arithmetic audit must use v_reconstructed, not new dissipated final velocity. Root updates public exports and shared bindings.

## Resources and scope

No capacity increase: worlds8/nodes512/edges1536, same settings/memory limit.
Active world device additions: coefficients P*4, v_reconstructed 12*N, trace128*P (reuse original status); total 12*N+132*P bytes. No separate node impulse device buffer; observer sums device trace. Explicit zero makes no device allocation.
Host constants include original parameter bytes and padded coefficient bytes; receipt reservation includes both velocity arrays, trace, two N*3 float64 impulses, and three E float64 energy arrays, plus original parameter ownership. Root accounts all scratch/readbacks conservatively in existing requirements before runtime allocation.
Exact/one-less reservation and late allocation failure must be tested. UNKNOWN_RELEASE remains fail/poisoned per original resource contract; no repeated release workaround.

## Multi-case tests before application

1. None versus explicit zero: original outputs, old row type and launch/allocation counts unchanged; source identities differ.
2. Two free unequal masses 1/2 kg, x=(-.5,0,0)/(.5,0,0), axial v=.25/-.125, c=2, both signs and h=1/128,1/256: independent ordered FP64 oracle, pair momentum and signed dissipation.
3. Fixed support edge: fixed reaction; T1s linear oscillator k64,m1,extension1/8,c16 independently chosen critical damping, two h; compare independent split recurrence. No contact-case fitting.
4. Noncollinear two-edge unequal-mass chain: every ordered stage, force/moment, kinetic accounting; expected order dependence explicit.
5. Noncoplanar rigid translation/instantaneous angular velocity null mode; rotated/translated/boosted/reordered stable-ID covariance. Test operator at fixed candidate geometry, not false finite-angle global invariance.
6. Five-point membrane snapshot: z=0 center-normal motion has no axial damping; z=1/8 instantaneous coefficient 4*c*z²/(2+z²) separately, full pass against ordered oracle (not that coefficient times h).
7. Parameters/model binding, nonzero-c zero-length geometry, fixed-velocity rejection, memory exact/one-less, late allocation cleanup.
8. Sparse two-world actual failure after candidate -> unchanged accepted states/last-step/queries -> real retry, checkpoint/reset/restore successors.
9. D3 sphere and inclined capsule short integration: entry contact law, original elastic phase and new damping/fixed reaction, true rigid response, shared late-failure atomicity.
10. Source-external installed consumer on interpret/CUDA, including at least D1 and D3 actual successors. Only accepted parts go into final wheel.

Velocity/impulse comparison atol2e-6,rtol2e-5 on these bounded cases; no fixture/tolerance edits from output. Energy bound derives from actual stored per-component update rounding (at most one half-ulp per primitive, with underflow handled explicitly) and keeps signed raw values. Full pass is first-order ordered splitting, not simultaneous solve.
No 3s N1-damped run until module/short-coupled acceptance and an independently declared application protocol. It is optional and does not replace old FAIL.

## Ownership

- source agent: NEW backends/deformable_dissipation.py and _tyx_distance_dissipation.py, NEW tests/collision/test_distance_dissipation_kernel.py, NEW lane report. Tests first, no execution.
- tests agent: NEW tests/collision/test_deformable_dissipation.py and _transactions.py, public D1 matrix above; independent oracle in NEW tests/collision/deformable_dissipation_cases.py. Tests first, no execution.
- audit agent: NEW tests/installed/check_deformable_dissipation.py and NEW tests/collision/test_deformable_dissipation_contact.py; self-contained D3/reference/installed path. No source/shared modifications.
- root only shared D1 batch/buffers, D3 contact, exports and new subtype; integration, all serial tests/build/install.
All use original isolated worktrees, based on current parent; preserve prior branches/WIP. No helper agents, no import/collection/probe/static/formatter/build by children. Any incompatible API/math issue report before changing this contract.

### Pre-execution clarifications at parent integration

- Observer masses are the registered stored FP32 inverse masses actually used by D1, not unrounded FP64 declarations.
- An additional pure device copy_dissipation_velocity(n, velocity, saved_velocity) kernel preserves the reconstructed velocity before the single damping pass. Both launches and their buffers are counted; no HOST-generated velocity is uploaded.
- The 2e-6 / 2e-5 gate applies to each same-candidate damping phase. For the independent continuous FP64 split oscillator trajectory, use absolute position2e-3m and velocity1e-2m/s, the already-declared D4 long-window engineering scale. Preserve every step/max error. The continuous analytic oscillator error is diagnostic, since the declared integrator is a first-order split. This clarification precedes every D6 execution; no case result has selected a threshold.

### D6-A first source results and the retained conservation limit

Runs102/103 pass the fixed-candidate kernel (33 interpret including 11 HOST,
22 CUDA); run105 passes 26 other public D1 cases, run106 both full-second
oscillators. Run104 retains the original strict free-pair momentum failure:
3.0517578125e-5 N s exceeds the authored 2e-5 gate. Run107 reads only that
closed saved prefix and verifies the entire old elastic FP32 primitive chain:
the raw imbalance closes as elastic predictor/position/reconstruction rounding
plus new damping rounding. It does not establish exact physical conservation.
The pair's replacement acceptance requires this independently bounded complete
arithmetic decomposition and retains the original strict pass/fail as a saved
diagnostic. Same-candidate D6 velocity/impulse gates and independent continuous
position/velocity gates stay unchanged. No mass, h, force, coefficient or
numeric threshold was tuned to this result. Original run104 remains FAIL.
This accepts the optional dissipative operator with documented finite precision,
consistent with the user's functional-first scope, not a repair of D1 conservation.

Run112 exposes the same existing elastic reconstruction precision limit in
D3: 4.31084045e-5 m/s at one component against the original2e-5 short gate.
That strict failure is retained. Before any subsequent run, D6-A's installed
consumer now reports FUNCTIONAL_PASS separately from strict_elastic_precision;
the latter records every original same-entry velocity violation. Exact saved
elastic primitives, all new D6 phase checks, original geometry/force/reaction/
transaction gates and the already-authored independent continuous2e-4 state
gates remain hard checks. This bounded functional qualification follows the
user's explicit finite-precision acceptance and does not claim the strict old
elastic gate passed. No parameter, mass, h or numeric threshold is changed.
