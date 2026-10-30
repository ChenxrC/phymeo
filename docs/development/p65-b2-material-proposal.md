# B2 proposal: composite distance and signed-dihedral material

Status: **PROPOSAL_ONLY — NOT A FROZEN IMPLEMENTATION CONTRACT**.
Read-only code investigation and handwritten derivation; no B1 rerun, production
or test edit, import, numerical probe, test, compile, lint/formatter, build,
installation, profiling, remote CI or helper execution occurred for this proposal.

Inspected parent integration: `712fdadcaff83c661f769372aee85f074b2f8830`,
`C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo`.
This proposal is written only in the existing bending worktree, branch
`codex/p65-bending-module`, based on `96cbf209564d4444b27ef83cf6817cbb64859110`.
No integration merge/reset was performed. Prior B1/S1/G1 passing counts are
context supplied by the parent, not new evidence produced here.

## Recommendation and alternatives

Recommend one additive `ClothMaterial`, composite program/participant, ordered
trace schema, actual native bending kernel and independent composite ledger.
Keep existing distance-only arithmetic, program and ledger behavior unchanged.
The parent alone modifies existing material/contact/replay/export seams.

| Alternative | Tradeoff and decision |
| --- | --- |
| Add bending to old ClothProgram and remove distance-only gates | Fewer APIs, but changes old signatures, evidence and acceptance together. Reject: the frozen architecture requires additive verified composition. |
| Separate bending participant over the same particle state | Independent code but two candidate owners, duplicated prediction/reconstruction and missing cross terms. Reject. |
| One new composite owner reusing old distance arithmetic | More explicit adapters/proof code, but one physical state and auditable ordering. Recommended. |

A host B1 call inside a tyx participant is not native bending. Two independent
final-state ledgers cannot certify an interleaved mixed path. An unbounded
numerical atan2 oracle is not independent ideal-angle evidence.

## Existing code and concrete constraints

References are relative to the inspected integration commit.

| Existing source | B2 consequence |
| --- | --- |
| `src/phymeo/constraints/bending.py:20,116,183,266`; `bending_trace.py:29` | Owned oriented hinges, positive compliance, analytic FP64 gradient, atomic bending-only sweep, endpoint traces. No native/timestep acceptance. |
| `src/phymeo/solvers/cloth.py:15,70,91,119,143` | Distance-only; one prediction, zero initial lambdas per workspace, accumulating sweeps, one final velocity reconstruction. |
| `src/phymeo/validation/material_path.py:33,82` | Strict DistanceConstraints/None; schema `xpbd-distance-path-v1`, `(m,18)` frames, complete order/continuity and original-input checks. Device results must be exact FP32 promotions. Do not weaken it. |
| `src/phymeo/validation/_material_path_math.py:105,121` | Independent distance energy bounds and actual K/gravity/work/momentum reporting are separate from equation acceptance. |
| `src/phymeo/solvers/_material_contact_balance.py:20,38` | Absolute energy/angular ceilings; per-domain angular L1 sum without cancellation. Current elastic helper is distance-only. |
| `src/phymeo/coupling/participant.py:31,78,120` | One substep owner; configuration loads and instantaneous impulses have distinct operations. |
| `src/phymeo/backends/tyx_participants.py:26,220,228,295,328` | Explicit FP64 storage/FP32 computation; one distance lambda vector and one native reconstruction today. |
| `src/phymeo/backends/tyx_node_math.py:38,103`; `_tyx_distance_trace.py:13` | Native ordered distance lane; per-call FP32 conversion and fixed FP64 position restoration. Mixed evidence must expose these boundaries. |
| `src/phymeo/backends/tyx_math.py:15,29,76` | Normal-or-zero FP32, i32 indices, private arrays, sync, finite-result checks, cleanup and counters already exist. |
| `src/phymeo/runtime/load_replay.py:118,181` | Every trial restarts from original state/history and applies held integral/h once. No lambda carry between trials. |
| `src/phymeo/solvers/_finite_contact_replay.py:93`; `_thick_contact_replay.py:101,149` | Audit held pre-terminal material state; terminal replay preserves complete selected held evidence. Aggregate kinetic cross terms remain mandatory. |
| `src/phymeo/solvers/dynamic_robot_thick_contact.py:90`; `_thick_contact_workspace.py:66` | Distance-only material gates/ledger creation; selected contact nodes have positive inverse mass. Parent adds explicit dispatch, not deleted gates. |
| `src/phymeo/solvers/articulated_coupling.py:113,141`; `runtime/scene_stepper.py:153,231` | Participant sweep before contact; cleanup on failure; later rejected substep returns whole original state/history. |
| `src/phymeo/core/scene.py:141,179` | StepSettings has no angle reference. StageMetrics geometric fields are metres: do not place radians there. |

Read-only dependency evidence: delivery `.venv` has tyx pin
`99f6df7c0fd8fdeef77f97df83914da27b64b7ee` in direct_url.json.
`tyx/intrinsics.py:220` declares atan2(f32,f32)->f32 mapped to NumPy arctan2,
std::atan2 and atan2f; its AD rule is unsupported. Lines 456ff show interpret
dispatch. This is source evidence only, not an actual backend test or an ULP
bound. The B1 host-test `.venv-py311` did not expose tyx in the directory read.
No dependency installation or pin change is proposed.

## Proposed public configuration and execution interfaces

These names and policies require parent freeze. Configurations/evidence own
read-only data. Scalars reject bool/NaN/infinity and invalid signs.

| Proposed interface | Contract |
| --- | --- |
| `ClothMaterial(distance=None, bending=None)` | Owned FP64 DistanceConstraints and B1 BendingConstraints; concrete two-law sum, not an arbitrary callback. Immutable order `("distance","bending")`. Empty blocks remain explicit. |
| `CompositeMaterialAuditSettings(reference_angle_rad=1.0, angle_terms=64, max_domain_intervals=4096, max_domain_depth=32)` | Separate angle scale and deterministic proof-resource budgets, all bound in the signature. Exhaustion rejects, never loosens a tolerance. |
| `CompositeMaterialBudgets(max_energy_gain_j, max_angular_defect_nms)` | Explicit positive per-substep physical ceilings; no output-derived defaults. Whole-trajectory ceilings are separately frozen. |
| `CompositeClothProgram(model,material,domain="cloth",*,audit_settings,budgets)` | New host program, one ParticleState; always records/audits mixed path. Old ClothProgram unchanged. |
| `CompositeClothParticipant(model,material,*,audit_settings,budgets)` | Existing ParticipantWorkspace contract and one ParticlePointPort state owner. |
| `TyxCompositeClothParticipant(model,material,mode,*,audit_settings,budgets)` | Same law, FP32 native calculation and FP64 storage, actual native bending/distance/prediction/reconstruction. |
| `CompositeMaterialPathLedger(model,material,initial,original_inputs,h,settings,audit_settings,context)` | `.require(candidate,effective_inputs,expected_impulses,diagnostics,*,sweeps)` audits new schema and returns immutable evidence. |
| `bending_bounds(nodes,rest_angle,compliance)` | Audit-only intervals for ideal theta, gradient, C and energy; never supplies replacement state/gradient. |
| `certify_bending_segment(start,end,material,*,budget)` | Certificate for an explicitly named affine segment; unknown/rejected is distinct from certified. |

x/v: `(n,3)` m and m/s; inverse_mass: `(n,)` 1/kg. Distance indices `(md,2)`,
rest m, compliance m/N, alpha_d=compliance_d/h^2 in 1/kg, lambda_d in kg*m.
Bending indices `(mb,4)`, rest rad, compliance 1/(N*m), alpha_b in 1/(kg*m^2),
lambda_b in kg*m^2. Both delta_lambda_d*direction and delta_lambda_b*gradient
are nodal configuration loads kg*m; divide by h for world impulse N*s.

Residuals have separate dimensionless distance/bending entries. Keep
geometric_error as distance metres; expose `angular_error_rad` separately.
Analytic local gradients do not establish a differentiable entire simulator;
global gradient capability remains unsupported.

Recommended first physical composite/contact scope: all inverse masses positive,
matching current thick contact. B1 and old distance fixed-node behavior remains
unchanged. A composite fixed-support extension needs a separately explicit
stationarity/reaction/work/moment contract. This support boundary is proposed,
not silently imposed on B1 or approved by this document.

## Ordered trace and transactional publication

New schema: `xpbd-composite-material-path-v1`, bound to material identity, h,
reference scales, precision and proof policy. A sweep copies x, both lambda
vectors and its trace checkpoint; runs all original distance rows, then all
bending rows; validates both blocks; publishes all state/lambdas/evidence only
when the complete sweep succeeds. A late bending failure discards distance
updates too. No prefix publication.

`CompositeMaterialSweep(sweep_index,blocks=(distance_batch,bending_batch))`
has two blocks even when empty. Sweep indices are consecutive and count actual
inner position iterations, not outer replay attempts. Each block owns:

- kind, element identity/order;
- `(n,3)` positions_in, positions_compute_start, positions_compute_end,
  positions_out;
- lambda_in/compute_start/compute_end/out in its own shape and physical units;
- original parameters/h, actual alpha and declared compute conversions;
- backend/dtype and operation-specific launch evidence;
- original distance `(md,18)` records or complete ordered B1-shaped bending
  observations; conversion/restoration and domain-certificate identity.

NumPy compute boundaries are identities. Tyx arrays/observations are exact FP32
promotions and original FP64 parameters are independently retained. Every
conversion and any admitted fixed restoration is checked explicitly. A distance
adapter may reconstruct compute-end by replaying observed post-position columns;
it must not compute synthetic observations from a replacement solve.

Native bending wire proposal: `(mb,45)` FP32 fields = before x 12 + after x 12
+ gradient 12 + theta before/after 2 + C before/after 2 + lambda before/after 2
+ delta 1 + alpha 1 + denominator 1. Separate `(mb,4)` i32 identity, `(mb,)`
i32 status and `(1,)` first-failure status/index. Host wrapping is observational.
Empty blocks explicitly bypass launches because TyxMath rejects zero-size arrays.

Bind final positions and both multiplier vectors to the complete path, and
check final residuals of both laws after the entire sweep. Bending changes
shared distance lengths and distance changes hinges. Missing/reordered/stale
rows must reject even when final x coincides. Initially reject unrecorded
configuration loads between material sweeps: the existing thick base has an
empty native contact set and held-input replay, which fits. Old node/sphere
position-load interleaving needs a typed load event before claiming support.

## Native execution and lifetime

New `project_bending_native(engine,x,w,bending,alpha_b,lambda_b)` should use the
same TyxNodeMath/TyxMath engine as the composite participant. A new ordered
one-lane @tx.kernel computes atan2, analytic gradient, denominator, delta,
position changes, post-state and observations. Follow existing checked FP32
arithmetic and finite-output/failure-status conventions. Private buffers survive
only until the whole mixed sweep is accepted; synchronization and cleanup use
the existing engine. Host bounds reject but never repair x, lambda, rest or sign.

Require actual operation-specific bending launch counts: unrelated completed
prediction/axpy launches cannot prove bending execution. Each nonempty bending
batch must have its native observations and corresponding candidate. Failed
attempts remain visible in counters without committing physical state/history.
Interpret/CUDA/bounded C++ remain separate claims; no mode or host fallback.

Prediction occurs once, lambda vectors reset per inner substep and replay trial,
reconstruction occurs once, and terminal impulses do not move x or rerun material.
Preserve complete held-pre fingerprint checks. Any later velocity/audit/close
failure must trigger existing whole-state/history rollback.

## Independent ideal angle, gradient and energy bounds

This is a constructive strategy, not implemented proof. The audit must not call
production bending_angle_gradient/bending_energy or use a native result as truth.
Treat binary stored coordinates/parameters as exact rationals with q. For
`e=b-a`, `u=c-a`, `v=d-a`, derive independently:

```
E = e dot e
D = det(e,u,v)
A = (e dot u)*(e dot v) - E*(u dot v)
S = -sqrt(E)*D
N_L = e cross u; N_R = v cross e
N_L dot N_R = A
|N_L|^2*|N_R|^2 = A^2 + E*D^2
Theta = atan2(S,A)
```

Certify E>0 and, for each oriented face,
`|N|^2 > q(1e-10)^2*scale^4`, where scale is its exact maximum absolute edge
component. This is B1's existing binary threshold, not a new epsilon.

Use rational sqrt enclosures; `_contact_replay_audit.sqrt_bounds` already has
an integer-square-root 256-bit construction. sign(S)=-sign(D). Compare |S| and
|A| by rational `E*D^2` versus `A^2` to reduce a ratio r into [0,1] without a
floating quadrant guess. D=0,A>0 gives zero; D=0,A<0 is the forbidden cut;
A=0,D!=0 gives signed pi/2. Both zero contradict certified nondegeneracy.

Apply twice `t <- t/(1+sqrt(1+t^2))`. Then t<=tan(pi/16)<1/4 and
atan(r)=4*atan(t). The alternating series through degree 127 (64 terms) has
remainder at most `4*(1/4)^129/129 < 2^-254` radians. Interval sqrt/operation
widths are additional, explicitly tracked errors. Bound pi independently using
`pi=16*atan(1/5)-4*atan(1/239)` with alternating remainder bounds, then restore
the atan2 quadrant using exact signs. Fraction(np.pi) and Fraction(float(atan2))
are stored observations, never ideal pi/theta.

Ideal gradient components are radical-rational expressions:

```
g_c = -sqrt(E)*N_L/(N_L dot N_L)
g_d = -sqrt(E)*N_R/(N_R dot N_R)
t_c = (u dot e)/E; t_d = (v dot e)/E
g_a = (t_c-1)*g_c + (t_d-1)*g_d
g_b = -t_c*g_c - t_d*g_d
```

This unscaled audit algebra is independent of the producer's scaled-face
implementation. Enclose `(Theta-rest)^2/(2*compliance)` with interval squaring
including zero for an interval spanning zero. Sum both material families once.
Hard distance energy exclusion and its original equation/physical-gain checks
remain unchanged.

Require strict ideal theta/C branch membership from enclosures. Proposed
internal normalized enclosure width is <=tau/64, while every final normalized
upper error is <= original tau. Exhausted precision/series budgets reject.
For each row audit the ideal residual interval
`C_I+a_I*lambda_before+(sum(w_i*|G_i|^2)+a_I)*delta_lambda` and separately bind
observed theta/C/gradient/alpha/denominator and lambda update. Audit actual dx
against w*delta_lambda*g_observed, with gradient, parameter, conversion and
storage errors accumulated in an h*reference_speed displacement budget. Do not
average errors over sweeps or accept a self-consistent but wrong producer.
Physical energy gates use outward intervals before rounding diagnostic floats.

Still unimplemented: a verified interval representation, all quadrant/zero
cases, dependency conservatism, precision-cap behavior and independent bound
tests. No backend atan2 ULP guarantee or performance claim is assumed.

## Continuous domain: separate proof and explicit blocker

B1 endpoint smoothness plus rejection of a >=pi angle jump does not prove an
affine segment avoids degeneration or crosses the cut and returns. A full
trace is a sequence certificate, not a continuous-domain certificate.

For a named affine nodal segment x(s), s in [0,1], e/u/v are linear; face N is
quadratic; |N|^2 and A are quartic; D is cubic and E is quadratic. For all six
oriented face-edge components f_j(s), certify
`|N(s)|^2-q(1e-10)^2*f_j(s)^4>0`. Requiring all six exactly expresses the
max-component-scaled face condition. Rational Bernstein bounds with deterministic
dyadic subdivision can prove positivity; root isolation detects failures.
A branch cut requires D(s)=0 and A(s)<0: isolate cubic roots and establish A's
sign on their rational isolating intervals, explicitly handling zero/common
roots. Subdivided ideal-angle enclosures then prove |Theta(s)-rest|<pi.
Exhausted interval/depth budgets, a strict boundary or unresolved roots reject.

Recommend separate certificates for every numerical material update/conversion/
restoration segment and for the declared physical affine x_initial->x_final
substep. Solver iterates are not physical time; neither certificate proves the
unknown exact continuous solution. A distance row can invalidate an incident
hinge before its bending turn, so inspect exact affected topology after each
row; no arbitrary neighborhood cutoff. This material-domain proof cannot replace
S2 collision/CCD. Parent may split B2-discrete and B2-domain checkpoints, but
without the latter no complete material-domain/folded-step claim is justified.
The polynomial proof, cost/support tradeoff and implementation remain open.

## Actual energy, work, power and NUM005

Let q_ri=delta_lambda_r*g_ri for bending and the original signed direction load
for distance. Aggregate actual recorded nodal configuration loads across all
rows, then `J_material_i=sum_r(q_ri)/h`. Do not apply them again: reconstructed
velocity already includes their effect. Scalar bending lambda is not a world
impulse, and nonlinear final lambda times final gradient is not the full path.

For bookkeeping `u_r=v_pred+(x_r-x_pred)/h`, stored dx gives du=dx/h and

```
Delta K = sum_i m_i*u_before_i dot du_i + (m_i/2)*|du_i|^2.
```

This is a discrete algebraic work identity, not a fictitious solver-time flow.
With exact dx=w*q it is mid-velocity work of q/h; record conversion/projection/
restoration/reconstruction discrepancies separately. Recompute whole potential
changes, including the other material's shared elements. At a frozen physical
state, independent conservative force `F_i=-sum(C/compliance)*g_i` satisfies
`dU/dt=-sum(F_i dot v_i)`; bound it independently for force/moment/power checks.
Do not infer that nonlinear finite XPBD conserves energy from this local identity.

Physical acceptance remains actual endpoint K+gravity+both elastic energies
minus original-input endpoint work. Audit held request work and represented-input
discrepancy separately. Preserve terminal full aggregate kinetic work
`p dot v + p dot W p/2`, with shared-node cross terms and unchanged potential.

For all-dynamic domains retain actual angular defect:

```
D_L = sum_i[x1_i cross(m_i*v1_i)-x0_i cross(m_i*v0_i)
            -x0_i cross(h*(m_i*g+F_original_i)+J_held_i)
            -x1_i cross J_terminal_i].
```

Sum per-domain componentwise absolute values; opposite domain defects cannot
cancel. Each local ideal gradient may have zero torque at its own x_r while
its impulse has a moment at x0. This includes
`sum_r,i (x0_i-x_ri) cross(q_ri/h)`, plus observed gradient and rounding/motion
errors. Report that mechanism, but never subtract it from D_L to pass a budget.

NUM005 stays open. Preserve the hard rotating edge's
`Delta_Lz=2/sqrt(1+h^2)-2`, existing h=.1 rejection and h=.025 acceptance with
fixed .001 N*m*s budget. Direct regressions are:
`test_material_ledger.py::test_angular_loss_is_reported_as_material_discrete_contribution`,
`test_tyx_material_path.py::test_actual_rotating_edge_keeps_original_discrete_loss`,
`test_initial_robot_material.py::test_fixed_angular_budget_rejects_coarse_rotating_material`.
These tests are proposed future direct-consumer selection, not executed here.

## Two independent non-handoff scenarios and proposed frozen gates

All following equations/ceilings are proposed for pre-execution freeze, not
observed or tuned. A failure requires diagnosis or an explicit parent contract
decision. Host equation tolerance 1e-8, native 2e-5; Lref=Vref=angle_ref=1 SI.
Physical ceilings are independent of these solver tolerances. Require actual
backend execution rather than skip/fallback when a mode is claimed.

### A. Unit-hinge elastic release

a=(0,0,0), b=(1,0,0), c=(0,1,0), d=(0,-1,0) m; masses all 1 kg, v=0,
gravity=force=0. One hinge, rest=1/4 rad, compliance=1/8 1/(N*m), explicitly
empty distance block. At h=1/8 s, the first sweep has
`g_a=(0,0,2),g_b=0,g_c=g_d=(0,0,-1)`, alpha=8, denominator=14 and
`delta_lambda=1/56 kg*m^2`. Independently:

```
a_z=1/28; b_z=0; c_z=d_z=-1/56; x/y unchanged.
delta=1/56
Theta_after=atan2(6*delta*sqrt(1+4*delta^2),1-5*delta^2)
U_before=1/4 J
K_after_reconstruction=3/49 J
U_after=4*(Theta_after-1/4)^2 J
P_after=0; L_after=0.
```

This first-sweep oracle does not assert final convergence or claim an accepted
one-iteration timestep. Then use the public runtime to T=1/4 s at
h=1/64,1/128,1/256, max 512 position iterations/substep. Require equation and
named domain certificates, actual K/U/P/L reports, zero external work and no
invented supports. Proposed per-step AND fixed whole-window positive energy
gain and angular-defect ceilings: 1e-3 J and 1e-3 N*m*s. Values retain the
existing material scale; the window bound is separately proposed and must not
be multiplied by step count. Reflection symmetry and zero net momentum are
independent controls, not a claim of exactly zero late-step angular defect.

An independent continuous-limit derivation is also available: rest=0,
compliance=6, z=(eta/3,0,-eta/6,-eta/6), eta=2^-10 and v=0 gives linearized
`theta''+theta=0`. Its solution is eta*cos(t), rate=-eta*sin(t), while the
linear discrete mode obeys
`theta_new=(theta+h*rate)/(1+h^2)`,
`rate_new=(rate-h*theta)/(1+h^2)`, both using old values. At T=1 its amplitude
error is bounded by `eta*(h/2+h^2/3)` for h=1/16,1/32,1/64. This is only the
linearized model: the nonlinear remainder and accumulated native rounding
bound are not yet supplied. They are explicit pre-freeze blockers for a
quantitative continuous-trajectory claim; do not use this formula as the exact
nonlinear production trajectory or silently choose a fitted envelope.

### B. Asymmetric six-node mixed strip with actual load replay

Triangles `(0,1,2),(1,3,2),(2,3,4),(3,5,4)`; hinges
`(1,2,0,3),(3,2,1,4),(3,4,2,5)`. Coordinates (m):

```
0=(0,0,0)       1=(1,0,0)
2=(0,1,1/16)    3=(1,1,-1/32)
4=(0,2,1/8)     5=(1,2,0)
inverse_mass=(1,1/2,1/4,2,4,1) 1/kg
```

Distance order is `(0,1),(0,2),(1,2),(1,3),(2,3),(2,4),(3,4),(3,5),(4,5)`.
Flat-reference rest=1 except diagonals (1,2),(3,4), whose stored FP64 rest is
1.4142135623730951. Distance compliance=1/64 m/N; bending rest=0 and
compliance=1/8 1/(N*m). Both families are active and share nodes.

With initial mass centroid Xcm, set
`v_i=(1/16,-1/32,0)+(0,0,1/8) cross(x_i-Xcm)` m/s, gravity=0 and original
force `m_i*(0,0,1/64)` N. One held replay requests J0=(1/64,0,0) N*s,
J5=-J0; other rows zero. Net impulse is zero, but prescribed moment is
`(x0-x5) cross J0`, generally nonzero. Do not incorrectly demand zero torque.
Evaluate p, zero, p from the identical original snapshot: first/third complete
state/history/trace must agree. Add terminal node-2 impulse (0,1/64,0) N*s and
check full kinetic work exactly once, without reapplying material.

Use h=1/128,1/256,1/512, T=1/8 s, 512 position iterations, unchanged equation
tolerances and the same proposed per-step/window 1e-3 J/N*m*s ceilings.
Independent expectations are the determinant angle identity, separately derived
energy derivatives, exact aggregate input/work/momentum relations and ordered
trace replay, not the production solver's output. Future tests may use centered
differences of a separate scalar-energy oracle; rigorous bounds decide acceptance.
Rotate/translate geometry, velocities, forces and impulses consistently.
Perturb node-4 z by +/-2^-12 m and all masses by factors 1/2 and 2 in separate
bound cases without retuning budgets.

Permuting material order changes the bound signature and may change the discrete
trajectory; require its correct new path and the original physical gates, not
bitwise order invariance. Two disconnected copies in one particle domain must
sum energy/loads without double state application. A hinge-compliance-to-sheet-
modulus calibration for mesh refinement and a quantitative long-duration/error
convergence envelope are not proved here. These remain full-acceptance blockers;
the finite windows above do not close them.

### Shared failure controls

Late hinge degeneration; explicit atan2 jump; intermediate degeneration with
admissible endpoints; conversion-induced domain loss; altered theta/gradient/
denominator/alpha; missing/reordered rows; hidden final mutation; interval
budget exhaustion; failed native launch; late velocity/diagnostics/close or later
substep failure. Each rejects the whole trial and preserves original history/
state; retry equals fresh. Opposite angular defects in two domains must not
cancel. Preserve the NUM005 controls. No such tests have run for this proposal.

## Exact proposed file ownership and future test selection

Only this proposal document may be written now. Future ownership is a proposal,
not permission to start implementation.

Bending lane, new files:

- `src/phymeo/constraints/cloth_material.py` — owned material/audit/budget data.
- `src/phymeo/constraints/material_trace.py` — complete mixed batches/sweeps.
- `src/phymeo/solvers/composite_cloth.py` — one host substep owner.
- `src/phymeo/coupling/composite_cloth.py` — host participant adapter.
- `src/phymeo/backends/tyx_bending.py` — native buffer/observation adapter.
- `src/phymeo/backends/_tyx_bending_kernels.py` — actual ordered FP32 operator.
- `src/phymeo/backends/tyx_composite_cloth.py` — one native composite owner/ports.
- `src/phymeo/validation/bending_bounds.py` — independent ideal bounds.
- `src/phymeo/validation/bending_domain.py` — affine-segment domain certificate.
- `src/phymeo/validation/composite_material_path.py` — original equations/sequence.
- `src/phymeo/validation/composite_material_energy.py` — actual physical budgets.
- `tests/test_bending_bounds.py`, `tests/test_bending_domain.py`.
- `tests/test_composite_cloth.py`, `tests/test_composite_material_path.py`.
- `tests/test_composite_material_physics.py`, `tests/test_composite_material_transaction.py`.
- `tests/test_tyx_bending.py`, `tests/test_tyx_composite_material.py`.

Parent alone, first thick-contact integration:

- New `src/phymeo/solvers/_material_dispatch.py`: explicit own/bind/ledger/energy
  dispatch, legacy DistanceConstraints/None versus verified ClothMaterial.
- Existing `src/phymeo/solvers/dynamic_robot_thick_contact.py`: bind new typed
  material/signature while retaining FP64, positive masses and topology checks.
- Existing `src/phymeo/solvers/_thick_contact_workspace.py`: ledger factory.
- Existing `src/phymeo/solvers/_material_contact_balance.py`: elastic-bound
  dispatch without changing original work or energy/angular ceilings.
- Existing `src/phymeo/__init__.py`: reviewed public exports.
- New `tests/test_composite_thick_contact.py`; targeted existing
  `tests/test_thick_robot_transaction.py`, `tests/test_thick_robot_tyx.py`.

Old distance, distance-trace, cloth, MaterialPathLedger and native distance
arithmetic are not lane edit targets. Initial/normal/finite-contact constructors
continue to reject unsupported composite materials until the parent grants their
own integration scope; do not broadly delete type gates. Existing SceneLoadReplay,
SceneStepper and terminal replay should need no semantic change. An actual need
to alter them is an interface finding for the parent, not unilateral expansion.

Future direct-consumer selection:

- New bounds/domain files: independent special angles/quadrants, near-cut cases,
  rigorous remainders, original conditioning and unknown rejection.
- New host/path/physics/transaction files: both scenarios, full order/units,
  original residuals, physical budgets and complete rollback.
- New native files: mandatory interpret/CUDA for claimed capability; bounded
  cpu_cpp under existing resource gates. No unsupported cpu_llvm promise.
- Existing material_path/material_ledger and their tyx tests only where adapters
  or dispatch directly affect them; old strict law and NUM005 remain unchanged.
- Exact targeted nodes:
  `test_tyx_participants.py::test_cloth_workspace_uses_ordered_constraints_and_device_reconstruction`,
  `test_tyx_participants.py::test_node_position_units_reconstruct_only_once_and_reject_bad_state`,
  `test_load_replay.py::test_actual_xpbd_hidden_multipliers_are_rebuilt_for_each_trial`,
  `test_load_replay.py::test_p_zero_p_uses_original_time_state_and_inputs`,
  `test_thick_robot_transaction.py::test_terminal_pre_replay_matches_full_selected_held`,
  `test_thick_robot_transaction.py::test_terminal_only_replay_rechecks_complete_selected_pre_evidence`,
  plus the three NUM005 nodes above.

Each future execution requires one immutable source, explicit import path,
fresh checked basetemp, raw output/exit and resource release. No queue grant,
remote CI, packaging, benchmark or old full-suite rerun is implied here.

## Precise remaining parent decisions and blockers

1. Ratify additive APIs, distance-then-bending order and always-complete trace
   versus old API extension; recommendation is additive.
2. Ratify explicit FP32 native computation, conversion records and operation-
   specific launch receipts. Declared atan2 availability is not verified execution.
3. Ratify all-dynamic first physical scope or supply fixed-support work/moment
   equations. Preserve current contact positive-mass checks.
4. Decide whether B2-discrete precedes B2-domain as a separate limited checkpoint.
   Without segment proof, no complete domain/folded-step certification.
5. Implement/verify independent rational atan/gradient/energy enclosures, strict
   unknown handling and polynomial segment certificates. No existing single-
   quadratic arithmetic field represents ideal angle or establishes this proof.
6. Freeze proposed physical/trajectory ceilings before results; derive the missing
   nonlinear small-mode remainder, native accumulation bounds and mesh-compliance
   calibration before quantitative continuum/time/mesh convergence claims.
7. Parent must bind the same material identity in participant and contact mapping;
   a correct ledger for different material cannot certify the actual native path.
   Preserve full selected held-pre evidence and one state owner.
8. NUM005 and known tyx lifetime/resource limitations remain open. Accurate
   bookkeeping must not relabel their defects as conservation or capability.

This is a concrete partial design checkpoint. The lane releases it after the
allowed document commit; the parent combines B2/S2/G2 contracts before code or
new executable validation begins.
