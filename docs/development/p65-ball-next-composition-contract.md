# Next ball composition: fixed-root contact before direct ball motors

Status: PROPOSED_CONTRACT / SOURCE_RESEARCH_ONLY / NOT_IMPLEMENTED / NOT_EXECUTED.
2026-11-12. This document proposes the next independently qualified slice; it is
not an implementation or execution grant. Parent product reading was at
3e37c1afed9802ddc19e9dbf63fb660ff764885f. BALL-A/ROB2b accepted results remain
limited to their reports. No old suite, G1 trajectory, source model or budget was changed.

## Decision and source facts

**Recommend BALL-C: fixed-root ball/hinge/slide trees interacting through the
existing primitive condim1 and pyramidal condim3 contact routes, with the
already supported scalar NONE/FILTEREXACT actors.** Direct ball motors remain
the next independent transmission slice. This ordering is an implementation
assessment from current source, not a claim that either new composition passed.

| Question | Current source fact | Consequence |
| --- | --- | --- |
| Why are ball contacts rejected? | [_model.py](../../src/phymeo/backends/mujoco_tyx/_model.py) 94–103 selects the no-contact ball route before contact dispatch; [_tree_model.py](../../src/phymeo/backends/mujoco_tyx/_tree_model.py) 136–149 rejects every potential pair when has_ball; [_actuator_model.py](../../src/phymeo/backends/mujoco_tyx/_actuator_model.py) 127–141 likewise forces a ball activation model to the bare tree. | Admission and composition routing are missing, despite reusable mechanics. |
| Are three tangent columns already available? | [_tree_device.py](../../src/phymeo/backends/mujoco_tyx/_tree_device.py) 42–78 preserves separate native joints/dofs, q width and all 8 velocity columns. [_normal_device.py](../../src/phymeo/backends/mujoco_tyx/_normal_device.py) 89–128,225–238 and [_constraint_device.py](../../src/phymeo/backends/mujoco_tyx/_constraint_device.py) 99–139,286–300 already select dof_ids for fixed mechanics and pass nq/njnt/native joint table to the quaternion solve. | BALL1's wrapper ABI is already connected; do not emulate a ball with three hinges or duplicate its integrator. |
| Is contact J scalar-joint-specific? | [_normal_geometry.py](../../src/phymeo/backends/mujoco_tyx/_normal_geometry.py) 306–327 and [_constraint_rows.py](../../src/phymeo/backends/mujoco_tyx/_constraint_rows.py) 127–210 iterate every velocity DOF, transport both COM Jacobians to the same contact point, subtract both sides, and rotate into the contact frame. | Reuse full J/M coupling and current normal/pyramid solvers. A three-column ball response still needs actual new qualification. |
| Why defer direct ball motor? | [_actuator_model.py](../../src/phymeo/backends/mujoco_tyx/_actuator_model.py) 24–29,60–114 and [_actuator_kernels.py](../../src/phymeo/backends/mujoco_tyx/_actuator_kernels.py) 34–86 encode one scalar qa/dof/gear; [_tree_model.py](../../src/phymeo/backends/mujoco_tyx/_tree_model.py) 60–67,114–133 rejects ball targets. [mujoco_actuation.py](../../src/phymeo/backends/mujoco_actuation.py) 38–64 binds exact scalar layout/schema. | Genuine three-component gear needs new transmission/scatter and stateless plus stateful paths; merely lifting the target rejection is incorrect. |

Fixed local upstream sources were read as text:

- MuJoCo3.3.7 CPU [engine_core_constraint.c 940–991](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_core_constraint.c#L940-L991):
  complete contact-point Jacobian, then normal row or opposing pyramid edges
  J_normal +/- mu*J_tangent. No contact special case turns a ball into a hinge.
  Cache: audit worktree
  artifacts/p65-mujoco-tyx-t4-research-20261110-01/upstream-3.3.7/engine_core_constraint.c.
- Fixed Warp0.0.2 local
  artifacts/p65-unified-scene-20261110-01/optional-deps/mujoco_warp/_src/smooth.py
  1814–1827 distinguishes ball gear[:3]/three moments/rotation-vector length
  from scalar gear[0]*q. Its constraint.py 1211–1360 uses the complete relative
  translational Jacobian for pyramid rows. These are source reuse references;
  CPU3.3.7 is the behavioral oracle.
- Installed Newton1.7.0 local
  .venv-newton-1.7.0/Lib/site-packages/newton/_src/solvers/mujoco/solver_mujoco.py
  7282–7362 creates one native ball (4 qpos/3 qvel) and per-axis actuator gears;
  1390–1396 and1868–1876 bind native condim/cone attributes.
  [Fixed Newton source](https://github.com/newton-physics/newton/blob/v1.7.0/newton/_src/solvers/mujoco/solver_mujoco.py#L7282-L7362).
  This is SolverMuJoCo support, not an assertion about all Newton solvers.
- Both upstream projects are Apache-2.0; preserve the package LICENSE/NOTICE and
  existing attribution. No dependency installation, source execution or online
  version substitution was performed.

## Public contract and admitted domain

Use existing public APIs, without a new robot wrapper or private state port:

1. compile_mujoco_xml(xml, require_asset=False).
2. MuJoCoRuntime(compilation, mode="interpret" or "cuda", implementation="tyx",
   world_keys=(7,41), initial_states={key:MuJoCoInitialState(q,v,act=act)}).
3. set_controls({key:MuJoCoControl(ctrl,qfrc_applied=full_nv,
   xfrc_applied=full_body_world_COM_wrenches)}); step(h).
4. integration_snapshot()/snapshot() preserve accepted q/v/act/held controls,
   complete last_step dynamics, contacts and any mixed row/actuation receipts.
5. checkpoint().to_bytes(), restore(...,expected_revision=...), clone(), and
   partial original reset() retain their actual next-step continuation semantics.
   Close owns all resources. No candidate-state read is a substitute for publication.

Fixed-root native BALL plus HINGE/SLIDE only; nv<=8, nq<=10, all physical native
bodies<=128; each ball is the only joint on its body. Existing scalar passive
terms, native address mapping, quaternion unit admission and complete physical
M versus Euler M+hD stay unchanged. Scalar JOINT actors retain nu<=32/na<=32,
NONE/FILTEREXACT, fixed gain and existing scalar bias/clamp/disabled rules.

For this new ball-contact composition, explicitly limit mixed condim to1 or3;
condim3 requires pyramidal cone. Use the existing nine primitive pair families
and native mask/weld/parent filtering, nonnegative mixed margin and original
solref/solimp/gap semantics. Declare <=128 geoms, <=16 potential pairs and
<=32 candidate slots for this slice; this is a stricter composition domain,
not enlargement of T3/T4 caps. Existing allocated contact/EFC caps and all
source-iteration/KKT gates still reject overflow, never truncate. Static
runtime default capacities and per-world/world-count limits are not increased.

Still reject FREE+BALL, direct BALL/FREE-target actors, JOINTINPARENT, tendon/site
transmission, same-body ball mixtures, ball springs, all joint limits and dry
friction, equality/flex/plugin/mocap, explicit pair/exclude, elliptic cone,
condim4/6, unsupported geometry/options and joint-wide summed-force clamps.
Keep all non-ball admitted domains and old rejection diagnostics unchanged.
Inactive/no-contact states inside an admitted potential-pair model must publish
zero current contact force, not stale force; never choose the route from overlap.

## Minimal shared seams; no new solver

Proposed product allowlist is only four MOD paths:

- _model.py: keep structural BALL precedence, but choose bare tree, normal, or
  pyramidal contact route from declared pairs/condim; na>0 still goes through
  the existing activation compiler. No failure-catching fallback.
- _tree_model.py: permit ball contacts only through existing explicit contact
  lowering, and check the narrower domain above before allocation. Keep its
  separate unconditional ball limit/dry/spring/target/structure rejections.
- _actuator_model.py: choose the same admitted contact base for BALL-A actors,
  preserving wrapper flags, native kind2, activation layout and hashes.
- _mujoco_tyx_runtime.py: truthful model-domain capability labels and declared
  contact dimensions; do not report ball joint limits/dry support from a generic
  mixed-row flag. Suggested labels: fixed-ball-scalar-normal-contact-v1 and
  fixed-ball-scalar-pyramidal-contact-v1, with explicit filterexact suffix.

Parent owns these shared routing/publication files. Existing normal/constraint
model/device/math files, actuator math, checkpoint schema, BALL mechanics and
integrator are read-only reuse in the first implementation proposal.
If a concrete source counterexample requires another seam, report its ABI and
the smallest added file before implementation; do not silently widen this list.
New author-owned files would be two focused ball-contact test files, one installed
public consumer and a result report. Freeze their exact XML/input tables before
the first execution; tests may reuse independent old oracle helpers, not product
kernels as expected values. This document grants no file implementation ownership.

## Fixed qualification matrix

All continuous cases: h=.001,24 real steps, two independent world keys7/41;
interpret andCUDA. Scalar controls follow BALL-A's registered +.8 for steps0–7,
-.75 for8–15 and0 for16–23; keep authored gains/gears/clamps. Original nonzero
q/v/act and requested full external loads are retained; load removal is not
an assertion that contact must disappear. No reset/warmup inside these windows.

| Cell | Predeclared composition | Required distinct evidence |
| --- | --- | --- |
| C1 | Eccentric anisotropic ball link, off-axis sphere against a static plane, condim1, nu=0; nonidentity quaternion and nonzero three-axis angular velocity | Nonzero contact moment about the ball, all three J columns and coupled response; loaded initial state and a separate initially inactive declaration. Contact cannot act only on a scalar coordinate. |
| C2 | Existing HINGE_BALL_SLIDE inertias/native address order with scalar NONE negative/shared gears; sphere or capsule tip against finite box, condim1 | Scalar qa!=dof, full M cross terms, actual normal force and both COM wrenches; one predeclared approaching state crosses from positive gap to contact during the window. |
| C3 | Same kinematic family with mixed NONE/FILTEREXACT, nonzero act, actearly variants; finite support, condim3, signed tangential incoming velocity | Full normal/tangent rows, force/motion and original activation clamp/disabled ordering on one continuous loaded/reversal/unload trajectory. No output-based frame or contact permutation choice. |
| C4 | Two dynamic ball branches plus scalar descendants, nq10/nv8 boundary, sphere-sphere contact, condim3; unequal masses/inertias and nonidentity orientations | Both dynamic bodies receive equal/opposite force and point-to-COM moments; native complete cross-coupling, not independent per-contact solves or prescribed partner motion. |
| C5 | C3 with sparse W4=(7,41,103,509) | Actual checkpoint/advance/restore successor; clone; partial original reset followed by a real step; untouched worlds exact. |
| C6 | C1/C3 final-world failure and constructor negatives | Original late status/readback/restore refusal preserves accepted state, act, raw clock, controls and complete receipt; real retry equals independent clone. Capacity, unsupported domain and wrong binding reject before allocation. |

C1–C4 exact source declarations must include analytical first-step sign/lever-arm
premises. C2 birth requires a pre-authored gap/closing-speed bracket, not fixture
tuning after execution. If that premise cannot be established, stop that case
during authoring and report it; do not manufacture contact or extend24steps.
The first implementation must not also add torsional/rolling friction, ball
limit cones, direct ball motors, a G1 run or a training-performance experiment.

## Independent reference and success/stop rules

CPU3.3.7 native model/data: same accepted entry q/v/act/control/applied forces,
fresh mj_forward fields, then native mj_step; a second independently advancing
CPU trajectory starts once from the same original initial state. Never overwrite
that trajectory with TYX output. Guard CPU forward/step only around actual TYX
step, separating oracle calls. Compare native body/COM/J/M/bias/passive/applied,
raw contacts, normal or full EFC J/R/aref/force, qacc and Euler integration.
Match contact geometry/identity before force comparisons; ordinal is not a
cross-precision identity. Independently reconstruct both-sided COM wrench and
J-transpose force, actuator generalized force/filter, and accepted v/q updates.

Keep BALL1 mechanical tolerances from test_mujoco_tyx_ball.py:20–32 and
BALL-A activation/force tolerances. For the newly composed contact solve use
the existing contact gates from test_mujoco_tyx_normal_contacts.py:32–34:
force rtol3e-3/atol3e-3, acceleration3e-3/1e-2, continuous q atol2e-4,
v3e-3 and time2e-6; mixed-row R/aref use the existing T4 5e-4/4e-4 gate.
These are the existing contact-domain gates, not retrospective changes to
BALL-A no-contact gates. Solver normalized KKT remains max(source tolerance,
2e-5), source iterations retain their current face-solve-budget semantics.
No equality of native iteration path/warmstart is claimed.

Success means every declared case passes both backends and the installed public
consumer preserves the same source/trajectory/receipt protocol; all first failures
and complete input/output/row histories are retained. Review only affected scalar
wrapper compatibility nodes; do not rerun old complete green suites. If the same
input exposes a semantic mismatch, missing column/address or refusal, classify
that concrete saved failure before a generic repair. Do not change XML, masses,
budget, tolerance or controls to obtain green. Two nonproductive rounds stop
this slice with the minimal reproducer; direct ball motors remain independent
follow-up work, not an excuse to hide the contact failure. No overall Newton
parity, general robot, grasping or G1-completion claim follows from this slice.
