# BALL1 actuator composition: next bounded increment

Status: SOURCE_RESEARCH_COMPLETE / PROPOSED_ONLY / NO_PRODUCT_CHANGE / NOT_EXECUTED.
2026-11-12. Read parent `b46ee0c05e16f77b137ba314c85be7f8ccee7913`; this is a
15-minute source decision, not a new implementation or execution grant. Existing
BALL1 and scalar activation qualification are reused only in their accepted scopes.

## Decision

Implement **BALL-A: scalar JOINT motors on a fixed ball/hinge/slide tree** first.
A hinge or slide before/after a ball can receive NONE or FILTEREXACT drive while
the ball moves through the existing full coupled M/J/b and quaternion integrator.
The actuator target remains scalar. This is a genuine connected feature, not a
HOST filter or a pretend three-hinge ball. No mathematical kernel change is
indicated by current source. Qualify the new composition rather than rerunning
all old BALL1/ROB2b cases.

A genuine ball JOINT motor is a separate **BALL-B** transmission increment.
Reusing filter integration is possible, but simply removing the BALL1 rejection
would interpret quaternion w as scalar position and apply force to only one of
three tangent DOFs. Do not admit it through BALL-A.

## Fixed source semantics

CPU [MuJoCo 3.3.7 transmission, lines 934–971](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_core_smooth.c#L934-L971)
uses native qpos/dof addresses. Hinge/slide length is gear[0]*q; its moment has
one nonzero DOF. Ball JOINT uses the first three gear entries: normalize its WXYZ
quaternion, convert to principal rotation vector r, set length=r dot gear and
place the three gear values in the moment row. JOINTINPARENT additionally rotates
the gear using the inverse quaternion and is a separate transmission mode.

The [rotation-vector conversion](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_util_spatial.c#L106-L119)
uses atan2 and wraps angles larger than pi. Near identity and the pi branch need
explicit handling; q[0] is never an angle. Native moment/velocity semantics must
be copied directly, not replaced by a guessed derivative of the rotation vector.

[Forward actuation](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_forward.c#L232-L254)
keeps the existing filterexact/clamp order. The same file lines 210–213 defines
actuator velocity as moment*qvel; lines 418–444 selects ctrl/entry act/actearly
next act, then fixed gain plus affine length/velocity bias; lines 516–521 clamps
scalar actuator force before moment-transpose accumulation. Lines 791–798 use
zero integration derivative for disabled actors. One ball actuator is still one
scalar input/activation/force with a 3-vector gear; three independent axis motors
are three actuators. The generalized contribution is gear*force, summed across
all actors. Neither force clipping nor activation clipping is a per-axis torque
clamp. Existing raw act_dot versus actual disabled integration-zero semantics stay.

Local fixed Warp0.0.2 `smooth.py:1814–1827` and `forward.py:524–571,764–774`
confirm ball/scalar transmission and full moment accumulation. Root-relative path:
`artifacts/p65-unified-scene-20261110-01/optional-deps/mujoco_warp/_src/`.
`smooth.py` SHA256 `9ccfe35e8e5b8d814047826f3e593690df62a5ac682c0a1f157b40b49771d98b`.
CPU 3.3.7 remains the behavior reference under the accepted FP32 gates.

Installed Newton1.7.0 `newton/_src/solvers/mujoco/solver_mujoco.py:7282–7362`
creates one native BALL and per-axis actuators with unit-vector gears; its
3959–4011 path forwards authored six-entry gear, dynamics, actearly and limits.
Its target-quaternion/velocity adaptation is additional functionality, not part
of BALL-A. This is the native SolverMuJoCo route, not a claim that every Newton
solver or its separate Actuator abstraction implements the same transmission.
Local root is `.venv-newton-1.7.0/Lib/site-packages/`; file SHA256
`0fb93442c5424e3e320a9b2a8c35308a9ac14be49fd04dcfb76073a32f9fe051`.
[Newton fixed source](https://github.com/newton-physics/newton/blob/v1.7.0/newton/_src/solvers/mujoco/solver_mujoco.py).
Both upstream projects are Apache-2.0; preserve existing attribution/LICENSE/NOTICE.

## Exact current gaps and reusable fields

| Current root file | BALL-A change or verified source-level reuse |
| --- | --- |
| `_model.py:compile_model` | Early any-BALL currently always selects `compile_tree_model`, before `model.na`. Keep structural BALL precedence, but select activation lowering when na>0; never catch a failed route and fall back. |
| `_tree_model.py:compile_tree_model` | Replace blanket ball nu/na rejection only with explicit scalar-target admission. Every actor must target a valid HINGE/SLIDE through JOINT. Keep all BALL1 geometry/constraint exclusions even when allow_activation=True. NONE-only actors keep old stateless validation and launch count. |
| `_actuator_model.py:compile_activation_model` | Existing ids already hold independent qa/dof/joint/actadr; scalar targets, native activation address cover, masks/ranges and capacities are checked. For has-BALL, choose `compile_tree_model(..., allow_activation=True)` directly before contact/free selection, so forbidden combinations cannot reach another lowering. |
| `_tree_kernels.py:mechanics` | Stateless force already indexes q by actuator qa and v/scatter by dof. No nq==nv assumption in this actor loop. Full physical M/J/b and original Euler M+hD remain unchanged. |
| `_actuator_kernels.py:update` | Same independent qa/dof metadata; scalar length/velocity, NONE/FILTEREXACT, negative/shared gear and ordering remain applicable. Do not change this kernel for BALL-A. |
| `_actuator_device.py`, `_tree_device.py` | Existing optional bank and one stage compose after complete mechanics(nu=0), before original solve. q width already covers nq<=10, velocity width8. Original NONE-only three stages; stateful four. No new buffer/ABI is indicated. |
| `_mujoco_tyx_runtime.py:capabilities` | Stateful kind2 currently reports fixed-hinge-slide-filterexact-v1; add truthful ball-tree/scalar-actuation domain labeling. Public q/v/act/forces/step receipt remain existing types. |
| `mujoco_checkpoint.py:binding/restore` | Stateful v2 binds na plus ids/parameters hashes and actual accepted act bank; full source/qpos bytes already included. Stateless v1 remains unchanged. New composition needs real continuation qualification, not a schema rewrite. |

Keep nv<=8, nq<=10, physical bodies<=128 and original world capacity. New BALL-A
actors are bounded by nu<=32 and na<=32 (NONE-only na=0; stateful na>=1). Keep fixed
gain, NONE/affine bias, finite positive tau and current mask/range rules. FREE,
ball-target motors, JOINTINPARENT/site/tendon transmission, contact, all joint
limits/dry friction, ball springs, same-body mixed joints and joint-wide summed
force clamps remain rejected before allocation. Do not loosen non-ball domains.

Minimal product allowlist proposal: `_model.py`, `_tree_model.py`,
`_actuator_model.py`, `_mujoco_tyx_runtime.py` (all in their existing owner lanes).
NEW two focused test files, e.g. `test_mujoco_tyx_ball_scalar_actuation.py` and its
transaction file, plus result report. No changes planned to mechanics/activation
kernels, device transport, public actuation schema or checkpoint implementation.
If runtime bank/state assumptions contradict this source reading, stop with the
specific ABI counterexample rather than broadening that list automatically.

## Predeclared acceptance for BALL-A

1. Original hinge-ball-slide geometry with nonidentity quaternion, nonzero all
   tangent/scalar velocities, eccentric COM/inertia; drive both scalar joints
   with NONE-only negative/shared gears. Include the scalar after the ball so
   qa!=dof is exercised. Full same-entry native M/J/b/applied and genuine ball
   motion through coupled response; 24 continuous steps with reversal/unload.
2. Same generic family with mixed NONE/FILTEREXACT native ordering, nonzero act,
   actearly both settings, control/activation/force clamps and disabled groups;
   preserve raw versus integrated derivative and actual force/motion publication.
   Two balls with following scalar at nq/nv capacity can be one bounded address
   case, not an unrelated enlarged domain.
3. Sparse W2/W4: checkpoint -> real advance -> restore -> same-control successor,
   clone, partial original reset, last-world/late-readback refusal and true retry.
   Retain q/v/act/raw clock/control/receipt together; no checkpoint from candidate.
4. Native ball target (including gear along its first axis only), FREE target,
   ball/contact/limit/dry/spring/same-body/JOINTINPARENT combinations and nu/na
   overflow reject before allocation. Revise only the old BALL1 blanket actuator
   negative that has become a scalar positive; real ball-target negatives stay.

Use existing BALL1 dynamics gates: pose/COM rtol2e-5/atol3e-5; J2e-4/5e-5;
M2e-4/3e-5; bias/passive/applied3e-4/2e-4; accelerations5e-4/8e-4;
continuous q atol5e-5, v3e-4, time2e-6. Activation rtol3e-5/atol3e-6 and
scalar/generalized force3e-4/2e-4 remain the ROB2a gates. Both backends need
independent CPU3.3.7 same-entry and continuous references plus analytic scalar
filter/gear expectations. Save complete original controls/fields/failures; guard
CPU forward/step only inside actual TYX step. No new G1 or learning run is needed.

## BALL-B successor: reusable integration, new transmission

Keep the complete mechanical solver. Add a source-bound transmission row with
joint kind, qa, dof start, width(1 or3), and full three-entry gear; publish actual
length, velocity and moment rows as well as scalar/generalized force. Current
ids[nu,9]/values[nu,12] encode one gear and one scalar DOF, and public actuation
schema/checkpoint hash binds precisely that layout. Version the extension; do
not reinterpret the existing scalar schema silently.

A narrow genuine torque-motor slice can initially require JOINT, FIXED gain and
BIAS_NONE on ball actors, NONE/FILTEREXACT dynamics, three noncollinear motors
and arbitrary finite signed/zero gears. Scalar actors can retain their existing
bias rules. This avoids pretending a full ball position servo has been ported;
affine ball length/velocity bias and its pi discontinuity require their own
transmission qualification. It still needs full three-DOF accumulation and must
handle na=0 NONE-only models without dummy activation coordinates or a fake
zero-sized CUDA buffer. Reuse filterexact math and accepted-bank publication;
new per-actor force scatter and evidence are real work, not admission-only.

Minimum successor cases: nonidentity anisotropic rotor under three independent
motors; oblique/shared and zero gear; parent/child ball coupling plus scalar
motor; filter/disabled/clamp interactions; complete force power equality
`v dot qfrc_actuator = sum(actuator_velocity*actuator_force)`; unchanged lifecycle
faults. Full M/J/b cross terms remain. Do not equate a single motor with three
independent controls, normalize authored gear, rotate JOINT gear into world,
or implement quaternion components as scalar motor lengths. JOINTINPARENT,
ball limits/springs/free-root/contact composition remain separate qualifications.

Decision gate: BALL-A is the smallest connected increment suitable for the
current window after owner/file approval. BALL-B should be the next separately
frozen slice because transmission layout, publication and checkpoint identity
change. No timing estimate or passing result is inferred from this research.
