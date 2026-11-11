# Native ball joint: next bounded slice

Status: RESEARCH_COMPLETE / FIRST_SLICE_PROPOSED / NOT_IMPLEMENTED / NOT_EXECUTED.
Baseline: parent `20850b876586125cc94c296b5d1fb930a150f766`. No project execution.
The memory diagnostic was separately handed off at `89f32e8e`; it grants no ball
qualification. This is a proposed development boundary, not an implementation grant.

## First connected slice

Recommend BALL1: fixed-root BALL + HINGE/SLIDE trees with actual generalized
applied forces and world COM wrenches, complete device M/J/b, Euler advance and
public observation/checkpoint/restore. Keep `nv<=8`, native bodies<=128 and the
existing world capacity. One or two balls fit the DOF budget; `nq=nv+nball` can
reach 10 without increasing the physical solve capacity. Qualify W1/W2/W4.

First scope: nu=na=0, no free root, possible contact pairs, limited joints, dry
friction, equality/tendon/plugin/mocap/userdata, or ball spring. Native nonnegative
damping and armature apply to all ball DOFs; existing scalar spring semantics
remain on scalar joints. Keep every body, eccentric COM/inertia and transform.
A ball-owning body has only that joint in BALL1; same-body slide+ball is a later
qualification. Mixed joints on separate bodies form real chains and branches.

`MuJoCoControl.qfrc_applied` drives the three ball torque coordinates and
`xfrc_applied` supplies real COM wrenches. Force, motion, receipts and accepted
state must publish together. This is not yet a motor-transmission or ball-limit
feature. A HOST quaternion utility or successful model import is insufficient.
The complete free/ball/contact/activation/limit port exceeds a safe last-minute
change: freeze this slice now; assign product ABI/files before implementation.

## Fixed primary-source semantics

- Ball uses WXYZ qpos[4], three tangent angular velocities and an anchored
  relative rotation. Native allows slides with a ball on one body, but no other
  rotational joint there. [MuJoCo 3.3.7 XML](https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#body-joint).
- FK right-multiplies local rotation and corrects the body origin about the
  anchor. Tangent axes are current child-body rotation columns. All three ball
  cdof_dot columns use the same pre-ball velocity before the complete rotational
  velocity is added. [smooth.c 107-124,219-227,1825-1836](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_core_smooth.c).
- Euler first advances velocity. Quaternion integration uses native q/dof
  addresses, normalizes the input and right-multiplies the local increment.
  [integratePos](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_support.c#L499-L525),
  [quatIntegrate](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_util_spatial.c#L215-L224),
  [Euler](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_forward.c#L768-L815).
- Ball spring uses a quaternion-difference three-vector, not four scalar springs;
  damping is per DOF. [passive.c 83-110](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_passive.c#L83-L110).
- Ball JOINT transmission uses the first three gear entries: length is rotation
  vector dot gear; the moment row has those three entries, and velocity is moment
  dot qvel. JOINTINPARENT rotates gear additionally. Preserve native moment
  semantics rather than inventing three scalar gear channels. [transmission](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_core_smooth.c#L943-L971).
- Ball limit is one angular-extent row: normalized-quaternion angle,
  `dist=max(range)-angle`, active below margin, Jacobian negative unit rotation
  axis across all three DOFs. It is not three hinge bounds. [constraint.c](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_core_constraint.c#L747-L788).

Fixed installed Warp0.0.2 source under parent's
`artifacts/p65-unified-scene-20261110-01/optional-deps/mujoco_warp/_src` confirms
these separate coordinates: smooth.py:126-131,478-482,1715-1724,1814-1824;
forward.py:99-108; math.py:160-170; constraint.py:_efc_limit_ball; passive.py:145.
Warp additionally normalizes the integration result. CPU3.3.7 ordering is the
reference under existing FP32 engineering gates, not an exact-rounding claim.
Both upstream projects are Apache-2.0; preserve LICENSE/NOTICE and attribution.

## Current source gaps and reuse

| Location | Necessary change / existing reusable part |
| --- | --- |
| `_model.py:90-134` | Ball-only misses the hinge route. Add structural ball dispatch with explicit first-scope rejection before allocation; no failed-lowering fallback. |
| `_tree_model.py:43-46,137-169` | nq==nv, one joint per DOF and one ancestor bit are assumed. Preserve native joint q/dof addresses and widths, all three ancestor columns, and separate native-joint from DOF metadata. |
| `_tree_kernels.py:79-200,270-289,316-329` | Scalar displacement/rate, spring q[d] and joints[d] rotational type are incompatible. Add one anchored SO(3) update; reuse full COM transport and coupled M/J/b/wrench accumulation. |
| `_tree_kernels.py:402,449` | Copy all nq; integrate scalar coordinates by qadr/dadr and every ball quaternion once from its three new velocities. Preserve physical M vs Euler M+hD solve. |
| `_tree_device.py` | q banks accept input shape, physical buffers remain width8. Wire explicit metadata and nq-aware kernel ABI; no copied integrator or CPU step. |
| `_mujoco_tyx_runtime.py:339-355` | Fixed-tree preflight compares all q against nv scalar limits. Select only native scalar q/dof addresses. Existing receipt input_q already uses nq; M/J/b use nv. |
| `mujoco_runtime.py:418-426`; mirror/checkpoint | Initial unit-quaternion checks already handle BALL. Complete integration signature/owned bytes/raw banks are reusable, but require actual continuation qualification. |
| `_floating_model.py:33-50,147-183`; `_floating_kernels.py:496` | One-free-plus-scalars assumes nq=nv+1 and q=dof+1; stays outside BALL1. |
| `_actuator_model.py:84`; `_constraint_model.py:143-146` | Ball actuator targets and ball limits remain explicit rejects, independent of accepted scalar activation/limits. |

Three serial hinges are not a ball. Derive the pre-joint anchor position/velocity/
acceleration, form the post-joint rotation, add the complete relative angular
velocity once, then transport back to origin and COM. The existing full physical
M/J accumulation consumes all three motion columns. No diagonal mass or frozen
bias approximation is justified by this research.

## Predeclared multi-case acceptance

Use unchanged fixed-tree gates from tests/test_mujoco_tyx_tree.py:21-35:
pose/COM rtol2e-5/atol3e-5; J2e-4/5e-5; M2e-4/3e-5;
bias/passive/applied3e-4/2e-4; acceleration5e-4/8e-4;
continuous q atol5e-5, v3e-4 and time2e-6. Freeze actual fixture numbers before
execution; failure is evidence to diagnose, not permission to relax these gates.

| Case | Required evidence |
| --- | --- |
| Centered anisotropic ball rotor | Nonidentity q and three nonzero rates; analytic 3x3 inertia/gyro bias; applied three-axis torque and reversal. |
| Eccentric ball pendulum | Nonidentity body/inertial frames, full COM J and force-plus-moment projection, gravity and true off-anchor motion. |
| Hinge -> ball -> slide and ball -> hinge with sibling | Complete q/dof identity, off-diagonal M and reciprocal acceleration; scalar spring, damping and armature preserved. |
| Two balls on distinct bodies plus two scalar DOFs | nv8/nq10 edge, second quaternion and following scalar indices, 24 continuous steps. |
| Sparse W2/W4 and transactions | Save -> actual advance -> restore -> same-control successor; clone, selected reset, last-world reject retains all banks/mirrors/ticks/controls. |
| Negative domain/state | Nonunit quaternion/nonfinite control, nv overflow, ball motor/limit/spring, free+ball, possible contact and same-body slide+ball rejected before device allocation. |

Both backends use independent fixed CPU3.3.7 models: same-input oracle binds the
actual accepted entry, and continuous native reference is never overwritten from
TYX. Keep full M/J/b/control/receipt/failure evidence; tiny analytic rotors prevent
sole reliance on a CPU oracle. Orientation comparisons respect quaternion sign,
but checkpoint and raw state identity must not canonicalize/delete coordinates.
Block CPU step/forward during TYX step; full snapshot keeps its declared CPU
observer. No G1 or PPO run is necessary to qualify this joint type.

## Ownership, order and stopping point

Proposed author MOD: `src/phymeo/backends/mujoco_tyx/_tree_model.py`,
`_tree_kernels.py`, `_tree_device.py`; NEW tests/test_mujoco_tyx_ball.py,
tests/test_mujoco_tyx_ball_transactions.py and result report. No alternative
integrator. First freeze native-joint vs DOF metadata and calls, then tests-first.

Parent seams: `_model.py` route; `_mujoco_tyx_runtime.py` pack/capabilities;
checkpoint binding only if its existing general widths need changes. Parent also
owns `_normal_device.py` and `_constraint_device.py` tree-call ABI adaptation if
needed; their scalar paths must remain compatible while ball composition stays
rejected. This cross-route impact prevents calling the change a tiny admission
patch. No actuator-kernel, contact-solver, dependency-pin or resource-cap change
is needed for BALL1. Exact signatures and final allowlist precede product work.

Success is the entire connected case/transaction matrix. After one reproducible
failure allow one bounded diagnosis; a second unproductive revision freezes the
minimal failing input and stops that slice. Do not tune fixtures/caps, use CPU
fallback, or repeat unaffected accepted groups.

Later independent qualifications: JOINT three-column torque transmission with
NONE/FILTEREXACT and shared gears; angular-limit row with original R/D/aref and
common solver; free-root/loaded-contact composition. JOINTINPARENT, ball spring
and same-body slide+ball require their own explicit cases. Native existence is
not device qualification. These are successors, not an all-at-once grant.
