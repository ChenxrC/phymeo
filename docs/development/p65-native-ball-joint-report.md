# BALL1 native fixed-root ball joints

Status: CODE_READY_TESTS_PENDING / AUTHORING_SLOT_RELEASED. This report records
pure source authoring, not a tested capability or a new execution result.

## Scope and source

This increment lowers native fixed-root BALL plus HINGE/SLIDE trees, preserving
native quaternion qpos and three tangent qvel coordinates. It admits nv<=8,
nq<=10, at most 128 native bodies and the original runtime world capacity.
BALL1 has nu=na=0 and no FREE joint, possible contact pair, limit on any joint,
dry friction, ball spring, equality, tendon, plugin, mocap or userdata. A body
with a ball has no other joint. Scalar springs, nonnegative damping and armature,
eccentric COM/inertia, welded bodies and separate-body chains/branches remain.
Every body and coupled matrix entry is retained. Generalized torque and COM
world wrench are the existing public control channels. This is not yet ball
motor transmission, ball limits, floating-ball, or ball-contact support.

Primary fixed sources and the deferred transmission/limit work are recorded in
[p65-native-ball-joint-next.md](p65-native-ball-joint-next.md). The implementation
keeps the existing Apache-2.0 notices. Native CPU 3.3.7 supplies independent test
models; no CPU solver is added to the TYX step.

## Mapping and kernel contract

`joint_ids[njnt,4]` remains indexed by native joint: `(kind,qadr,dadr,body)`,
with slide=0, hinge=1 and ball=2. A new keyword-only optional `dof_ids` field
preserves the independent `TyxFloatingModel` dataclass constructor. The fixed-tree
compiler always supplies its `(8,3)` table: `(rotational,scalar_qadr_or_-1,joint)`.
Three ball columns have rotational=1 and no scalar qadr. Every descendant's
ancestor mask contains all three columns. Body ranges still address native
joints, never pseudo hinges. Static vectors have `(9,max(8,nq))` extent; row 4
contains the entire native qpos0, including q[8:10]. Physical matrix/vector
scratch remains width 8; old scalar vector extent remains `(9,8)`.

Geometry ABI is unchanged. Mechanics changes only the sixth argument from the
joint table to the DOF table. Fixed-tree solve is now:

```
count, nv, nq, njnt, h, joints, vectors, mass, bias, passive, applied,
q, v, clock, qnext, vnext, tnext, factor, rhs, forward, acceleration,
integration_acc, constraint, rows, status
```

The parent-owned normal/constraint wrappers use this signature only for kind 2;
kind 3 keeps its floating ABI. Parent commit 880eedbf was imported as b8bd0da4:
early structural BALL dispatch, scalar qadr limit preflight and explicit BALL
capability naming. No fallback dispatch or contact/actuation admission is added.
The parent also owns the old ball-with-motor negative's diagnostic-code change:
it stays rejected, with `ball-actuator-domain` instead of `joint-domain`.

## Mechanics and publication

For a ball, form the pre-joint anchor position/velocity/bias acceleration from
the parent/body fixed transform. Set Rpost=Rpre*R(q), with a normalized scratch
copy. With u=Rpost*v_ball, omega_post=omega_pre+u and
alpha_post=alpha_pre+omega_pre cross u. The three Jacobian angular columns are
Rpost's columns at the same anchor. Move from anchor to the rotated body origin,
then reuse the original COM transport, full mass and bias assembly. This keeps
all parent/child and sibling cross terms; no three-hinge approximation is used.

The original two Cholesky solves are retained: physical M for reported smooth
acceleration, and M+h*diag(damping) for Euler integration acceleration. Integrate
scalar q by native qadr/dadr; integrate each ball once using its three accepted
new local rates and right-multiply the exponential quaternion. Normalize the
new quaternion on device as Warp 0.0.2 does, publish its real FP32 values, and
retain original input values in the receipt. The CPU oracle uses its own native
Euler integrator. No state repair, CPU dynamics, additional stage or budget is
introduced. Input FK/M/J/b and output motion/force continue to share the original
whole-batch publication boundary; a bad final world retains every world.

The existing public initial-state gate remains 64*eps64 from unit length. New
BALL scratch checks finite/nonzero norm and `abs(sum(q_i*q_i)-1)<=8*eps32`.
Let u=eps32/2. Rounding a unit FP64 vector to four FP32 components changes its
squared norm by at most 2u+u^2 (plus the existing eps64 admission allowance).
Four rounded squares and their sum contribute at most gamma4 times that norm,
so the total is below 7u and strictly inside 8*eps32. This is a packing-domain
guard, not an error allowance for physics. Zero/NaN/infinite quaternions still
fail; scratch normalization cannot turn those failures into acceptance. The
old free-root T2 quaternion gate is untouched.

## Authored cases and unchanged gates

Both new files were committed before implementation (`d131b529`, then explicit
address/negative additions `14155c38`). Decorator arithmetic declares 40 nodes:
26 in the dynamics file (12 backend parametrized +14 HOST-only) and 14 in the
transaction file (7 per backend). Thus the intended selections are 27
`not cuda` nodes and 13 `cuda` nodes. These are authored counts, not collection
or execution results.

| Case | Fixed input and independent check |
|---|---|
| Centered anisotropic rotor | Nonidentity quaternion, three nonzero rates; independent diagonal I+armature, v cross Iv, damping and three-axis torque |
| Eccentric ball | Nonidentity body/inertial frames, off-origin anchor, full COM Jacobian, gravity and force/moment projection |
| Hinge-ball-slide | Real native qadr/dadr divergence, scalar spring, complete M/J/b |
| Ball-hinge plus sibling slide | Shared-ancestor coupling with nonzero velocities and loads |
| Two balls plus two scalars | nv8/nq10, quaternion and scalar beyond old q width, full qpos0 and ancestor table binding |
| W1/W2/W4 | Sparse IDs; accepted checkpoint bytes, actual advance, restore, identical real successor, clone, partial reset and fresh selected-world successor |
| Late failure | Last-world returned status and post-readback exception after all three real launches; all-world retention and retry against independent clone |
| Domain/state negatives | Ball spring/limit/motor; any scalar limit/motor with ball; dry/contact/free/capacity/same-body slide-ball; zero/nonunit/nonfinite quaternion; finite wrong-width last control packet |
| No fallback | During step and integration observation forbid CPU forward/step/M/J, NumPy linear solvers and Warp modules; assert exactly three actual launches |

Each continuous case runs 24 steps, with nonzero applied control, reversal at
step 8 and reduced world wrench after step 16. Same-input CPU dynamics is rebuilt
independently at each actual entry. The second CPU state starts once from the
initial stored state and evolves continuously, never overwritten by device
outputs. Full per-step inputs, receipt, independent fields and trajectory are
saved; exception context retains the last entry and original error.

The original T1 `(rtol,atol)` gates remain: body position/rotation/COM
(2e-5,3e-5), J (2e-4,5e-5), M (2e-4,3e-5), bias/passive/applied
(3e-4,2e-4), smooth/integration acceleration (5e-4,8e-4). Continuous q has absolute
5e-5, v 3e-4, time 2e-6. No force/trajectory gate has been loosened.

## Pending parent validation and boundaries

Only text review, file IO and Git diff checks ran in the author lane. No import,
collection, numerical probe, pytest, lint, formatting, compile, build, install,
profile or native execution occurred. Parent must review and run original first
static checks, then the complete two new files under its single execution slot:

```
python -B -m pytest tests/test_mujoco_tyx_ball.py tests/test_mujoco_tyx_ball_transactions.py -m "not cuda" -q -x --tb=short -p no:cacheprovider
python -B -m pytest tests/test_mujoco_tyx_ball.py tests/test_mujoco_tyx_ball_transactions.py -m cuda -q -x --tb=short -p no:cacheprovider
```

Each actual run needs its own fresh absolute basetemp and source/command receipts.
Affected scalar fixed-tree, activation and normal/constraint call sites require
focused regression after the final shared ABI is integrated; old floating code
was not changed. New installed qualification is still pending. No G1, PPO,
contact-ball or general MuJoCo capability follows from this source handoff.

## Parent execution and integration (2026-11-12)

BALL1 source284e3c51 and parent shared routing completed run141 HOST/interpret27passed/13deselected,14.34s and run142 CUDA13passed/27deselected,6.32s, both raw0. Original gates and five24-step cases were unchanged. The source scope is fixed-root BALL+hinge/slide, nv<=8/nq<=10, no actuators/contact/limits/dry friction or ball springs; it is not floating BALL or G1 qualification.

run144 old compatibility stopped after26pass at an obsolete pre-T4 hard-limit-refusal assertion. Source review confirmed T4 already handles soft limit rows. The same original two-world XML/q/v/h now checks two native-matched steps: first crossing has no row at entry, second has the native inward limit row, without clipping. The old sibling-box rejection likewise became a positive full-tree case using the identical separated source. Other negative admission gates remain. run145 validates the46 not-yet-run selected compatibility nodes, and run166 validates the7 changed domain nodes; all raw0. Parent normalmerge a6cffc07 and merged reset/clone seam run16914passed/20.85s complete source integration. No prior green group was repeated merely to collect another pass.

Canonical receipts are artifacts/p65-parity-until-0720-20261112-01. ball-reset-triangle-source-qualification.json verifies all selected before/after and saved source bytes with zero mismatches and explicitly lists later file differences. Package consumer check_ball_tree.py is authored and statically checked; installed execution remains pending until the final new wheel.
