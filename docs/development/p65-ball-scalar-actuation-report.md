# BALL-A scalar actuation authoring report

Status: **CODE_READY_TESTS_PENDING / AUTHORING_SLOT_RELEASED**.
2026-11-12. This report records test authoring only. No project import,
collection, numerical probe, test, lint, format, compile, build or install was
performed in this lane. Parent owns all four product seams and the sole serial
execution slot. Existing BALL1 and scalar activation results are not results
for this new composition.

## Contract and file ownership

Authority: parent `4b7c9df7`, final BALL-A section in
[the bounded batch plan](../superpowers/plans/2026-11-12-newton-parity-until-0720.md),
and [the native ball actuator source research](p65-ball-actuation-next.md).

The exact author allowlist is two NEW tests and this NEW report:

- `tests/test_mujoco_tyx_ball_scalar_actuation.py`
- `tests/test_mujoco_tyx_ball_scalar_actuation_transactions.py`
- `docs/development/p65-ball-scalar-actuation-report.md`

No product, old test, kernel, checkpoint schema or integration method changed
in this lane. Parent separately owns `_model.py`, `_tree_model.py`,
`_actuator_model.py`, `_mujoco_tyx_runtime.py`. The former BALL1 negative
`scalar-motor-with-ball` is obsolete under this grant; parent was notified and
owns its removal. The new exact-XML/default-entry positive below replaces that
specific expectation. All genuine ball-target negatives remain negative.

The bounded feature is a fixed-root BALL/HINGE/SLIDE mechanical tree with
native independent qpos and tangent-DOF addresses, nv<=8, nq<=10, scalar JOINT
actors only, nu<=32 and na<=32. NONE has no activation state; FILTEREXACT owns
one native state per actor. Ball coordinates move under the existing full
coupled M/J/b and quaternion integration. No actuator applies a torque directly
to a BALL target in this increment.

Expected capability labels are exactly
`fixed-ball-scalar-actuation-no-contact-v1` and
`fixed-ball-scalar-filterexact-no-contact-v1`. Stateless paths retain three
actual stages, original receipt and v1 checkpoint; stateful paths retain four
stages, `native-scalar-filterexact-actuation-v1` evidence and v2 checkpoint.

## Frozen inputs and coverage

Original BALL1 `HINGE_BALL_SLIDE` and `MAXIMUM` geometry declarations are reused
without changing masses, eccentric inertia/COM, axes, anchors, gravity,
damping, armature or scalar springs. The old helper gives nonidentity unit
quaternions and nonzero velocities in every tangent/scalar coordinate. New
activation starts at 0.34+0.01*world_slot and time0.125; source initial activation
may be outside an enabled actrange, as in the accepted scalar activation tests.
Native behavior, including next-state clipping, remains the oracle.

| New node family | Fixed input and meaningful assertions | Manually enumerated nodes |
| --- | --- | ---: |
| Complete24steps | Stateless, filter-late, filter-early, disabled, capacity; each runs all24 steps on each backend | 10 |
| Exact retired input | Original HINGE_BALL_SLIDE plus `<motor joint="hinge"/>`, original default initial state/control, two real steps and full native comparison | 2 |
| Admission negatives |13 declared HOST-only preallocation checks; never allocate a device |13 |
| Lifecycle | Stateless sparse W2=(7,103); stateful sparse W4=(3,19,91,203), each backend |4 |
| Late failures | Two stateless and four stateful faults, each backend |12 |
| No CPU dynamics | Stateless and stateful actual step/integration observation, each backend |4 |
| Total | Manual decorator enumeration, **not collection or a passing result** |45 |

The main24-step path loads at ticks0..7 (`ctrl=.8`), reverses at8..15
(`ctrl=-.75`), and unloads all requested controls/qfrc/COM external wrenches at
16..23. All these values were authored before any execution. There is no
warmup/reset or reseeding in these paths. The enabled stateful dynamics retain
and decay activation after unload; zero control is not a reset operation.

Four ordinary actors include direct NONE gear-0.5, slide gear-2 with affine
bias and ctrl/act/force clamps, hinge gear1.5, and shared-hinge gear-0.25.
FILTEREXACT tau values are0.004/0.002/0.008. The disabled fixture preserves the
native positive force interval[0.1,0.25]: raw force is zero while final clamped
force is0.1; raw act_dot remains nonzero and integration derivative is exactly
zero. The capacity fixture uses two balls and both scalar joints with
(nq,nv,nu,na)=(10,8,32,32), 32 declared FILTEREXACT actors, alternating hinge/slide
targets and signed0.125 gears, gain0.02, tau0.004. It is an address/capacity
qualification, not a performance claim.

Negative inputs: NONE ball target, FILTEREXACT ball target, FREE target in a
free+ball tree, enabled contact, scalar soft limit, scalar dry friction, ball
spring, ball limit, same-body mixed joint, JOINTINPARENT, FILTER, INTEGRATOR and
33 actors/activations. FREE+BALL necessarily also exceeds this nv8 domain; that
case proves the whole unsupported combination is rejected, not that one
particular validation code is the sole reason. The device constructor is
replaced by a failing sentinel only for these preallocation checks.

## Independent oracles and original gates

Each new main step reconstructs a separate native MuJoCo3.3.7 state from the
actual accepted q/v/act/time, then copies the original FP32 control and external
force channels. Native forward supplies all body poses, COM Jacobians, full
physical mass matrix, bias, passive terms, actuator scalar/generalized force,
and raw act_dot. Independent applied force is assembled as
`qfrc_applied + qfrc_actuator + sum(J_com.T @ xfrc_applied)`.
The oracle separately solves physical M for smooth acceleration and M+hD for
Euler integration acceleration. No diagonal approximation or artificial ball
hinges are used. An independently evolving native state is initialized once
from the same accepted initial state and never follows/reseeds from later
TYX outputs.

The existing test-side scalar analytic helper independently interprets native
qpos/dof/activation addresses, ctrl clamp, exact exponential filter update,
act clamp, actearly, gain/affine bias, disabled groups, force clamp and signed
gear accumulation. It does not call product lowering or product kernels. New
checks bind the published ids to the native layout and verify all BALL tangent
columns of actuator generalized force are zero. Ball motion remains coupled
through complete mechanical dynamics. Entire receipt/control/state arrays are
saved, not just norms.

| Channel | rtol | atol |
| --- | ---: | ---: |
| Body position/rotation/COM |2e-5|3e-5|
| Full COM Jacobian |2e-4|5e-5|
| Physical mass matrix |2e-4|3e-5|
| Bias/passive/applied |3e-4|2e-4|
| Smooth and integration acceleration |5e-4|8e-4|
| Activation/raw derivative |3e-5|3e-6|
| Scalar/generalized actuator force |3e-4|2e-4|
| Continuous q |0|5e-5|
| Continuous v |0|3e-4|
| Time |absolute|2e-6|

The BALL quaternion squared-norm gate remains8*eps32. Source ids and held
control bindings are exact. Full observer q/v/act/ctrl/time/tick/last_step agree
with integration observations at predeclared ticks0,7,8,15,16,23; full observer
CPU forward is outside actual TYX-step guards. Dedicated guards forbid native
step/forward/fullM/J/RNE, NumPy solve/inverse/Cholesky and Warp imports only
inside actual TYX step plus integration observation.

Lifecycle tests compare complete encoded integration publications, including
original receipts and held controls. Checkpoint bytes roundtrip, actual advance,
CAS restore, same-control successor and clone successor are required. Partial
original reset is followed by a real step: selected worlds compare with a fresh
runtime at their original input, unselected worlds compare with the uninterrupted
clone. Late readback/status/input-act and late restore-readback injections retain
the entire accepted publication and exact checkpoint bytes, followed by a real
retry and native/clone successor. Tests deliberately use internal fault seams;
they do not implement a second runtime or expose new product fault APIs.

## Evidence and handoff

The tests reuse only old independent test declaration/oracle/serialization
helpers from `test_mujoco_tyx_ball.py` and `test_mujoco_tyx_activation.py`.
Each runtime closes in finally. Full gzip histories include source XML, actual
initial state, original entry/control, native fields and continuous outputs,
scalar expectations, complete typed receipt/output, counters and any failure.
Lifecycle cases additionally retain exact checkpoint bytes in a binary file and
hex within the complete saved history. No evidence from previous acceptance is
rewritten or reclassified.

Tests-first commit: `8a0f6758cc674d12d78b1c5a465c5f6837ddd147`.
Working-byte SHA256 at author handoff:

- main: `2ba8fb24d42774ebb215088694c9f1cd3ee518f2d4d501d8d19ae966c9a40fcb`
- transactions: `1bd94cf8b496e73d4d9a2144333f67d27df7dc25ed8b20ea316122fe234c7edd`

Only textual source review, explicit-path Git staging and `git diff --check`
were performed. Parent may first apply formatting and record qualified source
parity; this report does not claim current formatting or tests pass.

Pending focused parent selections, after integration and unique execution guard:

```text
python -B -m pytest tests/test_mujoco_tyx_ball_scalar_actuation.py tests/test_mujoco_tyx_ball_scalar_actuation_transactions.py -m "not cuda" -q -x --tb=short -p no:cacheprovider
python -B -m pytest tests/test_mujoco_tyx_ball_scalar_actuation.py tests/test_mujoco_tyx_ball_scalar_actuation_transactions.py -m cuda -q -x --tb=short -p no:cacheprovider
```

Expected selection from authored decorators is29 interpret/HOST plus16 CUDA,
subject to actual collection. Parent supplies existing pinned environment,
thread settings, fresh basetemp and complete receipts. Parent separately chooses
affected old compatibility nodes without rerunning unrelated accepted groups.
No extra source-external wheel, G1 run, learner run, ball actuator transmission,
ball constraints, free-root composition or performance claim is included here.
