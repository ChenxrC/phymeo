# BALL-MC: direct BALL JOINT motor and primitive contact composition

Status: **CODE_READY_TESTS_PENDING**. Author-only delivery; no import, collection,
numerical probe, pytest, lint, formatter, build or installation has run in this lane.
Parent owns all project execution and the shared device/runtime changes.

## Scope and implementation boundary

Base: b66504f3c82c868fddf3830bb0f9b2260bd3fd7b.
Plan: [2026-11-12 parity motor/contact/asset](../superpowers/plans/2026-11-12-parity-motor-contact-asset.md).
This composes the accepted BALL-M1 stateless native JOINT moment mapping with
the accepted BALL-C fixed-root primitive condim1 / pyramidal condim3 contact laws.
It is not a new integrator, friction law, solver, force reconstruction or CPU step.

Author owns exactly five files:

- MOD `src/phymeo/backends/mujoco_tyx/_tree_model.py`.
- NEW `tests/test_mujoco_tyx_ball_motor_contacts.py`.
- NEW `tests/test_mujoco_tyx_ball_motor_contacts_transactions.py`.
- NEW `tests/installed/check_ball_motor_contacts.py`.
- NEW this report.

Tests-first checkpoint 9f321387 contains only the two new test modules.
The product checkpoint dd2e5e80 changes one guard: direct BALL transmission may
have declared contact pairs only when the explicit `allow_normal_contacts`
route is selected. All earlier structural checks remain. Admission alone is not
completion: the parent's wrapper wiring is a necessary part of the same delivery.

The permitted domain remains fixed BALL/HINGE/SLIDE, nv<=8, nq<=10, nu<=32,
na=0, original 128-body/128-geometry limits, at most16 possible pairs and32 static
candidate slots. Native JOINT BALL targets use all three finite stored moments,
including zero columns, FIXED gain and BIAS_NONE. Scalar JOINT actors preserve
the existing stateless affine-force support. Every actor must be NONE when any
BALL target exists. FREE composition, activation, joint limits/dry friction,
quaternion springs, same-body BALL+another joint, equality/tendon, explicit
pairs/excludes, elliptic cones and condim4/6 remain outside this composition.
Existing iteration/resource caps, FP32 representation gates and scalar-only paths
are unchanged. A rejected node must fail before device construction.

## Exact shared ABI required from the parent

Both contact devices already call `TyxTreeDevice.__init__`; the plan wrappers'
`__getattr__` forwards the complete base, including `joint_transmission`.
Thus they already allocate the original CSR offsets/columns/moments and
`transmission_fields[world, actor, 3]`; their base download already publishes
`joint_transmission_fields` when enabled. No duplicate table, kernel or transfer
allocation is needed.

The parent's two fixed-tree mechanics calls must use the same five arguments as
the accepted tree device, immediately after `act_values` and before `controls`:

```text
int(transmission_enabled), transmission_offsets, transmission_dofs,
transmission_moment, transmission_fields
```

The floating kind3 call must still supply no such arguments. Existing disabled
fixed-tree calls use the original alias buffers and flag0. No new launch is added:
the declared fixtures use five normal or seven mixed-constraint stages per batch.
Extended primitive manifolds retain their existing extra stage outside these
particular fixtures.

The runtime already publishes contacts/constraints and then selects
`MuJoCoTyxJointMotorStepReceipt`; this must remain one atomic force/motion
publication. The existing versioned checkpoint already allows and source-binds
the complete typed transmission, original ctrl and accepted actuator-force mirror.
New capability labels are
`fixed-ball-joint-motor-normal-contact-v1` and
`fixed-ball-joint-motor-pyramidal-contact-v1`, with no activation/limit/dry promise.
The author does not edit runtime, checkpoint, device wrappers, old kernels or
their defaults.

## Fixed multi-case matrix

All six continuous cases use h=.001, W2=(7,41),24 actual steps, no reset/warmup,
nonzero original incoming velocities, and two independently evolving CPU references.
Commands are +.8 for steps0..7, -.75 for8..15 and zero for16..23. World41 adds.015
to nonzero commands. Generalized external inputs are signed .03*(dof+1);
the last-body COM wrench is (.3,-.2,.4,.01,-.03,.02)*(1+.2*world_slot).
Both external channels become zero at16. Zero command does not assert zero
passive bias or disabled-positive-clamped motor force.

| Case | Frozen geometry / mechanics | New composed behavior |
| --- | --- | --- |
| MC1 normal | Original C1 eccentric anisotropic m1.3 ball, sphere radius.1 at(.08,.03,-.18), root z=.13577777777777778, WXYZ=(2/3,1/3,2/3,0), plane condim1 | Direct gear(1,-2,.5), all three normal J columns, initial real load and nonzero COM moment |
| MC2 friction | Same C1 geometry and initial q/v, both geoms condim3 | Same complete motor with normal/tangent native EFC force and motion |
| MC3 mixed | Original C3 H/B/S full inertia/COM, scalar ref.07, gap-.001, slide speed-.02, finite box support, condim3 | Ball gear(1,-2,.5), hinge-.5, slide-2 with gain1.2 and affine bias(.01,-.3,-.04), original ctrl/force clamps |
| MC4 shared | MC2 sphere-plane | Two nonzero shared BALL actors: gear(1,-2,.5)/gain1 and(-.25,.5,-.125)/gain.6, plus exact zero three-column gear/gain.8 |
| MC5 two moving branches | Original C4 nq10/nv8, two sibling ball+slide branches, masses1/1.7 and distinct inertia/COM, spheres radius.1 | Two direct BALL gears(1,-2,.5) and(-.3,.2,.4), plus slide gears-.5/.5; both branches move and both COM moments are nonzero |
| MC6 disabled | MC3 identical geometry/state and scalar actors | BALL actor group1 disabled, raw force0 then positive clamp[.1,.25]; scalar channels still act |

Mass, damping, reflected inertia, source solver options, collision parameters and
entry geometry are authored before execution, not selected from runtime outcomes.
The original C1–C4 source definitions are reproduced explicitly in the standalone
consumer; it imports no source tests/examples or private bank.

The exact former M1 `negative_source("contact")` XML is separately tested from
the original default entry for two steps. It stays separated and therefore
requires zero contact force instead of fabricating a load. Parent should retire
only `NEGATIVES["contact"]` from the old M1 constructor-negative parametrization;
keep its source helper for this exact-input positive test. The old BALL-C
`ball-motor` source also has FILTEREXACT actors and therefore remains unsupported.

## Independent oracles and unchanged gates

Each step retains the full same-input native M/J/bias/passive/applied, all
original contacts, full normal or pyramid rows, R/aref/forces, source/body identities,
and the independently evolving native trajectory. Native `mj_step` / forward
are called only outside the guard around the actual TYX step. Full and integration
observations bind identical accepted q/v/act/ctrl/time/tick/last_step.

The transmission oracle independently reconstructs native joint→qpos/DOF addresses,
CSR1/3-column structure, scalar force/clamp/disable law and generalized force;
it also compares the dense native moment matrix and actuator velocity. It never
treats a quaternion component as actuator length. Full motor power equals both
generalized force dot native velocity and the sum of scalar actor power.

Original BALL-C gates are retained: contact/generalized constraint force
rtol/atol3e-3; acceleration rtol3e-3,atol1e-2; continuous q absolute2e-4,
v absolute3e-3, clock2e-6. Original motor force remains rtol3e-4,atol2e-4.
Complete M/J/geometry/bias tolerances are the unchanged field-specific TOL table.
No nonlinear trajectory/force gate is widened. Native source iterations128 and
tolerance1e-10 come from original C1–C4; the exact retired XML retains its own
native defaults. Published KKT must satisfy the original
max(source_tolerance,2e-5) gate and original budget.

The published transmission force is FP32, while the legacy receipt is FP64.
The consumer checks both declared dtypes then compares exact FP64 promotion;
it does not repeat the previous installed typed-byte mismatch or use a new
force tolerance. Every actual step checks the original5/7 kernel count and
the real completed/accepted-world counters.

All raw contact/constraint/transmission/state evidence is kept in typed gzip
records before numerical assertions. Same-input and continuous native paths
remain distinct; neither overwrites the actual accepted device state.
Failures preserve the original exception and capture errors separately.

## Lifecycle, failure and expected node identities

Each normal/pyramidal lifecycle uses an actual checkpoint after three steps,
portable bytes roundtrip, a real A successor, independently cloned successor,
restore then B successor, stale CAS refusal, partial original reset with exact
unselected world/raw-clock/mirror/receipt retention, and two real reset successors.
Each is eight actual batch calls including one clone call.

Each solver path has four late-failure cases: final-world status, complete
download refusal, final-world complete transmission publication refusal, and
real prepared/verified restore refusal. Every failure preserves whole accepted
state, checkpoint bytes and revision. Retry must equal an uninterrupted clone.
Another case refuses a checkpoint from the other declared model.

Hand enumeration, not collection or executed results:

- Six continuous cases ×2 backends:12.
- Fifteen constructor negatives:15 HOST nodes.
- One exact former negative two-step positive ×2:2.
- Two lifecycle cases ×2:4.
- Two solver paths ×four faults ×2:16.
- Foreign checkpoint ×2:2.

Total51, expected33 in a complete non-CUDA selection and18 CUDA.
Native dependency skips cannot count as qualification.
Suggested parent command is the two complete new test files, with its existing
serial runner selecting interpret/HOST and then CUDA separately.
No already-green full suite is requested.

Bounded changed-seam compatibility suggestions (each backend, not whole old files):
original `test_ball_contact_continuous[c1-loaded]` exercises the flag0 fixed-tree
normal caller, `test_ball_contact_continuous[c3-early]` exercises the unchanged
scalar FILTEREXACT pyramid caller, and
`test_joint_motor_complete_24_steps[oblique]` checks the no-contact direct BALL
route. Parent may add a kind3-only smoke if its actual shared diff warrants it;
this author has neither requested nor run a floating/full old-suite rerun.

Installed CLI:
```text
check_ball_motor_contacts.py --mode interpret|cuda --installed-root ABS --output freshABS
```
It independently runs MC1 and MC3 continuous windows plus MC3 W2 lifecycle:
56 actual batch calls across the runtime instances,112 actual worldsteps per
backend if successful. All loaded Phymeo modules must belong to the supplied
installation, with no checkout path or source helper import. This qualifies the
two installed compositions; MC2/4/5/6 and fault injection remain source-test scope.
The final gzip records inputs, outputs, native oracles, all public receipts,
portable checkpoint bytes, counters and raw failures; no all-MuJoCo claim.

Stop rather than tune if a declared native/physical comparison fails. Parent
preserves the first failure and identifies implementation, oracle, API or saving
layer before any revision. Two unproductive diagnosis rounds trigger the plan's
bounded stopping rule. This report and code do not assert a test or physical pass.
