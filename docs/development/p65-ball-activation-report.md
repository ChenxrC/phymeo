# R1 direct BALL activation composition author report

Status: CODE_READY_TESTS_PENDING / AUTHORING_RELEASE. This lane performed no
project import, collection, test, numeric probe, lint, formatter, compile,
build/install, profile or CI. Parent alone owns every such execution.

## Delivery

Baseline 79e3e0892f364c3681d642f14df23f946c7617e8,
branch codex/p65-ball-activation-20261112.
Tests-first c639839e; five-product implementation 661fddce; independent consumer
and legacy input qualification 6466d191. Final report commit follows.

Five MOD, no other product changes:
- src/phymeo/backends/mujoco_tyx/_tree_model.py
- src/phymeo/backends/mujoco_tyx/_tree_device.py
- src/phymeo/backends/mujoco_tyx/_actuator_model.py
- src/phymeo/backends/mujoco_tyx/_actuator_device.py
- src/phymeo/backends/mujoco_tyx/_actuator_kernels.py

Five NEW: the two tests below, tests/installed/check_ball_activation.py,
p65-ball-activation-design.md and this report in docs/development/.
Parent's model/runtime/public records/checkpoint/export changes are separate;
they were not copied into or overwritten by this author branch.

The only untracked old file is docs/development/mujoco-tyx-t1-review.md,
preserved untouched. Source writes and Git diff checks are not physics results.
One staged diff check found a consumer EOF blank line before 6466d191; the final
author suffix removes it. There was no project execution failure in this lane.

## Connected behavior

The actual original activation device invocation receives five appended fields:
flag, CSR offsets, DOF columns, moments, actual transmission output.
Every native actor writes both actuation and transmission fields from the same
local ctrl/raw-force/final-force values. All CSR columns are accumulated once in
actor order, including zero moments and shared target rows. The original physical
M/J, damping solve, q/v integrator, contact law and double-bank commit remain.

Flag zero aliases existing storage and preserves scalar arithmetic. A BALL row
has three columns and uses gain times activation, with BIAS_NONE required before
allocation. A scalar row retains its complete original affine length/velocity
bias expression. No BALL quaternion component is used as a length. For example,
gear (1,-2,.5) maps scalar force f to (f,-2f,.5f) in the three native local angular
DOFs; power is f times that same moment row dotted with incoming qvel.

The original filter uses clamped control, native raw derivative, disabled
integration-zero, tau_eff, exact-filter update, activation clamp, actearly,
force generation and force clamp in the original order. Initial act is nonzero
and deliberately outside actrange in the first accepted step. Disabled raw force
zero with a positive lower force bound is retained as a nonzero final force.

New layout schema native-joint-filterexact-v1 only labels plans with complete
joint transmission; old scalar layout/schema remain unchanged. Parent implements
MuJoCoTyxJointActivationStepReceipt with native-joint-filterexact-actuation-v1,
same-stage four-field byte equality, original ctrl binding, checkpoint v2 typed
restore, and precise three domain labels in the design.

## Authored matrix and exact pending commands

Manual decorator enumeration, NOT collection:
- tests/test_mujoco_tyx_ball_activation.py: 34 nodes:
  8 fixed continuous cases times 2 modes, 12 preallocation negatives,
  3 exact legacy input cases times 2 modes.
- tests/test_mujoco_tyx_ball_activation_transactions.py: 36 nodes:
  3 lifecycle paths times 2 modes; 2 contact routes times 5 late faults times
  2 modes; 2 foreign-checkpoint cases; 2 routes times 2 independently tampered
  records times 2 modes.
- Total 70; 41 non-CUDA and 29 CUDA marked selections.

Parent pending commands, only under its unique serial slot and fresh guard:
`python -B -m pytest tests/test_mujoco_tyx_ball_activation.py tests/test_mujoco_tyx_ball_activation_transactions.py -m "not cuda" -q -x --tb=short -p no:cacheprovider`
then the identical file list with `-m cuda`.
These commands have NOT been run by the author; parent's actual environment,
absolute fresh basetemp and complete source before/after receipts remain required.

Each continuous case has two sparse independent worlds (7,41), h=.001,
24 actual steps, nonzero v and act, forward/reverse/unload controls and external
loads. The eight cases, exact source clamp/filter/gear values and original
numerical gates are in the design and self-contained consumer. Physical source,
mass, geometry, integrator, contact policy and thresholds are not fitted to results.

Native same-entry full M/J/bias/passive/applied/forces, normal or full pyramidal
contact/EFC fields, and independent continuous CPU trajectories remain distinct.
Independent expm1 filter and complete JOINT CSR/power formulas do not call the
product producer. Force+act+q/v/time are compared on the same actual step.
CPU mj_forward/step guards cover only actual TYX physics, not its independent
oracle or explicit full endpoint observation. No claim of zero HOST reads.

Lifecycle saves accepted checkpoint bytes, runs a real successor, restores and
repeats it; clone and selected original reset have real successors. Late status,
readback, last-world actuation/transmission publication and verified restore
failures require unchanged accepted bytes/revision and a real uninterrupted-clone
retry comparison. Each single-record checkpoint tamper changes only one scalar
force in actuation OR transmission, recomputes the outer checksum, and must be
rejected without changing the next physical result.

Retire only these obsolete negative parameters in parent:
BALL-M1 negative_source('activation'), BALL-MC negative_source('activation') and
BALL-MC negative_source('scalar-activation'). Their exact original XML/default
entry has new two-step native-positive coverage. The old ROB2a feature='ball'
and BALL-A 'ball-target-filter' still contain affine BALL bias and stay rejected.

Suggested affected compatibility: one original scalar FILTEREXACT case per
tree/free-root/normal/pyramidal route exercises appended flag-zero ABI; representative
stateless BALL no-contact and contact cases establish unchanged launch/force path.
Do not rerun all unrelated already accepted suites or use this suggestion as a
claim that compatibility has executed.

## Installed protocol and limits

Copy tests/installed/check_ball_activation.py outside the checkout, then run:
`python check_ball_activation.py --mode interpret --installed-root ABS_INSTALL --output FRESH_ABSOLUTE_OUTPUT`
and separately cuda. Each mode declares three independent 24-step windows
(no-contact, normal, pyramidal) plus eight actual lifecycle calls: 80 batch calls,
160 world steps. Provenance rejects checkout discovery and records every loaded
phymeo module's installed path/SHA. No imports of tests or examples.

The JSONL gzip retains fixed sources/initial states, every native comparison,
actual attempt/return/state/receipt, checkpoints, reset and failure prefix.
An actual physics-return event precedes endpoint observation. Primary and cleanup
exceptions are preserved by continuous/lifecycle and top-level finalization.
No package build, installed execution or saved evidence audit has occurred here.

Original fixed BALL capacities and FP32 gates remain: nv8/nq10, actor/act32,
16 pairs/32 slots, condim1/pyramidal3. This matrix does not independently qualify
every maximal activation/CSR combination. No FREE+BALL, direct BALL affine bias,
limits/dry, tendon, continuum/soft-body, G1, training, broad MuJoCo parity or speed
claim follows. A completed author handoff is not acceptance.


## Parent source qualification — 2026-11-12

Integrated source passed run 021 (41 non-CUDA nodes, including the 12 one-time preallocation negatives; 29 CUDA nodes deselected) and run 022 (29 CUDA nodes; 41 other nodes deselected), both RAW_EXIT=0. The 70 new nodes include the exact three previously rejected activation XML inputs as positive real-successor controls. Run 023 additionally passed the affected stateless direct BALL and scalar FILTEREXACT source cases on both backends (four robot compatibility nodes; the other two nodes in that run belong to scene compatibility).

The parent implemented composed activation/transmission receipts and checkpoint binding, preserving both reports from one actual actuator publication. Independent text review tightened the consumer EFC check to be selected by the pyramidal solver schema rather than nonempty reported rows, including a correctly shaped empty wrench array. This was fixed before the first functional run; no physical gate or fixture changed. Initial import/format findings and all original sources remain in `artifacts/p65-three-track-a-20261112-01`. Source-external qualification remains pending until the unified Batch A wheel is tested.
