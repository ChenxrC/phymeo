# FREE+BALL FILTEREXACT: next bounded composition

Status: SOURCE_RESEARCH_AND_TEST_DESIGN_ONLY, 2026-11-13. Parent source inspected at
3e8d0df73720b34777d3529034f179c48c05c1ce. No product/test edits, project imports,
collection, numerical probes, tests, static tools or builds were performed.
This document proposes a contract; it does not grant implementation or claim PASS.

## Decision

A useful bounded next slice is feasible: one FREE root with native BALL/R/P
children, mixed NONE/FILTEREXACT direct JOINT actors, with no contact, primitive
condim1, or pyramidal condim3. Existing devices already own activation banks and
invoke the complete CSR actuation before the original floating/contact solve.
No new integrator, mechanical kernel signature, activation kernel signature or
contact law is needed according to this source inspection. There are five real
admission/publication gaps, including a combined receipt that would reject an
otherwise correctly executed free-root stateful step. Simply deleting a guard
would therefore be incomplete.

## Existing support and exact gaps

| Layer | Existing source fact | Required bounded change |
|---|---|---|
| Route | `_model.py` BALL+FREE branch runs floating/normal/constraint before the later `model.na` branch. | Route stateful FREE+BALL to the common activation compiler, with explicit structural validation and no fallback. |
| Structure | `_floating_model.py:54` rejects any BALL tree with `na` or non-NONE dynamics even when `allow_activation=True`. | Permit only explicit `allow_activation` plus existing NONE/FILTEREXACT actor validation; keep all independent structure/contact/limit/dry gates. |
| Activation compiler | `_actuator_model.py:135` sends every BALL model to fixed `compile_ball_model`. | Distinguish FREE+BALL first; run full floating structure with normal permission, then choose no-pair/normal/pyramidal base using the same complete declared pair set. |
| Combined publication | `mujoco_joint_transmission.py:86` accepts only the fixed transmission schema, whose capacity is nv8. The free transmission already has its own nv35 schema. | Admit a specific free activation/transmission schema pair, keeping fixed schema and cap strict. |
| Capability | Runtime's later FREE+BALL and contact blocks overwrite the earlier stateful label with a stateless label. | Publish distinct free stateful no-contact/normal/pyramidal labels and preserve limits/dry=false. |

Current accepted subsets are fixed BALL FILTEREXACT (including original contact),
FREE+R/P scalar FILTEREXACT, and FREE+BALL NONE with original normal/pyramidal
contact. Their union is not evidence that this new combination has already run.

No dimension-specific missing stage was found in these actual interfaces:

- `_floating_device.py` validates `(W,na)` FP32 input, owns the two act banks,
  full native q/nv fields, CSR, and invokes mechanics with `nu=0` when stateful.
- `_actuator_device.py:advance` already passes `(W,nu,na,nv,h,...)` and the same
  five CSR arguments: enabled, offsets, columns, moments, actual transmission
  fields. It adds actuator force exactly once before the original solve.
- `_normal_device.py:150,273` and `_constraint_device.py:160,334` invoke that same
  stage and use their actual base download. Activation and transmission fields
  survive both wrappers; nested plan delegation preserves kind3 and native layout.
- Runtime prepare binds input act to the accepted mirror, constructs candidate
  q/v/act and complete force/contact receipts, and only then commits one active
  bank/revision. Checkpoint v2 already saves/restores the fourth accepted act bank.
  No separate activation commit or per-step CPU physics is proposed.

## Public contract and ownership proposal

Keep the existing public construction and controls:
`compile_mujoco_xml` -> `MuJoCoRuntime(..., implementation="tyx")`,
`MuJoCoInitialState(..., act=...)`, public control/step/snapshot,
`integration_snapshot`, clone/reset and portable checkpoint/restore.
No new caller-side actuator or state API is necessary.

Use the existing `MuJoCoTyxJointActivationStepReceipt` type. Add exactly one
actuation schema, `native-free-ball-joint-filterexact-actuation-v1`, paired only
with `native-free-ball-joint-motor-transmission-v1`. Keep the old fixed schema
pair and scalar schema unchanged. Publication and checkpoint restore must select
and validate the pair from the actual plan, not from supplied payload labels.
Retain byte equality of clamped_control/raw_force/scalar_force/generalized_force
between both same-attempt records, original control binding, and accepted force
as the exact FP64 promotion of the published FP32 vector. Reject a fixed/free
schema substitution even when a particular small model fits both dimensions.
The internal activation layout can retain `native-joint-filterexact-v1`: its
native address table is already generic; public domain distinction is sufficient.

Proposed labels:

- `free-ball-joint-motor-filterexact-no-contact-v1`
- `free-ball-joint-motor-normal-contact-filterexact-v1`
- `free-ball-joint-motor-pyramidal-contact-filterexact-v1`

Exact proposed product allowlist, subject to parent grant:

| Owner | MOD files relative to `src/phymeo/backends/` | Responsibility |
|---|---|---|
| Robot author | `mujoco_tyx/_floating_model.py`, `mujoco_tyx/_actuator_model.py` | Finite admission and composition of the existing plans. |
| Parent shared seams | `mujoco_tyx/_model.py`, `_mujoco_tyx_runtime.py`, `mujoco_actuation.py`, `mujoco_joint_transmission.py`, `mujoco_checkpoint.py` | Route, exact capability and paired schema, joint publication/restore binding. |

No changes proposed to `_joint_transmission`, `_floating_kernels`,
`_actuator_kernels`, device modules, normal/pyramid geometry, solver or resource
ledger. If an actual missing ABI appears, stop and amend the contract before
expanding this list. Existing class exports suffice; no new public type is needed.

Proposed NEW independent qualification files:
`tests/test_mujoco_tyx_free_ball_activation.py`,
`tests/test_mujoco_tyx_free_ball_activation_transactions.py`,
`tests/installed/check_free_ball_activation.py`, and a bounded result report.
The parent can delegate independent tests/consumer; source owner must not edit
those inputs after a failure without recording a genuine fixture/protocol defect.

## Preserved native semantics, capacity and resource budget

MuJoCo 3.3.7 primary source, Apache-2.0:
[engine_forward.c](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_forward.c),
locally read at `artifacts/p65-mujoco-tyx-t4-research-20261110-01/upstream-3.3.7/engine_forward.c`
in the research worktree. Lines 231-257 define nextActivation; 289-334 ctrl clamp
and raw act_dot; 375 and 418-459 disabled force/actearly/gain/bias; 519-521 native
moment-transpose force; 788-809 activation then semi-implicit q/v; 832-876 retain
physical M versus Euler M+h*diag(damping). These are native source facts, not a
new physical model or an assertion of identical native solver iteration history.

For each FILTEREXACT actor, tau_eff=max(mjMINVAL,tau), raw derivative is
(clamped_ctrl-entry_act)/tau_eff. Actual disabled integration uses zero derivative;
activation range is still applied. Actearly chooses the clamped next activation,
otherwise entry activation drives the force. Fixed gain and original scalar bias
precede force clamp; BALL remains BIAS_NONE. Disabled raw force zero can still be
raised by a positive lower force bound. Reuse the already accepted small-ratio
FP32 filter expression; do not replace it with a different exponential algorithm.
BALL has all three native local angular moment columns, scalar one; shared
actors add in native order. FREE cannot be an actuator target. Root world-linear
and local-angular velocity conventions and all quaternion integration stay intact.

Preserve nv<=35, nq<=45, bodies<=128, nu<=32, na<=32, complete CSR<=96 entries;
1<=na for this stateful route and every activation coordinate has one owner.
Keep implicit original primitive pairs, geoms<=128, pairs<=16, raw slots<=32,
nonnegative mixed margin and only dim1/3 (pyramidal for3). The generic T4 larger
static capacities must not leak through this route. No limits, dry friction,
BALL/free springs, BALL affine bias, explicit pairs/excludes, tendons, plugins,
multiple free roots, unsupported flags or shapes. Existing scalar affine bias
and NONE actors remain supported; there is no new source mass/control adjustment.

Each stateful step adds the existing one activation launch to its original route.
Activation storage already allocates four `(W,na)` FP32 banks/derivative arrays,
`(W,nu,4)` fields, `(nu,9)` int32 IDs and `(nu,12)` FP32 parameters: logical added
payload `16*W*na + 16*W*nu + 84*nu` bytes over the same stateless CSR plan, before
allocator overhead. CSR and mechanics are not duplicated. Report actual allocation,
launch/readback counters and close ownership, not this formula as measured peak.
No world/model capacity extension, training integration or performance claim.

## Predeclared multi-case proposal

Use the existing FREE+BALL sources, mass/geometry/entry/h/contact budgets intact.
Only explicit new actuator dynamics/clamp fields and nonzero initial act are new
source declarations. Two sparse worlds `(3,29)`, h=.001, 24 continuous calls:
8 forward, 8 reverse, 8 unloaded, preserving the existing incoming nonzero root
and child velocities and prescribed qfrc/xfrc sequence. References never follow
the actual trajectory by overwriting their accepted state.

| Family | Fixed variants | Required observable result |
|---|---|---|
| A: R2 asymmetric no-contact FREE+BALL+R/P chain | late and early filter; shared oblique/negative/zero BALL gears plus a scalar actor, mixed NONE/FILTEREXACT | Full free-root/child coupling and all CSR columns; nonzero act and motor power; zero actual contacts. |
| B: R3 unequal feet on finite primitive support | original normal and original pyramidal source, filtered BALL plus scalar drive | Full contact force/COM moment and actual act/q/v together; normal row or every four-edge friction block, reverse and unload. |
| C: R3 two moving BALL siblings | original normal and original pyramidal source; one disabled filtered actor with positive force lower clamp plus a live filtered actor | Both body wrenches, common free-root response and asymmetric drive; disabled raw derivative versus actual integration/force distinguished. |

This is six continuous sources, both backends, not a proposed long G1 trajectory.
Reuse R1's declared filter values: tau=.004, second shared actor tau=.007;
ctrlrange[-.45,.45] unless the original scalar range[-.4,.4] exists;
actrange[-.3,.3]; initial act by native address
`(-1)^address*(.34+.01*worldslot+.005*address)`. Retain every original gain/gear
and force range except the explicitly declared C disabled positive-clamp variant,
whose source is fixed before execution. Exact XML/default-entry upgrades of old
activation negatives require separate two-step positive records; do not silently
remove an old rejection because another altered source passed.

## Qualification and stop conditions

For each actual entry, independently build native data from the saved q/v/act and
original control, call native forward/step outside the TYX guard, and save full
M/J/bias/passive/applied/actuator/contact/EFC fields. Separately advance an
independent continuous native world from the declared initial state. Independent
activation arithmetic uses native addresses and expm1; it does not call the
product activation producer. Dense native moment and CSR power obey
`qfrc_actuator dot v = sum(force_i * (moment_i dot v))` at the same entry.
Save step work using the declared h and entry velocity separately from kinetic
energy change; activation state is not an invented mechanical energy reservoir.

Preserve R1 activation rtol3e-5/atol3e-6 and actor-force rtol3e-4/atol2e-4;
FREE+BALL q/v absolute1e-4/1e-3 and time2e-6 (do not substitute looser fixed-BALL
contact q/v gates). Preserve the full original R2/R3 geometry/M/J/bias TOL tables,
no-contact acceleration rtol2e-3/atol4e-3, contact acceleration rtol3e-3/atol1e-2,
contact force rtol/atol3e-3 and original EFC/COM moment/power checks. Distinguish
physical qacc from damped integration acceleration. Raw bytes, clocks and identity
comparisons remain exact where the old protocol is exact.

Lifecycle in each route: accepted checkpoint bytes -> real successor -> restore
-> identical held-control successor; clone and partial original reset each advance.
Check unselected world raw compensated clock, act, controls and complete receipt.
Inject after actual final-world readback, after actuation or transmission publication,
and after replacement-bank verification before restore commit. Complete q/v/act,
force/receipt/revision must remain unchanged and a real retry must match a clean
clone. Independently corrupt either same-attempt record, act address/size, or the
fixed/free schema pair with a valid outer checksum: reject before publication.
True unknown device release failure remains sticky; no invented retry guarantee.

Before device allocation retain all genuine unsupported domains above plus
nonfinite/tiny unsupported actuator parameters, invalid quaternions, na/nu/cap
violations and missing activation owners. Direct compiler allow flags may not
bypass structural gates. Parent retires only demonstrably supported old activation
negative inputs; affine BALL/FREE target/limits/dry remain negative. Focused old
compatibility: one fixed BALL stateful route, one scalar free stateful route and
one stateless FREE+BALL contact route, without rerunning complete green suites.

Authoring is a moderate seven-file route/schema change with unchanged numerical
kernels; bounded tests and package validation are the dominant work. It is a
reasonable next functional slice in the current roughly two-hour window only if
shared-seam ownership is frozen immediately and the source/consumer lanes fit
before the 06:15 new-feature cutoff. Do not promise installed acceptance tonight.
After two unsuccessful diagnoses of the same new failure, freeze the concrete
remaining defect instead of tuning tau/mass/controls/tolerances or adding more
trajectories. If mechanical/kernel changes become necessary, defer this slice.
Success means both source backends, affected compatibility and fresh installed
public consumers with complete force/activation/motion and rollback evidence;
it does not confer FREE+BALL limits, arbitrary robots, training, long support or
G1 dance qualification. No original D4/D6 failure is superseded by this work.
