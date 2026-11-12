# R3-N: FREE-root BALL trees with primitive normal contact

Status: **CODE_READY_TESTS_PENDING / AUTHORING_RELEASE**. Parent baseline:
6dcf99c2. No import, collection, numerical probe, pytest, Ruff, formatter,
build, install, profile or CI was executed by this author. A manually
enumerated node count is not a result.

## Change and ownership

The only product change owned here is
[_floating_model.py](../../src/phymeo/backends/mujoco_tyx/_floating_model.py).
Its explicit allow_normal_contacts=True permission admits FREE+BALL potential
pairs only after the shared native mask/weld/parent predicate, complete pair
dimensions and geometry families, and original 128 geom / 16 pair / 32 raw
candidate-slot limits. It reuses the established PAIR_SLOTS table. The normal
compiler still checks and constructs the complete geometry and contact law.

Activation, every joint limit, dry friction, explicit pairs/excludes,
unsupported transmission, BALL affine bias/springs, multiple FREE roots and
invalid joint layout remain rejected. Passing all three private permission
flags cannot bypass these gates or admit mixed condim3.

Parent owns _model.py structural routing and runtime capability labels:
full floating structural pass, then nonempty pairs to compile_normal_model,
otherwise reuse the no-contact result. There is no failed-lowering fallback.

Original floating dynamics, native DOF/ancestor table, full M/J/bias, CSR motor
application, physical M versus Euler M+hD, and root/child quaternion integration
are unchanged. Normal wrapper ABI already passes complete DOF/CSR and native
layout/nq/njnt/joint integration arguments. No kernel, solver, bank or public
receipt type is added.

Capabilities are free-ball-joint-motor-normal-contact-v1 and, without actors,
free-ball-normal-contact-v1. The existing
native-free-ball-joint-motor-transmission-v1 schema is retained.

## Fixed sources and first-step reasoning

The [installed consumer](../../tests/installed/check_free_ball_normal.py) contains
both independent XMLs and oracles. Source tests load that harness; the consumer
itself imports no checkout tests/examples and never injects sys.path.

| Case | Native coordinates | Authored mechanism |
| --- | --- | --- |
| feet | nq12 / nv10 / nu2, FREE + BALL + SLIDE | 2.6 kg offset-inertia root, unequal .4/.5 kg feet, BALL gear (1,-2,.5), SLIDE gear -2; two sphere feet contact a plane. |
| siblings | nq15 / nv12 / nu3, FREE + two BALL | Same 2.6 kg root, unequal 1/1.4 kg branches with different offset COMs; two oblique motors share one BALL and a different gear drives the other. |

Sphere radius .1 m. Feet use root z=.4 plus foot z=-.301: center height
.099 m, initial gap -.001 m. Root and BALL initial rotations are yaw, preserving
these heights. The second world changes horizontal translation and velocities,
not contact depth.

Sibling centers are .198 m apart before the common root yaw. Equal child
rotations and equal joint anchor offsets shift both centers equally, retaining
the -.002 m initial gap. Different COM offsets produce two nonzero contact
moments. Internal sibling contact has zero free-root generalized resultant but
nonzero child-joint response; this does not fix or freeze the free root.

Both cases use world keys (7,41), h=.001 s, initial time=.125 s and genuine
nonzero root/child velocities. Each has 24 uninterrupted steps: 8 positive,
8 reversed and 8 zero actuator/generalized/external controls. This is control
unloading; gravity remains present in feet. No reset or replaced initial state
occurs inside that trajectory. It does not assert steady support or separation.

The first step must have both foot contacts or the sibling contact active with
positive force. Every world retains complete native references, not sampling.
Both sibling moments, both BALL motions, actual root motion and internal
resultant cancellation are checked. No parameter was selected by a probe.

## Complete independent reference and unchanged gates

At each actual device entry, separate MuJoCo 3.3.7 MjData receives stored FP32
controls. Native forward/fullM/jac/contactForce and fields provide full M, COM J,
bias/passive/applied/actuator force, normal J/R/aref/force and geometry/successor.
A separately owned native trajectory advances continuously from its own initial
state; it is never overwritten with later device states.

Independent native-address/gear formulas construct every CSR row, dense
moment, clamped control, scalar/generalized force and power. Two-sided COM
wrenches independently reconstruct the same generalized contact force.

| Gate | Original value |
| --- | --- |
| FREE+BALL q / v / time absolute | 1e-4 / 1e-3 / 2e-6 |
| position/rotation/COM rtol/atol | 3e-5 / 4e-5 |
| full J; M | 2e-4 / 5e-5; 3e-4 / 5e-5 |
| bias/passive/applied | 4e-4 / 4e-4 |
| smooth acceleration | 2e-3 / 4e-3 |
| actuator force; contact force | 3e-4 / 2e-4; 3e-3 / 3e-3 |
| physical/integration acceleration | 3e-3 / 1e-2 |
| normal KKT / face budget | max(source tolerance, FP32(2e-5)) / original source128 |
| root and child quaternion squared norm | 8*eps32 |

Full restored state/receipt comparisons preserve all fields and types. The
FP32 motor force is promoted exactly to compare the original FP64 public force.
CPU step/step1/step2/forward are forbidden during each actual TYX step. Separate
native oracles and the explicit full endpoint observer run outside this guard;
this is not called a device-only or zero-readback observation pipeline.

## Predeclared source matrix

[Main tests](../../tests/test_mujoco_tyx_free_ball_normal.py): **33** manually
enumerated, UNEXECUTED nodes:
4 continuous (2 models x 2 backends), 15 source/domain/capacity rejections before
allocation, 4 direct-compiler flag bypass negatives, 2 exact original static
capacity boundaries, 8 root/child zero/nonunit quaternion negatives.

[Transactions](../../tests/test_mujoco_tyx_free_ball_normal_transactions.py):
**22** manually enumerated nodes:
4 checkpoint/clone/partial-reset cases; 16 two-model/two-backend late status,
actual readback, final-world transmission publication or verified restore
failures; 2 bad-control/foreign-capsule atomicity cases and actual successors.

Total planned nodes: **55**, not a collected or passed count. Failure retains
complete old force/motion, capsule and revision. Real retry must equal a real
uninterrupted clone successor. Rejected candidate work is still counted.

Static boundary cases validate exactly 16 pairs and 32 raw slots; they do not
claim maximum-capacity dynamics. Actual continuous layouts are nq12/nv10 and
nq15/nv12, W2. Existing nv35/nq45/body128/actor32 bounds are not enlarged and
their maximum Cartesian product with all contacts is not qualified here.

Original FREE+BALL negative_source("contact") has default condim3; original
BALL-MC "free" also has condim3. Both remain rejected with free-ball-contact.
No old negative needs retirement.

## Source-external qualification and complete evidence

Copy the standalone consumer outside every checkout; public CLI:

    python check_free_ball_normal.py --mode interpret --installed-root <fresh-site-packages> --output <fresh-absolute-directory>
    python check_free_ball_normal.py --mode cuda --installed-root <fresh-site-packages> --output <fresh-absolute-directory>

The two continuous cases contribute 48 worldsteps each; the real
checkpoint/clone/restore/partial-reset successor workflow contributes16.
Expected per backend: **56 actual calls / 112 worldsteps**, not rollback tick
differences. These are planned counts, not observed results.

Origin checks reject checkout root/src, source cwd/script and preloaded Phymeo;
existing environment site-packages dependencies remain allowed. All loaded
phymeo modules must come from the installed root. Floating and normal source
modules are explicitly required in final provenance.

Every step saves original entry/control before the oracle, complete native
evidence before physics, the actual return immediately, then full accepted
state/receipt and observation before assertions. Lifecycle capsules and all
unselected raw clock/mirror fields are retained. Gzip uses complete typed array
bytes, compressed/logical hashes and sizes. Primary and cleanup errors are
preserved separately, with raw failed prefixes. Nothing turns a refusal into
a pass.

## Parent-only pending validation and boundaries

Parent alone runs the two complete new files, relevant old route compatibility,
static checks and new wheel/installed two-mode qualification. No unrelated
accepted suite or archived evidence should be repeated. Parent records any
style repairs and their source binding.

Author checkpoints: tests-first63ae790d, product194b5c11; consumer/report follow.
Original untracked mujoco-tyx-t1-review.md is untouched.

This is normal-contact composition, not pyramidal friction, FILTEREXACT,
limits/dry friction, multiple FREE roots, general robot completion or P6.5
completion.

## Parent source validation — 2026-11-13

The parent connected the explicit FREE+BALL normal route and public capability
label without changing numerical kernels. Run `22-r3-n-complete-first` completed
both full new test files: **55 passed in 41.34 s**, RAW_EXIT 0. The matrix includes
interpret and CUDA cases in the tests themselves; it was executed once, without
skip or deselection. Runs 18/19 completed the exact new-file Ruff checks after
formatting and the recorded import-order repair.

Status: **SOURCE_VERIFIED_INSTALL_PENDING**. This is not yet the unified wheel
acceptance. Original native reference, numerical gates, rejected combinations,
capacity bounds and transaction requirements remain unchanged. Evidence is in
`artifacts/p65-feature-until-0730-20261113-01`.
Run31 then passed all seven selected affected compatibility nodes in15.65s:
FREE+BALL without contact, fixed BALL motor normal contact, original FREE/scalar
two-foot normal contact (both modes), and the unchanged old pyramidal FREE
negative. This does not rerun or broaden the old accepted full suites.