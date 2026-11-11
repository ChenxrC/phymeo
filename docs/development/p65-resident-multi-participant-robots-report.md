# ROB1c articulated participants: authoring handoff

Status: **CODE_READY_TESTS_PENDING / AUTHORING_RELEASE**. This lane ran no
project import, collection, numerical probe, test, formatter, linter, native
kernel, build or installation. A text review and Git diff check are not execution
acceptance. The parent owns the sole execution slot.

## Implementation boundary

The public resident scene now consumes actual full-body COM linear/angular
velocity from the parent service's `body_twist[b,6]` packet for all four
families. It retains those immutable arrays in
`ResidentParticipantReceipt.entry_body_twist/accepted_body_twist`, alongside
the original typed geometry, full M/J/b, input external load, FP64 contact-law
wrench, actual uploaded FP32 wrench, full physical generalized contact and
original family diagnostic. Affine K coordinates do not truncate physical N
rows or graph reactions.

This lane changes no integrator or family solver. The parent service owns the
four storage groups and the actual launch order:

1. Stage every used storage's entry mechanics once.
2. Read one common entry; assemble all HOST finite-patch loads, with one
   equal/opposite pair force and the corresponding moment about each real COM.
3. Each original storage projects the complete loads and solves once.
4. After every candidate FK/status is available, collect each physical row's
   status into its unique mechanical slot. One lane per logical world reduces
   these slots. Scatter marks originally successful peers for retention.
5. Each original family finalizer executes once; all complete immutable world
   publications are prepared before the shared bank commit.

The three new status-only kernels are
`collect(count,row_slots,info,flags,failed)`,
`reduce_slots(world_count,world_offsets,world_slots,failed)`, and
`scatter(count,row_slots,failed,info)`. The earlier single-storage `reduce`
entry point is retained. Slot sets are disjoint across storages and worlds; all
slots are overwritten during each collect phase. Scatter preserves a member's
original failure code, and gives originally successful peers code 6 solely to
request whole-world retention. It does not assert that each peer had an
independent FK failure. There is no state copy, force computation, bank choice
or publication in these kernels.

The unchanged common-entry explicit contact law is not a cross-tree implicit
contact-force/KKT solve. This scope remains single collision model/page, at most
8 participants and 33 full physical bodies per logical world, original candidate
and site limits, HOST patch geometry and TYX mechanics. It inherits neither
W256 scene qualification nor device BVH/contact-path/CCD qualification.

## Fixed cases authored before implementation

The two new files declare 22 parameterized nodes per backend (14 robot/window,
8 restore/transaction). This count is textual, not collected.

| Case | Fixed scope | Independent success checks |
|---|---|---|
| R02 | Ordinary Rz/Px arm and eccentric free box; nonzero q/v/control and body load; h=1/1024, 40 continuous steps; mu=0 and 1/4 | Full non-diagonal M/J/b, both COM moments, true free-box motion, bounded control, same-entry and independently continued trajectory |
| R03 | Two actual 2-joint Rx/Px arms and free tool; symmetric and asymmetric; outward control 48 steps then inward 80, no state reset | Initial load, strict separation during steps 32..47, reentry/load during 96..127; asymmetric tool rotation; complete continuous records |
| R04 | Fixed, affine, floating and floating-affine, each separately paired with a free box; 16 steps | Full physical projection and original family diagnostic, graph reactions and reduced equations; nonzero full joint contact whose independent joint projection is exactly zero at entry |
| R05 | W2/W4 sparse IDs, all worlds at the same coordinates, reversed registration, comparison with each independently built W1 scene | No cross-world interactions, exact accepted checkpoints and generalized loads per corresponding world; one actual stage/solve/finalize per used storage |
| R06 | All four storage families simultaneously, two worlds, five participants/world, 11 full bodies/world | Final floating-affine member rejection retains every peer only in its world, true retry, final publication exception, tick19 checkpoint plus 21-step exact replay, partial reset, late restore fault and true retry, missing/wrong/stale checkpoint rejection |

R02 uses link masses (2,1), principal moments 1/8, root COM (1/8,0,0),
child COM (1/16,1/32,0), child joint offset (0,1/4,0), initial
q=(1/16,1/32) and v=(1/32,-1/64). Since the child prismatic axis is local x,
the independent transform derivative gives

`M01 = -(1/4 + 1/32) = -9/32`.

The tip sphere's local center is (1/8,1/8,0), radius 1/8. The box has half
extents (1/8,3/16,1/8), the same initial yaw, and its geometric center is
7/32 along the rotated x axis from the sphere. The initial depth is 1/32,
so k=64 gives exactly 2 N before storage rounding. The box collider center
is offset 1/16 in local y from its real COM. For mu=0 the two actual
COM moments are +3/16 ez on the tip and -1/8 ez on the free box. This
tests a physical lever arm and full joint projection, not a resultant repair.

R03's hinge axis is the actual local x push axis. Its nonzero speed/inertia
remains in mechanics while a spherical center on that axis isolates the
control-driven separation direction. Initial depth is 1/128. Prescribed
velocity targets are +/-1 m/s (the asymmetric right arm uses 3/4 m/s),
with implicit damping gain1024, finite cap4096, h=1/1024. The outward
48-step target travel is at least36/1024 versus depth8/1024; the later
80-step return target exceeds that travel by at least24/1024.
The heavy free tool is still unconstrained and moves under the actual
unbalanced force/moment. These are predeclared geometric/control margins;
the original numerical assertions must diagnose a failed event rather than
retune input parameters after observing a result.

R04's literal two-prismatic graph uses P=(1,-1)^T. Its two equally loaded
physical contact rows have nonzero full generalized load, while the independent
joint component cancels. In the floating-affine case the root force remains:
zero reduced joint load never means zero free-root response.

## Oracle and evidence provenance

The new tests use the existing independent reference files
`tests/reference_newton/robot/resident_affine_reference.py` and
`resident_floating_reference.py` only for explicit homogeneous-transform
derivatives and full physical M/J/b; no production FK or solver supplies the
expected state. The new sphere/oriented-finite-box oracle uses the analytic
closest point and common midpoint, regularized tangential traction and both
COM moments. A separate small exhaustive actuator active-face calculation
solves the declared bounded implicit PD problem. The continuous reference
starts once from the authored initial state and never chases the tested
trajectory. Same-entry comparisons are stored separately.

All normal window records include original input dictionaries, selected
controls, actual before/after public world capsules, full receipts including
family diagnostics and graph reactions, complete same-entry/continuous
oracle records, status and per-storage/transfer/allocation counters. Gzip
encoding retains dtype, shape and original array bytes as well as readable
values. A window assertion failure also saves its last actual entry,
controls and complete last receipts; it is not mislabeled a pass.

Transaction cases save the actual pre-failure publication/checkpoints, injected
identity, observed retained/retried state, epoch/bank and all successful step
receipts. The late restore hook belongs to the parent service and executes
after all genuine stage/upload/check work but before the bank swap. The test
does not emulate restoration by reconstructing an initial scene.

The explicitly enabled intra-participant path has two additional targeted
qualifications per affine family: positive sibling contact with an independent
8 N first force, and direct-parent rejection before stepping. Affine parent
tables are read from the original physical compiler output; reduced-coordinate
wrappers are not treated as physical trees. Default cases are unchanged.

## Public example

`examples/resident_arm_push.py` uses only public Phymeo APIs and NumPy/stdlib.
It independently constructs the R/P arm, free eccentric box, finite sphere/box
bindings, nonzero incoming velocities and commands. No test import, private
bank access, position prescription or HOST integrator is used. It saves full
inputs, public states, control/force receipts, all diagnostics, counters and
failure details, then verifies owned device bytes are zero after close.

Pending, to be run only by the parent's sole serial runner:

```text
python -B -m pytest tests/scale/test_resident_multi_participant_robots.py tests/scale/test_resident_multi_participant_restore.py -q -x --tb=short -p no:cacheprovider
python -B examples/resident_arm_push.py --mode interpret --steps 40 --friction 0.25 --output <fresh-absolute-directory>
python -B examples/resident_arm_push.py --mode cuda --steps 40 --friction 0.25 --output <different-fresh-absolute-directory>
```

The pytest mode and environment are supplied through the repository's existing
TYX runner, once for interpret and once for CUDA. This lane does not grant or
claim those runs. Existing ROB1a/ROB1b green groups are not rerun by this author.

## Source ownership and history

- Parent prerequisite: shared four-family transaction adapter, deferred
  tree-finalize hook, original restore pre-commit seam and body-twist packet
  under parent commit `d13e9fcf`; not copied into this child tree.
- Tests-first `c8164542`: only the two new test files.
- Product `558d69da`: contact adapter/receipt, status-only transaction kernels,
  public façade documentation and the new body-twist assertions.
- Targeted tests-first `bba26532` adds the approved affine parent-table cases
  and failure context; its following product commit only fixes that lookup.
- Final authoring adds the public example, this report and the explicitly
  approved failure-context-only test save supplement.

No layout compiler, existing integration kernel, old test group, G1 asset,
physics budget, collision cap, expected fixture trajectory or external
dependency was modified. The old untracked
`docs/development/mujoco-tyx-t1-review.md` remains untouched.
