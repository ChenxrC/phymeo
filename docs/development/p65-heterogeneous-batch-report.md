# Heterogeneous MuJoCo-TYX batch author report

Status: **CODE_READY_TESTS_PENDING**. Authoring only; no import, collection,
numerical probe, pytest, lint, formatter, build, installation, or benchmark was
run in this lane. The parent owns all actual execution and shared runtime seams.

## Scope and source

Baseline: `5d3b4c50`; branch `codex/p65-heterogeneous-c-20261112`.
Contract: [heterogeneous design](p65-heterogeneous-batch-design.md).
The implementation schedules exact-source homogeneous buckets and publishes one
heterogeneous result. It neither fuses kernels nor adds cross-world contacts.
The fixed MuJoCo 3.3.7 physics, original TYX kernels, per-model capacities and
numerical gates are reused.

Author commits:

- `1c597a85`: design and source-backed scope.
- `597bc19b`: both complete test files before product authoring.
- `fbe7bdd7`: two product modules, installed consumer and bounded transaction
  additions. Source is now frozen for parent integration.

Only seven NEW paths belong to this lane:

| Path | Purpose |
| --- | --- |
| `src/phymeo/backends/mujoco_heterogeneous.py` | Immutable declarations, exact-source buckets, atomic facade |
| `src/phymeo/backends/mujoco_heterogeneous_checkpoint.py` | Bounded composition of original checkpoints |
| `tests/test_mujoco_tyx_heterogeneous.py` | Source identity, complete independent physical comparisons |
| `tests/test_mujoco_tyx_heterogeneous_transactions.py` | Failure, restoration, clocks, cleanup and reentrancy |
| `tests/installed/check_mujoco_tyx_heterogeneous.py` | Self-contained source-external public consumer |
| `docs/development/p65-heterogeneous-batch-design.md` | Frozen contract |
| This report | Authoring and qualification boundary |

The original untracked `docs/development/mujoco-tyx-t1-review.md` is unchanged.
No old product or test file was edited by this lane.

## Public and internal contracts

Public types are `MuJoCoTyxModelSpec`, `MuJoCoTyxWorldSpec`,
`MuJoCoTyxWorldLayout`, `MuJoCoTyxHeterogeneousBatch`,
`MuJoCoTyxHeterogeneousStepReceipt`,
`MuJoCoTyxHeterogeneousBoundaryReceipt`, and
`MuJoCoTyxHeterogeneousCheckpoint`. The parent owns public exports.

A batch admits 1–8 declared model keys and 1–16 sparse world keys. Model keys
cannot be unused, world keys cannot repeat, and all native timesteps must agree.
Native structural preflight of every model occurs before any device constructor.
Grouping compares the actual primary XML, archive and compiled model bytes,
native version, mode and capacities. Equal dimensions alone do not merge models.

All bucket locks are held in canonical order. Reentrant operations are refused.
The parent runtime supplies prepare/validate/commit/discard for steps and
controls. Ordinary steps reuse the existing active/inactive device banks; no
clone is performed. Every draft and the complete aggregate are prepared and
validated before the first commit. Commit only publishes the already prepared
objects and accepted-bank indices. A late status, download or aggregate failure
retains every previously accepted world and receipt.

Boundary operations may allocate. Reset clones only affected buckets, restores
their complete original leaf checkpoint, then applies the original selective
reset. Restore prepares replacement buckets from all saved leaves. No replacement
becomes visible until the whole operation is ready. This preserves unselected
raw q/v/act, two-component clocks, CPU integration mirrors, initial states, held
controls and original full receipts. Historical aggregate controls remain tied to
their own accepted step even if held controls changed later.

The aggregate checkpoint has a 128 MiB encoded-byte ceiling and at most eight
original leaf checkpoints. It uses checksum-bound JSON plus base64 leaf bytes;
the original leaf decoder validates the physics schemas and source bindings.
This is a trusted application checkpoint format, not authenticated hostile data.

## Cleanup and accounting

A failed close retains its strong owner for an explicit later close attempt,
subject to the existing runtime/resource-ledger semantics. Actual uncertain
underlying releases keep their existing sticky quarantine behavior; this facade
does not retry an unknown device free. Construction plus cleanup failures retain
the facade in a module quarantine and preserve both exceptions.

Counter observation cannot skip a real close, mask its error, or turn an already
committed control into an apparent refusal. Failures are recorded separately.
Counter snapshots and positive cumulative deltas expose the original runtime
counters; facade successful calls/world steps, failed batches, control operations,
boundary clones and bucket preparations have separate explicit meanings. These
numbers are not a measurement of all transient native allocations inside clone
or a process-wide live-memory census. No speed claim is made.

## Fixed functional matrix

The shared timestep is 0.001 s. World order is
`(41, 7, 305, 11, 203, 59)`; the second positive ordering is its reversal.

| Model | Original fixed source | Qualified behavior reused |
| --- | --- | --- |
| Scalar | `test_mujoco_tyx_activation.source("coupled", early=True)` | Hinge/slide, four NONE/FILTEREXACT actors, affine scalar force law |
| Fixed BALL | `check_ball_activation.source("r4-pyramid")` | Direct BALL FILTEREXACT, shared target, primitive pyramidal contact |
| FREE+BALL | `check_free_ball.source("mixed")` | Mixed FREE/BALL/R/P, direct NONE JOINT motors, no contact |

Each model has two independent worlds with nonzero original initial velocities
and controls. The 24-step sequence preserves the original reversal/unload inputs.
Same-entry MuJoCo fields and a separately advanced continuous native trajectory
are both checked. Complete M, COM J, bias, passive/applied loads, scalar and
generalized actuator forces, activation/transmission records, contact/EFC fields,
both-body COM wrenches and q/v/time are kept. CPU `mj_step`/`mj_forward` are
forbidden only around the actual TYX step and integration observation; the
independent references run explicitly outside that guard.

No numerical gate was changed. In particular:

- Scalar activation: rtol 3e-5 / atol 3e-6; actuator force: 3e-4 / 2e-4;
  q/v absolute limits 5e-5 / 3e-4.
- Fixed BALL contact uses its original complete field-specific gates, contact
  force 3e-3 / 3e-3, and q/v absolute limits 2e-4 / 3e-3.
- FREE+BALL uses its original field-specific gates and q/v absolute limits
  1e-4 / 1e-3. Root and child quaternion norm gate remains 8 FP32 eps.
- All native time comparisons retain the original 2e-6 bound.
- Checkpoint successors, single-bucket equivalence and unselected saved fields
  use exact original typed values/bytes, not approximate equality.

Manual decorator enumeration, **not executed collection**, is 42 nodes:
18 main-file nodes (8 backend-parametrized functional and 10 HOST admission),
24 transaction nodes (22 backend-parametrized and 2 HOST cleanup-owner checks).
The late-failure cases are last-bucket status, actual download, final aggregate,
both reset/restore replacement boundaries, invalid last control, wrong binding,
stale revision, reentrant writers and postpublication cleanup. Whole-batch retry
is compared with a separately constructed original trajectory.

## Installed protocol and pending execution

The standalone consumer imports installed public interfaces and NumPy/MuJoCo,
never checkout tests or examples. Its oracle equations and fixed XML declarations
are self-contained copies of the qualified protocols. It rejects source checkout
entries, records every loaded product module path and SHA, and requires all
product modules under the supplied installation root.

Per backend it performs 24 continuous calls, one actual successor, an exact
checkpoint-restored successor, and one selective-reset successor: **27 actual
physics calls / 162 accepted world steps**. This count includes real work after
restore and is independent of rolled-back tick values. Wrong-world controls and
stale restore are negative boundary operations without extra physics. Full
attempts, immediate return prefixes, states, original native expectations,
aggregate/leaf receipts, controls, checkpoints, counters and exceptions are
written to one gzip stream. Cleanup preserves the original failure.

Pending parent commands, with fresh evidence directories and the parent's
single execution guard:

```text
pytest tests/test_mujoco_tyx_heterogeneous.py tests/test_mujoco_tyx_heterogeneous_transactions.py -q -x --tb=short -p no:cacheprovider
python <outside-source>/check_mujoco_tyx_heterogeneous.py --mode interpret --installed-root <fresh-site-packages> --output <fresh-output>
python <outside-source>/check_mujoco_tyx_heterogeneous.py --mode cuda --installed-root <fresh-site-packages> --output <fresh-output>
```

The parent performs static checks, actual collection, both backend tests,
affected shared-seam compatibility and a new source-external installed package.
Authoring success is not physical or installation acceptance.

## Remaining limits

This first heterogeneous batch is a functional 8-model/16-world ceiling, with
actual positive fixtures covering three distinct models and six worlds. It is
not a scale qualification. Models retain their independently declared domains:
FREE+BALL contact, activation, limits and dry friction remain unsupported here.
There is no public child runtime, dynamic insertion, multirate stepping, shared
contact graph across worlds, training tensor ABI or per-step boundary clone.
Failures before common publication are atomic; retired-resource cleanup after
a committed boundary is separately reported rather than undoing accepted physics.
