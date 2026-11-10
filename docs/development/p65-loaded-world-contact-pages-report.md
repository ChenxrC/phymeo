# C1 world-complete loaded contact pages

Current status: **FOCUSED_INTERPRET_AND_CUDA_VERIFIED**. Parent jobs 12/13 passed
all 17 cases per backend on source 6ba6ec40, including loaded W127/128/129/256,
four layouts, ordering/reset/capacity and last-page rollback. Public CUDA job 18
also completed 16 continuous steps at W1/32/128/256; comparable single-page runs
matched complete states and receipts. This establishes scale support, not a
Newton speed advantage. Evidence is in `artifacts/p65-parity-20261111-01` and
[the rolling report](p65-parity-2026-11-11-progress.md). New wheel checks remain pending.

The original authoring account below is historical. The parent owns
the only execution slot. This extends the accepted resident contact transaction;
it is unrelated to the separate MuJoCo-TYX mixed-constraint solver.

## Public contract and scope

`ResidentContactPageSpec(model, bindings, materials_by_pair, candidate_capacity=4096)`
owns a tuple of bindings and an immutable material mapping. Its CollisionModel
retains the original shape, vertex, revision, filter and exclusion contracts.
`TyxWorldBatch.prepare_contact_bundle(pages, site_capacity=128,
sampling="discrete_entry", tree_dynamic_pairs=False)` returns a
`ResidentContactBundle`. The parent integrated the three facade exports for the
spec, bundle and `ResidentContactBundleResult`.

A bundle contains zero through eight pages. Each page retains the original
256-shape model bound and candidate-capacity range 0..32640. The aggregate
candidate budget is exactly the sum of validated page capacities; there is no
truncation or independently enlarged per-page cap. A contact-bound world must
belong to one page in its entirety, even when two pages would use disjoint shape
keys in that world. Material/domain errors, wrong owners/topologies, duplicate
world registrations, shape bindings, site-cap mappings and page limits reject
before any device allocation. Original per-world site capacity remains 0..4096.

Page order and batch world order have no physical meaning. Equal coordinates
and reused local body/shape numbers in distinct worlds do not create a pair.
World-complete partitioning means all relevant shapes, materials and sites for
a given world reach the original HOST narrow phase and load accumulator in one
call. This is still entry-sampled HOST narrow phase, with device broad phase and
resident mechanics; it does not add device narrow phase, CCD or friction history.

`batch.step_contacts(bundle, h)` and `ResidentContactStepper(batch, bundle)`
use the existing batch transaction. Empty bundles and empty-model pages perform one ordinary free
step. A bundle owns its page/scratch resources, never its batch.

## One mechanics transaction

The original `_resident_contact_plan.py` mechanics methods now live in private
`_ResidentContactMechanics`. The single-page public plan retains the same
constructor and collision superclass. Its validation/allocation wrappers call
the shared helpers; its force, Jacobian, solver, state and work algorithms have
not been copied or changed. Original and floating frame readers enumerate the
prevalidated per-world shape tuple instead of scanning one global model.

Internal `_ContactCollisionPage` uses the original collision validation but
defers `_prepare_device`. Every page first finishes HOST registration. Only
then does a fresh scoped batch state view allocate the page broad-phase storage
and one shared contact scratch group for each participating ordinary, affine,
floating or floating-affine storage. No borrowed view survives construction.

For each attempt:

1. Stage all page queries and pending BVH layouts at the same accepted epoch.
   Candidate overflow rejects the whole attempt before material/mechanics work.
2. Assemble entry mechanics once and read each participating storage once for
   entry packets. All of a world's sites accumulate through the old FP64 load
   path before its one FP32 upload.
3. Project and solve each complete storage once, including its unbound worlds
   with zero contact load. Candidate geometry and candidate packets use the old
   implementation.
4. Build every world receipt and all page publication data. Validate the
   original entry owner, topology/revisions, epoch, bank and held controls.
5. The batch performs its existing single bank swap. Bundle publication only
   assigns preconstructed page layouts/results and receipt/status mappings.

Per-world site overflow retains that world's original complete state while
other valid worlds can advance, as before. A page candidate failure, allocation
failure or later receipt/finish exception cannot publish earlier pages or any
new physical state. The counterexamples retain old receipts and support retry.

## Publication and accounting

`last_accepted` and `last_status` cover the same world semantics as a single
plan. `last_result` / `last_query` is the complete aggregate entry query;
`last_page_results` exposes the ordered complete page results. Each page result
retains the original owner/registration, topology/revision, ticks, shapes, bounds,
registered pairs and candidates. Aggregate `require_current` verifies every
page and the single batch epoch, including the empty-bundle case. An accepted
step makes its entry query stale for future work, as in the original plan.

`stats()` includes all page counters, page/world membership, each candidate cap
and their sum, combined live device bytes/allocations, and shared contact
allocation/upload/download/HOST-read/kernel counts. The batch records live
bundle count separately from its owned collision pages. Closing a bundle closes
every page and shared contact resource; batch closure also closes live bundles.

## Authored validation matrix

New `tests/scale/test_resident_contact_bundle.py` declares 17 parameter cases per
backend (34 interpretation/CUDA cases if both modes are selected):

- W127/128/129/256, with two finite shapes and real nonzero load per world.
  Three continuous steps retain nonzero incoming velocity and external/motor
  input; a literal 2 kg slider law independently checks the state. Spies require
  exactly one mechanics assembly and one solve per storage and two packet reads
  per storage/step (entry and candidate).
- Single-page versus original-plan equality, page/world reorder and identical
  coordinates, plus eight loaded pages versus one old plan.
- Four coordinate kinds in one batch, each on a different page, with complete
  reused independent affine/floating/floating-affine mechanics/contact oracles.
  The new fixture retains ordinary velocity/input, F02 motor plus root velocity,
  and A02 root velocity plus external root force/torque. It does not invoke old
  test functions or rerun the accepted historical suites.
- Partial reset/stale epoch and modified result revision/page count/order/capacity rejection; site overflow
  retains only the affected world and its old receipt.
- Last-page candidate overflow before any mechanics, and final-world load-finish
  failure after solving but before publication; unchanged full states, receipts
  and page layouts, followed by a valid retry.
- Six registration errors: duplicate shape, same world with disjoint shape keys,
  later-page missing material, later-page foreign topology, ninth page, invalid
  per-page candidate cap. A forbidden-allocation sentinel proves preflight order.
- Actual second-page allocation failure closes the first page, immutable specs,
  close/borrowed-stepper lifetime, and zero-page/empty-model free stepping.

Saved test artifacts retain full receipts, inputs, per-page numeric query data
and counters. Opaque in-process owner/registration objects are not serializable;
the saved page record explicitly retains the complete value/stamp fields rather
than pretending to serialize those identities.

## Pending parent execution

No execution result is claimed. In the sole parent slot, after its fresh guard,
snapshot and facade exports, run exact changed-file Ruff checks and format checks,
then the complete new test file in interpretation and CUDA separately:

```text
python -m pytest tests/scale/test_resident_contact_bundle.py --tyx-mode=interpret -q -x --tb=short -p no:cacheprovider --basetemp=<fresh-absolute-interpret-dir>
python -m pytest tests/scale/test_resident_contact_bundle.py --tyx-mode=cuda -q -x --tb=short -p no:cacheprovider --basetemp=<fresh-absolute-cuda-dir>
```

Use the repository's existing isolated environment/thread/import protocol. The
parent also selects minimal original single-plan/stepper compatibility checks
because the mechanics extraction and explicit accepted type seam changed. Do
not rerun unrelated accepted suites or claim W256 until actual new tests pass.

## File ownership

New: `src/phymeo/backends/_resident_contact_bundle.py`, the new test above and
this report. Modified: `_resident_contact_plan.py`,
`_resident_floating_contact.py` (world-shape enumeration only),
`tyx_world_batch.py`, `training/resident_stepper.py`.
The bridge's FP64 accumulation code needed no modification. No kernels,
physical law, caps, solver algorithm, precision gates or dependency pins changed.
The old untracked T1 review draft is untouched and excluded from commits.
