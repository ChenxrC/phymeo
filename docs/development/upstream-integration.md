# PHYMEO upstream integration — 2026-11-16

This isolated candidate combines upstream `76a240d8` and local integration
`5ec30b9d`. It is not merged into either mainline and has not been pushed.
The old source trees, environments and qualification records remain intact.

## Current consolidation checkpoint — 2026-11-17

The active review branch is `feature/wh/phymeo-candidate-20261117`, based on
`fb4286b1`. A fresh fetch still resolves official `origin/codex/dev` to
`76a240d8499a1b6d117e47650015e85a73239b26`. It is an ancestor of this candidate;
no further upstream merge is needed at this checkpoint. No push or mainline
merge is authorized. Earlier batch sections below retain their dated results,
not an aggregate current qualification statement.

### Retained changes and workspace disposition

The inherited history contains 75 local commits relative to upstream. A
path-based inventory groups 58 as touching implementation, nine as tests/evidence,
and eight as documentation. This is a navigation aid, not 75 completed semantic
reviews. The upstream-to-candidate delta before this checkpoint spans 126 files. Keep
upstream capability additions and the integrated release-outcome resolution;
do not replay the pre-integration four conflicting versions over them.

Retain local model/identity/compiler, PhySolve runtime/storage/execution,
resident prepared I/O, contact/measurement, visual and native-scene interfaces.
Transport interfaces remain reusable library changes; retaining them does not
claim completion of the Transport consumer task. Keep regression tests and dated
failures. External diagnostic logs, caches and wheels remain outside source.

The same three-file uncommitted frontend-pair patch existed in four historical
worktrees (pairing, FANUC state, robot Graph and FANUC cadence). It was copied
byte-for-byte into this candidate and committed as `46f867ab` here only. Those
historical working trees are preserved unchanged. All eight pre-existing
worktree HEADs are ancestors of the candidate; ancestry alone does not prove
they can be deleted. The upstream checkout and two original integration
environments are retained. Historical diagnostic trees are archive candidates
only after checking their source-path consumers and local/ignored files.
No branch, worktree, environment, cache or evidence is removed by this batch.

### Dependency and acceptance boundary

The payload loader accepts exactly a7/TYX 0.1.0a0 and a8/TYX 0.2.0a1. This is
distribution-version admission, **not** source-commit or wheel-hash certification.
Merged TYX `e2bea343` and local fixes `948e63f` / `68f86a9` share version 0.2.0a1.
Robot results require the actual `68f86a9` wheel (SHA256
`38051b9f5bfd8a532fb4733e1aee433f0e2f01207e90d30d2be063689e1983b1`), not merely
an environment reporting that version. The two local compiler fixes are not
included in PHYMEO and are not represented as merged upstream changes.

| Area | Evidence/state | Remaining review or acceptance |
| --- | --- | --- |
| Public contracts and release ownership | Earlier scoped and installed passes retained; this batch source 13 and installed 13 passes, overlapping | Final overall candidate review; not a full suite result |
| Frontend pairing | Exact-pair rejection and CPU owner tests pass; prior simple-model native direct/Graph/reset results retained | Reproducible frontend source/artifact delivery and exact TYX provenance |
| Robot compilation | Multicontact and actuator compiler rejection resolved by local TYX fixes | Deliver/review those fixes separately before claiming official-TYX robot support |
| Robot direct | Existing explicit FANUC N=1/8/64 tests passed unchanged | Short-window result only; extended FANUC trajectory is unstable |
| Robot Graph | Previous full run 23 passed / 5 failed; separate reset run 1 passed / 5 failed | Keep failures open; no numerical investigation added by consolidation |
| Curved/trace regression | Earlier bounded cases passed; broader run terminated with 0xC0000409 | Incomplete qualification; crash investigation remains paused |
| Documentation | 504 historical missing targets, unchanged from the previous recorded local baseline | Do not label the documentation gate green; decide scoped repair/disposition separately |
| Product/consumer | Prior consumer evidence belongs to its original artifact combination | This candidate does not inherit current P2PE/GUI/full product qualification |

The FANUC cadence audit is recorded in [the runtime topic](physolve-runtime.md).
It refutes a Graph-only explanation, not a shared-integration defect. Native
CUDA 32 MiB retention, GPU performance admission and historical crash/coverage
limits remain open. No checks, physical parameters, control, CUBINs or official
dependency pins are relaxed.

### Next review sequence

1. Use this single candidate for subsequent changes; preserve old trees as evidence.
2. Finish the exact TYX/frontend delivery manifest and upstream disposition of the
   two compiler fixes before updating the official dependency pin.
3. Review the large upstream-to-candidate delta by model/compiler, runtime/owner,
   resident and observation interfaces. Inventory is not a substitute for review.
4. Run only missing acceptance required by the declared experimental scope, and
   preserve failures explicitly. Do not restart solver microdifference work.
5. Prepare a PR description with implementation scope, tested artifact identities
   and unqualified combinations. Push/merge require a later authorization.

Evidence for this consolidation is outside source in the sibling `evidence/`
directory: worktree/WIP hashes, commit inventory, source/installed JUnit, the
600-file source-to-installed comparison and independent review. No new wheel
was built: production bytes equal the already isolated installed artifact.

## Public and resource contracts

The public exports retain upstream heterogeneous, resident scene and deformable
capabilities and add the local PhySolve provider. Importing the root does not
load TYX, CPU MuJoCo or PhySolve. The optional compiler profile remains separate
from the official MuJoCo extras.

Resident cleanup distinguishes two outcomes:

- `RetryableReleaseError` is an explicit guarantee by the releasing implementation
  that no release occurred. Its callback and dependencies remain available for retry.
- Any other callback exception (including BaseException) has an unknown release
  outcome. The stack is quarantined with strong ownership, and the callback is
  never invoked again. This is retained memory after a failure, not a successful
  cleanup or an automatic recovery path. Restart is required for that owner.

Registration after cleanup starts and recursive cleanup are rejected. Successful
callbacks are not repeated. Child owners close before their parent runtime;
counts are cleared only after successful release. Boundary leases, loan/reentry
rejection and UNKNOWN_RELEASE quarantine remain the upstream authority for
boundary resources. Completed boundary cleanup clears pending before the local
prepared-plan cleanup phase.

Convex diagnostics keep the upstream `resources=` allocation seam and curved
support/query algorithms. A failed trace release is retained by its plan;
context exit cannot silently retry it, and no query/trace is published. Ordinary
query failure with successful cleanup remains retryable. Upstream contact packets
retain `body_twist` alongside the local release accounting changes.

The public TYX import guard now explicitly declares the two existing upstream
Torch consumers (heterogeneous I/O and manipulation environment); symbol and
private import negative checks remain in force.

## Qualification and limits

The scoped lifecycle/publication/architecture run passed 79 tests, including
N=1/8/64 CUDA release retry and all-world publication. An independent reviewer
ran 19 synthetic owner tests and verified all 419 root exports and optional
import isolation. This is not an independent GPU or full-product qualification.

A source-external PHYMEO wheel with TYX `e2bea343` (0.2.0a1) passed 136 model/owner
contracts; its one FANUC asset-configuration skip was separately rerun and passed.
The installed resident profile passes pip check, contains no Newton/Warp/
MuJoCo-Warp and imports both packages from its site-packages directory.

The old PhySolve frontend a7 and PHYMEO payload gate both require TYX 0.1.0a0.
Installing a7 beside 0.2.0a1 correctly fails dependency checking; the loader also
rejects it before loading the native Host. Neither check is bypassed. A new
frontend dependency artifact and paired Host/Graph qualification are still
required before this candidate can claim PhySolve support with the merged TYX.
The six CUBIN hashes and native Host hash match the preserved baseline.

The official PHYMEO TYX pin is deliberately unchanged until downstream
qualification finishes. The 32 MiB native CUDA retention, GPU performance
admission and historical crash/coverage limits are not closed by this batch.
No tolerances, physical algorithms, checkpoints or CUBINs were changed.

## Reproduction and evidence

External sibling `evidence/` contains JUnit/logs, installed identity, dependency
blocker and frozen payload hashes; `artifacts/` contains wheels built from the
frozen TYX source and this candidate. It is not a second production test system.
The formal tests remain under `tests/scale`, `tests/collision` and the existing
compiler/assembly modules. New release-outcome cases use synthetic owners only;
actual CUDA release and physical publication retain the existing tests.

The initial combined convex run used the default interpreter and was interrupted
before a result; it is unqualified, not passed. Prepared I/O completed with 193 passes and one old pre-release fault-injection
failure; the explicit safe-retry injection was corrected without changing its
assertions, and that test plus the selected-readback retry test passed (2/2).
Public interface/architecture tests passed 34/34, and installed sphere field reads
passed 6/6 across N=1/8/64 interpret/CUDA.

Three bounded CUDA convex/trace cases passed (curved query, late allocation failure
with rollback, analytic round query). The subsequent full two-module CUDA run
terminated with Windows status 0xC0000409; only two completed cases appear in its
partial JUnit. It did not qualify the full curved/trace regression. Its root cause
is undetermined; neither upstream, TYX nor hardware is blamed from this result.
Crash deep investigation remains paused. The previous interrupted/unknown runs
remain in evidence. Later runs deduplicate only repeated TYX AST DeprecationWarning
messages; they do not remove tests or alter numerical/ownership assertions. Whole-tree documentation checks retain missing
historical artifact/link failures; this batch does not generate fake artifacts.

## Paired frontend candidate — 2026-11-16 follow-up

The isolated `feature/wh/tyx-frontend-pairing-20261116` worktrees add frontend
`0.0.0a8` with exact `tyx-compute==0.2.0a1` (merged TYX `e2bea343`). The
previous paragraph describes the prior batch's blocker; a7 remains pinned to
0.1.0a0. The payload gate admits these two exact pairs, never their Cartesian
product or an unknown patch version. Official extras and TYX pins stay unchanged.

Frontend-owned symbolic `ArraySpec` replaces mutation of TYX's immutable array
annotation. Batched fields borrow the original storage through public
`Array.from_ptr` with the existing zero first-axis stride; they do not copy
physical state or own another simulation. Host constraint initialization uses
public scalar `numpy_dtype`. All 27 original algorithm sources still pass the
manifest's reverse-transformation hash check; native Host and six CUBIN hashes
match the frozen baseline. No TYX IR/AD, solver algorithm or tolerance changed.

Source qualification: 25 native Host tests passed for N=1/8/64 direct, fixed
Graph, force/contact, forward, reset, external streams and failure cleanup.
Six metadata/storage tests and one exact-pair rejection test passed. The new
metadata tests require explicit native-profile configuration, and CUDA cases
carry the CUDA marker; a7 does not provide this new frontend metadata API.

Robot qualification remains blocked: all six Panda/FANUC N=1/8/64 cases fail
before physical advance while compiling convex `multicontact`, with TYX
`LoweringError: undefined scalar variable ..._array_local`. This is a captured
compiler rejection, not the earlier 0xC0000409 crash. It does not establish a
numerical regression or its root cause. No collision, model, threshold or CUBIN
was changed to evade it. The curved regression and crash investigation remain
in their prior state. Simple-model passes do not qualify robot/GUI/C ABI use.

The batch uses sibling `evidence/` for actual source/installed JUnit, module and
artifact identities, unchanged payload hashes and independent review. This
candidate remains local and unmerged; complete installed results are recorded
in the batch report. No full PHYMEO or downstream product qualification is claimed.

Installed qualification completed with 32 passes (169.18 s), including the same
25 native cases and seven metadata/pair checks. All runtime Python modules came
from the independent environment's site-packages; pip check and reverse source
verification passed, and Newton/Warp/MuJoCo-Warp were absent and not imported.
Installed Panda N=1 independently reproduces the same compilation rejection
(1 failure, 28.10 s). This closes source-path ambiguity, not the robot blocker.
The first source run accidentally used the default user TYX compilation cache;
subsequent source and installed runs explicitly used isolated cache directories.
No old environment dependencies were changed or cache contents deleted.
