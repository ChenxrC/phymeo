# PHYMEO upstream integration — 2026-11-16

This isolated candidate combines upstream `76a240d8` and local integration
`5ec30b9d`. It is not merged into either mainline and has not been pushed.
The old source trees, environments and qualification records remain intact.

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
