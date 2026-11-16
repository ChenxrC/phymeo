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
