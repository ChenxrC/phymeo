# N3 / ROB1c / COL trace batch — 2026-11-11

Base: `ed6f6d847c3d64abb13ab28f5f7042832c5b2f3e`. The three original authors reuse isolated worktrees; parent owns shared seams and all serial project execution. No deadline, recurring task, remote CI, pin change or unrelated cleanup. Preserve the three old untracked K1 files.

## Actual outcome

| Lane | Delivered and executed | Remaining boundary |
| --- | --- | --- |
| N3 | Actual CUDA combined PPO checkpoint/restore: 24 passed. Two W8 MLP tasks reproduce update3 after save at update2; GRU reproduces update2 after nonzero-hidden save. Pre-CAS rejection and post-CAS poison verified. | Same device/dependencies/config, complete update boundary. Learning quality and N4 throughput remain unqualified. |
| ROB1c | Interpret union and complete CUDA each cover 20 passing nodes and 2 R03 precision failures. Arm/eccentric box, four graph families, sparse worlds, complete mixed-world rejection/restore pass. Old shared-seam regression 11 passed per backend; public CUDA arm push completes40steps and frees ownership. | R03 symmetric/asymmetric acceleration errors about2.40e-5/2.44e-5 remain; full128step R03 qualification is open. HOST patch, <=8participants/<=33physical bodies/world; no implicit inter-tree QP or new wheel. |
| COL trace | Interpret18 and CUDA14 protocol nodes passed; six fixed CUDA failure captures preserve same-state ordinary-query geometry. Old affected query/patch/source regression18 passed per backend. | Diagnostic success does not repair geometry. FourY distance/normal exits and twoHN repeated simplex reductions are documented; caps, thresholds and fixtures unchanged. |

Details: [N3](p65-mujoco-tyx-n3-report.md), [ROB1c](p65-resident-multi-participant-robots-report.md), [trace protocol](p65-convex-query-failure-diagnostics.md), [actual six-input findings](p65-convex-query-trace-findings-20261111.md).

Canonical evidence is `artifacts/p65-parity-n3-rob1c-col-trace-20261111-01`:
N3 index `2992e4f6e1262f1b2e66745c6f24652b81ed6f14c49ad0e7c8387c1e60e5cf0a`
(12 positive files), ROB index `a0af3e60aa110cc95ff5949a9dd1b199ffbbfbd11f3e690fa4d9cfe20f9ab979`
(89 saved files including original failures), COL index `fc683830e6777aee9a4fa99b31706e41c031982e42f97f1aaee698e546a17abb`
(42 saved files). All functional execution input before/after bindings are unchanged.
No producer, integrator or numerical audit was rerun to make these file indices.
The original first import-sort/format check failures and run14 pre-execution Node path-parser failure remain in the same root; N3 functional validation passed on its first complete run.
ROB harness fixes preserve raw failures: normalized PD units, the original fixed-affine binding constructor, physical rather than cross-owner checkpoint equality, and both wrong-world constructor/restore boundaries.
The actual R03 FP32 PD cancellation precision failure is retained, not hidden by those harness repairs.

Next: N4 freezes multiple tasks/seeds/learning and scale measurements; ROB1d packages the qualified public paths and keeps R03's saved-input arithmetic investigation separate; COL tests a generic affine-simplex reconstruction hypothesis before separately addressing the outgoing normal stop condition. No larger iteration budget, new threshold, fixture tuning or G1 long rerun follows from this batch.

## Original frozen batch contract

| Lane | Concrete next capability | Success evidence | Difficulty / stop boundary |
| --- | --- | --- | --- |
| N3 | Versioned update-boundary PPO session checkpoint | Two original W8 tasks: save after update 2, continue update 3, actually restore same instance and reproduce the successor; GRU nonzero hidden and partial reset; corrupt/stale/mid-update/late-failure cases | High. Same device/config only. No arbitrary learner serialization or cross-platform bitwise claim. CAS-after-refresh failure poisons the session instead of claiming rollback. |
| ROB1c | R02 arm/free-box, R03 arms/tool, independently qualified R04 graph families | Original full M/J/b, full physical loads/reactions, real incoming velocities/control, separation/reentry, W1/W2/W4 and whole-world restore/faults | High. Reuse existing four storage families and integrators; HOST patch and <=33 physical bodies/world remain. No cross-tree unknown-force QP. |
| COL trace | Explicit bounded failed-iteration recording | Same four Y source failures and two CUDA HN failures, plus trace/source/capacity/publication protocol tests | Medium. Diagnose actual simplex/support/weights/termination before selecting a repair. No threshold/cap/fixture changes or full-matrix rerun just to collect scratch. |

N3 owns new `mujoco_tyx_checkpoint.py`, checkpoint tests/report and restore example, with necessary public composition methods in its existing env/adapter. Parent owns exports and any backend seam. The proposed `MuJoCoTyxPpoSession` wraps actual RSL 5.5.2 PPO; it does not reimplement optimization. Save requires a completed update, cleared rollout/transition and no pending asynchronous work. It binds physics/controls, episode/observation/action state, learner/Adam/normalizer, global and explicitly registered RNG, real RNN attributes and next iteration. Candidate construction must undo its temporary RNG consumption on every pre-CAS exit. Publish a new hash-checked no-pickle bundle by directory rename; existing destinations are not overwritten.

ROB author owns participant-facing scene/contact/layout/kernel additions, new robot/restore tests, example and report. Parent extends `_resident_batch_transaction.py`, original read helpers and `TyxWorldBatch` narrowly. Entry/candidate `body_twist` is actual device full-body COM velocity. Entry M/J/b and projected contact load retain full physical coordinates for affine families; original reduced/constraint diagnostics remain separate. All storages solve candidates first; an internal post-stage hook gathers per-row status to canonical mechanical slots, reduces by logical world, scatters only rejection into originally successful rows, then finalizes each storage once. No bank is visible before all publication construction/checks complete. A restore fault seam occurs after real upload/check and before the original swap.

COL author owns `_device_convex_query.py`, `_tyx_convex_query_kernels.py`, an optional new trace schema module, dedicated trace tests/runner/report. Explicit `query_with_diagnostics(pairs=...)` selects 1–8 canonical pairs and returns its normal query plus immutable trace; default query/result semantics stay unchanged. At most the original 64 GJK iterations are recorded: actual pre/post simplex, material IDs/weights, support direction/points, subset decisions and termination values. Additional diagnostic-only quantities cannot drive physics branches. Temporary device arrays, clear launches, transfer bytes and limits are accounted; late trace failure preserves the old query/trace publication. Curved patch/load remain unsupported.

All authors are CODE_AND_TEST_AUTHORING_ONLY until parent handoff review. Tests and literals precede implementation execution. Parent performs fresh entrypoint-aware guards before every import/static/test/build, keeps first failures and freezes source snapshots. New source or unresolved concerns justify affected reruns; unchanged green suites and finished evidence indices are reused. Independent lanes continue when one reaches its two-attempt diagnostic stop.

References: [training N3](../superpowers/plans/2026-11-11-mujoco-tyx-gpu-training.md), [ROB1 contract](../superpowers/plans/2026-11-11-resident-multi-participant.md), [COL remaining failures](p65-device-convex-support-family-report.md), [current panorama](phymeo-newton-panorama-2026-11-11.md).
