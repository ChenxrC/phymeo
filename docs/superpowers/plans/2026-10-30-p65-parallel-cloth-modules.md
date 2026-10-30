# P6.5 Parallel Cloth Modules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. User explicitly chose parallel code development and a single serial test queue.

**Goal:** Develop bending, self-contact infrastructure and finite gripper modules concurrently, then integrate and verify the complete physical capabilities serially.
**Architecture:** Per-lane managed worktrees, additive module ownership, one parent integration owner, one execution queue. Current delivery branch and its source manifests remain untouched.
**Tech Stack:** Existing Python/NumPy FP64, Fraction geometry, current tyx pin; no new dependencies.
**Spec:** ../specs/2026-10-30-p65-parallel-cloth-modules-design.md

## Global Constraints

No tests, numerical imports/probes, compilation, profiling or packaging without a queue grant. Code and independent tests may be written before execution as expressly requested by user; statuses must distinguish written/unexecuted/passed. No private state writes, pins/welds, scenario-specific physics, old shared module edits, workflow/pin changes or push by children. One candidate for each actual state domain; parent owns shared integration and audits. Current Task3 numerical/index owner must not be interrupted.

## Review Focus

- Signed-angle orientation/branch and gradient signs: B1 flat/right-angle/rigid-transform/finite-difference tests.
- Partial sweep mutation on a later failure: B1 two-hinge transaction test.
- Broad-phase AABB wrongly advertised as safe CCD: S1 crossing and false-positive controls; S2 proof remains explicit.
- Self-contact double application and lost cross terms: S1 real common base operator response/apply/count/kinetic-work controls.
- Builder or prescribed trajectory mistaken for real grasp: G1 only model/input tests; G2 finite-effort hold/slip and integrated folded acceptance separately required.

## Task B1: bending module code

Files/interfaces exactly spec B1. Read existing distance.py and distance_trace.py only as interface conventions; do not alter their laws.
- [x] Record actual worktree/base/allowed files and deferred test policy in lane report.
- [x] Author independent tests and formulas with the prescribed oriented hinge and analytic gradient.
- [x] Implement immutable constraint data, angle/gradient/energy, transactional one-sweep projection and complete immutable trace.
- [x] Self-read branch/degeneracy/error/ownership paths without executing code.
- [x] Commit allowlisted files, label CODE_READY_TESTS_PENDING, report unresolved issues; no automatic test/build/CI.
- [ ] After explicit queue grant only: original-baseline missing-capability evidence if still meaningful, then current tests. Do not label post-implementation evidence as historical TDD RED. Parent audits and schedules B2 native/composite integration.


## Task S1: self-contact module code

Files/interfaces exactly spec S1. Reuse existing public TriangleOffsetPointPort and MultiDomainPointOperator over one physical base.
- [x] Record worktree/base/allowed files and deferred test policy.
- [x] Author incidence, exact swept bounds, false-positive and shared-node/load/candidate tests.
- [x] Implement immutable topology, exact-rational conservative candidate filtering, and public same-domain offset attachment composition.
- [x] Self-read preservation of affine-path bound and distinction between candidate and collision certification.
- [x] Commit allowlisted files with CODE_READY_TESTS_PENDING and report S2 proof gaps explicitly.
- [ ] Run only with parent queue grant, then freeze S2 multi-normal geometry/CCD/response contract; no silent reuse of one-Q field for two independent roots.


## Task G1: finite jaw module code

Files/interfaces exactly spec G1; implemented by parent while coding slots are occupied, or a third lane after a slot frees.
- [x] Implement 3-prismatic massive carriage/two-finger RobotModel builder and finite patches.
- [x] Author independent FK/input-limit/ownership/invalid-configuration tests.
- [x] Commit as CODE_READY_TESTS_PENDING; no simulation or builder-as-grasp claim.
- [ ] Queue independent tests then write G2 actual hold/slip acceptance from independent force/energy equations. Full folding depends on B2/S2; simple no-self-contact finite hold can precede them.


## Task I: one integrated execution and delivery queue

- [x] Observe current Task3 interpret/CUDA completion and explicit resource/index release; do not duplicate its already completed validation.
- [ ] Parent original Task4 evidence/build/outsidewheel/newhead exact CI remains first delivery gate.
- [ ] Merge reviewed code checkpoints only into the isolated integration branch; no existing source/version evidence silently relabeled.
- [ ] Grant one queued B1/S1/G1 test group at a time with immutable hash, fresh basetemp/rawexit, revoke after natural completion. Dependency installation also uses queue.
- [ ] Freeze B2 composite material/native trace, S2 complete contact/CCD and G2 real load-bearing contracts with their own independent cases, using reviewed B1/S1/G1 APIs.
- [ ] Code independent portions concurrently, shared seams by parent only. All tests and physical execution serial.
- [ ] Run integrated current-source hold/slip/fold/transaction and backend gates, one fresh review of the new integration, packaging and accurate full CI. H2/P6.5 remain open until their entire acceptance is met.
