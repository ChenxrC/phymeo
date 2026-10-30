# P6.5 exact contact quotient implementation plan

**Goal:** Deliver a bounded independently verifiable exact redundant-owner impulse decomposition, initially unconnected to physical solvers.

**Architecture:** One standard-library module owns immutable rational input binding, resource checks, exact nullspace/mass algebra, certificate generation and an independent verifier. One focused test file uses independent rational oracles. Physical contact admission and parent shared dispatch remain deferred.

**Tech Stack:** Python standard library fractions/dataclasses/hashlib/json; pytest only in tests.

**Spec:** [Frozen algebra contract](../specs/2026-10-31-p65-contact-quotient-design.md).

**Global Constraints:** Existing original B2 owner only, four NEW paths. Focused host validation and parent acceptance are complete; the execution slot is released. The historical grant authorizes no further run. No helpers/reviewers, old-source changes, parent merge, build/native or remote action. Complete P6.5 remains open.

**Review Focus:** Complete exact nullspace and all owner rows; full positive W; strict H conditioning; bit/operation bounds with no bypass; independent verification of altered evidence; off-range and cross-term behavior; no physical pressure or energy-credit claim.

## Task 1: Author the independent contract tests

- [x] Read spec and original proposal/adjudication from the parent absolute tree.
- [x] Verify original B2 branch is clean at its documentation release; preserve all existing history.
- [x] Create tests/test_contact_quotient.py with all ten control categories and fixed rational oracles. Keep all tests unexecuted.
- [x] Commit test authoring with explicit UNEXECUTED status; no fabricated RED.

## Task 2: Implement the standalone certificate

- [x] Create src/phymeo/validation/contact_quotient.py with exact owned inputs and checked resource policy.
- [x] Implement deterministic exact elimination, complete nullspace and independent W/H positive-LDL proofs.
- [x] Implement generator, full cross-term decomposition, explicit remainder and uncertainty.
- [x] Implement independent verifier with all bindings/identities and mutation rejection. No producer call inside verifier.
- [x] Check source text against allowlist/domain without executing it; commit implementation.

## Task 3: Hand off immutable authored evidence

- [x] Write docs/development/p65-contact-quotient-report.md and current SDD task-1-quotient-report.md, both CODE_READY_TESTS_PENDING.
- [x] Record exact public names/fields, four paths/hashes, authored nodes, resource limits and physical limitations.
- [x] Commit both reports, confirm clean Git and report handoff to parent. Do not take an execution slot.

## Task 4: Parent acceptance after separate serial grant

- [x] Parent audit exact authored four-path diff and saved hashes, then freeze a focused validation grant after S2 formal release.
- [x] Original owner performs first environment/import, scoped static and complete new host file serially, with original first outcomes retained.
- [x] Diagnose any failure within allowlist and unchanged mathematics; finalize raw exits/source/evidence/clean/process release.
- [x] Parent accepts only the standalone algebra scope. Ordinary-mass nonlinear contact success proofs and shared physical integration remain required.

Authoring checkpoint7673f670 was received and text/hash audited by the parent; Tasks1-3 above mean authored completion only. Task4 and all executable results remain pending. The prepared validation selection is not yet activated because S2 retains the sole slot.


The parent has now verified S2's final release and integrated its accepted
response as1396df42e16d057d3ed939b7fa6aafe08f42ea6e. The previously prepared
task-1-quotient-validation-grant.md is activated for the original B2 owner on
clean authoring report7673f670. Task4's first prerequisite is complete; all
quotient executable results and final acceptance remain pending. No parent
merge or old physical suite is part of this focused grant.


Final local algebra acceptance: tested80d2a09/report0b55b831, first host83/0.34s/0, exact formatting-only AST/import equality, parent four-source/53-evidence/eight-exit/six-ZIP checks all matched. Normal merge44456ea902ac08c4ce9eab92f9898d0fdd0c5d36. All Task4 items are complete only for this standalone scope. See [parent acceptance](../../development/p65-contact-quotient-parent-acceptance.md). Execution is released; physical integration remains open.
