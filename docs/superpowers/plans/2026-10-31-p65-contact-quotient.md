# P6.5 exact contact quotient implementation plan

**Goal:** Deliver a bounded independently verifiable exact redundant-owner impulse decomposition, initially unconnected to physical solvers.

**Architecture:** One standard-library module owns immutable rational input binding, resource checks, exact nullspace/mass algebra, certificate generation and an independent verifier. One focused test file uses independent rational oracles. Physical contact admission and parent shared dispatch remain deferred.

**Tech Stack:** Python standard library fractions/dataclasses/hashlib/json; pytest only in tests.

**Spec:** [Frozen algebra contract](../specs/2026-10-31-p65-contact-quotient-design.md).

**Global Constraints:** Existing original B2 owner only, four NEW paths, code/test authoring only until separate serial grant. S2 retains the sole execution slot. No helpers/reviewers, old-source changes, parent merge, imports/probes/tests/lint/formatter/build/native or remote action. Complete P6.5 remains open.

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

- [ ] Parent audit exact authored four-path diff and saved hashes, then freeze a focused validation grant after S2 formal release.
- [ ] Original owner performs first environment/import, scoped static and complete new host file serially, with original first outcomes retained.
- [ ] Diagnose any failure within allowlist and unchanged mathematics; finalize raw exits/source/evidence/clean/process release.
- [ ] Parent accepts only the standalone algebra scope. Ordinary-mass nonlinear contact success proofs and shared physical integration remain required.

Authoring checkpoint7673f670 was received and text/hash audited by the parent; Tasks1-3 above mean authored completion only. Task4 and all executable results remain pending. The prepared validation selection is not yet activated because S2 retains the sole slot.
