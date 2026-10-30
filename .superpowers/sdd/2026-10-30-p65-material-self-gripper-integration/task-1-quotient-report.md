# Task1 exact contact quotient prerequisite report

Status: **CODE_READY_TESTS_PENDING**. Original B2 owner only. No executable slot
was taken; S2 retains its queue through saved audits and formal release.

## Contract, scope and sequence

Read parent contract072f818866e32bbcff37952fe4643ff26a54846b and its exact spec,
plan and brief from the absolute integration checkout. No parent merge. Starting
child HEAD43b7f846ef8cf85080ed83fb752efbfa27064d63 was clean and all four new files
were absent. Worktree is
`C:\Users\38347\.codex\worktrees\p65-bending-module\phymeo`, branch
`codex/p65-bending-module`.

Only these NEW paths belong to this task:

1. src/phymeo/validation/contact_quotient.py
2. tests/test_contact_quotient.py
3. docs/development/p65-contact-quotient-report.md
4. .superpowers/sdd/2026-10-30-p65-material-self-gripper-integration/task-1-quotient-report.md

Task1 test authoring checkpoint:
`bc8ba7be57bbb420e52e126396f4936cd2826505`. Tests were authored before source and
explicitly marked UNEXECUTED. No observed RED exists.

Task2 source checkpoint:
`ac1b59b05d236e9322c656f7798f15ffd8f2e71b`. Standalone module and added independent
cross-term/mixed-nullspace/fresh-budget controls, still UNEXECUTED.

Task3 documentation checkpoint: this report plus the lane report; exact final
commit/report hashes and clean status are supplied in the final handoff. There
is no self-referential report hash inside either report.

Ruling: the explicit code/test-authoring-only grant governs generic skill steps.
No test/import/static execution, helper/reviewer, task-start script, extra ledger,
remote action or old/shared edit was performed. No mathematical contract change
was needed. Field names and canonical hash encoding are routine implementation
choices permitted by the frozen spec.

## Exact implemented interface

Public names: QuotientUnresolved(ValueError), frozen QuotientProofLimits, frozen
deep-owned ContactQuotientCertificate, certify_contact_quotient and
verify_contact_quotient. Producer/verifier use the exact positional/keyword
input contract; verifier has certificate first and returns True only after full
independent checking. No public package export or solver wiring was added.

Certificate fields:
`owner_ids`, `coordinate_ids`, `difference_map`, `inverse_mass`,
`delivered_impulse`, `impulse_uncertainty`, `limits`, `input_fingerprint`, `rank`,
`nullspace_basis`, `response_matrix`, `nullspace_projector`, `response_scale`,
`gauged_response`, `gauged_inverse`, `pressure_map`, `pressure`, `remainder`,
`pressure_uncertainty`, `remainder_uncertainty`, `condition_number`, `schema`,
`status`. Schema is exact-contact-quotient-v1; status exact_algebra_certified is
ONLY algebraic. A raw dataclass constructor does not certify its contents.

Stdlib only. Owned Fraction tuples; strict finite built-in list/tuple and
int/Fraction inputs; no bool/float/generator/array coercion. ID/shape checks occur
before scalar copies. Limits192/256/4096/2,000,000 are default AND hard ceilings;
caller policies may only decrease them. Every rational binary operation and
proof check is metered, with bit checks on normalized scalar results. Positive
LDL of full W/H, strict kappa_inf(H)<10^12, complete exact rank/nullspace and all
cross terms are retained. Zero rank explicitly rejects.

The verifier independently checks external identity and all algebra, including
complete alternate nullspace bases; it never calls the producer. Producer final
identity checking shares its construction counter. Every public verification
gets a fresh counter. Failure cannot publish a partial certificate. Fingerprints
use framed UTF-8 tokens with exact numerator/denominator values, shapes, ordered
IDs, fixed domain/units/schema/status and complete policy.

## Hashes and authored controls

Source SHA256:
`3ad9bc3c2ec91b41aeafd7cdf82881d87815813775104e04b38d4b4cd54fd336`

Test SHA256:
`da0d85341fea55970a23a4ed861372e4f40d2c2af3c0765403086d9f07d05fa8`

The lane report `docs/development/p65-contact-quotient-report.md` gives the complete
28-function inventory with83 statically enumerated authored parameter cases.
These are NOT collected test counts. Node bases are the new test file plus those
function names; no test, branch coverage or numeric outcome has been observed.

Coverage authored: explicit diagonal, duplicate, opposed, off-range and full-W
cross-term rational oracles; exact condition boundary; scaling/permutations;
scaled/mixed complete bases; producer independence and fresh verifier counters;
19 forged certificate controls and7 external bindings; valid-shaped incomplete
nullspace; invalid shapes/scalars/IDs/policies/mass/zero rank; input/intermediate
bit and operation exhaustion; immutable ownership and equivalent fingerprints.
Expected algebra is handwritten; no producer output generates expected values.

## Verification and remaining gates

Performed ONLY: file IO, source-text inspection, Git whitespace/scope checks and
pure file hashing. Git diff --check had no whitespace findings. This does not
establish syntax, import, lint, format, collection, test or numerical correctness.
All such checks are pending. No B1/B2/R7/S2/G2/adapter historical pass is acceptance
of this new code. First future outcomes must be preserved honestly.

Parent must issue the separate sole serial validation grant after S2 releases.
The planned future group is child source/import provenance, scoped Ruff/format,
and the complete new host file with fresh absolute basetemp. No native/C++ or old
physical suite is implied for this unconnected exact stdlib module.

Complete B2 physical support is still NOT_FROZEN. This certificate cannot accept
contact, supply cone/phase capacity, modify state or requests, claim unique
physical pressure, remove residuals, or credit external work. Two ordinary-mass
nonlinear fidelity controls, ideal extrema, robot predictor provenance and full
energy/angular accounting remain unresolved. S2 R1/R2, the proposal, old schemas,
R7/native evidence and all physical budgets remain unchanged.

Authoring-only release: CODE_READY_TESTS_PENDING. No executable process was
started by this task; no execution release or process0 claim is manufactured.