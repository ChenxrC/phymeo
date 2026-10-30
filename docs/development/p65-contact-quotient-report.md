# Exact contact quotient: authoring handoff

Status: **CODE_READY_TESTS_PENDING**. This is an unconnected standalone algebra
prerequisite, not accepted B2 contact physics. No imports, collection, numerical
or offline probes, pytest, lint/formatter, compile/build/install/profile, native
execution, helpers/reviewers, parent merge, push or remote action ran. S2 retains
the execution slot; this task took none. No observed RED or passing result exists.

## Authority and checkpoints

Parent contract072f818866e32bbcff37952fe4643ff26a54846b:
`docs/superpowers/specs/2026-10-31-p65-contact-quotient-design.md`, matching plan,
and `task-1-quotient-brief.md`, read from the absolute parent integration tree.
Starting child HEAD43b7f846ef8cf85080ed83fb752efbfa27064d63 was clean; all four
allowed paths were absent. The existing proposal/history and old source stay intact.

- `bc8ba7be57bbb420e52e126396f4936cd2826505`: independent tests authored FIRST,
  explicitly UNEXECUTED. This is the authorized code-first schedule, not TDD RED.
- `ac1b59b05d236e9322c656f7798f15ffd8f2e71b`: production source and further
  independent nullspace/uncertainty/budget controls; all remain UNEXECUTED.
- This report and the SDD report form the final documentation checkpoint; its
  commit and final four working-file hashes are supplied in the handoff.

Worktree: `C:\Users\38347\.codex\worktrees\p65-bending-module\phymeo`.
Branch: `codex/p65-bending-module`.

## Exact four-path scope and hashes

| New path | Role / SHA256 at implementation checkpoint |
| --- | --- |
| src/phymeo/validation/contact_quotient.py | Stdlib standalone evidence; `3ad9bc3c2ec91b41aeafd7cdf82881d87815813775104e04b38d4b4cd54fd336` |
| tests/test_contact_quotient.py | Independent unexecuted oracles; `da0d85341fea55970a23a4ed861372e4f40d2c2af3c0765403086d9f07d05fa8` |
| docs/development/p65-contact-quotient-report.md | This authoring report |
| .superpowers/sdd/2026-10-30-p65-material-self-gripper-integration/task-1-quotient-report.md | SDD handoff |

Final report-file hashes live in the final handoff rather than self-referential
report bodies. No old source, tests, package exports, reports, proposals, solver,
constraints, runtime, kernel or dependencies were changed.

## Implemented API and evidence fields

Public names are `QuotientUnresolved(ValueError)`, frozen `QuotientProofLimits`,
frozen deep-owned `ContactQuotientCertificate`, `certify_contact_quotient` and
`verify_contact_quotient`. Function argument names/order follow the frozen spec:
`difference_map, inverse_mass, delivered_impulse, impulse_uncertainty`, keyword
`owner_ids, coordinate_ids, limits`; verifier additionally takes certificate first.
The verifier returns True only after checking all evidence, otherwise it raises
QuotientUnresolved. Construction of the dataclass alone is not verification.

Certificate field spellings:

| Fields | Bound meaning |
| --- | --- |
| owner_ids, coordinate_ids | Ordered unique UTF-8 identifiers |
| difference_map, inverse_mass | Exact owned B and full W |
| delivered_impulse, impulse_uncertainty | Exact ell and nonnegative e |
| limits, input_fingerprint | Owned policy and canonical external binding |
| rank, nullspace_basis | Exact rank(B) and complete N in ker(B.T) |
| response_matrix, nullspace_projector | A=B W B.T and Pi_N |
| response_scale, gauged_response, gauged_inverse | s, H and H^-1 |
| pressure_map, pressure, remainder | L, p_eff and unchanged r_impulse |
| pressure_uncertainty, remainder_uncertainty | abs(L)e and abs(I-B.T L)e |
| condition_number | Exact kappa_inf(H), strictly below10^12 |
| schema, status | exact-contact-quotient-v1 / exact_algebra_certified |

Fingerprint uses SHA256 over length-framed UTF-8 tokens: schema/status, fixed
translational/mass/units domain, all policy fields, ordered ID arrays, matrix/vector
shapes and normalized numerator/denominator pairs. List versus tuple and equivalent
rational spellings bind identically. No floating JSON conversion or input coercion.

## Exact arithmetic, ownership and independent verification

Only exact built-in int/Fraction in finite built-in list/tuple containers enters.
Bool, float, generators, implicit arrays/coercions, malformed/nonempty shapes,
nonunique/empty/over256-byte IDs and negative e reject. ID and shape checks precede
scalar copying. Certificate input and derived matrices/vectors are owned Fraction
tuples; the frozen policy is copied too. A verifier re-owns and bounds submitted
certificate fields, including a possibly corrupted frozen policy.

The fixed hard/default limits are192 owner rows,256 coordinates,4096 fraction
bits and2,000,000 rational scalar operations; only smaller positive int policies
are allowed. Each add/subtract/multiply/divide uses `_Budget`, including elimination,
dot products, LDL, norms, propagation and all identity checks. Each normalized
input and scalar-operation rational result is bit-checked. Negation, absolute
value, comparisons, index/shape bookkeeping and hashing are not scalar operations.
Fraction's internal integer normalization is not a separate rational operation.

The producer uses deterministic first-nonzero RREF to establish rank and complete
nullspace, an exact Gram inverse for Pi_N, exact positive LDL for W/H, and a fixed
strict H condition ceiling. Zero rank rejects, even when unloaded. Full W cross
terms and all duplicate/opposed owners remain. No epsilon, regularization,
self-stress fit, pressure clipping or additional W condition ceiling is introduced.

The independent identity checker re-establishes external binding, exact rank and
complete nullspace span, positive W/H, A/Pi_N/H/inverse/L, pressure/remainder,
uncertainty, projector/mass-adjoint identities and the quadratic energy equality.
It characterizes Pi_N by action on complete N and range(B), allowing scaled/mixed
complete alternative bases rather than comparing against producer basis vectors.
It never calls the producer. The producer also runs that checker before return,
within the SAME construction budget; public verifier uses a fresh full counter.
No partial certificate is returned if either computation or checking exhausts.

Routine implementation choices stayed within the frozen contract: explicit field
names above, length-framed hash encoding, strict built-in container/scalar types,
and sharing the independent identity checker with final producer checking. No
mathematical condition or physical threshold was changed.

## Authored test inventory — NOT collection or execution

File: `tests/test_contact_quotient.py`. Source text contains28 authored functions.
The83 statically enumerated cases below are an authoring inventory only, not an
observed pytest collection count. No test outcome or code coverage is claimed.
Each row's node base is `tests/test_contact_quotient.py::` plus the name shown.

| Authored test function | Parameters / authored cases |
| --- | --- |
| test_full_rank_diagonal_independent_oracle | 1; I3, diag(1,2,4), exact errors/kappa4 |
| test_duplicate_rows_preserve_both_owners_and_exact_projector | 1; exact half projector |
| test_opposed_rows_do_not_fit_positive_self_stress | 1; negative component retained, no proposal API |
| test_off_range_remainder_and_uncertainty_are_retained | 1; residual half-energy9/2 |
| test_full_inverse_mass_cross_terms_and_quadratic_identity | 1;8=2+6 and nonzero propagated errors |
| test_strict_condition_ceiling_rejects | 2; equality, above |
| test_condition_immediately_below_ceiling_succeeds | 1;10^12-1 |
| test_uniform_inverse_mass_scaling_preserves_quotient | 2; fractional, integer |
| test_nontrivial_owner_and_coordinate_permutations_with_ids | 1 |
| test_verifier_accepts_complete_scaled_noncanonical_nullspace | 1 |
| test_verifier_is_independent_of_producer | 1; producer replaced by forbidden call |
| test_forged_certificate_rejects | 19; rank, incomplete/wrong N, A, projector, scale, H, inverse, L, p, r, both errors, kappa, fingerprint, both IDs, schema, physical status |
| test_external_binding_changes_reject | 7; B, W, ell, e, owner, coordinate, policy |
| test_nonexact_scalar_inputs_reject | 7; True, False, float, NaN, infinity, string, None |
| test_invalid_shapes_mass_uncertainty_and_zero_rank_reject | 10; empty, ragged, mass/ell/e shapes, negative e, asymmetric/singular/negative W, zero B |
| test_generators_and_implicit_conversion_are_not_consumed | 1 |
| test_invalid_owner_identity_rejects | 6; duplicate, empty, shape, type, ASCII and UTF-8 byte limits |
| test_coordinate_ids_and_utf8_boundary_are_bound | 1 |
| test_invalid_or_expanded_resource_policy_rejects | 8; zero/negative/bool/float and four ceilings |
| test_shape_limits_precede_scalar_copying | 1 |
| test_input_and_intermediate_fraction_bit_exhaustion_reject | 1; input, A=27, forged N |
| test_scalar_operation_exhaustion_is_fail_closed | 1 |
| test_certificate_and_policy_deep_ownership_and_frozen_fields | 1 |
| test_equivalent_fraction_spellings_have_identical_binding | 1 |
| test_incomplete_nullspace_with_valid_forged_shapes_rejects | 1; false positive rank with shaped incomplete N |
| test_verifier_accepts_mixed_complete_nullspace_basis | 1; nullity2, noncanonical mixing |
| test_verifier_has_a_fresh_metered_budget_each_time | 1; separate counters observed by authored spy |
| test_nonpolicy_objects_reject | 3; None, mapping, integer |

## Validation boundary and remaining work

Only file IO, source-text inspection, Git diff whitespace/scope checks and hashing
have run. Git text checks report no whitespace errors; that is not lint, syntax,
import, test, runtime or numerical evidence. No previous B1/B2/R7/S2/G2/adapter
result is reused as acceptance of this new module.

Parent must first audit the authored scope/hashes and wait for S2's formal slot
release, then issue the separate focused grant. That later scope is environment /
actual child import provenance, these two files' Ruff/format and complete new host
test file under a fresh absolute basetemp. Preserve first actual outcomes and fix
only allowed paths without weakening exact mathematics. No old host75, physical,
R7/native/C++ or unrelated suite is required for this unconnected stdlib module.

Limits are a fail-closed domain, not claimed coverage of every allowed matrix
size or mesh. A valid problem/basis may exhaust operations/bits; no performance or
pass claim is made. No pressure uniqueness, nonnegative cone admission, friction
capacity, ideal-extreme/nonlinear held-path proof, robot predictor provenance,
external-work credit or physical publication follows. The two ordinary-mass
nonlinear fidelity controls and all phase/energy/angular integration obligations
remain unresolved. No solver or package export is wired.

Authoring ends CODE_READY_TESTS_PENDING with a clean committed handoff. Execution
remains deferred to a separate sole-slot grant.