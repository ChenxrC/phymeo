# P6.5 S1 cloth self-contact infrastructure

Status: **S1_FOCUSED_VERIFIED** on source commit
`9a26fc7cf28ca0e6e900107adb0608de43eed25c`: 80 complete focused tests passed,
Ruff lint passed, and all five Python files passed the formatting check. This is
host NumPy S1 infrastructure verification, not complete self-collision or P6.5
acceptance. The original checkpoint `8d6a4036e92a7b5bfefd126eb264fb1b1ad25c45`
was authored under the human's parallel-code/serial-execution instruction and
correctly reported **CODE_READY_TESTS_PENDING**. Execution began only after the
parent explicitly granted the sole local slot. There is no historical TDD RED
claim. Raw first-run and final evidence are recorded below.
## Source ownership

- Frozen base: `1ccab18635a3b89c09a0c2e3f735bca2f823e65d`.
- Worktree: `C:/Users/38347/.codex/worktrees/p65-self-contact-module/phymeo`.
- Branch: `codex/p65-self-contact-module`.
- Contract: `docs/superpowers/specs/2026-10-30-p65-parallel-cloth-modules-design.md`,
  S1/shared contracts/parent rulings; matching implementation plan Task S1.
- Changes are limited to three new production modules, two new test modules, and
  this report. No shared source, exports, workflow, dependency pin, delivery
  checkout, integration checkout, or existing physics law was changed.

## Interfaces and decisions

`phymeo.constraints.cloth_self_contact.ClothSelfContactTopology(domain,
triangles, half_thickness_m)` owns an oriented face topology. Integer input is
range checked before copying to int64. Faces must have three distinct vertices;
unoriented duplicates, edges with more than two incident faces, and equal
orientations on a shared edge reject. Globally reversed winding is valid.
Half thicknesses use finite, strictly positive FP64 metres. A frozen object and
owned read-only query copies protect receipts even when a caller re-enables
writing on its returned copy. The recorded policy is
`incidence_policy == "exclude_shared_vertex"`.

`phymeo.geometry.cloth_self_candidates.candidate_self_pairs(topology,
start_positions, end_positions=None)` returns only a sorted tuple of `(i,j)`
with `i < j`. All supplied positions must be finite FP64 `(n,3)`; endpoints must
have identical shapes, and topology indices must be in bounds. Static input
uses one endpoint. For affine nodal motion, each axis interval is the union of
all three vertices at both endpoints, expanded by that face's half thickness.
Every endpoint and thickness is converted with `Fraction.from_float`; addition,
subtraction and overlap comparisons are exact for these stored binary values.
There is no epsilon, rounded expansion, normal evaluation, or global cache.

The inclusion proof is only an enclosing-box statement: affine nodal components
stay between endpoint values, and each component of `rho*N` lies within
`[-rho,rho]` wherever the normal exists. Inclusive overlap preserves touching
bounds. O(f**2) enumeration is deterministic. Every shared-vertex face pair is
excluded, even if the faces have no shared edge. Degeneracy anywhere on the path
is not certified safe, and an overlapping box is not certified collision.

`phymeo.coupling.cloth_self_contact.make_self_contact_operator(base, triangles,
weights, offsets, pairs)` returns `SelfContactBinding`. Triangles index public
point rows of the supplied base; pairs index attachment rows, not topology face
numbers. Repeated attachment faces and reversed/identical pair endpoints retain
existing graph semantics. The complete attachment list constructs exactly one
public `TriangleOffsetPointPort`, then one
`MultiDomainPointOperator({base.domain: port}, left, right)` with both endpoints
in that same domain. There is no independent base apply/response implementation,
second state, private runtime access, or time advancement.

The receipt exposes `.operator` and owned read-only `.triangles`, `.weights`,
`.offsets`, `.pairs`, `.world_points`, and `.jacobians`. Weight receipts are the
actual public offset-port weights, including its existing normalization policy;
input arrays are untouched. Geometry and maps remain host FP64 and the wrapped
base retains its existing backend. This adds no device geometry capability.

`binding.node_impulses(impulses)` returns owned read-only FP64 `(n,3)` requested
loads in the supplied base's public point-row order (`n=len(base.positions())`),
in N*s. It first uses the public graph's signed attachment aggregation, then the
recorded transpose Jacobians and compensated shared-node accumulation. It never
calls base response/apply. Nonfinite/invalid impulses and nonrepresentable
products or sums reject. The actual operator owns its existing single base apply,
full shared-node cross terms, candidate validation and candidate audit.

This is a requested-load audit, not a proof that a candidate applied those loads.
Fixed-node reactions remain in the request even when that node has zero response.
For separated endpoints the requested pair moment is
`(xA - xB) cross p`, which need not vanish; only coincident equal/opposite requests
have a generally vanishing pair moment. Independent full-domain kinetic work
must retain the off-diagonal terms. Existing mapping and ledger acceptance laws
are unchanged.

## Independent tests authored before the serial grant

The following are the 30 authored source node IDs. Under the later explicit
parent queue grant, all parametrizations in both complete files were collected
and executed: 80 passed on the original code and 80 passed on the final source.

`tests/test_cloth_self_candidates.py`:

- `test_unequal_thickness_equality_is_inclusive_without_epsilon`
- `test_affine_swept_union_finds_crossing_with_separated_endpoint_boxes`
- `test_negative_and_large_translations_preserve_candidates`
- `test_exact_bounds_do_not_overflow_or_round_a_small_gap_closed`
- `test_folded_strip_excludes_all_shared_vertices_and_retains_nonincident_faces`
- `test_disconnected_overlapping_sheets_and_aabb_false_positive_remain_candidates`
- `test_face_order_and_complete_winding_reversal_have_deterministic_receipts`
- `test_intermediate_degeneracy_is_not_reported_safe`
- `test_topology_owns_read_only_configuration`
- `test_invalid_incidence_is_rejected`
- `test_invalid_topology_inputs`
- `test_invalid_position_inputs`
- `test_endpoint_sizes_must_match_and_topology_type_is_explicit`
- `test_empty_mesh_is_supported_without_bypassing_input_validation`

`tests/test_cloth_self_operator.py`:

- `test_single_base_apply_and_separated_point_actual_moment`
- `test_coincident_opposite_requests_cancel_without_duplicate_state_advance`
- `test_reversed_and_identical_pairs_cancel_before_jacobian_products`
- `test_full_shared_node_off_diagonal_response_and_kinetic_work`
- `test_adjacent_triangles_match_independent_geometric_differences`
- `test_binding_owns_immutable_receipts_and_preserves_input_arrays`
- `test_receipt_records_the_public_ports_actual_normalized_weights`
- `test_late_candidate_rejection_publishes_no_state_and_preserves_snapshot`
- `test_invalid_attachments_are_rejected_before_any_apply`
- `test_invalid_impulses_do_not_change_source`
- `test_nonrepresentable_audit_request_is_rejected_without_applying`
- `test_empty_attachments_and_pairs_preserve_one_base_candidate`
- `test_compensated_pair_accumulation_retains_small_shared_node_request`
- `test_fixed_node_requests_remain_visible_without_claiming_applied_momentum`
- `test_fp32_base_storage_is_rejected`
- `test_underflow_in_audit_products_is_rejected_without_applying`

The independent unit-XY-triangle Jacobian in the tests is written from the
normal's individual height derivatives. A separate adjacent, nonplanar two-face
case uses centered differences of directly evaluated material points. Tests
compare full `(J_A-J_B) M^-1 (J_A-J_B).T`, actual aggregate base requests, moment,
virtual power, kinetic work, snapshot isolation and deliberate late candidate
rejection. The candidate cases use distinct sheets and a folded strip, exact
representable contacts/gaps, swept crossings, and explicit false positives.

The full two-file commands, fresh basetemps, pinned source commits, runtime
versions and raw exits are recorded in the serial-verification section below.

## Mandatory S2 integration still open

S1 is not completed self-collision support or P6.5 acceptance. Remaining work is
full primal triangular-prism geometry with two independently oriented normals
and two independent radicals; an appropriate exact number-field or interval
proof; symmetric SAT/clipping/normal selection; complete continuous certificates;
midstep and endpoint degeneracy/unknown rejection; owned manifold/anchor friction;
and coupled full-step contact response. The older one-quadratic-field or
patch-priority geometry cannot stand in for these proofs.

The shared-any-vertex exclusion also leaves adjacent faces' far-end crossings
outside S1 coverage; the complete capability must address them before folded
cloth acceptance. Parent integration owns the single physical substep, common
material/contact path and history, full kinetic/potential accounting, actual
force/moment/power audit and atomic commit/rollback. Native response/geometry,
independent scenarios, backend parity, step-halving, duration/repeat/mesh/
perturbation/failure gates, finite-grasp load bearing and integrated folded cloth
remain required. None is inferred from this code checkpoint.

## Serial verification after the explicit parent grant

The parent granted S1 the sole local executable-validation slot after B1 released
it. Before execution this worktree was clean at
`8d6a4036e92a7b5bfefd126eb264fb1b1ad25c45`, and the global inventory found zero
Python/pytest/Ruff/uv processes. A second pre-final-run inventory also found zero.
No helper/reviewer, installation, old full suite, profile, native execution,
build, push, remote CI poll, dispatch or rerun occurred. The optional delivery
policy merge was not taken; its intervening changes were only docs/reports and
workflows, with no production/test/dependency changes. The remote-CI deferral was
honored directly and no workflow file was modified in this lane.

Evidence directory (local raw artifacts retained for the parent):

`C:/Users/38347/AppData/Local/Temp/phymeo-p65-s1-8d6a403-20261030-serial`

Existing interpreter:
`C:/Users/38347/Documents/ChatGPT/phymeo/.venv-py311/Scripts/python.exe`.
Versions: CPython 3.11.15, NumPy 2.4.6, pytest 9.1.1, Ruff 0.17.2. tyx is not
installed in this interpreter; no native capability is claimed. Imports of
`phymeo` and all three S1 production modules were asserted to resolve beneath
`C:/Users/38347/.codex/worktrees/p65-self-contact-module/phymeo/src` in both the
original and final environment logs.

Every test invocation explicitly set that worktree `PYTHONPATH`,
`PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, and
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=
VECLIB_MAXIMUM_THREADS=RAYON_NUM_THREADS=1`. Both test runs disabled pytest's
cacheprovider and checked their distinct absolute basetemps absent before use.
All Ruff commands used `--no-cache`; repair and final Ruff runs additionally
set `RAYON_NUM_THREADS=1`.

### Original post-code evidence, preserved without relabeling

Original source: `8d6a4036e92a7b5bfefd126eb264fb1b1ad25c45`.
`01-original-source-manifest.json` records all six input hashes;
`02-import-check.log` records import assertions, packages and raw exit 0.
`03-original-ruff-check.log` records raw exit 1: two I001 import-spacing findings
and E741 for the test constant `O`. `04-original-ruff-format-check.log` records
raw exit 1: five files would be formatted. These failures are retained, not
reported as initial lint/format success.

Complete first test command, from `05-original-pytest.log`:

```text
C:\Users\38347\Documents\ChatGPT\phymeo\.venv-py311\Scripts\python.exe -m pytest tests/test_cloth_self_candidates.py tests/test_cloth_self_operator.py -q -p no:cacheprovider --basetemp C:\Users\38347\AppData\Local\Temp\phymeo-p65-s1-8d6a403-20261030-serial\05-original-pytest-temp
80 passed in 0.40s
RAW_EXIT=0
```

Original input SHA-256 values (raw working-file bytes):

| Input | SHA-256 |
| --- | --- |
| `src/phymeo/constraints/cloth_self_contact.py` | `5c184257a8c78b95297824662fecba10cf6ce46020b43dc54b2f0ee9c9d939cf` |
| `src/phymeo/geometry/cloth_self_candidates.py` | `dbb66a6f3b968bd03ccf6ca41a0538f96fdf0fd22ff1dcb6ea628f733a2745f2` |
| `src/phymeo/coupling/cloth_self_contact.py` | `5d5c56b279115b591baac9c57c8acec38c7f6526989a1c095a21e388ddaee707` |
| `tests/test_cloth_self_candidates.py` | `b1cdb4cdf68910cb281809956a00e1b9b8bf4f0ebd92140f62f558dd361b7cd8` |
| `tests/test_cloth_self_operator.py` | `609760710efb5711efb7e9d2b82c3499a51e182f93acffe9ce8ff92108685162` |
| `docs/development/p65-self-contact-module.md` | `24fcfc98f517789a30284f87e277ca746e65cffc66d559d670060633090cbf42` |

### Narrow cleanup and final same-snapshot evidence

The only cleanup was test constant identifier `O` to `OFFSETS`, import spacing,
and formatting. Commit `9a26fc7cf28ca0e6e900107adb0608de43eed25c` contains only the
two test-file layout/name changes. Raw worktree source bytes also acquired
normalized LF endings; their production Git blobs and ASTs did not change.
`06-repair-record.txt`, `07-import-fix.log`, `08-format-apply.log`, and
`11-cleanup.diff` preserve the repair. `11-focused-ast-equivalence.log` records
all five ASTs equal to the original (permitting only the declared test constant
rename), raw exit 0. No formulas, values, thresholds, test selection, ownership,
units, cross terms, transaction behavior, or unknown-result rules changed.

Final tested source: `9a26fc7cf28ca0e6e900107adb0608de43eed25c`.
`12-final-source-manifest.json` records six input hashes before the final run;
`13-final-import-check.log` again records correct worktree imports and raw exit 0.
`14-final-source-ruff-check.log` records `All checks passed!`, raw exit 0.
`15-final-source-ruff-format-check.log` records `5 files already formatted`,
raw exit 0. Both commands explicitly name exactly the three S1 source files
and two test files; their exact commands remain in the raw logs.

Complete final command, from `16-final-pytest.log`:

```text
C:\Users\38347\Documents\ChatGPT\phymeo\.venv-py311\Scripts\python.exe -m pytest tests/test_cloth_self_candidates.py tests/test_cloth_self_operator.py -q -p no:cacheprovider --basetemp C:\Users\38347\AppData\Local\Temp\phymeo-p65-s1-8d6a403-20261030-serial\16-final-pytest-temp
80 passed in 0.39s
RAW_EXIT=0
ALL_SIX_TESTED_INPUT_HASHES_UNCHANGED=true
```

Final tested input SHA-256 values:

| Input | SHA-256 |
| --- | --- |
| `src/phymeo/constraints/cloth_self_contact.py` | `5b685dfd0b30ff3a871189d1cb38393db78b9d84f3378b112bc1c0aacbeea72b` |
| `src/phymeo/geometry/cloth_self_candidates.py` | `f15474020c709ae4eaa2a050d7bfa490510978a8541f5de70948a754cc075e67` |
| `src/phymeo/coupling/cloth_self_contact.py` | `660a14c65ec3d6305478105762a0c2093120fc7d666c2f0075b038502e5f395d` |
| `tests/test_cloth_self_candidates.py` | `e43f1c555536686170e8d5e26739c48453ea8666fa94dcf8d52413c6b71d27f6` |
| `tests/test_cloth_self_operator.py` | `6406237d6ac29ca148cf942b4fe986c0370fd05e7305658d8cadca76ff25acab` |
| `docs/development/p65-self-contact-module.md` | `24fcfc98f517789a30284f87e277ca746e65cffc66d559d670060633090cbf42` |

All six hashes matched again after the final test process exited naturally.
Only this report is edited afterward to record the evidence; all five tested
Python files remain byte-for-byte unchanged. The report-only commit SHA, final
report hash, clean Git state and fresh process inventory are recorded in the
parent handoff and `17-final-release.json`. This host S1 evidence does not close
any mandatory S2 integration gap listed above.
