# P6.5 S1 cloth self-contact infrastructure

Status: **CODE_READY_TESTS_PENDING**. This is an unverified additive code checkpoint;
no test, project import, numerical probe, compilation, formatter, lint, build,
dependency setup, or CI command has been executed for this lane. The human's
parallel-code/serial-execution instruction takes precedence over ordinary
red/green or pre-commit execution. There is no historical TDD RED or passing claim.

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

## Authored independent tests, not executed

The following are source node IDs (parametrizations expand only when collected).
No collection has been run. The parent must grant the serial queue before using
these test modules or any dependent imports.

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

After an explicit queue grant, a focused initial command is the parent's chosen
Python runner with `-m pytest tests/test_cloth_self_candidates.py
tests/test_cloth_self_operator.py` and a fresh parent-assigned `--basetemp`.
The queue record must pin this lane's eventual commit, backend, runtime/pins,
raw command and raw exit. The command above is a proposal, not execution evidence.

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
