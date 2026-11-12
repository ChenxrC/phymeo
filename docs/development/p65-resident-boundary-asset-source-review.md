# M1-A asset source and installed-consumer review

Status: **TEXT_REVIEW_COMPLETE; no remaining blocking finding in the reviewed increment**. This record is not execution acceptance. The reviewer performed only source/file/Git reads and this document commit; no project import, collection, numeric probe, tests, static tool, build or installation.

## Reviewed snapshots and responsibility

- Asset author tree: `p65-h1-actuator/phymeo`, contract `p65-boundary-asset-contract.md`; admission module `d3de4cd1ac0f821b58c52dde5781378370b6af41`; complete new reference/test source through `2c48960a9eaa04d2e2f897f5a302a3f6d9bb2c06`.
- Parent WIP reviewed directly: `_resident_boundary_source.py`, `_resident_tree_binding.py`, `_resident_floating_binding.py`, specifically the original-source admission branch and the two factory shape-set changes. No parent file was edited or copied into this tree.
- Installed consumer `8bfab63123ea5d2f4a5feb8b754a414da6d2269f` was read completely, followed only by its fixing increments `3372e5f20709d57ffbdacc131ff3d07402cff828` and `49e60d4ac8e3725a174a90d3e4e763c44e5415d3`. Author-reported final SHA256: `fcd695d956bb52c49f55033e8b183c47f5322a67dbd486ecd2f074603f0d7d83`; this review did not duplicate the author's byte audit.
- Necessary existing definitions were read for the four binding/spec families, folded authored frames, physical storage ownership, host array accounting, BVH raw fields, and the original independent finite-geometry test helper. The existing numerical geometry and physics algorithms were not re-reviewed.

## Product conclusions

No concrete blocking defect was found in the limited product increment.

1. The original shape is compared as a complete `CollisionShape` after only the declared key remapping. Point IDs, all facets, revision, radius and filter fields are retained. Original model identity and affine graph identity are checked independently of the remapped collision body ID.
2. Each of the four exact binding types must match its physical/owning spec type. For an affine family, `owning_spec.physical is physical_spec` binds the actual full-body storage; the affine storage constructors pass those same physical spec objects to their underlying storage. Full physical group indices are retained rather than interpreting reduced coordinates as bodies.
3. The folded asset classes declare collider points in authored link frames. The existing two factories perform the authored-site transform once. The new admission copies the already transformed immutable FP64 points without applying another frame transform; the parent registration then uses its original local-point packing and accepted device frame path.
4. Direct and static installations do not enter the new adapter. Their definition fields and original topology/static-anchor paths remain unchanged. The factory allows mesh/triangle provenance, but `_FINITE_SHAPES` and the separate convex compilers retain their original restrictions. A mesh is not relabeled as a convex hull.
5. Admission runs before the first M1 owned allocation. Material/body aliases remain rejected by registration. Source provenance includes original shape/model/graph, physical topology and immutable local points; the existing recursive `_constant_bytes` accounting includes the additional ndarray storage. This is HOST array accounting, not a claim about Python object overhead or total process/GPU peak memory.

## Findings and closure

All three findings concern the new installed consumer, not a demonstrated product numerical failure.

- **P1, incorrect BVH stop expectation, closed by `49e60d4a`.** Original consumer line 304 compared `bvh_raw[0,3]` with `1` and called it a validity flag. `TreeBuffers.hits_device` in `_tyx_collision_bvh.py:103-106` verifies this final column against the root's escape/stop index. The original independent M1 test helper also checks `2 * facet_count - 1`. The consumer now expects `2 * len(FACES) - 1`, which is 7 for its four-facet full tree, and names the field correctly. No kernel or tolerance changed.
- **P2, private self-comparison in a public consumer, closed by `3372e5f2`.** Original lines 281/283 used binding private `_source_shape` and `_model_token` as expected values. The consumer now takes the original public `source_asset.collisions` and every public physical model field in the documented token order; it also compares physical topology with the accepted public checkpoint. It no longer relies on those private binding fields as its provenance oracle.
- **P2, missing physical attempt/return on a failing capture, closed by `3372e5f2`.** Original lines 382-393 performed the physical step and post-step snapshot before writing any step record. A step exception or a snapshot exception after successful return could omit that attempt or accepted result. The consumer now records the full pre-step attempt, records the real return and counts immediately, then independently captures post-state/statistics. Original step and evidence failures are both retained. Query outcomes and cleanup continue to preserve primary failures.

The complete correcting diffs were read against the original schema and field writers. No fixture, physical gate, candidate capacity, algorithm or nominal three-step trajectory changed.

## Evidence and qualification boundary

The source tests have predeclared ordinary, affine, floating and floating-affine cases, plain/fixed-tool/triangle material, sparse reordered worlds, original MJCF capsule material, model/graph/source rejection, capacity/memory/publication faults and filter-table controls. Expected moving material comes from the authored XML/OBJ literals and accepted public coordinates; the original finite triangle/segment reference is reused rather than calling product FK/admission/search as the oracle. Factory/source-constructor/registration failures are distinguished explicitly, including the original floating-affine graph guard order.

The installed consumer covers ordinary fixed-root original URDF/OBJ material, two sparse worlds, three actual advancing steps, all four tetrahedral facets, independent point-triangle distances/witnesses and stale packet rejection. It saves the full source/raw query and transfer/resource counters, validates installed module provenance, and closes plan before batch while retaining cleanup errors. Its intended six accepted world-steps and 24 rows are expectations, not observed results in this review.

This review does not establish actual interpret/CUDA passes, four-family installed qualification, contact forces, solid inside/outside, CCD, deformable source banks, or ParticipantScene-to-M1 accepted-bank composition. Private-field forgery is outside the frozen immutable asset contract. Parent remains the sole execution and integration owner; no further review expansion or rerun is requested here.
