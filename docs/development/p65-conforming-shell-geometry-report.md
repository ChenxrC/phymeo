# S2-FG conforming geometry: focused host validation

Status: **FOCUSED_HOST_GEOMETRY_VALIDATED; PARENT_ACCEPTANCE_PENDING**. The complete six-file host selection passed once, 99 tests in 39.52 seconds, RAW_EXIT=0. The same-run saved F1 audit also passed. This is strict geometry and complete director-map validation only, not physical contact, runtime trajectory, native/backend, general fold, or P6.5 acceptance.

## Validation authority and immutable evidence

Original clean authoring checkpoint: a3f80fb478a824f2c6b66c9e15ff194d93f81735. The earlier authoring checkpoints were genuinely UNEXECUTED then; none is retroactively TDD RED/GREEN. Parent's sole serial validation grant is debc1565e5c4109c5972dbb33bc74e77082417f4, grant SHA256 b7c749ce5caf092d1295862607f5713b36ab3d6bef8dbb286d73715649f2251b. The unchanged design SHA256 remains f3e5295a5a553e1b876b1b946a9eb6b465997c9bd58e1c5559798e8ad3a7776d. No parent merge was performed.

Evidence root: C:/Users/38347/AppData/Local/Temp/phymeo-p65-s2-fg-a3f80fb-serial. Original snapshots, parent contract copies, 204-entry initial source/config/report manifest, original dependency file hashes, every command/environment/launch/completion/process/raw-output/raw-exit receipt, each before/after manifest and before-source snapshot are retained there. The initial fourteen SHA256/HEAD blob/filtered working blob checks had zero mismatches; initial 204 copied snapshots also had zero mismatches. No evidence was compressed, removed, overwritten or regenerated through another geometry run.

Two preflight attempts were blocked BEFORE launch: 01 at 09:47:14+08 by foreign Python3.13 PID60136; 02 at 09:48:05+08 by external CRM archive QA PIDs49324/42640. Their guards remain, with no invented raw executable exit. Those processes ended naturally without intervention. Guard observation 05 at 09:51:44+08 was zero matching execution, followed by each command's own immediate guard. Ancillary Node services were recorded separately and untouched; matching execution zero never means machine process zero.

Environment run05 used the existing original .venv-py311/Scripts/python.exe: Python3.11.15, NumPy2.4.6, pytest9.1.1, Ruff0.17.2. PYTHONPATH explicitly names this child/src and all SIX actual module __file__ values were asserted to match it. PYTHONDONTWRITEBYTECODE, PYTEST_DISABLE_PLUGIN_AUTOLOAD, PYTHONNOUSERSITE and all four OMP/OPENBLAS/MKL/NUMEXPR thread caps were1. No producer/verifier was called during this environment check. No installation or pin/backend change occurred.

## Original outcomes and scoped cleanup

The run directory numbers reflect prepared commands; actual temporal order is05,03,04,06,07,08,09,10,11,12. Each command is serial and has its own immutable source/guard receipts.

| Run | Actual outcome | RAW_EXIT |
| --- | --- | --- |
| 05-environment | Six child imports and existing environment asserted | 0 |
| 03-ruff-check | FIRST outcome: five import-block findings plus E731 test-local lambda | 1 |
| 04-ruff-format-check | FIRST outcome: twelve files would reformat | 1 |
| 06-ruff-import-fix | Five scoped import findings fixed | 0 |
| 07-ruff-format | Twelve allowed files formatted | 0 |
| 08-ruff-check | All twelve-file checks passed | 0 |
| 09-ruff-format-check | Twelve files already formatted | 0 |
| 10-style-ast | Explicitly qualified AST/import comparison passed | 0 |
| 11-host-complete | Complete six files, 99 passed in39.52s | 0 |
| 12-saved-f1-audit | Same-run complete payload/coverage/sidecar audit passed | 0 |

The only post-code edits to Python were import ordering, formatting and the single test-local twice lambda rewritten as a def with the identical argument/Interval return expression. This is not raw AST identity: seven files have identical raw AST; the other five differ by four import-order ASTs and that one local lambda-to-def. The saved comparison sorts only contiguous import blocks and explicitly transforms that single lambda; all twelve then match. It makes no independent execution or general semantic-equivalence claim. Original bytes and all first static failures remain available. No oracle, limit, threshold, fixture, input grid or assertion was changed. There was no numerical failure, numerical repair, exploratory collection or repeat of an unchanged green suite.

Cleanup/tested-code commit: **be650fda9254666be20fabdee23e0b772293d23f**. It was clean before the host run. The command was the existing Python interpreter followed by -m pytest tests/test_conforming_shell.py tests/test_shell_intervals.py tests/test_shell_directors.py tests/test_shell_embedding.py tests/test_shell_sweep.py tests/test_shell_certificate_verifier.py -q -x --tb=short -p no:cacheprovider --basetemp C:/Users/38347/AppData/Local/Temp/phymeo-p65-s2-fg-a3f80fb-serial/11-host-complete/basetemp. This absolute basetemp was checked absent immediately before launch. Raw result:99 passed; no skip, deselection or fallback. The 204 source/config/report entries before and after run11 are identical, as are those surrounding audit12. Final report changes are the only later changes; all202 non-report entries remain checked against the tested snapshot in the final receipt.

## Same-run F1 evidence and scope

The saved-file reader imports no Phymeo module and performs no new geometry/physics call. Its authored pairing is fixed first: pytest's h decorator is recorded before variant, yielding h1/64,1/128,1/256 blocks of identity/rotation/translation/permutation. _mk_tmp truncates the function name to30 characters, then appends ordinals0..11. For each ordinal it requires the exact whole and every step filenames of rebuild0 and rebuild1. It rejects absent/extra files, duplicate resolved physical directories, a missing rebuild and mismatched complete bytes. No matching output digest is used to choose membership; no field is removed or normalized.

Audit12 verified12 authored cases,24 independent reconstruction windows,24 whole-T certificates,448 substep certificates,472 complete JSON payloads and472 SHA256 sidecars. Each of236 fixed payload pairs is byte-identical. It independently checks the frozen original reference IDs/order/coordinates/rho/faces, affine endpoint input coordinates and exact durations, every certificate's reference binding identity, complete accepted time spans/tree leaves, six tetra IDs, fifteen pair incidences, both endpoint publication receipts, original operation ceiling and computed component k<33/64. Complete payload byte comparison retains all other geometry/publication/domain/charge/reference/partition fields. The executed tests separately ran the independent verifier on every generated certificate. The complete same-run directory and raw audit output, including every payload digest and byte size, are preserved.

F1 remains the frozen analytic free-fold geometry inputs T1/8, h1/64,1/128,1/256, four transforms and two rebuilds. It is not a runtime replay of free-particle dynamics. F2's complete five-node J,65/32 response, hold/slip impulse, work/reaction/torque assertions passed in the same99-test selection; no physical F2 trajectory artifact is invented. All authored adverse topology/director/incidence/reference/path/publication/partition/resource/binding controls belong to that same full selection. The exact rational nonmidpoint partition and metered unary-negation cases were executed without changing their original limits.

## Exact final tested Python bytes

The two final report SHA256/Git blobs are supplied in final-source-manifest.json and the out-of-band clean handoff; no report attempts a self-referential digest. final-source-parity.json binds the202 unchanged source/config entries to run11. evidence-inventory.json and its SHA256 sidecar cover every retained evidence file other than their own self-referential entries. Final Git/status and fresh execution-zero release are recorded after this report commit in final-release.json; the parent independently audits before accepting/integrating.

| Path | Tested working-file SHA256 | Tested Git blob |
| --- | --- | --- |
| src/phymeo/constraints/conforming_shell.py | ec9337348ea5ac8ca5bde5aaad0fe3707adb6afdfc92c8cb54d1ff32b0b9ce3c | e90d84e11c6a350cc7b9a15b46c55b9d1755cce4 |
| src/phymeo/geometry/_shell_intervals.py | 644a42b69f27159a5c13ed2a0d24afdad54ab064686887c2eeca64c24b331209 | b671b3ea243c52d1626df129caa0fc3b848e931b |
| src/phymeo/geometry/shell_directors.py | 93a625f3fcd53f27051c5c470f430cd5130947118d1a5a7fec559b7c12ca8014 | cf8a1c288a5651742ffb2cf1ccc2460f69a3e8f6 |
| src/phymeo/geometry/shell_embedding.py | ed833268ad8a0de789d3c9e1841d3ff757b41113af183e2f1fe789c8d9bf2fc5 | 460d4d692d3ed0164bd93f1111b989d3002e4edf |
| src/phymeo/geometry/shell_sweep.py | 4e677947c3c3b09f29148e9670d302b0b646cad4c2667717240734b4f3f0f8c4 | 565bbb0027f9c67d14f3055c67b8ee25ed2db9c5 |
| src/phymeo/validation/conforming_shell.py | 372c0bfcf46d83012bd842a02d92371ef2b40b6f200e81e5363f7ceb3b464ad4 | b158975e5bf37759378134e768f933c58a1fd466 |
| tests/test_conforming_shell.py | 82f4ae30d5739c27c3acb1bda2cf65931f9b1e206d1feeeb4ce7a7170799f9d5 | 413a8cef7e978cfdfb9fa4dddc3efa4e13157dc2 |
| tests/test_shell_intervals.py | 6af8a900c0e8cf8bb8713b1b66c1508c80e79a254b535dc4ac5e676b4efc0752 | 4af13b2c4f850c0b8dd35c57ee2d49d9b386d0e7 |
| tests/test_shell_directors.py | 0b21a5eed09e3844ce3382a8785d2a8042f5aa375d190e3def865fdff2d3b9d2 | f5783d181b8f4f1f4645bf4e6ac44a29464079aa |
| tests/test_shell_embedding.py | fc716c10848c9a0980b25b35148e45ff6313260a50770e6a435d025e5778f0fd | 16f81decda33d2f1dd351efdb13d6e2ad7abec17 |
| tests/test_shell_sweep.py | a703d8f0bc8cbaac7ccf7417d7ebeb5b77d225b755fec61f493bffb7eae86a5b | 74835c516bfba61f94140c03576d6f6d79504f26 |
| tests/test_shell_certificate_verifier.py | d2cbd4d715922c56d8792a44f376d42b5ea0f57c4964cdab835d062518bf8006 | eb20f1bfbd304a109ff326768331358a785e65d9 |

Only the original14 NEW paths relative to c35a13a changed. No shared seam, export, old source/test/report, dependency, parent contract or runtime file was edited. No helper/reviewer, remote action, old suite, interpret/CUDA/C++, compiler/build/profile or install was used. Weak remote boundary touching, loaded rotating directors, contact capacity/cones/transactions, B2 ordinary-mass nonlinear integration and robot/fold/grasp/full P6.5 remain open. Positive local tetrahedra alone and blanket incident exclusions are still insufficient; unknown remains unknown.

## Preserved original authoring record (historical)

Everything below records the UNEXECUTED a3f80fb authoring handoff and its original hashes/names. Its code-only status statements apply to that checkpoint, not to the later host results documented above.

# S2-FG conforming geometry: code-only handoff

Status: **CODE_READY_TESTS_PENDING**. This is authoring of geometry and complete director-map prerequisites only. Every import, collection, numerical probe, test, lint, formatter, compilation, build, install, profile, native command and remote action remains UNEXECUTED. No helper/reviewer was created. No execution slot is held. AUTHORING_SLOT_RELEASED at the final clean Git handoff; parent must grant the later sole focused-host slot explicitly.

## Authority and lineage

Started clean c35a13a645bc68e912d390779ec29732ef07b644 on codex/p65-self-contact-module in C:/Users/38347/.codex/worktrees/p65-self-contact-module/phymeo. Read the parent adjudication, geometry design, matching plan and SDD brief by absolute parent paths, without merging. Frozen parent contract checkpoint:825f7ec14e9448518d6d5f3b77082c0a79697e9d. Design SHA256:f3e5295a5a553e1b876b1b946a9eb6b465997c9bd58e1c5559798e8ad3a7776d.

Only the fourteen NEW paths below changed. No old source/test/export/proposal/report/shared seam/runtime/pin or parent plan/spec/ledger was edited. Existing accepted S1/LG/S2-L/B2/G2/adapter/quotient evidence was not repeated.

Authoring checkpoints:

- 01860d2: initial six test files and two allowed reports before any production source existed. These are written expectations, not observed TDD RED/GREEN.
- 8a69058: owned topology, law/address/limits, metered interval/polynomial arithmetic and complete star map.
- befb88b: reference coverage, strict static/sweep producers, same-counter publications and additional explicit covariance/reference controls.
- Final verifier/report checkpoint is identified by the final Git handoff. Supplemental adversarial tests were added during completion, all still unexecuted.

The explicit authoring/execution split overrides generic skill commands to run tests or launch reviewers. Only source reads, manual derivation, document/code file IO and Git were used. Git whitespace/scope checks are not Python/static/test validation. SDD is ignored by repository rules, so only the explicitly authorized SDD report was staged with git add -f. A delayed read-only text-list command completed naturally; it did not run Python or a numerical lane.

## Implemented callable surface and exact schemas

All public producer signatures use keyword limits=ShellProofLimits(). Invalid type/shape/input construction rejects; proof uncertainty raises ShellUnresolved(ValueError), carrying bounded category and detail. There is no partial accepted state or fallback collision answer.

constraints/conforming_shell.py:

- ShellDirectorLaw(kind='raw-area-star-director-v1', projected_ratio_min=Fraction(1,4), shape_policy='invariant-1e-10-v1'). Only this frozen law is admitted.
- ShellProofLimits fields: max_nodes64, max_faces40, max_tetrahedra128, max_star_faces16, max_pairs8192, max_time_cells4096, max_time_depth32, max_fraction_bits4096, max_scalar_operations2000000, initial_sqrt_bits64, max_sqrt_bits4096. Positive built-in ints only; callers can only tighten ceilings and must keep initial<=maximum.
- ShellMaterialAddress(reference_id, tetra_id, counts, generation). Four built-in nonnegative int counts sum exactly2^52; tuple ownership, stable tetra ID and bounded nonnegative generation. The material-point function canonicalizes the positive-count lifted simplex and exact weights.
- ShellTetrahedron(tetra_id, vertices); ShellBoundaryFacet(vertices, owner).
- ConformingShellReference(domain, node_ids, face_ids, oriented_nodes, reference_positions, half_thickness_m, *, law=ShellDirectorLaw(), limits=ShellProofLimits()). Exact ndarray dtypes/shapes: int64[N], int64[F], int64[F,3], FP64[N,3], FP64[N]. Fields additionally law, limits, tetrahedra, boundary_facets, components, boundary_cycles, reference_id, input_fingerprint. Every array has immutable byte backing. All IDs are stable IDs, not storage-row indices.

geometry/_shell_intervals.py:

- Interval(lo,hi): frozen normalized Fraction endpoints.
- Publication(intervals,values,errors,shape): flat exact interval/error tuples plus owned FP64 ndarray of the declared shape. Each stored proposal is float of the exact rational midpoint, with its exact binary value and maximum endpoint distance checked; this is not ideal correctly-rounded or native delivery evidence.
- Arithmetic(limits, *, bits=None): one bounded operations/created_cells counter, checked rational +,-,*,/, explicit vector/matrix/cofactor operations, exact polynomial operations and Bernstein ranges, integer-square-root enclosures, publication and cell visits. Sqrt refinement follows legal doubling steps; every exposed integer/rational intermediate is bit-capped before allocation/arithmetic. Failed/visited cells and pending children are charged without resetting budgets.

geometry/shell_directors.py:

- shell_director_geometry(reference, positions, *, limits) -> ShellDirectorGeometry.
- shell_material_point(reference, positions, address, *, limits) -> ShellMaterialPoint.
- DomainWitness(faces,stars,corners). Face entries=(face_id,area_norm_interval,max_edge_squared_upper); star entries=(node_id,star_norm_interval,sum_face_norms_interval); corner entries=(node_id,face_id,projected_ratio_interval).
- ShellDirectorGeometry fields: schema, precision, status, reference_id, binding, policy, bits, node_ids, face_ids, areas, sums, directors, offsets, lifted, domain, director_publication, lifted_publication, operations. Schema conforming-shell-directors-v1; precision ideal-from-stored-fp64; status director_geometry_only. Canonical node/face order in result; lifted order is node then sign(-1,+1). Publications have shapes[N,3] and[N,2,3].
- ShellMaterialPoint fields: schema, precision, status, reference_id, binding, policy, bits, canonical_address, generation, node_ids, point, jacobian, domain, operations. Schema conforming-shell-point-v1, same precision/status. point publication shape[3]; jacobian publication shape[nnz,3,3], complete stable-node sparse blocks. No impulse application, contact owner/capacity or runtime operation exists.

geometry/shell_embedding.py:

- certify_shell_embedding(reference, positions, *, limits) -> ShellEmbeddingCertificate.
- ReferenceChart(nodes,face_ids,boundary,axis,normal,polygon,twice_area). axis=(nonzero normal coordinate,projection u,projection v); normal is a rational unnormalized reference normal.
- ReferenceGradient(tetra_id,vertices,lambda_gradients,determinant). Four interval vector gradients in physical reference coordinates and interval determinant.
- SeparatorWitness(left,right,axis,left_support,right_support). Components are stable-node tuples; axis is an arbitrary bounded nonzero rational covector, not a normalized contact normal. Support intervals cover ALL lifted vertices relative to the left component's anchor; only strict sign/separation is used.
- ReferenceWitness(charts,gradients,separators,boundary_facets,pair_incidence). All unordered tetra pairs appear, with exact shared lifted simplex IDs; no incidence pair is dropped.
- RankOneCharge(node_id,middle_norm,director_norm,beta_norm,gamma_norm,middle_term,director_term).
- TetrahedronWitness(tetra_id,determinant,beta_gradients,gamma_gradients,charges,k).
- ComponentWitness(nodes,tetra_ids,k).
- CellWitness(start,end,bits,domain,tetrahedra,components,separators,pair_incidence); k_max is the maximum component bound property.
- ShellEmbeddingCertificate(schema,precision,status,reference_id,binding,policy,reference,cell,geometry,operations). Schema conforming-shell-embedding-v1, precision ideal-from-stored-fp64, status strict_geometry_certified. geometry is the original static state's complete director/publication receipt, produced under the SAME arithmetic counter.

geometry/shell_sweep.py:

- certify_shell_sweep(reference,path,*,h,limits) -> ShellSweepCertificate. path is exact FP64[2,N,3]; h is finite positive FP64 seconds.
- TimeCell(start,end,depth,accepted).
- ShellSweepCertificate(schema,precision,status,reference_id,binding,policy,h,reference,cells,tree,endpoint_geometry,operations). Schema conforming-shell-sweep-v1, same ideal precision and strict-geometry-only status. tree includes failed parents as well as accepted leaves; cells are the complete accepted partition. endpoint_geometry has exactly two director/publication receipts bound separately to the two original endpoint arrays, with all work on the same producer counter. Endpoint publications are not substituted for the nonlinear cell proof.

validation/conforming_shell.py:

- verify_shell_director_geometry(receipt,reference,positions,*,limits).
- verify_shell_material_point(receipt,reference,positions,address,*,limits).
- verify_shell_embedding(certificate,reference,positions,*,limits).
- verify_shell_sweep(certificate,reference,path,*,h,limits).

Each independently returns True only after original-input reconstruction, otherwise ShellUnresolved. Each uses ONE fresh counter throughout topology/reference/law/cell/publication checks. It imports immutable receipt types and checked low-level arithmetic, but never calls a producer, its topology, Jacobian, reference-chart or coverage routines. Domain/reference objects do not authenticate certificates.

## Mathematical and interface decisions retained

Topology is an oriented simplicial two-manifold with interval/cycle vertex links. The global-ID staircase and orientation parity establish identical common side triangulations and cancellation of internal facets. Canonical reference_id normalizes only storage order and cyclic face order while preserving winding, stable IDs, exact binary reference data and law. input_fingerprint additionally retains caller order. State/path/h/address/policy bindings retain exact original byte input. Signed zero has its exact rational value in mathematics but may remain distinct in byte bindings.

All raw-area/star weights are1. Exact edge polynomial coefficients are built before raw-area products, and star/H-vector coefficients are summed before intervals. J includes every incident-star neighbor and shared-node accumulation. Translation/rotation/virtual-work/resultant/torque identities follow from this independently reconstructed analytic map; neither producer nor verifier uses a residual interval containing0 as an identity proof.

The positive reference domain is the independently checked planar convex constant-thickness disk per component. General manifold topology and variable thickness remain constructible/differentiable; unsupported reference charts do not receive embedding certificates. Producer uses rational closed polygon clipping; independent verifier enumerates included triangle vertices plus all edge intersections, retaining collinear/point intersections. Both re-establish all triangle-pair incidence, simple convex boundary, contained faces and exact area coverage. Reference gradients are obtained through checked cofactors/inversion of every actual tetrahedron, never a floating inverse.

Every cell proves the original law gates, positive current tetra determinants and each positive rank-one term. A component anchor's exact affine displacement is removed before bounding. Identical raw star polynomials cancel their director differences symbolically. The proof requires max k_T<1; positive tetrahedra alone are not embedding. All same-component pair incidence follows from the global lower-Lipschitz theorem on the proved convex reference body. Different components require complete strict support separation. Coordinate axes, both reference raw normals and the midpoint mean difference form the finite producer search; the verifier accepts any bounded rational separator it independently proves, including alternatives outside that producer choice.

Time bounds use an exact power-to-Bernstein transform after rational interval substitution. Middle paths are affine; c/S are quadratic and Q quartic; normalization is nonlinear throughout. Bisection is left-first with a complete pending sibling stack and explicit failed-parent tree. Static/sweep proof uses its declared initial sqrt precision; director/point publication can refine legal sqrt precision for representability/division uncertainty while retaining the same counter. Unsupported or insufficient enclosures reject conservatively rather than expanding limits or using a sampling/frozen-velocity fallback.

The independent checker requires claimed intervals to CONTAIN its independently recomputed outward enclosure, not merely overlap it. It independently checks positive charges and their sums, complete topology/tetra/component/pair memberships, full partition tree and original bindings. Conservative wider point intervals with valid storage errors are admitted. Alternate valid partitions and rational separators are not rejected for differing producer bytes. Reported producer counters are metadata; the standalone verifier charges its own complete reconstruction instead of trusting those counters.

All physical budgets remain untouched future gates. Current point/J publication error is strictly below2^-40; it is not permission to spend host1e-8/actual2e-5 mapping or host1e-8/actual1e-5 energy/angular budgets. No penetration allowance, diagonal regularization, epsilon-expanded body, approximate-equality topology, single-root/two-root relabeling or R7/native claim was introduced.

## Written independent acceptance controls, all UNEXECUTED

F1 retains the exact six tetrahedra and the analytic free middle path q(t)=1/8+t for T=1/8, unit masses, rho1/16, no original force/gravity/elastic energy. Tests request the whole window AND every substep for h1/64,1/128,1/256, two separately rebuilt references and identity/rotation(-y,x,z)/translation(4,-2,1)/stable-ID storage permutation variants. Each actual computed certificate must report k<33/64, all six tetrahedra and all fifteen pair incidences. Same-run future JSON evidence hooks preserve complete original reference/path/h/certificate and SHA256 sidecars; paired rebuild comparisons use those saved bytes. These are analytic geometry inputs, not evidence that runtime generated the physical path.

The nonflat F1 point-map tests independently differentiate y_top,2=(0,1-rho*q/s,q+rho/s), s=sqrt(1+q^2), along the free tangent, giving dy/dq=(0,-rho/s^3,1-rho*q/s^3). They apply all four transforms at q1/8,3/16,1/4; no finite difference supplies the oracle.

F2 retains the four-face star and all five exact blocks: J0=I, J1=-(1/8)e_x e_z.T, J3=+(1/8)e_x e_z.T, J2=-(1/8)e_y e_z.T, J4=+(1/8)e_y e_z.T. Tests check two rebuilt references/all four transforms, shared-parent address invariance, complete virtual work/resultant/moment, and W=diag(65/32,65/32,2) including the independent unit point. Hold u65/256,t=-1/8,DeltaK=-1089/4096; slip u65/64,t=-1/4,slip65/128,DeltaK=-451/1024. They are pure map/impulse algebra tests, not delivered contact or trajectory passes. The original future physical T1/16, three h1/16,1/32,1/64 and two rebuilds remain required later.

Negatives include N1/N2 connected positive-local-tetra overlapping stars; N3 separately certified endpoint configurations with interior crossing; N4 exact zero/inverted tetra with nonzero valid directors; N5 zero star with nonzero faces; N6 forged diagonal/orientation/boundary/reference/address/state/path/policy/coverage; N7 missing opposite-node or barycentric-only/zero J and undercharged director term. Resource, unsupported chart, collinear boundary, exact zero-area contact, unknown proof, remote boundary touch, partial partition and missing endpoint-publication cases are written. The separate half-turn collapse path has analytically valid rigid endpoint bodies and an interior collapse; it does not claim that the sufficient k<1 backend admits the final half-turn relative to that reference. N3 is the distinct negative with both endpoint certificates required.

Remaining work is explicit: first authorized host static/six full-file execution and same-run evidence audit; weak remote boundary-touch admission; rotating loaded held work/path/capacity; noncollinear cones and physical quotient pressure ownership; external robot manifold/CCD/full-star PointPort; actual once-only replay/terminal coordinator and transaction/energy/R1/R2 gates; typed active B2/R7 contact; full native F1/F2 windows; continuum/inertia/convergence claims. Strict geometry cannot be relabeled completed folded self-contact or full P6.5.
Parent text-only corrections before any execution: the verifier now accepts arbitrary resource-bounded exact rational binary split points derived from complete child spans/depths, rather than requiring the producer's midpoint strategy. A new constant-body1/3 partition test constructs its alternate proof without a second producer oracle; missing siblings, wrong depths, overlap and isolated nodes remain rejected. Every actual existing-Fraction unary negation now uses metered qneg, including interval negation, polynomial/H coefficient signs and the negative branch of qabs for storage-error reconstruction. A subtraction's already charged internal sign is not double-counted. Literal signs, integer topology bookkeeping and bit-length metadata are not misreported as rational model operations. Both changes retain the original2,000,000 ceiling and all fixed controls. Tests were written before these corrections and remain UNEXECUTED.
## Exact fourteen-path inventory

The twelve Python file hashes below describe the final authored bytes; no execution has occurred. Git blobs are computed with the path attributes applied. The last two rows are the two identical-content reports; their final SHA256/blob are supplied in the final out-of-band Git handoff, avoiding impossible self-referential report hashes.

| NEW path | Working-file SHA256 | Git blob |
|---|---|---|
| src/phymeo/constraints/conforming_shell.py | b15fcd5e9f4b44b95f99bcf98d7b5f9fc25e75c25dc3bead7280c0303c7f100c | f4ffbd911f0b12096549f8287f4925c05dab8383 |
| src/phymeo/geometry/_shell_intervals.py | b7c0799153eaee6a7998dc2ae2edfa93376d6dfdc4215bdaf9731590d4b93524 | fff08586438deef516ecbe2af583c54173fcfb8c |
| src/phymeo/geometry/shell_directors.py | 09f0feb63182435a88864929bc96e161d028cbbe75d48f442a175a25b82c42b9 | 76c30fe162241bc8e31a229fdc055ac5f76ed663 |
| src/phymeo/geometry/shell_embedding.py | 90121e9689aa80f8d9560a931bca955ce3eef645eb97bc4aa828a8d0373e85b8 | 0beac6b065627be3e62539707a5aaa2ca6a66b58 |
| src/phymeo/geometry/shell_sweep.py | 6d878f0c50ac6b8f49aa7e9b33450beff60a71f73ea7073dc6a16d84812a5956 | cf5c9076aa7b0a83a59c4a09d22f8eb3019e0e7c |
| src/phymeo/validation/conforming_shell.py | 52a9f4cfc22ef39d039074ab9abcebafc65754dbc8ad4812fa869b39e53dd5de | 33a7f18e19d51e00e8a7acb26fb9bb5c0a20eb6c |
| tests/test_conforming_shell.py | 13c663da0e21751a396d9bcc0be4c627f660ed41c0301a9728962e7afdc1bd63 | ef707cb73c1d2e4961ead3726c0331eb931d5fb8 |
| tests/test_shell_intervals.py | b6b945c87457d5410349eb67889b3934ae204ffd65a574afc9c71e88c70a990f | bf3e5f8ad2b9c82cb75e5ed17e6909e27a1dc0b0 |
| tests/test_shell_directors.py | 87483a8c4cb091119f105a5885f9b7ff249df5b96383f6afe4a2ca9dcd84fce1 | 6a09f9877d50c94eefbddae23134845ac068359f |
| tests/test_shell_embedding.py | 8cd439d1efd9ff2043fac6cf3434fc66113ef6eaa9191d8a968512a966743fb2 | 30439a1773fdcc132d684202c17f87c9f4476639 |
| tests/test_shell_sweep.py | 44fe0898faf8c474f2db2c91f40abf61c57c2f903dfa0612c9625830eae90e77 | 0fe1ee8f18965342100e3343123b258ff001246f |
| tests/test_shell_certificate_verifier.py | f94f805ce7412bd30d14bbe7931816fb3cbce4833e00fce75ab862403b98098c | baa6c2d1e98ccc7093e166bb90a156db77ff810b |
| docs/development/p65-conforming-shell-geometry-report.md | final out-of-band handoff | final out-of-band handoff |
| .superpowers/sdd/2026-10-30-p65-material-self-gripper-integration/task-2-conforming-shell-geometry-report.md | final out-of-band handoff | final out-of-band handoff |

## Authored, uncollected test function node IDs

These names are extracted as text, not through pytest collection; parametrized node IDs and a collected/pass count are deliberately not claimed. Every listed test is UNEXECUTED.

- tests/test_conforming_shell.py::test_f1_exact_six_oriented_tetrahedra_and_shared_diagonal
- tests/test_conforming_shell.py::test_reference_owns_arrays_with_immutable_backing_and_canonical_identity
- tests/test_conforming_shell.py::test_rejects_malformed_owned_inputs
- tests/test_conforming_shell.py::test_disconnected_vertex_link_rejects_bow_tie
- tests/test_conforming_shell.py::test_nonmanifold_edge_rejects_before_geometry
- tests/test_conforming_shell.py::test_law_limits_and_address_are_exact_owned_values
- tests/test_conforming_shell.py::test_cyclic_face_order_preserves_canonical_identity_but_binds_input_order
- tests/test_conforming_shell.py::test_constructor_shape_ceiling_precedes_owned_copy
- tests/test_shell_intervals.py::test_sqrt_exact_squares_and_nonsquare_integer_bounds
- tests/test_shell_intervals.py::test_interval_arithmetic_and_denominator_sign
- tests/test_shell_intervals.py::test_polynomial_bernstein_encloses_interior_extremum
- tests/test_shell_intervals.py::test_publication_has_exact_storage_error_not_correct_rounding_claim
- tests/test_shell_intervals.py::test_operation_and_intermediate_bit_budgets_are_enforced
- tests/test_shell_intervals.py::test_created_cells_count_failed_parents_and_depth
- tests/test_shell_intervals.py::test_unary_fraction_negation_and_negative_absolute_value_are_metered
- tests/test_shell_directors.py::test_f2_complete_five_node_jacobian_and_exact_independent_response
- tests/test_shell_directors.py::test_f2_independent_impulse_squares_virtual_work_and_reaction
- tests/test_shell_directors.py::test_shared_material_parent_address_does_not_change_point_or_star
- tests/test_shell_directors.py::test_f1_nonflat_full_map_along_independently_differentiated_free_tangent
- tests/test_shell_directors.py::test_geometry_only_status_storage_errors_and_vanishing_sum_rejection
- tests/test_shell_embedding.py::test_f1_static_all_tetra_pair_coverage_and_strict_bound
- tests/test_shell_embedding.py::test_n1_n2_positive_local_geometry_does_not_certify_overlapping_star
- tests/test_shell_embedding.py::test_n4_exact_degenerate_or_inverted_tetra_despite_valid_directors
- tests/test_shell_embedding.py::test_general_reference_remains_owned_but_unsupported_chart
- tests/test_shell_embedding.py::test_ordered_collinear_boundary_subdivision_is_valid
- tests/test_shell_embedding.py::test_wrong_reference_point_incidence_rejects_even_without_positive_area_overlap
- tests/test_shell_embedding.py::test_strict_remote_component_boundary_touch_excluded
- tests/test_shell_embedding.py::test_resource_limits_are_not_silently_reset_in_reference_proof
- tests/test_shell_embedding.py::test_concave_reference_is_supported_topology_but_not_this_proof_chart
- tests/test_shell_embedding.py::test_closed_triangle_intersections_retain_exact_point_and_edge_contacts
- tests/test_shell_sweep.py::test_f1_whole_window_every_substep_two_independent_rebuilds
- tests/test_shell_sweep.py::test_n3_legal_endpoint_components_cross_interior
- tests/test_shell_sweep.py::test_pure_large_common_translation_is_factored_before_interval_bounds
- tests/test_shell_sweep.py::test_interior_time_face_or_director_collapse_not_endpoint_sampling
- tests/test_shell_sweep.py::test_failed_parent_cells_and_depth_are_charged_not_dropped
- tests/test_shell_certificate_verifier.py::test_verifiers_do_not_call_producers_or_constructor_topology
- tests/test_shell_certificate_verifier.py::test_forged_full_star_material_point_rejects
- tests/test_shell_certificate_verifier.py::test_wider_containing_interval_is_valid_but_merely_overlapping_is_not
- tests/test_shell_certificate_verifier.py::test_static_coverage_topology_and_binding_forgeries_reject
- tests/test_shell_certificate_verifier.py::test_alternative_rational_separator_not_producer_byte_identity
- tests/test_shell_certificate_verifier.py::test_alternative_complete_dyadic_partition_and_missing_cells
- tests/test_shell_certificate_verifier.py::test_full_material_and_charge_forgeries_are_not_authenticated_by_constructor
- tests/test_shell_certificate_verifier.py::test_sweep_rejects_missing_endpoint_publication_and_mutated_path
- tests/test_shell_certificate_verifier.py::test_point_address_tetra_and_winding_are_bound
- tests/test_shell_certificate_verifier.py::test_independent_nonmidpoint_rational_partition_is_accepted_without_new_oracle_run

Final authoring verification is limited to manual source/math/contract reading, exact file/Git hashes, Git whitespace/scope inspection and the clean commit receipt. No claims of syntax/import/static/test/native success are made. All first executable outcomes belong to the later separately authorized focused-host grant.

**CODE_READY_TESTS_PENDING. AUTHORING_SLOT_RELEASED at the final clean commit. No executable slot was held or consumed.**
