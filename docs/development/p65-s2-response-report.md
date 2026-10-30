# S2-L self-response implementation and serial validation report

## Serial validation and current delivery

Status: LOCAL_VALIDATION_COMPLETE; parent acceptance remains pending. The sole execution slot is released only by the final clean/process receipt saved with this report. Tested production/test snapshot: bcf4c1d343df2ee0a15be9f62f1bd1e50ca087d4. All subsequent repository changes are these two reports.

The parent grant d806f28b53505cb045c9c1a346f8e62ed81fe27f authorized serial validation. The initial fifteen authored working hashes/Git blobs were saved at 7f68bdea32cc816352a5cfbd839c1eec532d3b65, then parent 174e2ab19d17652448109dad953e91c4654d2cea was merged normally as ee6370a991bfac90ca8e12cc34c7c438ff995da2 without conflicts. All fifteen authored files stayed unchanged at that merge. The adapter source and two tests matched accepted 721b5c162f1f7e42dd618c70378bba098af0698e byte hashes and Git blobs. No later dependency merge or shared-source edit occurred.

All commands use the child src explicitly. Environment preflights assert eight imported-module paths. Executions set PYTHONDONTWRITEBYTECODE=1, disable plugin autoload, set all configured thread counts to 1 and disable pytest cache; each pytest group has a checked-absent absolute basetemp. Host used the existing original .venv-py311: Python3.11.15, NumPy2.4.6, pytest 9.1.1, Ruff0.17.2. Actual used the existing .venv: Python3.12.14, NumPy2.5.4, pytest 9.1.1, Ruff0.17.2, tyx-compute0.1.0a0 at pinned 99f6df7c0fd8fdeef77f97df83914da27b64b7ee. No installation was performed.

| Required selection | Saved run | Observed outcome |
| --- | --- | --- |
| Scoped Ruff on all thirteen new Python files | 21-current-ruff-check | RAW_EXIT0 |
| Scoped format check on the same thirteen | 22-current-format-check | RAW_EXIT0 |
| Complete five host files together | 18-host-face-index-fix | 136 passed, 2193.93 s, RAW_EXIT0 |
| Complete actual file, interpret, --require-tyx | 27-interpret-first-executed | 80 passed, 1565.79 s, RAW_EXIT0 |
| Complete actual file, cuda, --require-tyx | 32-cuda-first | 80 passed, 1749.03 s, RAW_EXIT0 |
| Exact bounded terminal node, cpu_cpp, --require-tyx | 33-cpp-bounded-first | 1 passed, 11.37 s, RAW_EXIT0 |

The full host selection is test_cloth_self_response.py, test_cloth_self_anchors.py, test_cloth_self_mapping.py, test_cloth_self_physics.py and test_cloth_self_transaction.py. Both full actual selections use test_cloth_self_response_tyx.py; C++ uses only ::test_actual_bounded_cpp_terminal_control. Every pytest command used -q -x --tb=short -p no:cacheprovider. There were no skips or fallback substitutions. C++ proves only the specified bounded one-step terminal control, not all windows or backend lifetime/TYX003 closure.

### Original outcomes and repairs

All original post-code failures, commands, source snapshots and RAW_EXIT files are retained. They are not historical TDD RED. First Ruff run 03 found 25 issues; first format run 04 found 13 files. Style commit 6ea6c760e3984c865b8b0c7407831a90137937a7 sorted 19 imports, renamed two ambiguous LDL locals and replaced four assigned lambdas by local functions. Formatting-only ASTs matched for all 13; import-normalized ASTs also matched for all 13. The local rename/lambda transformations are separately preserved and are not mislabeled exact-AST equality.

Host run 14 reached 40 passes, then found a fixture using a generic StateRegistry where the public participant-owned registry was required. Commit 8b86b10 fixes only the new fixture to use base.registry. Host run 16 again reached 40 passes, then exposed tuple-based topology indexing that selected a scalar. Commit bcf4c1d uses triangles[list(pair)] to retain both oriented face rows. The original failure receipts and small diffs remain. The complete host run 18 and every later physical selection passed on the same resulting snapshot. No physical inputs, oracle equations, h, windows, proof resources, tolerances or physical gates changed.

Native preflight 23 failed in the observer's distribution metadata lookup (tyx versus installed tyx-compute); corrected preflight 25 passed without product changes. File-only observer 19 failed at final grouping because empty first-step owner evidence combined distinct B cases. The preserved v2 was not executed. V3 binds pairs from authored collection order before output comparison; it passed host run 28. Observer 30 then found that pytest's truncated native C directory names retain the variant and restart local numbering. V4 derives exactly that naming from the authored variant names and pytest 9.1.1 _mk_tmp behavior; no physics or source changed. Original observer failures and all versions remain preserved.

Launches 26, 29 and 34 stopped at broad process guards before execution and have no fabricated RAW_EXIT. Foreign processes were identified, never killed or treated as Phymeo validation; their results are not claimed. The initial native-preflight approval-review rejection, based on a stale host-running condition, is also retained alongside the already-finished host receipt and subsequent allowed command. These incidents did not cause physical resampling. The last identified foreign process ended naturally; the fresh global zero snapshot at 2026-10-31T05:00:26.7314786+08:00 is retained in 34-process-ended-global-zero.json before audit 35.

### Same-run saved evidence

Evidence root: C:/Users/38347/AppData/Local/Temp/phymeo-p65-s2-response-7f68bde-serial. Raw logs, exact commands, source manifests, exits, before-source snapshots, failed attempts, observer versions and process receipts remain there. The final same-snapshot audit checks 328 before/after/current source/test/config hashes for every listed passing selection and observer, plus all 13 authored SHA/Git blobs and the three accepted adapter files. Its source and adapter mismatch counts are zero.

| Saved physical group | Audit run/version | Windows / steps / exact rebuild pairs | Maximum positive energy / absolute angular window sum |
| --- | --- | --- | --- |
| Host | 28 / v3 (RAW_EXIT0) | 78 / 1434 / 39 | 0 J / 3.3306690738754696E-16 N*m*s |
| Interpret | 31 / v4 (RAW_EXIT0) | 79 / 1435 / 39 | 0 J / 5.9604644775390625E-08 N*m*s |
| CUDA | 35 / v4 (RAW_EXIT0) | 79 / 1435 / 39 | 0 J / 5.9604644775390625E-08 N*m*s |
| Bounded C++ | 36 / v4 (RAW_EXIT0) | 1 / 1 / 0 | 0 J / 0 N*m*s |

The saved-data observer checks complete actual state/history, original inputs, requested/represented/actual phase transfers, selected-pre binding, path/CCD receipts, independent R2 phase evidence and native FP32/launch/upload/download receipts. It compares the entire stable serialized payload for each prescribed pair, excluding only the exact typed SceneSubstepReport.timings field. Pair membership is fixed by authored function/case and ordinal block/stride before any output comparison; each pair contains exactly two distinct resolved physical directories. Aliases are deduplicated. This avoids vacuous singleton or output-selected rebuild equality.

All saved full-window energy and angular sums stay within their original noncancelling budgets: host 1e-8 and actual 1e-5. Capacity enclosure and complete phase gates retain the original 1e-8/2e-5 tolerances. No extra numerical run was used to generate this report.

To preserve full evidence within disk limits, the parent authorized lossless compression of closed per-step JSON only. Each file retains its original path, byte length, SHA/name sidecar, compressed SHA and independently streamed decompression byte/SHA verification in a .storage.json receipt. Atomic publication of the compressed file and receipt precedes removal of the verified redundant original. Raw logs, exits, commands, manifests, source snapshots and last-attempt.json remain uncompressed. The saved audits reconstruct and verify the exact original bytes. The first 352 sidecar filenames were independently rechecked; the later filename/regex corrections and original safe-stop evidence are retained.

### Exact scope of the ideal-map capacity enclosure

For actual stored B, L=(B M^-1 B^T)^-1 B M^-1 and p=L ell, the selected-address discrepancy is E=B_star^T-B^T, enclosed by |E|<=Gamma. With nodal delivery uncertainty |delta|<=e, the ideal-map certification witness is defined in the stored-map gauge by p_star=p+L delta-L E p_star. Its remainder is r_star=(I-B^T L)(ell+delta-E p_star). G=|L|Gamma and strict gamma=||G||inf<1 give the uniform contraction bound and the propagated pressure/remainder enclosures used by _ideal_capacity_error. Unknown or exhausted enclosure rejects.

This p_star is a stored-map gauge certification witness. It is not asserted to be the ideal-map mass-weighted pseudoinverse, a unique measured physical pressure decomposition, or a state correction. Gamma encloses exact-versus-stored J only at the plan's selected material addresses. Differences at all complete clipping-extremum addresses are separately checked by _extreme_velocity, the ideal velocity bounds and _residual_speed over both maps. The Gamma contraction alone does not certify every ideal extremum or a nonlinear held-position sensitivity. The parent's hand derivation of these identities is file-only mathematical evidence, not an additional numerical test or independent acceptance of the whole response implementation.

### Remaining scope and release

The delivered claim is the frozen nonincident unique-exposed-ray, constant-normal loaded held domain with None/legacy Distance, including the full A/B/C controls and strict gates above. Full incident folds/S2-F, arbitrary rotating loaded continuation, typed B2/v2 coupling and overall P6.5 integration remain open. Blanket shared-vertex exclusion or rejecting every folded case does not satisfy full folded-cloth contact. No general-tree energy or arbitrary convergence claim is made. Remote CI is user-deferred and unpassed. Accepted LG69/S1/G2/B2/adapter groups were not repeated; no helper/reviewer/push/remote or profiling occurred.

Final report/source manifests, evidence inventory and exact final clean/process 0 release receipt are returned out of band to avoid self-referential hashes. Parent acceptance is a later decision after independent inspection of these saved bytes.

---

## Preserved authoring record: 7f68bde

Historical authoring status: CODE_READY_TESTS_PENDING. No executable validation had been authorized or performed at this checkpoint. All execution/dependency/release statements in this preserved record describe that earlier checkpoint; the current results and dependency merge are above.
Starting lineage: a075d1826bb8ba29edb08a91460940184e0346e7, branch
codex/p65-self-contact-module, existing isolated managed worktree.
Contract read from the parent integration tree at d55182ea99b8c8a079acb4ea6ca4afa9833fa234,
including R1 conditioning and the subsequent pre-execution R2 capacity ruling at parent174e2ab19d17652448109dad953e91c4654d2cea. The accepted S1 and
LG69 files, old reports and parent adapter have not been modified or rerun.

## Implementation and independent equations

Seven new source modules implement immutable law/budget/phase/load data, exact
material ownership, an independent pair-field Jacobian and mapping ledger,
an independent actual-response/physical ledger, and a public replay coordinator.
Every held proposal enters one SceneLoadReplay force addition from the same
original snapshot. Every terminal proposal enters one aggregate
PointVelocityImpulses batch through TerminalImpulseProgram. The proposal graph
never applies impulses. Its complete W includes AA+BB-AB-BA and every shared-face
cross block. No duplicate particle domain, dummy robot, host candidate splice,
prior-step capacity or second nodal application is used.

R1 admission uses exact rational LDL of the explicitly symmetrized stored full
response, strictly positive pivots and strict ||W||inf ||inverse(W)||inf < 10^12.
Equality, singularity and unknown reject. The raw response and exact skew-bound
receipt remain available. Relative asymmetry above 10^-12 rejects. Existing local
friction effective-response checks remain unchanged. Normal active-block and
conditional tangent-disk solves are proposals only; complete actual replay and
independent exact/outward phase gates decide acceptance.

Owners retain all maximal ties, require one exact oriented exposed ray when
loaded, retain every complete ideal clipping extreme, bind the explicit domain,
face IDs and cyclic-oriented triples, and allocate one capacity per material
owner/generation. Counts are nonnegative int64 with exact sum 2^52; offsets have
signed FP64 bit identities. Actual addresses must remain on the exact selected
finite support and inside the prism, with world error at most 2^-40 m in the
controls. Unknown quantization, clipping, tie, area or proof rejects.

For charged held paths, exact raw-area coefficients are proved parallel to the
initial area and their oriented quadratic has strictly positive minimum. Every
material support equality is checked at both endpoints; with that proved constant
normal it is affine throughout. The final complete ideal material extrema and
oriented ray must match. Root transport into the final pair field is attempted
only after this full-path proof; incompatible or unknown radicals reject. The
independent phase ledger rebuilds both actual-address and ideal-extreme velocity
maps from source geometry and exact material coordinates. No production J or
reported point velocity serves as its oracle. Across unrelated pairs, only
rational outward enclosure endpoints are accumulated.

The response ledger independently recomputes candidate coverage and binds every
sweep domain, oriented node triple and thickness to topology. It runs the LG
independent sweep verifier on the actual full path. Positive retained SAT evidence
can certify unloaded separation even if complete static clipping is unresolved;
that unresolved status is not relabeled as penetration. All faces receive an area
path proof even when broad phase returns no candidates.

Mapping distinguishes exact ideal J, stored J, requested nodes, represented held
input differences, actual native terminal output, and published world state.
All node impulses are aggregated before mass response and kinetic squares.
The actual couple is sum((xA-xB) cross p), including a diagnostic nonzero-couple
control. Held internal work is actual ellH dot dx/h and is checked against material
point chord work. Terminal internal work is ellT dot vpre + one-half sum(w|ellT|^2).
Neither is external work credit. Acceptance gates Delta(K+U_material)-W_original
and the absolute full angular defect, with noncancelling step/window history.

## Hand-derived controls fixed before execution

A: XY=(0,0),(3,0),(0,4); A z=1/8,rho=1/8,m=2; B z=-1/4,rho=1/4,m=4.
Original forces are -2ez/+2ez per node. Wn=3/4 and free endpoint gap is
-3h^2/2, hence held pressure is 2h per corner. All tangent and terminal loads
are zero, state is unchanged, total pressure is 6h per step and 3/2 N*s over
T=1/4. K, material U, original work and angular change are zero. Authoring includes
None and exact-rest Distance edges with lengths 3,5,4 and compliance 1/16.
The material path remains exercised and recorded for both.

B: XY=(-1,-1),(1,-1),(0,1), rho=1/4,m=1. T=1/16. Initial faces have opposite
x shifts +/-uT/2, z=+/-(rho+T), and v=+/-(u/2,0,-1) with the explicit signs
in the fixture. Gap is 2(T-t)>0 before the terminal birth. Held pressure is zero;
terminal normal is one N*s at each of three corners. Gradients are
(-1/2,-1/4),(1/2,-1/4),(0,1/2), giving sum(g g^T)=diag(1/2,3/8).
The full uniform-x tangent coefficient is 35/16, not the barycentric-only value2.
With mu=1/2, u=35/64 yields t=-1/4 and zero point slip while centroids retain
+/-3/128 m/s. u=35/16 yields t=-1/2 and point slip35/32. Both triangles have
outgoing vz_i=3*rho*t*g_i.x. DeltaK is -3-105/512 J (stick) and -3-315/128 J
(slip). Full orbital plus spin Ly=3*rho*u is conserved. The window ends at impact;
this does not certify a subsequent rotating loaded step.

C: three copies at z=0,-1/2,+1/2, rho=1/4,m=1, velocities0,+ez,-ez, h=1/64.
For each corner Wn=((2,-1),(-1,2)), un=(-1,-1), held=(1,1) N*s.
Middle-face aggregate cancels, outer faces stop, terminal load is zero,
DeltaK=-3 J. Dropping cross terms gives pressure(1/2,1/2), closing speed-1/2
and endpoint gap-h/2. The original full-base control rejects that approximation.

Full A h=1/64,1/128,1/256 and B h=1/16,1/32,1/64 windows are authored.
Every control includes two independent fresh rebuilds and identity, signed proper
rotation(-y,x,z)+translation(4,-2,1), and exchanged/cyclic face declarations.
The oracle transforms forces, velocities and angular quantities with the same
proper rotation; it does not call the production pressure, geometry or J builder.

## Budgets and evidence

Controls use penetration0 m, anchor error2^-40 m, length/speed reference1 m/1 m/s,
impulse reference1 N*s. Host phase/map tolerance1e-8, actual tolerance2e-5;
host energy/angular1e-8 J/N*m*s, actual1e-5 J/N*m*s. The same ceilings apply per
step and to noncancelling whole-window sums. Independent enclosure contributions
are charged and capped at 1/64 of the corresponding budget. Resource defaults
remain128 contacts,64 anchors/pair,8 rebuilds,512 outer replays,16 line halvings,
512 normal iterations,96 tangent iterations and LG64/4096 bits,4096 cells,32 depth.
Inner replay uses one substep and original512 material iterations. No gate,
window, fixture, stiffness, dependency or native precision was tuned to output.

Each physical test writes the complete same-run state/history/report/replay/native
receipt and an SHA256 sidecar before asserting acceptance. Failed attempts retain
immutable original inputs, intended plans/requests and whatever actual runtime
receipt was produced; an operation interrupted before native publication is not
claimed to have a complete native receipt. No extra physics run is scheduled for
report generation. Retry/fresh comparisons are explicit transaction controls.
Only the precise SceneSubstepReport.timings field is observational in selected-pre
comparison; arbitrary dictionary keys named timings remain bound.

The six new test files author algebra/rank/conditional-disk/ownership tests,
all ideal extrema, map/J/couple forgeries, complete A/B/C physical windows,
late diagnostics/close/selected-pre/material/path/topology/phase faults, later
frame rollback, and actual interpret/CUDA plus bounded C++ controls. Actual native
receipts require FP32 promotion, matching launched/completed counts, and the
operation-specific terminal delta (one node-load kernel and one axpy for a
nonempty batch). No fallback is accepted or claimed.

## Boundaries and remaining work

Only None/legacy Distance and the frozen nonincident constant-normal held domain
are admitted. Composite/B2 v2, general rotating loaded continuation and lawful
incident folds/S2-F remain future required integrations; blanket incidence
exclusion is not full cloth self-collision safety. No general-tree energy or
arbitrary convergence theorem is claimed. Exhaustion remains rejection.

Parent owns the generic PointParticipantProgram adapter and old shared seams.
The adapter is a dependency to be integrated by the parent before validation;
it has not been copied or merged into this authoring tree. No imports, test
collection/execution, numerical probes, lint, formatter, compile/build/install,
profile, remote CI, push, helper or reviewer was run for this checkpoint.
No historical TDD RED or pass is asserted. First executable outcomes remain
pending a separate sole validation grant.

## R2 actual-delivery capacity witness

R2 was frozen and read before any execution. The proposal is never called exact
delivered native pressure. For the complete actual stored attachment map B and
original inverse masses, the code forms exact rational A=B M^-1 B^T and
L=A^-1 B M^-1, requiring positive LDL and strict kappa_inf<10^12. It records
p_eff=L ell, delta=p_eff-p_requested and r=ell-B^T p_eff. This is a named
mass-weighted certification decomposition, not measured physical ownership or a
state correction. Raw response versus exact Gram remains independently bounded.

Held ell is reconstructed as exact h*(q(F_effective)-q(F_original)), with the
reported integrated-load array checked separately. Predicted-velocity residual
abs(q(v_pred)-q(v0)-h*(q(g)+q(w)*q(F_effective)))/q(w) gives a conservative
native impulse enclosure. It includes conversion/arithmetic effects without
pretending to isolate contact-only native error. Terminal ell is reconstructed
from exact (q(v_post)-q(v_selected_pre))/q(w_original); requested batch, actual
native operation receipt and published output remain separate. The final ledger
rebinds source snapshots and rereads the mandatory legacy material path.

Native uncertainty propagates through every entry of |L|. Pair-field outward
bounds also enclose the ideal/stored-map discrepancy. A sufficient contraction
bound || |L| |DeltaB^T| ||inf<1 encloses its feedback; failure is unknown. The
same coordinate enclosure enters robust normal LOWER bounds, tangent UPPER
bounds and worst-case capacity/KKT. A separate response-scaled uncertainty gate
uses the original phase speed ceiling, not a new capacity allowance. A negative
normal lower bound rejects; no max(0,...) erases it. Open birth combines its two
bound phase projections exactly once. Zero-load rounding creates no safe capacity.

The nodal remainder is neither removed nor redistributed. Exact nominal r and
its enclosure are checked for resultant, couple, virtual/physical work and
response at every actual and ideal extreme. Their effects remain in actual state
energy and angular balance. Those physical gates do not subtract residual work.
Bounds are listed by atomic source; the robust disk uses one enclosure rather
than spending separate native/ideal allowances as extra friction.

New original-form controls include B=(I,-I), A=2I for on-range and equal-node
off-range delivery; and the sufficient-admission adversary
B=diag(2^-16,1,1), A=diag(2^-32,1,1), kappa=2^32<10^12. Its nodal native
uncertainty2^-30 N*s passes the1e-8 map ceiling, but |L|e=2^-14 N*s exhausts
the original1e-8 phase-speed gate at response scale1/kg. This is a matrix bound
control, not a claimed triangle fixture. Changed actual delivery with an unchanged
proposal and changed held receipt at open birth invalidate the old witness.
All A/B/C physical inputs/oracles and budgets remain exactly as above.

## Authoring release

This report accompanies the exact thirteen source/test file hashes in the task
report. Test function names there are text-discovered authored nodes, NOT pytest
collection results or passes. The parent adapter dependency was read only. The
only operations performed for this task were text/file IO and lane Git. No
execution slot was held. Authoring is released after the final clean report
commit; a separate sole validation grant is still required.

Code authoring snapshots: c93e3dd96ef87956545750cdd72fd92e5f1bef0b (algebra/anchors/mapping), then b22b79e03e92a75ea7d09d3c8d63bda1d2a54e6e (replay, actual-response ledgers, R2 and remaining tests). Both are unvalidated authoring commits.
