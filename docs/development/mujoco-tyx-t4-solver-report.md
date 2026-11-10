# MuJoCo-TYX T4 solver authoring report

Status: **CODE_READY_TESTS_PENDING / AUTHORING_SLOT_RELEASED**.

Source commit: `fcbed62a33f67f3f743f4dc191c1005a249cdf04`.
Branch: `codex/mujoco-tyx-t4-audit`, based on the accepted full source `73a85f8affccde5ebf21a4629845d06af6949479` and research commit `b0e25476876218d2649e4129c457eb28405188a2`.

This lane authored exactly one new implementation module, `src/phymeo/backends/mujoco_tyx/_constraint_solver.py`, plus this new report. It did not change existing product source, tests, shared runtime, geometry, policy, assets, or accepted reports. The old untracked `docs/development/mujoco-tyx-t1-review.md` remains untouched.

## Contract and interface

Authority is the parent plan `docs/superpowers/plans/2026-11-10-mujoco-tyx-mixed-constraints.md` at parent `2fe2e3f8`, read in full before authoring. Subsequent parent clarifications preserve logical efc_capacity=0 with at least one allocated row, the fixed ten diagnostics, and the stated mass-residual gate. The research [derivation and fixed-source references](mujoco-tyx-t4-research.md) is background; the parent's frozen ABI wins wherever its fields differ.

The new module exports exactly one `@tx.kernel solve` with the frozen argument order:

```text
count,nv,efc_capacity,budget,tolerance,
mass,bias,passive,applied,efc_data,efc_J,counts,
physical_factor,mass_work,smooth_acc,
face_matrix,face_factor,face_rhs,face_forward,trial_acc,trial_force,
working,previous_working,previous_force,
constraint,total_applied,physical_acc,diagnostics,status
```

The kernel only reads actual compact rows counts[w,2]. It updates efc_data force/gradient slots 9/10 and the provided output/scratch arrays. It has no contact primitive, body, material, cone-dimension, source joint-kind, or native API branch. All normal, pyramidal, limit, and dry-friction behavior comes from the caller's full J/aref/R/bounds. It imports only TYX.

Logical EFC capacity may be zero. Storage width is max(1,capacity); DOF width is max(8,nv). Force/gradient, row scratch, matrices, DOF scratch, and ten diagnostics are cleared before any solve. Empty-row states still factor physical M, compute smooth/physical acceleration, check its equation, and return zero constraint force. Upper=0 with finite_upper=0 is not an upper bound.

## Implemented algorithm

1. Validate actual counts, active flags, bound masks, zero-force feasibility, positive finite R/D, finite J and physical inputs. Positive R is authoritative; the face uses its reciprocal without editing R or substituting the saved D field.
2. Factor physical M once. Compute smooth acceleration from applied+passive-bias.
3. Start at feasible zero force. Interior dry-friction rows start free; ordinary unilateral rows start at their lower bound.
4. From the current stored force, rebuild J-transpose force, solve physical M for acceleration, and independently evaluate full KKT and physical mass residual. The last primal-face trial acceleration is never the final checker.
5. After a complete feasible face, release all KKT-violating bound coordinates together. During an internal blocking repair, do not immediately release the coordinates just bound.
6. Assemble H=M+J_F-transpose D_F J_F and rhs=rhs0+J_F-transpose D_F aref_F+J_B-transpose f_B. Factor the nv-dimensional H and recover all free forces. No E-by-E matrix, inverse, or factor is allocated.
7. Find the minimum feasible step over all lower/finite-upper bounds. Bind every coordinate with that same computed blocking ratio. Keep all other coordinates on the same feasible segment; no blanket force clamp or regularization repair is performed.
8. Count every face factorization against the original budget. An extra check pass permits success after the last allowed solve. A repeated identical force/working set is marked stalled and independently checked once before failure.
9. A successful publication candidate is built only from recomputed physical acceleration and stored force. Actual stored normalized diagnostic values decide acceptance.

Each face costs O(E*nv^2+nv^3), with O(nv^2+E) additional scratch beyond the retained J. This is a loop/dimension description, not a benchmark. A block release avoids a mandatory one-row-per-solve startup, but does not guarantee convergence in a given native iteration budget.

## Diagnostics and failures

The frozen ten fields are unchanged:

| Index | Meaning |
| --- | --- |
| 0 | Actual face factorizations attempted |
| 1 | Raw KKT violation divided by the declared global scale |
| 2 | Raw maximum KKT violation |
| 3 | Maximum force-bound violation, zero for feasible force |
| 4 | Actual EFC count |
| 5 | Current free-row count |
| 6 | Converged flag |
| 7 | Original face-solve budget |
| 8 | Normalized physical mass residual |
| 9 | KKT scale |

KKT scale is max(1,max_row(abs(aref)+sum_dof abs(J*a)+abs(R*f))). Lower-bound gradient must be nonnegative, upper-bound gradient nonpositive, and interior gradient zero. Classification uses the actual stored force's bound equality, not just the working-set label.

Physical mass normalization is max_i abs(M*a-rhs0-constraint) divided by max(1,max_i(abs(rhs0_i)+abs(constraint_i)+sum_j abs(M_ij*a_j))). Both normalized gates use the passed FP32 tolerance, without changing its value. Only normalized mass residual is directly stored in diagnostic slot 8; raw mass residual and its scale are independently reconstructible from the retained full M, inputs, force/J and physical_acc. This report does not claim two nonexistent device scalar outputs.

Status 1 is capacity, 2 invalid/nonfinite data, 4 physical-mass factor failure, 5 stall/force feasibility/mass residual failure, 6 face-factor failure, and 7 budget exhaustion. Matrix placeholder arithmetic following a diagnosed bad pivot cannot produce success: status remains failed and no subsequent physical solve is admitted. No positive-pivot floor, R adjustment, hidden force bound, host solve, or fallback exists.

The kernel only signals per-world status. The parent runtime owns whole-batch atomic publication and rejects all worlds on any failure. Failed intermediate scratch is diagnostic material, not an accepted complete solution. This lane did not implement or independently execute the parent transaction.

## Independent hand-worked checks, not executed tests

All examples below use a single DOF, physical M=1, rhs0=0, and exactly stated rows. They exercise the common kernel mathematics without relying on native solver output.

| Case | Exact solution or transition |
| --- | --- |
| Zero rows | a=0, constraint=0, KKT=0, no face factorization; nonzero rhs0 similarly gives a=rhs0 |
| One unilateral row J=1,R=1,aref=2 | H=2, a=1, f=1, gradient=0 |
| Two unilateral rows J=(1,1),R=(1,1),aref=(3,1) | Block release gives trial f=(5/3,-1/3). Alpha=0 binds row 2. Next face gives f=(3/2,0), a=3/2, gradient=(0,1/2). With budget 1 the final check must reject with status 7 |
| One dry row J=1,R=1,aref=3,bounds[-1/2,1/2] | Unconstrained f=3/2; alpha=1/3 binds upper. f=a=1/2 and g=-2 satisfy upper KKT. Reversing aref gives lower f=-1/2,g=2 |
| One dry interior row J=1,R=1,aref=1/2,bounds[-1/2,1/2] | f=a=1/4,g=0 |
| One unilateral plus two dry rows, all J=R=1,aref=(4,0,0), dry bounds[-1/4,1/4] | First H=4 gives a=1 and trial f=(3,-1,-1). Alpha=1/4 binds both dry rows simultaneously. Repaired H=2 gives a=7/4 and f=(9/4,-1/4,-1/4); gradients=(0,3/2,3/2) obey lower KKT |
| Zero J retained, R=2,aref=1, unilateral | H=M, a=0, f=1/2, g=0 despite no generalized force; deleting this row would be a different regularized problem |

These calculations establish intended algebra and branch behavior only. FP32 compilation, source-language lowering, numerical conditioning, progress on G1, and test expectations require the parent's sole execution slot.

## Verification and handoff

Source SHA256: `7be83c6776002dfda33f26666a401abc113f45768ced9b97121dc8118db53398`.
The complete source was manually read; file bytes were compared with the complete authored string and matched exactly. Git staged scope contains only the new module, and `git diff --cached --check` passed before the source commit. These are text/file checks, not lint, compilation, imports, or functional tests.

No project command was executed. In particular there was no TYX import, offline numeric probe, native call, collection, pytest, formatter, lint, build, install, or profile. Runtime semantics remain CODE_READY_TESTS_PENDING. The parent must integrate the separate row assembler and shared device ABI, preserve first failures, run the predeclared generic oracle matrix, and only then run the original G1 integration cases.
