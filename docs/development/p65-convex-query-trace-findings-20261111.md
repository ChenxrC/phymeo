# COL1 convex-query trace findings — 2026-11-11

Scope: **diagnosis only** from the six completed CUDA captures at
`e7c247c15647724cc622872aa85c6e33443bde9d`. No new query, geometry
recomputation, numerical probe, test, import, lint or algorithm edit was run by
this review. Bounded .NET gzip/JSON reads inspected saved FP32 fields and their
raw bytes. All six records say `completed=true` and
`same_state_geometry_equal=true`: enabling diagnostics preserved the ordinary
same-state query geometry in these captures. This is not acceptance of the
remaining geometric failures.

Field numbers refer to [the trace protocol](p65-convex-query-failure-diagnostics.md).
Below, iteration numbers are zero-based. Recorded quantities are quoted rather
than recomputed. Proposed repairs are explicitly hypotheses.

## Evidence identity

Parent canonical root:
`artifacts/p65-parity-n3-rob1c-col-trace-20261111-01/`.
Each entry is the original `complete.json.gz`; no decompressed copy was saved.

| Run / fixed identity | SHA256 |
| --- | --- |
| 29-col-capture-0 / Y-ordinary | c9feefb123f160f929868a46ee6d17da00101ecb730d303e75b64f5069e396d3 |
| 30-col-capture-1 / Y-affine | 04ddd58b91888943a4ac8fbf8623ed0a350627375168b18e46788c1a74426503 |
| 31-col-capture-2 / Y-floating | b57bca54f09d1bc8610cd86eed3f6c6127b6e621bc351860fb45d8a9ff8eb066 |
| 32-col-capture-3 / Y-floating_affine | 1d1aa3414655902f14595ba1efc7a1fc94d7f55389dbbe81e842d7abb3941778 |
| 33-col-capture-4 / sep-HN | e1b95d4bd6de8f90fed63f1273373da58516d862e0cd9903068d2771501744c0 |
| 34-col-capture-5 / rotated-reversed-HN | 22ebf2914e2a29f0a49403c80705fdb9440d1757f80299522bb0e793b9e30999 |

The `source-2-_tyx_convex_query_kernels.py` snapshots in these directories
contain the actual executed source. Relevant snapshot lines: 748 duplicate
check; 846/856 triangle Gram solve/raw weights; 890–917 clamping, normalization
and weighted-vertex q reconstruction; 924 equal-square tie; 1007 separation
exit; 1040 simplex compaction; 1078 GJK budget exit.

## Four Y failures: distance success is not normal success

All four fixed world17 histories end after **7 support calls**, with
`meta=[1,1,0,5,7,0,7,1,0,7,...]`. Last iteration 6 has:
entry simplex 2, appended support 1, candidate simplex 3, **duplicate=0**,
selected mask **4** (the newly added point), selected count 1, decision **2**
(the support-plane distance exit). It is not a duplicate exit, EPA failure,
capacity exhaustion, or a SUPPORT_BALL exit.

| Recorded final value | ordinary / affine | floating / floating-affine |
| --- | --- | --- |
| closest distance | 0.03125206008553505 | 0.0312444306910038 |
| support-plane bound | 0.03125182166695595 | 0.031244194135069847 |
| distance minus bound | 2.384185791015625e-7 | 2.3655593395233154e-7 |
| saved independent normal error | 0.0031187900921125196 | 0.0031195516802129354 |
| saved independent gap error | 1.5273690223693848e-7 | 1.5273690223693848e-7 |

Both distance widths satisfy the unchanged 8e-7 internal exit. Both normal
errors exceed the unchanged 0.002 external gate. Ordinary/affine publish
`(0.9999951124191284, 0, -0.0031187862623482943)`; the two floating layouts
publish its corresponding cyclic orientation with transverse component
`-0.003119547851383686`.

The ordinary last iteration records incoming q
`(-0.03128558024764061, 0, -2.4394152205786668e-5)`, but chooses outgoing q
`(-0.03125190734863281, 0, 9.746849536895752e-5)`. The bound was computed from
the incoming q and that iteration's support; the published normal uses the
outgoing q. Candidate edges 5 and 6 have actual negative raw weights
`-0.08310019969940186` and `-0.16698014736175537`, and triangle 7 has negative
raw weights too; all are marked unusable. Thus the final point selection is
visible, not an inferred missing simplex branch. The cylinder support stays at
the same rim point in these last iterations; the opposite sphere support is
still moving.

**Conclusion from the recorded decision:** a valid-looking distance interval
allowed a still-inaccurate outgoing normal to publish. A depth-only success
cannot serve as this normal gate. Four shared termination signatures do not
justify four layout-specific fixes. The trace does not prove that tightening
the distance threshold or restoring the withdrawn strict FW gate would work.
Field23 (about 1.056e-6 here) is explicitly a diagnostic incoming-q expression,
not an actual gate and not a certificate for the outgoing normal.

## Two HN failures: a repeatable simplex-reduction cycle

Both HN captures report **GJK_LIMIT**, 64 support calls, 3 retained vertices,
0 EPA iterations and 0 EPA faces. Starting at iteration 3, every one of the
416 stored float fields is **byte-identical through iteration 63** within each
capture. This comparison uses slices of the saved raw FP32 bytes, not
recalculated geometry.

The unrotated repeated state is:

- incoming and outgoing q =
  `(-8.177109123153059e-8, 3.292074879368556e-8, -0.03125)`;
- squared norm = `0.0009765625`, distance = `0.03125`;
- bound = `0.031248489394783974`, width = `1.5106052160263062e-6`;
- new support w =
  `(0.548933207988739, -0.07002487778663635, -0.03125)`;
- duplicate=0, append=1, candidate count=4, selected mask=7, retained count=3.

The newest point is discarded every time. The next iteration asks for exactly
the same support again, but the duplicate test considers only the retained
simplex, so it correctly reports zero under its current definition. Merely
raising the iteration budget cannot change this recorded cycle. Treating
historical support repetition or equal material IDs as success would not be
a valid geometric certificate; smooth support points can share material IDs.

The decisive competing triangles are fully recorded:

| Field | mask 7, retained | mask 13, rejected |
| --- | --- | --- |
| usable | 1 | 1 |
| actual Gram denominator | 0.2209833413362503 | 0.30157914757728577 |
| candidate q | (-8.177109123153059e-8, 3.292074879368556e-8, -0.03125) | (-3.980295559813385e-8, 1.062319743283524e-8, -0.0312500037252903) |
| candidate square | 0.0009765625 | 0.0009765627328306437 |
| better flag | 1 | 0 |
| stable difference diagnostic for mask13 | — | 2.328245651828098e-10 |

All four actual simplex/support points have the same exact normal coordinate
`-0.03125`. Mask13 nevertheless reconstructs that coordinate one FP32 ULP
away, while reducing both tangential components. Its squared norm is therefore
strictly larger, so the existing **equal-square** comparison is not entered.
The recorded positive diagnostic difference also does not justify choosing it
by simply applying that comparison unconditionally. The tetrahedron has
determinant 0 and is rejected; that is consistent with this coplanar simplex.

The rotated/reversed capture has the same denominators, flags, weights,
width and termination, with the q/support coordinates cyclically permuted.
The trace therefore identifies a generic reconstruction/reduction sensitivity,
not a cone/hull registration omission or an EPA-capacity shortage.

**Inference, not a newly executed result:** the current normalized
`sum(w_i * vertex_i)` reconstruction leaks out of a constant-coordinate affine
hull and leaves enough tangential noise to steer the support query toward a
distant rim point. The captures establish the comparison and discarded-point
cycle. They do not separately isolate normalization rounding from weighted
product/sum rounding, or prove that one alternative arithmetic form is enough.

## Bounded next repair hypothesis

1. First test a **generic affine-simplex reconstruction/projection**: form the
closest point using a chosen simplex vertex plus weighted edge differences
(and, if needed, a bounded residual correction), instead of summing all
weighted absolute simplex coordinates. Apply it by simplex dimension, not
shape label, world axis, or fixture. Preserve original material weights and
reconstruct both witnesses consistently. This directly targets the recorded
affine-hull leakage and repeated rejected support; it is not yet a demonstrated
fix. Keep degeneracy, weight and resource gates.
2. After that reducer is stable, separately check normal convergence against
the **same outgoing candidate** whose normal would publish. If an additional
support is needed, it must consume the existing support/iteration budget.
The four Y records require this separate stopping question; improving the HN
arithmetic alone is not shown to solve them. Do not blindly restore the strict
FW gate already withdrawn after regressions, or use depth as a uniqueness proof.
3. Start with the six unchanged captured inputs and original gates. Any proposed
generic change must retain edge/face ties, bilateral witnesses and truthful
UNKNOWN/resource outcomes. No increase to 64 GJK / 192 EPA / 64 vertices /
128 faces, no tolerance change and no fixture retuning is justified by this
diagnosis. If the bounded candidate repair fails, retain the concrete gap
rather than introducing a third unbounded algorithm.

This review changes only this document. Product repairs and their execution
remain a separate parent decision.
