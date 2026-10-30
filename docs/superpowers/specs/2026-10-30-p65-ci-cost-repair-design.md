# P6.5 CPU CI cost repair: exact arithmetic equivalence

2026-10-30. User decision: retain the CPU job limit of 30 minutes and optimize runtime. This repairs delivery of the existing [thick friction contract](2026-10-29-p65-thick-friction-design.md); it introduces no new physical capability.

## Evidence and choice

Baseline HEAD127cd28, product307 digest cd6d77917dee8af09f13a430f4fdd4a5b744461874364a142aff0d8e6722d84b. Exact CI40950637127 for5150c3e was cancelled: all8CPU jobs reached30 minutes. Its incomplete test result is neither a numerical failure diagnosis nor a pass. Documentation40950637280 succeeded.

The finished single-step cProfile baseline (artifacts/p65-ci-optimization/baseline-A.json/.prof/.log/.exit) accepted A, mu=.25, h=.125, with53 held and1 terminal replay. Profiling wall46.55563850002363s included instrumentation. Quadratic.__post_init__ ran1,275,544times and accumulated18.3744367s; Fraction arithmetic dominated. State fingerprint8c992db91af1bb46e8498b3b4a8d75e2cf50377ace340cb5665b89afd0aa0551; report fingerprintfb348514fb1879fe04998fc93253cd54c3cf691cb12151a96433266ad9237880.

Options considered: reuse more test fixtures, cache geometry by trial, or remove redundant exact scalar work. Existing immutable reference caches already avoid repeated normal trajectories. Trial/geometry caching changes invalidation obligations. Choose exact scalar specialization first because pure canonical values have no state, lifecycle or cache identity. No global cache or geometry changes are needed for this attempt. If measured improvement is insufficient, report the next profile before broadening scope.

## Binding invariants

- Production scope: only src/phymeo/geometry/_prism_exact.py. Tests: new tests/test_prism_exact.py; existing tests are unchanged. Reports/artifacts may be added. No shared old solver/runtime/kernel/port/law/history, dependency/pin or workflow edits.
- Preserve all accepted constructor conversions, positive-radicand validation (including b=0 with c<=0), perfect-square reduction, canonical Fraction a/b/c, cross-field errors, division-by-zero behavior, exact signs/order/hash, outward rational bounds and promoted float bits. No epsilon, float intermediary, looser bounds, larger physical tolerance or rounded axis.
- Rational operations may avoid zero arithmetic and root classification. Operations on already canonical same-field operands may avoid reclassifying the unchanged non-square radicand using a private invariant-preserving constructor. All public construction remains validated. Cancellation must restore b=0,c=1. Different nonzero fields still reject; rational/irrational combinations preserve the nonzero field. Preserve expression results exactly, including negative values, huge/subnormal binary inputs and near cancellation.
- Preserve every physical proposal, score, native replay, iteration/rebuild budget, selected candidate, accepted/rejected result, state/history, diagnostics, impulses, CCD interval/certificate and mapping audit. R1-R4 and diagonal reachability limitation remain unchanged.
- CI timeout30/matrix/commands/examples/required/fail-fast are immutable for this repair. Workflow baseline SHA256 c917835c91aeb4dad328bd49fc266f9238b638b5327346d06d8e9edad7ba7297 must remain identical.

## Acceptance

Meaningful mathematical tests precede code: independent coefficient/rational identities, canonical forms, comparisons, strict exceptions, exact bounds and float bits. Add a deterministic cost control that rational construction/bounds and already canonical arithmetic do not repeatedly classify unchanged roots; observe real RED on the old implementation rather than invent a functional RED. If a new correctness test already passes, record it as an initial GREEN.

An artifact-only differential harness must load the frozen baseline module from127cd28 (record exact byte hash) and compare deterministic rational/irrational/mixed/cancellation inputs, all supported operations and exceptions, coefficient tuples/hash/sign/order/bounds/float.hex. Include independently calculated cases rather than only same-algorithm comparison. No baseline interpreter inside production.

Repeat the identical single-step profile with a new tag and require the exact state fingerprint, held53/terminal1 and complete CCD contents to match. For reports use the R5 typed-timings exception below: capture raw full reports/fingerprints, and require exact complete canonical report equality across the planned fresh old/new observations; do not require equality of inherently variable raw timing fingerprints or claim normalized parity with the unavailable original baseline report payload. Also make a small paired unprofiled old/new comparison in fresh processes (three runs per version, same fixture/configuration, resources closed, old baseline injected only in the observation process). These are bounded CI-cost observations, not formal solver P95 or a cross-backend speed claim; no900 sample rerun.

Because every thick consumer uses this arithmetic, run all four host thick test files plus the new arithmetic tests once on the final snapshot, including prescribed A/C windows/rebuilds and all12late faults. Run all tests/test_thick_robot_tyx.py on actual interpret thenCUDA with --require-tyx, fresh basetemps, no skips/fallback. Old finite/normal/friction modules do not import this arithmetic; do not repeat their whole suites or old long windows. Record source manifests, raw exits, nonzero collection and actual snapshots. Any failure requires diagnosis, no gate removal.

After the original implementer releases numerical/index ownership, main inspects the narrow diff and evidence against these invariants (no repeat whole-stage fresh review), checks build/external-wheel source parity at the new source, commits/pushes, and verifies the new exact head CI and Documentation. Only successful original full CI within30 minutes closes delivery. Local observations cannot substitute for it.


## R5 parent observation ruling

Existing unmodified SceneStepper._timed records perf_counter wall seconds in SceneSubstepReport.timings. Therefore the raw complete report fingerprint is inherently different between runs: baseline profilefb348514fb1879fe04998fc93253cd54c3cf691cb12151a96433266ad9237880 versus optimized profilec518c6b1048cd2a0ee5b0490834513fd35a030fdade015ab1f03364638c84eb9, despite equal state,53held/1terminal and completeCCD. Preserve these raw values; do not claim normalized equivalence with the original baseline report, whose complete payload was not saved.

For the already planned3old/3new fresh-process observations, capture full raw and canonical reports. Artifact-only canonicalization may omit ONLY the `timings` field on exactly the `phymeo.runtime.program.SceneSubstepReport` dataclass, recursively wherever such typed reports occur. No generic dict-key removal. Preserve raw full reports/fingerprints/timings separately. All other report fields, diagnostics, residuals, iterations, inputs/state/history and completeCCD remain exact and must match old/new. This corrects the observation contract only, without changing runtime, physical gates or production scope. Original baseline profile is never rerun.
