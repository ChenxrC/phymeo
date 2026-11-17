# I3e: TYX clipping topology with exact HOST boundary refinement

2026-11-17, exp07c310a3. Root owns shared source and the sole serial execution slot. Original agents only author/review. No dependency upgrade, remote CI, installs or dev changes.

## Design

Installed TYX supports f32/i32 arrays, not f64. Preserve the original exact-from-stored-FP64 contact contract: a new opt-in persistent intersector uploads positively normalized tetra barycentric coefficients evaluated at each rigid triangle vertex. One TYX lane per cell/face performs bounded polygon clipping using lane-owned two-bank scratch (at most seven output vertices). It returns active halfspace descriptors, not trusted world coordinates or pressures.

HOST reconstructs each positive polygon vertex from two independent original exact lines. Check all seven halfspaces, every claimed tight constraint, unique vertices, positive global convex orientation, and a genuine nonconstant common support line for every edge. Constant-zero coplanar constraints cannot certify edges. These checks prove both containments of the convex intersection. Then reuse original canonicalization, shared-face owner/gradient rules and triangle publication.

Empty, low-dimensional or ambiguous output is not a negative collision certificate: explicit fallback='exact_host' recomputes that pair with the original exact clipper and records a reason; fallback='reject' raises device_unresolved. No silent device-success label. Input/triangle/solver capacities never fall back or truncate. Failed calls retain last_result/routes; transient scratch and diagnostic cost counters may change.

## Integration and tests

Both single and heterogeneous scenes may receive an explicit intersector with matching runtime mode. Default remains existing HOST geometry; lifecycle remains caller-owned and reset/restore never caches device physics owners. Intersector state is scratch and is not part of physical checkpoints.

Fixed tests compare all original seven static cases, transformed/finite/shared boundaries, canonical triangles/pressure/area/wrench, strict device-positive path without HOST clipping, explicit fallback and rejection, forged boundary certificates, capacities, and failed readback. New short scene tests compare same inputs under HOST versus mixed clipping for continuous heterogeneous steps and record complete step wall time plus intersector phase/transfer counts. No speedup requirement; do not tune inputs or tolerances to win.

## Exit

Focused interpret and CUDA tests, changed seam checks, source review, raw receipts and same-source snapshots. Publish measured one-time/JIT and warm costs separately where available, exact fallback counts and remaining HOST work. No claim of all-device geometry, CCD, GPU scale superiority, friction or FFT material coupling. Next step depends on measured costs, not kernel timing alone.

## Completed 2026-11-18

Implementation 4489f456: 37 new + 16 affected compatibility tests passed, seven Python files pass final static checks. Positive topology is exactly refined; explicit fallback remains necessary (36 device-certified polygons / 156 fallback pairs in the 192-pair cost window). CUDA warm11 full steps: HOST0.9153292s, mixed0.9268371s; no speedup claim and no default switch. Resource admission now requires OPEN, including clone/separate scene bindings. See [report](../../development/area-pressure-device-topology-report-20261118.md).

Next batch priorities, based on these measurements: cache immutable asset preparation while preserving current geometry/identity validation; certified empty-pair filtering and batched world/pair submission; repeat same-output full-step cost comparison on fixed expanded scene sizes before any automatic/default policy change. Do not optimize a standalone kernel while HOST preparation/fallback dominates.
