# MuJoCo-TYX T3b: fixed primitive manifold research

2026-11-10; source research and manual control-flow review only. Baseline `275872beca81767bdc2ee8a1d0ca52fb5ae9c100`; no project import, numeric probe, collection, test, formatter, build or installation. This extends [T3a fixed-source review](mujoco-tyx-t3-research.md) along the [MuJoCo-TYX expansion plan](mujoco-tyx-expansion-plan.md). The T3a coupled normal solver, condim1 admission, nonnegative mixed margin, implicit-pair filters and existing option restrictions remain unchanged.

## Fixed sources and frozen interface

Primary references are the MuJoCo **3.3.7** functions in [engine_collision_primitive.c](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_primitive.c), [engine_collision_box.c](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_box.c), and the dispatch/postprocessing in [engine_collision_driver.c](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_driver.c). These three pairs dispatch directly to `mjc_CapsuleCapsule`, `mjc_CapsuleBox`, and `mjc_BoxBox`; they do not require a new convex/GJK route, MULTICCD option, or replacement contact law.

Comparison files are fixed installed **mujoco_warp 0.0.2** `collision_primitive_core.py` and `collision_primitive.py`, in the parent's existing optional-deps archive. Relevant functions begin at core lines 181, 564, 1074 and wrappers 779, 1380, 1463. Warp is useful translation scaffolding; the differences below preclude treating those wrappers as the CPU manifold oracle.

| Pair | Frozen result slots per possible pair | Identity |
|---|---:|---|
| Capsule-capsule | 2 | Current native compact output ordinal |
| Capsule-box | 2 | Current native compact output ordinal |
| Box-box | 24 | Current native compact output ordinal after native bad-point/duplicate removal |

The parent's final decision is **2/2/24**, replacing the initial eight-slot suggestion derived from Warp. The existing total candidate capacity remains 32; a declaration whose reserved slots exceed it is rejected, not truncated. A box-box raw enumeration has at most `4 lines * 2 rectangle axes * 2 sides + 4 rectangle corners + 4 incident corners = 24` candidates in either branch. A 32-point scratch region therefore covers the enumerator without claiming a final eight-point theorem. Native [mjMAXCONPAIR is 50](https://github.com/google-deepmind/mujoco/blob/3.3.7/include/mujoco/mjmodel.h); it does not truncate this bound.

The agreed device interface adds an `extended_contacts` kernel producing `raw[world,slot,11]` with scratch `[world,64,3]`: candidate storage in the first 32 vectors and working transforms/vectors in the second half. Existing normal row construction consumes these geometric results and supplies the unchanged full current J/R/aref and coupled solve. Existing six-family models retain five launches; models with an extended pair use six. The feature index is an instantaneous compact ordinal, **not a persistent material address across frames**.

## Capsule-capsule

CPU uses scaled half-axes and a two-segment closest-point calculation. The nonparallel branch, selected by `abs(det) >= mjMINVAL`, produces one sphere-sphere candidate after the native sequential clamping. The parallel branch tests endpoints in order: capsule 1 positive, capsule 1 negative, capsule 2 positive, capsule 2 negative. Each projects onto the other segment; detection stops once two contacts have been obtained. A failed margin test does not consume an output ordinal. Preserve duplicate sites if this primitive emits them; box-box cleanup is not a global deduplication rule.

Sphere-sphere supplies distance, midpoint, and the geom-orientation-aware coincident-center fallback already reviewed in T3a. Warp 0.0.2 instead returns one closest-segment candidate and explicitly leaves the parallel branch unimplemented. Do not inherit its single-result layout or epsilon-modified closest-segment routine. Positive native shape dimensions and finite uploaded geometry remain prerequisites; arithmetic failure is a failed candidate, not a sphere substitution.

## Capsule-box

The CPU implementation first checks the negative then positive capsule endpoints against box faces, and then searches box edges with **axis j outermost, corner i inside**. It retains strict-improvement tie decisions, including the `bestdist - mjMINVAL` edge test. Warp reverses those edge loops and uses a different initial best-distance bound, so tied configurations need the CPU ordering.

The retained closest feature determines whether a second segment location is constructed. A closest endpoint inside the box has `clface == -1` and no second point. The optional second sphere location uses `bestsegmentpos + secondpos`, not `secondpos` alone. Each sphere-box test is margin-filtered independently and compacted in production order. The maximum is two. The old diagnostic-only loop between closest-feature search and second-point selection has no live output dependency and need not be translated; copying its unused divisions would create artificial failures.

## Box-box

Preserve native SAT face/edge selection order and strict tie tests, then the selected clipping branch. The cross-axis preference includes the literal `1-1e-12` factor. This is FP32 device arithmetic, so near-tie branch boundaries are not a bitwise FP64 promise.

Three concrete CPU/Warp differences matter before comparison:

- In the CPU face branch, the point's local depth is halved for the contact midpoint and that halved value is also assigned to `con.dist`. Warp returns the earlier full `depth` value. Preserve the fixed CPU observable; do not silently correct it or compensate in the solver.
- CPU's first incident corner is appended without the XY strict-interior checks applied to later corners. Warp applies those checks to every corner.
- Native raw candidates then pass the driver cleanup below. Warp's eight-result wrapper writes its contacts without that cleanup, and its edge branch additionally differs in projected/unprojected coordinate usage. Reuse requires line-by-line reconciliation, not a wholesale copy.

## Required driver cleanup

Only box-box receives this postprocessing. Let `s = box_halfsize + mixed_margin` and let `a` be the absolute local contact position. Native [mju_outsideBox](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_util_misc.c) classifies it as outside (`1`) if any `a > 1.01*s`, inside (`-1`) if every `a < s/1.01`, otherwise boundary band (`0`). The comparisons are strict as written; they are not an epsilon contact test.

Reject a raw point when `(outside1 == 1 and outside2 != -1) or (outside2 == 1 and outside1 != -1)`. Among the remaining points, compare all three **world position components for exact equality**. If a later good point equals an earlier good point, mark the earlier point bad and stop that earlier point's search. Finally compact the survivors in original order. Thus the later duplicate survives; sorting, epsilon merging, keeping the first duplicate, or applying this removal to capsules changes native behavior. There is no subsequent generic recomputation of the distance or normal after this cleanup.

## Independent tests and review targets

The source path is ready to implement; these cases identify likely translation defects without requiring an additional theory stage:

- Capsule-capsule parallel side support with two forces, near-parallel and crossing configurations, endpoint-only contact, unequal half-lengths, separation/reentry, and compact ordering when an earlier endpoint candidate is absent.
- Capsule-box side/face, endpoint, edge/corner, interior endpoint, oblique second site, and ties whose result depends on CPU edge-loop order. Check full detected manifold before force comparison.
- Box-box rotated face manifold, edge-selected branch, equal-depth axes, exact duplicate positions, outside/boundary-band removal, positive margin, and complete finite-face withdrawal/reentry. CPU face-distance expectations must come from fixed native observations, not a textbook OBB gap substituted into the oracle.
- Apply the unchanged common QP to all retained rows; verify current full generalized J, R, actual constraint force and both body COM wrenches. Include eccentric motion and coupled material/robot DOFs rather than only centered translational examples.
- Preserve the raw candidate/count distinction from active constraints and positive force. Exercise reserved-layout capacity and actual detected/active capacity independently, selected reset, clone's next step, and late-world rejection without partial state/force publication.

Plain fixed-source caches are under the audit worktree's ignored `artifacts/p65-mujoco-tyx-t3b-research-20261110-01/upstream-3.3.7/`. Source files were read as text, never built or run. Author code review and actual parent-only validation are separate from this research result.
