# Experimental M16 TYX mesh geometry queries

This opt-in implementation runs mesh vertex transformation, broad-phase
compaction and convex GJK/EPA queries on TYX CUDA. It is a geometry development
interface, **not a contact response, dynamics or learning backend**. Existing
MuJoCo-TYX dynamics admission and the original simulator remain unchanged.

## Interface and ownership

```python
from phymeo.backends.mujoco_tyx._mesh_catalog import compile_mesh_query_catalog
from phymeo.backends.mujoco_tyx._mesh_query import MeshQueryWorkspace

catalog = compile_mesh_query_catalog(mujoco, model)
with MeshQueryWorkspace(
    catalog, worlds=8, capacity=16384, tile=4096,
    cache_root=task_cache_directory,
) as query:
    # TYX CUDA FP32 arrays on device 0, shapes [W,G,3] and [W,G,3,3].
    # Synchronize an external producer before entry and retain its buffers.
    loan = query.query_device(device_positions, device_rotations)
    # Optional detached host download for diagnostics, not a simulation step:
    diagnostics = loan.read()
```

The catalog owns immutable copies of the compiled box/mesh support vertices,
original geometry/body IDs, implicit collision filters and margins. It does not
simplify the source hulls. An explicit geometry subset is a diagnostic selection,
not full-scene qualification. Unsupported geometry types, explicit pairs and
non-default collision flags are rejected.

Static vertices and reusable output/scratch arrays remain resident. The device
entry borrows poses without uploading them. The `query()` convenience entry
uploads host poses explicitly. A `(worlds + 1)` int32 count/error vector is still
read back to schedule bounded narrow-phase tiles, and each query synchronizes
before returning. This is not yet a fully asynchronous device simulation loop.
The current private result exposes host diagnostics; a stable device contact
consumer contract is still needed before force assembly.

A result loan expires on the next query attempt, including a failed attempt, or
workspace close. A downloaded result owns its read-only arrays. Candidate order
is unspecified; consumers must use world/geometry IDs. Capacity exhaustion raises
instead of silently dropping pairs. The workspace checks its computed allocation
against an explicit memory budget (256 MiB by default). This bounds owned arrays,
not the entire process, CUDA runtime or JIT cache. `tile=128` remains the default;
4096 is an explicit batching choice, not a new physics parameter.

Each pair is translated and scaled into local coordinates before the existing
generic GJK/EPA kernel runs. Outputs are transformed back into metres and world
coordinates; original support vertex IDs are preserved. This prevents the
generic kernel's unit-scale numerical floor from dominating micron-scale gaps.
The generic kernel itself and its iteration budgets are unchanged. UNKNOWN and
RESOURCE outcomes remain explicit. Even RESOLVED is the kernel's outcome, **not
an independent certificate of the requested physical accuracy**.

## Validation on 2026-11-27

Pinned TYX: `99f6df7c0fd8fdeef77f97df83914da27b64b7ee`, installed with the
repository's hash-locked CUDA dependencies in an isolated task directory.
Hardware: RTX 5090, driver 610.88. Reference MuJoCo: 3.3.7.

The new test module passed 17 cases: eight host cases and nine actual CUDA cases.
Coverage includes original vertex ownership/filtering, invalid layouts, resource
preflight, separation/penetration, world isolation, stale result loans, invalid
poses, overflow, the device pose entry, scale regression, transformed compiled
mesh frames, witnesses and resource release. The scale regression first failed
at millimetre geometry scale before the local-coordinate correction. After
extending only the permitted explicit tile size, the targeted resource case
passed again and the original-scene query below exercised the larger tile.

One saved original reset batch was used: eight worlds, all 1939 geometries,
including 1930 meshes and 14465 support vertices. The native implicit filter
produced 871424 static possible pairs, not simultaneous contacts. The GPU broad
phase produced 51666 candidates. Every native reference contact pair was present
in that candidate set in all eight worlds; this is a recall check for these
saved poses, not a proof for future trajectories.

| Fixed saved-pose query | Unresolved pairs | GPU launches | Query wall time |
| --- | ---: | ---: | ---: |
| Initial global-coordinate implementation | 15296 | 413 | 1.833 s |
| Pair-local coordinates, tile 128 | 1131 | 1231 | 1.028 s |
| Same arithmetic, explicit tile 4096 | 1131 | 52 | 0.271 s |

The last two saved results agree byte-for-byte after sorting by world/geometry
IDs, except for the transient compacted output-row address. The larger tile owns
98827756 bytes of arrays. These are single fixed-query measurements, with query
timing including any first compilation; they are **not** physics-step, PPO or
32/256-world training throughput measurements. Full diagnostic downloads are
excluded from query timing.

Independent FP64 support-plane/witness checks for separation and convex SAT
checks for penetration were applied offline to the saved results, at a fixed
5 micrometre tolerance. They share the compiled support catalog; they do not
independently reimplement asset compilation or certify dynamics.

* 49813 of 50535 RESOLVED pairs passed these geometric checks.
* 722 RESOLVED pairs failed the gap/bound or closure checks; a plausible gap
  estimate alone does not establish a sufficiently narrow error bound.
* 1131 remained unresolved: 986 iteration-budget outcomes, 144 degenerate
  simplex outcomes and one nonunique-normal outcome. Of these, SAT classified
  439 as clearly separated by more than 5 micrometres, zero as overlapping by
  more than 5 micrometres, and 692 within the boundary band.
* The native `mj_geomDistance` comparison also disagreed on 1667 RESOLVED pairs.
  In selected cases it returned zero for shapes independently shown separated
  by a support-plane/SAT bound. Those disagreements are retained; the native
  comparison is not treated as an infallible oracle, nor are all disagreements
  declared GPU-correct.

**Original-scene geometry qualification is therefore incomplete.** No solver
force, friction impulse, time integration, policy sampling or PPO update was
performed. The source model, physical thresholds, checkpoints and consumed RNG
state were not modified. No CPU fallback was added to the GPU query path.

## Remaining engineering work

1. Use the saved unresolved pairs and wide-bound examples to correct thin-hull
   convergence and near-contact witness/normal publication. Keep the numerical
   accuracy criterion fixed and retain uncertain outcomes explicitly.
2. Establish the device result lifetime/order contract and contact manifold
   generation, then integrate normal/friction response and validate dynamics.
3. Cover the original model's equality, mocap, actuator force clamp and
   IMPLICITFAST/elliptic-friction requirements, which the dynamics compiler
   still rejects. This geometry feature does not bypass those checks.
4. Connect resident PD/observations/reward/reset/rollout to the existing model and
   Adam state, then measure actual M16 PPO and 8/32/256-world end-to-end behavior.

Detailed local artifacts: `phymeo_tyx_contact_20261127` under the AlphaNIST task
root, especially `m16_result003.json`, `tile_comparison.json`,
`certificate004.json`, `unresolved004.json` and the retained CUDA test logs.
