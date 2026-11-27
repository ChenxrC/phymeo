# M16 device manifold and elliptic row assembly

`MeshContactWorkspace` connects `MeshQueryResult` to the existing
`EllipticDynamicsWorkspace` through scoped device buffers. It generates contact
positions/frames, relative COM Jacobians, material mixing, impedance, reference
acceleration and elliptic friction parameters. It retains the existing native
box-box multiple-point kernel; box/mesh and mesh/mesh use one witness midpoint.
The current explicit domain is implicit rigid box/mesh pairs, zero margin/gap,
condim3 elliptic contact, and default collision flags including MULTICCD off.
Other domains are rejected before allocation. This does not enable the full
original model compiler, integrator, rollout or training backend.

```python
contact_result = contacts.assemble_device(query_result, motion)
with contact_result.device_view() as rows:
    # Supply the original resident M/smooth/scalar constraint inputs as well.
    acceleration = dynamics.solve_device({**mechanics, **{k: rows[k] for k in (
        "counts", "jac", "aref", "D", "friction", "mu", "eligible"
    )}})
```

`motion` contains same-epoch `com`, `body_jac` (linear then angular COM rows),
and `qvel`. The caller owns these buffers until synchronous assembly returns.
Geometry views now expose the pose buffers that produced the query. They are
read-only loans, not copies; external owners must not mutate them while used.
Each result expires on the next attempt or close. Outstanding loans block
overwrite/close; overflow, invalid motion and invalid geometry cannot be loaned
to the solver. Full host downloads are explicit diagnostic operations. Small
query count/status and contact world-status readbacks remain.

## Validation and the remaining precision boundary

12 new tests cover actual CUDA box multiple contacts and native rows, priority,
direct solref, flat/linear impedance, mesh contacts, native point Jacobians,
mixed box/mesh, an empty world, resident input/output loans, stale results,
overflow, NaN motion, resource preflight and unsupported contact settings.
40 tests pass with CUDA selected, including the 12 new tests plus direct query
and acceleration regressions. Host admission checks are included in that count.

At the eight saved original M16 poses, the new GPU manifolds contain 4,158
contacts, versus 4,154 native contacts. Independent FP64 row assembly at the
GPU's own contact points gives maximum Jacobian error 1.295e-9, R relative error
5.983e-7 and reference acceleration error 1.121e-8. The manifold workspace owns
2,012,220 bytes for eight worlds and capacity 640 per world. Body COM/Jacobians,
velocity, M/smooth and scalar constraints still come from saved native forward
captures. Geometry/manifold/contact rows are newly generated on CUDA.

Direct resident rows feed the existing solver: all eight worlds converge in
9–10 iterations. Independent native constitutive evaluation on those generated
rows gives normalized force-balance residual at most 1.819e-7, below the unchanged
2e-6 check. This proves the assembly/solver boundary for these inputs, not native
full-model acceleration equivalence. Native acceleration componentwise checks
still fail; maximum absolute difference is 7.262435 (mixed coordinate units).

There are 4,145 common pairs, nine missing native pairs and 13 extra GPU pairs.
All 22 activation disagreements have nanometre-scale gaps. Independent SAT at
the original FP64 poses agrees with native presence in all 22. Rounding poses
to FP32 changes the SAT sign in 11; subsequently rounding world vertices adds
five sign flips. This is specific evidence that a 5-micrometre geometry error
bound does not guarantee a zero-gap contact activation decision. SAT is not a
solver contact/force measurement and the rounded-vertex diagnostic is not a
claim of bitwise reproducing the CUDA transform.

Two fixed offline interventions are retained, never used as production fallback:
replacing common contact points alone does not resolve the major world-5 error;
restoring the nine missing native rows/removing 13 extra rows reduces world-5
maximum error from 7.262435 to 0.014205. World 6 still has point/direction-related
sensitivity. Neither intervention passes complete native acceleration comparison.
The latter restores whole missing rows, so it is not a pure sign-only experiment.

Next work should preserve low parts of poses and pair-local transforms, then
resolve the saved contact activation cases and world-6 witness/normal sensitivity.
Do not add a contact margin, loosen success/failure thresholds, inject CPU native
contact rows in production, or treat a converged solver as learning qualification.
There were zero integration steps, policy samples or PPO updates in this work.

The source equations are pinned to MuJoCo 3.3.7:
[collision driver](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_driver.c),
[convex collision](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_convex.c),
[constraint assembly](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_core_constraint.c).

Development evidence is in `phymeo_tyx_manifold_20261127` in the task workspace.
The first test run's duplicate kernel-decoration error and the first independent
check's native contact-view slice error are retained; final checks use the fixes.
