# Original IMPLICITFAST and process-local state commits

The new experimental `compile_implicit_model` / `ImplicitWorkspace` component
consumes the original resident mass and actual forces. It uses IMPLICITFAST
rather than substituting Euler or changing the original damping/PD parameters.
It is not yet a complete engine or a learning release.

## Semantics

For the admitted scalar-joint, fixed-gain/affine-bias, no-fluid domain, the
velocity derivative is diagonal:

`D = -dof_damping + sum(gear**2 * affine_bias_velocity)`.

As in native MuJoCo 3.3.7, actuator and joint-force saturation do not remove
this derivative. RNE velocity derivatives and contact derivatives are excluded
from IMPLICITFAST. The integration solve is
`(M - dt*D) * integration_acceleration = smooth_force + constraint_force`.
Nonpositive D is required by this bounded component; unsupported models fail.

`prepare_coupled` assembles actual contact `J.T @ force` and scalar-constraint
forces from converged device outputs. It does not substitute `M @ qacc`, captured
native forces or a CPU dynamics solve. Native functions compile static data and
serve as independent test oracles only. Mass, force, acceleration, qpos and time
retain high/low FP32 lanes. Velocity remains FP32. The quaternion update
normalizes its input and right-multiplies the local angular increment; bounded
cos/sinc series handle zero spin without division by zero. A half-angle over
0.5 radians is rejected as an unsupported numerical domain, never clipped.

## State and ownership

Two owned state banks hold qpos, qvel and time. `prepare_device` or
`prepare_coupled` writes only the inactive bank, then checks factorization,
residual and finite state. `commit` changes the active bank and generation for
the entire batch. A failed world cannot partially commit the other worlds.
Expired candidates, generation mismatch, invalid input and explicit discard
leave active state unchanged. Live loans prevent replacement or closing.

This is process-local transactionality, **not** durable crash recovery or
rollback of already executed physical steps. The caller must keep rigid,
smooth, contact and acceleration results in the same current state scope;
shape/size checks alone do not prove provenance. The component has no local
reset adapter, activation dynamics, native warmstart state or training rollout.
Existing whole-model compiler gates are unchanged.

## Verification and the actual stopped M16 case

24 new checks passed with CUDA required (including host admission checks):
saturated/negative-gear derivatives, modified mass, quaternion/clock, invalid
state/mass, loans/generation/discard, mismatched capacities, contact and scalar
force assembly, full-batch rejection, and a 32-step small-model sequence whose
moving force computation and integration both remain on GPU. A separate
32-step integrator-only test uses native forces as diagnostic input and is not
claimed as a complete GPU chain.

A single original M16 probe was registered for eight saved reset states and
at most four 1 ms steps, independent cap 64 world integration steps split
32 GPU / 32 native reference. No policy was run. The first GPU step generated
all transforms, contact geometry/rows, smooth/scalar forces, acceleration and
the new state on device. Against an independent native trajectory:

| Quantity | Maximum error / residual |
| --- | ---: |
| qpos | 1.1706125e-10 |
| qvel | 3.0881553e-7 |
| time | 8.6736174e-19 s |
| integration linear residual | 4.9881427e-15 |
| original native force balance | 2.7317845e-7 (limit 2e-6) |

The next query was rejected **before** a second integration. Of 51,650
candidates, world 1 / geom 1249–1487 exhausted 64 GJK iterations. These original
hulls have 98 and 9 vertices; the existing finite-feature fallback accepts at
most 16 vertices per side. Its output remained unresolved rather than being
treated as no contact. The retained state is the first committed step.

A read-only original-vertex support calculation gives a positive separation
lower bound 0.0010202469333074893 m; native distance is
0.0010202469333075008 m. This identifies a separated geometry convergence case,
not a penetration event, force measurement or justification to suppress an
unknown result. The stopped case's qpos/qvel and original local/world vertices
are retained for a focused geometry regression.

Actual M16 integration: 8 GPU + 8 native world steps. Two batch reservations
charged 32/64, settled with 16 known; remaining 32 is not used for retries.
No new AV, policy samples or PPO updates. The four-step M16 qualification
**did not pass**. One successful step is not long-term stability or 32/256-world
training qualification.

## Next boundary

Fix the retained 98-by-9 GJK convergence case, or implement a rigorously
separate GPU separation certificate that does not invent distance/contact
witnesses. Keep the original vertices and contact criteria; do not simply
increase iteration budgets, treat UNKNOWN as separated, or run CPU narrow
phase in production. Verify the saved case before registering a new bounded
moving-state probe. The stopped probe remains closed.

Only after moving-state validation should the same chain gain resident PD,
observations/rewards/done/local reset, rollout and Torch/Adam integration.
No old learning runner, model/RNG reset or direct large training is enabled.
