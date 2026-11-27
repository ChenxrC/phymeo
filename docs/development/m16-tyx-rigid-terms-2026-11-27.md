# Original rigid terms on TYX CUDA

The next component removes captured body transforms, COM Jacobians and mass
matrices from the saved-state GPU contact/acceleration check. It adds
`compile_rigid_model` and `RigidWorkspace`, without changing the full-model
compiler's integration admission. No policy sampling or integration is enabled.

Only original static model fields are processed on CPU. Per-state FREE,
HINGE and SLIDE transforms, root mocap poses, geometry transforms, COM and
Jacobians, physical mass (including armature), and gravity/Coriolis bias run
on TYX CUDA. The bounded tree supports one joint per body and world-child
FREE/mocap roots; BALL, multiple joints per body, flex and plugins are rejected.
Hinge displacement must be within 32 radians, an explicit numerical domain
check, not a clamp or change to the original physical joint limits.

Quaternion composition, normalization, angle reduction, sine/cosine and pose
transforms preserve a pair of FP32 lanes. This is compensated FP32 arithmetic,
not native hardware FP64. Device qpos/mocap inputs retain both lanes. Velocity,
Jacobian, mass and bias currently use FP32. The explicit host convenience
method splits host state; the device path never downloads state for calculation.
It reads one status word per world to admit the result. Buffers are reused;
results expire on subsequent attempts, and synchronous loans prevent workspace
reuse/close. The output velocity is an owned epoch snapshot, so later producer
buffer reuse cannot silently change an existing result.

## Verification

17 focused tests pass with required CUDA execution (including host rejection
tests): moving FREE/HINGE/SLIDE trees, nonunit quaternion inputs, offset anchors
and COM, nonzero Coriolis, moving mocap poses, angle-reduction quadrants,
mass symmetry/SPD/kinetic energy, input immutability, result/loan lifetime,
device input validation and unsupported domains. Earlier 16-test results
overlap; they are not an additional count.

The same eight original M16 states pass independent native MuJoCo 3.3.7
forward comparisons. Criteria were registered before execution: pose 2e-11
absolute, Jacobian 2e-6 absolute, M/bias component error at most
`2e-6 * (1 + abs(reference))`, plus symmetry/SPD/kinetic energy.

| Quantity | Maximum absolute error |
| --- | ---: |
| Geometry position | 1.0214052e-14 m |
| Rotation component | 2.3203662e-14 |
| COM position | 9.7699626e-15 m |
| COM Jacobian component | 5.2115811e-8 |
| Mass component | 3.8231784e-7 |
| Gravity/Coriolis bias component | 5.2137937e-6 |

Jacobian, mass and bias entries have coordinate-dependent units; these are
not all distances. Every M is symmetric positive definite. The eight-world
workspace owns 3,818,656 bytes and uses three launches per evaluation. This is
a correctness measurement, not a long-run throughput or 256-world admission.

The new resident poses/COM/Jacobians feed the existing mesh and manifold
workspaces directly, and the new resident M feeds the coupled elliptic solve.
All 4,154 native contacts match, with no extra/missing contacts and no point
disagreement above the unchanged 5 micrometre criterion. All eight solves
converge. An independent native FP64 response check using native contact rows
passes the original force-balance threshold 2e-6 (maximum 3.6891836e-7) and
the original per-component acceleration threshold `.005 + 5e-4*abs(native)`.
Maximum acceleration difference is 2.2484259e-4 in mixed generalized units.
Atomic compaction order is not asserted bitwise deterministic.

## Remaining boundary

The composed acceleration check still uploads the previously captured smooth
acceleration (both lanes) and scalar constraints. The new bias is independently
validated, but has not yet replaced the smooth-acceleration construction.
Original actuator transmission/total joint force clamps, passive forces,
equality/limit/friction rows, IMPLICITFAST, transactional integration, reset,
observations/rewards and Torch rollout remain separate work. This is not a
full engine or PPO qualification, and does not prove the historical access
violation fixed.

Next, construct generalized force and the free-acceleration solve on device,
consume the new M/bias, and generate the original scalar rows. Validate those
against these same original states before moving to integration. Preserve the
latest consumed V63 S2 state, optimizer, RNG and unresolved budget entry.

Evidence is in `phymeo_tyx_rigid_terms_20261127`: `rigid002.json`,
`manifold_check002.json`, `acceleration_check002.json`,
`original_balance002.json`, and `pytest002.log`. Initial unsupported rank-5
array and integer-condition lowering errors are retained; both were software
development failures before results, with no integration or training samples.
The unchanged box kernel's verified cache artifact was reused; no old lock was
removed and no old test suite or runner was restarted.

Algorithm semantics are checked against
[MuJoCo 3.3.7 smooth dynamics](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_core_smooth.c)
and [spatial utilities](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_util_spatial.c).
Source attribution is retained in the package NOTICE.
