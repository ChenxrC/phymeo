# MuJoCo-TYX T3 independent normal-contact cases

2026-11-10. Parent baseline `d3f2dbba437c9a6f03300a8dee53d03984a97036`. **PREDECLARED / AUTHORING_ONLY / UNEXECUTED**. The parent owns every project execution. This document precedes the new test implementation and any T3 product result. No tolerances or fixture parameters are selected from product output.

## Fixed contract and independent oracle

The admitted six primitive families are plane-sphere, plane-capsule, plane-box, sphere-sphere, sphere-capsule and sphere-box. Only implicit filtered condim=1 pairs with nonnegative mixed margin are admitted. Native-legal negative mixed margin is explicitly rejected as contact-margin, including a sphere-box negative test; this capability boundary was frozen during authoring before product execution. Each world solves all active rows jointly with physical M; source iteration budget is bounded and the TYX solver is explicitly named normal-dual-active-set-v1. It is not a claim to reproduce native Newton iteration history.

Independent native MuJoCo **3.3.7** compiles the original XML separately. Main models use Euler, h=0.001 s, source Newton, iterations=128, tolerance=1e-10 and dense Jacobians. At each actual stored input q/v/control, fresh native mj_forward supplies geometry, contact parameters, full J, efc_R, aref, smooth and constrained acceleration. Native mj_step separately advances a continuous CPU trajectory from the first actual stored initial state. CPU q/v are never copied back into TYX.

The independent dual matrix is H=J M^-1 J^T+diag(R); b=J a_smooth-aref. For at most four rows a small test-only exhaustive active-set solve independently checks f>=0, Hf+b>=0 and complementarity. Native force remains a second oracle. No product geometry, mass matrix, regularization or solver is used to generate expected results. Default warmstart is retained; comparison is of converged force, not iteration equality.

Native condim=1 uses translational body_invweight0 in its regularization, not the exact diagonal of J M^-1 J^T. Detected contacts at the inclusion boundary can have no active row and still consume capacity. Source references: [constraint construction and parameters](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_core_constraint.c), [constraint model](https://mujoco.readthedocs.io/en/3.3.7/computation/index.html).

Forward acceleration uses physical M. With native Euler damping, integration separately uses M+hD with the complete smooth-plus-contact generalized force. Both must be compared. Source: [Euler implementation](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_forward.c).

## Fixed matrix

All dimensions are SI; scalar q uses native m/rad and free quaternion uses native WXYZ. The geometry-only rows still take a real one-step device path.

| Group | Fixed coverage | Window |
|---|---|---:|
| six primitive families | plane-sphere; horizontal capsule with two contacts and tilted capsule with one; plane-box flat four corners and tilted subset; sphere-sphere; sphere-capsule side and end; sphere-box face, edge, corner and inside with unique nearest face | 1 step each |
| redundant normals | one 1 kg vertical slider with two equal spheres at x=+/-0.15, radius0.1, center z0.099; explicit inertia, no duplicated mass | 32 steps |
| eccentric free box | box halfsizes(.18,.12,.08), root COM(.03,-.02,.01), mass2, nonprincipal inertia, initial z=.079; nonzero local omega and root load | 48 steps |
| two feet, shared root | anisotropic free root plus two hinged/slide feet, real incoming velocity, armature and damping; two distinct floor contacts and full root-joint reaction | 48 steps |
| sibling self contact | two moving sibling branches in one tree, sphere-sphere or sphere-capsule, nonzero approach speeds; no independent free-root extension | 24 steps |
| finite edge and withdrawal | free sphere initially at(.19,0,.149) over box halfsizes(.2,.2,.05), radius.1; vx=1.0 with no horizontal force, h=.001; must leave finite surface, not remain supported by an infinite plane | 128 steps |
| 29 scalar tree | binary hinge/slide tree, original body identities including static marker and welded tool; contact on last body, last matrix column active | 8 steps |
| detection/filter boundaries | exactly representable margin/gap equality; OR mask positive and blocked negative; same-body/weld and parent exclusions; impossible condim3 pair does not poison no-contact route | 1 step each |
| batch and failures | sparse W4 with different contact counts/controls; selected reset plus clone next real step; actual late readback failure, all-world rollback then identical retry; no CPU/Warp/HOST/linalg during step | bounded 1-4 steps |

Six-family scenes use explicit inertias, so adding a geom never silently changes mass. Geometric finite edges use native sphere-box face/edge/corner behavior. Plane-box follows CPU max-four lower corners and plane-capsule may emit both endpoint contacts. Source: [primitive collisions](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_primitive.c), [sphere-box](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_box.c).

The exact coupled counterexample has J=(1,1)^T and mass1. If both rows have regularization r and rhs beta>0, the joint force is beta/(2+r) per row while independent single-row solves give beta/(1+r). The actual fixture compares native r; the symbolic result is independent of any product output. It must discriminate a wrong independent solver by more than 5% of total joint force.

Capacity clarification before any T3 execution: the hard32 ceiling is static feature slots, not the detected-contact count. Four plane-box pairs consume32 slots and emit at most16 contacts; a fifth box is rejected at construction with contact-layout-capacity. Sixteen horizontal plane-capsule pairs fill32 slots and can emit32 contacts. Within one admitted layout, runtime contact_capacity=N/N-1 and constraint_capacity=active/active-1 check exact fill versus atomic failure. Sixteen plane-sphere pairs fill pair cap16; seventeen reject. G128 is exercised with masked static markers; G129 rejects. Detected inactive contacts consume contact capacity. No truncation is accepted. This corrects the provisional layout assumption in preregistration commit60fc8422 after the parent clarified the hard layout boundary; no product output was observed.

## Predeclared engineering gates

| Quantity | rtol | atol |
|---|---:|---:|
| point, distance, body pose/COM/rotation | 3e-5 | 4e-5 m or dimensionless |
| contact unit normal | 0 | 1e-4 |
| COM/contact Jacobian | 2e-4 | 5e-5 native SI |
| physical M | 3e-4 | 5e-5 native SI |
| bias/passive/applied | 4e-4 | 4e-4 N or N m |
| R, reference acceleration, normal velocity | 5e-4 | 4e-4 native SI |
| full dual matrix H | 7e-4 | 5e-4 native SI |
| normal force and generalized contact force | 3e-3 | 3e-3 N or N m |
| constrained and integration acceleration | 3e-3 | 1e-2 m/s^2 or rad/s^2 |
| smooth acceleration | 2e-3 | 4e-3 native SI |
| continuous q | 0 | 2e-4 native m/rad/WXYZ |
| continuous v | 0 | 3e-3 m/s or rad/s |
| clock | 0 | 2e-6 s |

Independent KKT uses the frozen backward-error scale: g=Hf+b, absolute=max_i(f_i>0 ? abs(g_i) : max(0,-g_i)), scale=max(1,max_i(abs(b_i)+sum_j abs(H_ij*f_j))); normalized=absolute/scale<=5e-5, with f>=0 and no deletion of weak/inactive rows. The runtime diagnostic compares to its published FP32 stored solver_tolerance, avoiding a one-ulp Python-literal mismatch. The runtime's declared converged diagnostic must separately satisfy its <=max(source tolerance,2e-5) gate, iterations<=source budget and finite nonnegative forces. Positive and negative normal forces are not cancellation-averaged.

Every generic contact is checked against native geom/body identities, same-input world point and geom1-to-geom2 normal. Multi-contact matching is a complete bijection within the declared pair, with uniqueness or known native feature ordering; missing/extra contacts fail. Contact-frame tangents may differ under condim1, so compare normal and frame orthonormality, not an arbitrary unused tangent orientation. Both world forces are equal/opposite; COM torques are reconstructed from the common contact point; J^T f equals the published generalized constraint force. Inactive contact force is exactly zero.

Selected first/middle/final steps save complete original input, native oracle, actual receipt, q/v/control and metadata in compact JSONL; every step retains q/v/control and contact counts. Failure details are appended before assertions. No full repository snapshot or oversized proof artifacts are required.

The result will establish finite normal contact only. Native friction, general mesh/CCD, ball joints, equality/tendon, multi-free-body admission and G1 standing remain later gates.


## Author handoff

The new test file contains fourteen test functions, a twelve-case primitive matrix and six continuous scenarios (288 main trajectory steps per backend). These are authored counts, not a collected or executed result. The 29-scalar binary tree uses native DFS order: j14 is the last scalar and carries the floor-contact point; the label28 is not assumed to be the final native DOF. Test-only JSONL and exact-state comparison reuse T2 serialization helpers, while geometry/dynamics/contact oracles are independently generated from original native XML. No import, collection, numerical probe, pytest, lint, formatter, compile, build or install was run in this author lane.
