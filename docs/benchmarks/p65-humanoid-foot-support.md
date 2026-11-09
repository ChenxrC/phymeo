# Finite-foot sustained support: authored acceptance and public recipe

## Parent results, 2026-11-10

Normal support is verified for the actual free-root two-foot fixture: three
128-step stance/load-moment windows and one continuous 528-step lift/reentry
window. A separate late-failure/retry test passed. The six adjacent box-patch
cases and two new subresolution-orientation regressions also passed.
Evidence is `artifacts/p65-h1-20261109-01`, tasks 13–18.

The first stance attempt failed when a ~1e-22 stored orientation changed a face
load into a full-force corner load. The parent repaired only the OBB load
manifold's coordinate-resolution admission: reconstructed vertex displacement
must stay within the existing FP64 coordinate bound. Actual material points,
site gaps, velocities and reciprocal application points are retained. The
admission and its displacement/bound are exposed in contact details. Exact
convex collision predicates are unchanged; resolved tilt 2^-25 still gives the
original edge result. This is an engineering load-manifold convention below
coordinate resolution, not proof that the nonsmooth maximum-gap potential has
an unchanged angular derivative.

The planned 128-step lateral-history target **failed on its second request**
when real tilt changed the foot patch. Its original positive target and raw
failure remain in task 14. The current test explicitly verifies rejection and
unchanged accepted state for this unsupported transition; it is not counted as
successful lateral support. General resolved-tilt distributed pressure and
rolling/feature-changing material history remain open. No matcher tolerance,
mass, friction coefficient, root constraint or physical acceptance gate was
relaxed.

Actual G1 standing is assessed separately by
`examples/g1_phymeo_support.py`; success on the small fixture does not establish
humanoid balance or dance. The author-only handoff below is retained as history.

Status: **AUTHORING_ONLY / UNEXECUTED**. Base `666dd66184869ff36401551abcd0904e7564128a`, branch `codex/h1-foot-support`. The parent owns all project execution. This checkpoint changes three new files only; it does not change existing dynamics, collision, material history, limits or numerical tolerances.

The task is reusable physical foot support, independent of the G1 demonstration. A real floating trunk and two massive articulated feet exercise finite box-ground contact, asymmetric force/moment balance, unloading/reentry, explicit friction history and atomic rejection. The parent separately integrates the official G1 model and its actual foot capsules.

## What the current source already supports

`MultiRobotScene` samples every normal contact at a common entry state, passes reciprocal loads to the robot and separately registered rigid floor, and publishes accepted robot/rigid state and contact events together. Floating root motion is physical: six root velocity coordinates are retained, and child drives use the full root-child mass coupling.

`MaterialStationRegistry` accepts real floating root/link owners. `_scene_tangent_ports._robot_port` builds full root-plus-joint point Jacobians and takes the participant's prepared mass/bias. Parent-owned work in this batch adds a distinct response metric for implicit joint damping while preserving the physical mass used for energy; this authoring lane must not reconstruct or discard the actuator terms.

The default contact friction is velocity-regularized. It cannot supply nonzero static horizontal force at exactly zero slip. Static lateral hold needs explicitly declared material stations and `enable_coupled_tangent`; registering ordinary named foot colliders is supported, while floating pads are explicitly unsupported. Shared-scene execution remains HOST in this base.

## Public material-station construction

The new `foot_material_registry` helper consumes a real initial `scene.sample()`, not a synthetic force record. Declare actual names and all colliders belonging to each foot. For a floating URDF/MJCF asset the computational child body is its link index plus one:

```python
from phymeo.experiments.humanoid_support import foot_material_registry, read_foot_support

feet = {
    "left": (scene.shape_key(robot_id, left_link, left_collision),),
    "right": (scene.shape_key(robot_id, right_link, right_collision),),
}
ground = scene.body_keys[(0, floor_id)]
asset = scene.assets[(0, robot_id)]
pose = ph.floating_kinematics(asset.model, scene.state.robots[(0, robot_id)])
frames = {}
for link_name, keys in ((left_link, feet["left"]), (right_link, feet["right"])):
    body = asset.link_names.index(link_name) + 1
    for key in keys:
        frames[key] = (pose.position[body], pose.rotation[body])
shapes = scene.geometry().model.shapes
registry = foot_material_registry(
    scene.sample(), feet, ground,
    frames_by_shape=frames,
    shape_vertex_ids={shape.key: shape.vertex_ids for shape in shapes},
    chart_ids={shape.key: i for i, shape in enumerate(shapes)},
)
scene.enable_coupled_tangent(registry, stiffness_n_per_m=200.0)
```

Enable once before the first step. The stiffness above belongs to the small authored fixture; it is not an uncalibrated recommendation for G1. Each positive initial site gets one stable token sorted by the canonical shape pair and true foot-local material anchor. All original shape vertex IDs and explicit chart IDs are retained. No contact-force magnitude selects a token, and no quadrature fraction is multiplied twice.

Important current limitation: this is fixed-local-anchor matching, **not rolling/feature-transition matching**. A rotating G1 capsule changes its radial local witness even while its axial material weights stay unchanged. An initially declared registry does not prove that new witness is the same material station. A registration rejection during such motion must be diagnosed in the matcher/history lifecycle; do not increase match tolerances, add pads or hold the root to conceal it. A box foot can also change its contact patch under tilt, so the lateral test is a real acceptance target, not a previously measured success.

## Read-only accepted support diagnostics

```python
support = read_foot_support(
    scene.observe(), feet, ground,
    floor_rigid_rows=(scene.rigid_rows[(0, floor_id)],),
    robot_address=(0, robot_id),
    reference_root_position_m=initial_root_position,
)
```

`FootSupportReading` retains sampled-load and committed-state times, per-foot force/couple/friction and contributing site-side counts, their summed force/couple, selected opposite ground reaction, separately published floor force, loaded feet, root displacement, root COM velocity, root world up-axis and tilt angle. Moments use one declared world origin. Normal force is the up-component on the declared flat support. The published floor channel can include other participants; it is therefore separate from the selected reciprocal foot reaction. The helper changes no state and never invents contact impulse from an undefined time integration.

## Authored cases and success criteria

| Test | Cases | Evidence and intended acceptance |
|---|---:|---|
| `test_h1_two_finite_feet_support_the_real_free_root` | 3 | 128 steps each: symmetric stand; reversed shape-owner order; 1 Nm external pitch with independently derived unequal foot reactions. Sum supports 4 kg gravity, floor reaction is opposite, resultant moment balances the actual external torque, and free-root/joint state remains at the analytic equilibrium. |
| `test_h1_liftoff_and_reentry_use_one_continuous_trajectory` | 1 | One continuous 528-step stand/lift/return trace. Actual mass-proportional external COM forces lift both feet; actual normal-load-feedback joint effort preserves relative foot coordinates. Observe zero support after separation, then both feet reenter and decelerate without any pose/velocity repair. |
| `test_h1_lateral_load_release_uses_explicit_foot_material_history` | 1 | 128 steps under a real 2 N lateral load then release. Explicit material history supplies opposing traction, obeys each foot Coulomb bound, retains elastic traction after load removal and bounds root drift. Any material-matching or motion/work rejection remains a failure to diagnose. |
| `test_h1_late_tangent_failure_keeps_the_published_foot_state` | 1 | Complete one real material-motion audit, inject a subsequent rejection, confirm robot state/history/events/geometry stay at the prior publication, then retry from that state. |

These are six intended pytest nodes, not collected or passed nodes. The fixture uses real masses 2+1+1 kg, actual finite 0.24 by 0.12 m feet, a finite 4 by 4 m floor, gravity 9.81 m/s2, h=1/1024 s and a declared spring/damper law 20000 N/m and 120 Ns/m. The static reactions and motor efforts come from force/moment equilibrium before execution. Load/release thresholds are engineering acceptance targets, not fitted output. The toy feedback is an explicit two-coordinate task controller, not a claim of general humanoid balancing.

Every authored test saves the input URDF/law/masses, attempted commands, accepted entry/end state, actual drive and contact effort, actual body wrenches, per-site force/material weights/application points and diagnostics. An exception preserves its successful prefix and attempted request. No test writes a root, joint or contact state, pins/welds a body, or branches on world position/time to bypass contact.

## Parent validation and remaining scope

The parent should review and merge this checkpoint, run first exact-file lint/format checks, record any diagnosed style change, then execute this focused file under its sole HOST slot with a fresh absolute basetemp. Example functional selection: `pytest tests/test_humanoid_foot_support.py -q -x --tb=short -p no:cacheprovider --basetemp=<fresh-absolute-path>`.

No command above was run by this author. There is no CUDA, interpret, G1 standing, dance, long-window or multi-environment result in this checkpoint. The remaining material-feature lifecycle, coupled stiff normal-contact stability, genuine G1 balance, actuator model parity and GPU 29-joint capacity are separate acceptance obligations owned by the parent and other lanes.
