# G1 current IMU-site observation protocol

Parent decision 2026-11-11, within the approved until-08:00 work. Source-only research and run59 establish an observation-axis mismatch; they do not establish the cause of the old 112.372 s XY-gate failure.

Fixed research: original C author report g1-observation-protocol-research.md SHA256 2d406b0510166c23a1e19dbfabcf35e94237756a01e54c20414ddd37fafe958f. Unitree 1425b15f training actor uses builtin_sensor("robot/imu_ang_vel"); mjlab v1.2.0 G1 attaches it to the unrotated pelvis IMU site. MuJoCo 3.3.7 distinguishes SITE site_xmat from BODY ximat. Current harness uses BODY inertial axes.

## Narrow authoring grant

Reuse original mtyx_t2_audit, new isolated branch from parent 90951628 in its existing worktree, preserving the previous branch and unrelated old T1 draft. Exactly two product/test paths:

- MOD examples/g1_mujoco_tyx_policy.py
- NEW tests/test_g1_mujoco_observation_protocol.py

Tests first. No project import/collection/pytest/numeric probe/lint/format/build/install/policy/physics execution by the author. Parent keeps all execution. No new helpers.

Introduce explicit protocol IDs g1-current-body-inertial-round50-v1 and g1-current-imu-site-round50-v2. New runs default to v2; explicit v1 remains available for reproducibility. Do not edit old payloads or claim they originally contained these names. Only observation 64:67 changes on the same input state. All other 151 words, previous raw action, round(time*50), stored time, 50 Hz update, PD/control hold, initial state, assets, physics, gates and budgets remain unchanged.

Bind the actual named gyro sensor to its site/body/local quaternion using the compiled model. For v2, current free-root-local angular velocity is expressed in that site's local axes, R(site_quat).T @ qvel[root angular dofs]. Admit only the verified root-mounted sensor case; reject missing/wrong-type/wrong-body/non-free-root bindings explicitly. Do not use stale native sensordata/cvel or invoke CPU FK/physics to supply device state. A rotated site must work under the same formula; an articulated descendant is unsupported in this narrow helper. Keep the formula and binding in the example, not a claim of a complete engine sensor API.

Record protocol ID, sensor/site/body IDs and names, local quaternion, angular convention, current-state timing, frame rule and control period in new source/initial metadata. Both controllers use their own states/history. Ensure output/saved schema can identify the chosen version without changing old evidence.

## Meaningful tests and bounded validation

Unit/semantic cases: same state v1/v2 differ only in 64:67; current v2 equals independent native SITE gyro observation for an inline free-root model with nonidentity inertial frame and rotated site; native BODY differs as expected; root rotation and nonzero angular velocity; missing sensor, wrong type/body and unsupported child binding; explicit legacy retains the previous formula. Use actual compiled address/identity data rather than a fixed G1 ID shortcut. Native forward is only the test oracle, never the runtime producer.

Author returns exact two-file small commits and ignored report/manifest. Parent reviews all changed lines, validates exact-file static checks and the new semantic tests in the sole serial slot. If they pass, run exactly one new v2 two-second/1000-step trajectory at original h=.002 and initial assets with both independent lanes and all original engineering gates. All original failures remain failures. This qualifies only the sensor connection and bounded continuation; it does not automatically grant or claim the full 131.48 s acceptance. Any later full run requires a concrete parent decision based on these results and remaining time, without parameter search or relaxed gates.

Six fixed saved-state comparisons from run59 already bind the original v1 vectors and v2 site-axis formula; do not reread the entire gzip. A later small comparison may consume the retained six full records if needed. Fixed source links and research belong in the report. No policy retraining, absolute-XY controller patch, frame-clock change, model/pin change, or broad sensor subsystem expansion.
