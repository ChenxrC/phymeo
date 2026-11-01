# Robot limits and link contact implementation plan

> Execution: parent implements inline in the existing isolated worktree; one serial execution slot. User authorized the next framework node and practical compliant accuracy. No new agents or approval cycle.

**Goal:** Reusable joint stops and robot-body reactions, then the existing force-gated handoff with recorded collision geometry.
**Architecture:** Independent immutable constraint/shape data and a robot load evaluator; a robot-only midpoint program and the cloth program consume the same loads. IK obeys configuration bounds; physical state is never clipped.
**Tech stack:** Existing NumPy FP64, SceneStepper transactions, no new dependencies.
**Baseline:** 457c3e9. Existing strict/native research remains historical.

## Design and choices

Soft physical stops and capsule contacts are the chosen implementation here. Hard position clamping would lose reaction impulses; a new complementarity solver would expand this node substantially. The current small-substep compliant integrator already provides a useful integration point.

- JointLimits has per-coordinate lower/upper, stiffness and damping (rad or metres, with corresponding generalized effort units). Bounds are finite; lower < upper. Stops act only outside limits, restoring by k*depth plus closing-only damping. Store signed effort, violation and spring energy.
- solve_point_ik(..., limits=None) projects candidate targets onto these bounds, reports an unsolved target when constrained, and does not modify physical q/v.
- LinkCapsule binds two local centerline points and a positive radius to a named robot/link. CollisionPlane is explicitly an infinite stationary half-space, not a finite table or CCD.
- RobotContactSystem validates bindings and explicit excluded pairs. Same-link shapes are one rigid body; parent/child exclusions require named pairs. All remaining capsule pairs and capsule/plane pairs are examined. Closest segment points yield the gap. One shared contact point receives opposite world forces and moments through full point Jacobians. Ambiguous coincident centerlines fail explicitly.
- ContactMaterial sets normal stiffness/damping, regularized Coulomb friction and slip speed. Positive overlap gives k*depth + d*max(-normal_rate,0); separation gives zero. No attraction. Small overlap is measured, not hidden.
- Independent robot-only and cloth/robot programs sample loads at the same midpoint used for velocity updates. Start/mid/end overlap guards reject a whole SceneStepper transaction. This is sampled collision response, not continuous collision detection.
- Recorded shapes, exclusions, limits, reaction samples and frame world endpoints are carried with models. Viewer draws actual collision shapes.
- Handoff hardware may define a fixed spatial offset for the two arm mounting planes and wrist extensions; the stage policy, cloth, friction and force criteria stay shared. Finger/cloth contact remains the existing finite pads. This node does not implement general cloth/capsule collision or collision-avoiding motion planning.

Research: [Newton collision primitives and filters](https://newton-physics.github.io/newton/latest/concepts/collisions.html), [MuJoCo limit/contact constraints](https://mujoco.readthedocs.io/en/stable/computation.html). These motivate shape separation and soft reactions; this implementation is independently tested.

## Tasks and verification

### 1. Joint stops and bounded IK (medium)
Files: constraints/robot_contact.py, control/point_ik.py, tests/test_robot_contacts.py, tests/test_point_ik.py, public exports.
- [ ] RED: lower/upper force signs, zero interior load, energy; out-of-limit target stalls at a bound while seed is unchanged; invalid dimensions rejected.
- [ ] Implement immutable data and bounded target updates. Test inactive joints and seed outside a soft stop without silently publishing repaired state.
- [ ] GREEN, focused checks, logical commit.

### 2. Reusable collision loads and common physical step (medium/high)
Files: geometry/capsule.py, dynamics/robot_contact.py, solvers/compliant_robot_contact.py, solvers/_compliant_motion.py, solvers/compliant_cloth_contact.py.
- [ ] RED: oblique/parallel/point-segment closest witnesses; pair action/reaction and common-point torque; friction power; named exclusions; coincident axes rejection.
- [ ] Integrate reusable loads into robot-only and cloth programs. Same substep state and physical input; no extra apply or state correction.
- [ ] Real two-moving-mass collision reverses closing velocities with conserved total momentum; driven joint reaches a soft-stop equilibrium; plane collision and rollback are recorded.
- [ ] GREEN and changed-source regression tests; commit.

### 3. Same handoff and recorded body geometry (high)
Files: experiments/handoff_grippers.py, compliant_handoff.py, compliant_cloth.py, replay exporter/viewer; tests for bindings; results JSON and existing delivery plan.
- [ ] Add bounded hardware and collision shapes, retain shared phase/control criteria.
- [ ] Run one continuous articulated handoff; report loads, transfer, table dwell, limit excursions and contact penetration. A meaningful failure changes geometry/controller diagnosis, not hidden success thresholds.
- [ ] Collision-active obstacle and joint-stop counterexamples are covered by actual robot-only dynamics. Reuse old zero-friction evidence unless its affected behavior needs a new run.
- [ ] Save model/shape-bound offline replay; focused static checks, independent saved-data inspection, outside-source wheel verification of changed interfaces and a physical handoff boundary.
- [ ] Record measured result, limitations, clean commits. No remote CI or push.

## Review focus / stop rule

Finite/shape validation, duplicated/unknown exclusions, rigid pair force/moment balance, singular contact geometry, and transaction rollback must each have independent tests. Two attempts without new evidence stop that branch. No new mathematical-proof ladder. Success is a reusable module with physical positive/negative examples plus an honest handoff result, not a claim that all P6.5 is complete.
