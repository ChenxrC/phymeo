# I2b: regenerate finite area contact from accepted TYX state

2026-11-17, codex/exp. Root owns all project execution. Original agents only
author/review in isolated trees. I0/I1/I2a accepted suites are reused.

## Contract

BodyAreaPressureAsset stores a full zero-boundary tetrahedral pressure field and
a finite rigid triangle mesh in their respective body-local frames. The new
MuJoCoTyxAreaPressureScene consumes the exact declared ordered body pairs. Each
step holds the runtime lock, verifies revision/h, runs current accepted-bank TYX
forward kinematics, transforms both assets, runs existing I1 HOST intersection,
then invokes I2a once. No CPU dynamics, old receipt pose, or synthetic motion.
The first domain is full pressure assets only; unsupported gradient/capacity
propagates as an explicit failure. No automatic fallback to point contact.

Geometry becomes visible only after a successful physical commit. External
reset/control/step revision invalidates last_geometry. Generation or physical
failure preserves the previous accepted state and geometry. Scene assets are
owned/immutable; samples carry original body/material/cell/face identity.

## Fixed validation before running

- Full tetra outer vertices (-.12,-.12,0),(.12,-.12,0),(0,.12,0),(0,0,.08).
  Interior vertex is their mean, pressure 400000 Pa; outer pressures zero.
  Four tetra cells cover the full parent. Bottom cell has g=2e7 Pa/m.
- Finite triangle (-.015,-.045,0),(.015,-.045,0),(0,-.015,0);
  area .00045 m2, normal +z. It stays in the bottom positive-gradient region.
- Two z sliders m=(.5,1.25) kg, initial gap .002 m, v=(-.2,0) m/s,
  no gravity/joint damping. h=.0005/.00025 s, T=.08 s, both execution modes.
  Check empty entry, positive contact, peak compression, separation and zero
  final load. Bound depth <.004 m; conserve momentum to 1e-5 kg m/s.
  Compare with independently authored scalar BE stepping (no geometry oracle);
  per-step force error <= 1e-3 N + 1e-3 relative and velocity <= 3e-5 m/s.
  Positive total mechanical energy excursion <=2% initial energy; final
  energy loss <=30%; peak force/depth h difference <=8%; first/last
  contact event h difference <=2 coarse steps. These are engineering gates.
- Eccentric y-hinge + z slider uses the same full field/finite triangle,
  pressure body's local field translated by (.06,0,-.025), pivot at z=.027,
  initial angular speed 2 rad/s; hinge inertia .02 kg m2, second mass1.25 kg;
  h=.0005, T=.08, both modes. Verify current rotation/translation from q,
  positive force, actual joint response, changing area and output moments,
  local frozen-step work/motion consistency. Do not claim a global energy
  theorem for this finite-face rotational example.
- HOST protocol: identity/order/ownership/full-domain, sparse worlds, stamped
  transforms, stale revision, per-world generation failure, physical failure,
  late publication, reset/revision invalidation. All-or-nothing publication.
- Save complete compact per-step states, input/output stamps, area, forces,
  generalized load and pressure source IDs. Plot these actual records.

## Boundaries and stop rules

No CCD claim: entry-sampled contact may enter the layer within one step; measure
that discretization rather than requiring exact no penetration. No friction,
large mesh acceleration, device-only clipping, area observation ABI/checkpoint
or automatic algorithm selection. If cases fail, diagnose once against the
independent model; do not tune masses/loads/steps/tolerances to manufacture pass.


Pre-execution clarification: the energy gates are on relative mechanical energy
(the center-of-mass kinetic energy is constant). Erel0=1/140 J. Coarse final
loss <=30%, fine <=18%, upward excursion <=2%. These reflect known BE
algorithmic dissipation, not calibration of a physical damping coefficient.
