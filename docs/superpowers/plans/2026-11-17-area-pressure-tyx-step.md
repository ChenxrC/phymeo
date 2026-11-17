# I2a: area-pressure rows in the actual MuJoCo-TYX step

Date: 2026-11-17. Branch: codex/exp. Root owns the sole execution slot.
Existing I0/I1 and external FFT qualifications are reused, not rerun.

## Scope and discrete law

Opt-in declared ordered body pairs own their normal contact mechanism. All native
geom pairs between those bodies are removed before device route construction.
Each step requires current, explicitly stamped AreaPressureSurface data for every
world and pair, including empty surfaces. This is a supplied-current-surface port;
automatic pressure-asset refresh and CCD are not implemented by this batch.

Device geometry supplies both actual point Jacobians. For each quadrature point:
k=dA*g, phi=-p/g, J=n dot (JA-JB), R=1/(h*h*k),
aref=-phi/(h*h)-J*v/h. Solve f>=0, J*a+R*f-aref>=0 with complementarity.
The response operator is M+h*D, matching the existing Euler velocity update.
For zero joint damping, unrelated native contacts/limits/dry-friction may share
the solve without changing their equations. With nonzero damping, this first
route refuses any remaining native contact pair or native constraint: using a
different mass operator for just one row family would invalidate the symmetric
solver. Real passive joint damping and actuator force remain active.

No host dynamics or post-step force application. Inputs use the owned allocation
manifest, bounded capacity, and existing candidate/commit transaction. Publication
distinguishes type-7 area rows from native rows and retains sample/body identity,
actual normal forces, world force/moment, and input-state binding.

## Work and validation

1. Add opt-in model admission, stamped step API and device row append.
2. Use effective response in area route and preserve physical acceleration output.
3. Independently test two movable bodies, unequal mass, entry velocity, damping,
   off-centre hinge torque, unrelated native contact with D=0, separation/empty
   surface, stale/missing/extra pair inputs, capacity and publication rollback.
4. Run only the new focused HOST protocol and interpret/CUDA cases serially;
   include one warranted existing T4 seam regression after shared edits.
5. Save actual outputs and report limits; commit small verified changes.

The first route is frozen entry geometry, elastic normal contact only. It does not
claim nonlinear end-geometry consistency, friction, arbitrary contact birth,
resident-device surface generation, training observation completeness, or full I2.
No inference of universal material suitability from the FFT experiments.
