# Generated contact-load FP32 delivery

The user accepts finite numerical loss to make the general physics pipeline
usable. Run59 exposed a concrete boundary failure: CUDA returned 51 accepted
N1 steps, then attempt52 was rejected when the device-generated aggregate
torque contained binary32 bits `800000ac`, approximately -2.41023336e-43 N m.
Saved-only run60 located the original full bytes in the failed attempt. This
was neither a nonfinite force nor a physical parameter/steady-support failure.

The new boundary is explicit and narrow. `normalize_contact_fp32` accepts only
an already-generated finite native-float32 array and its exact declared shape.
Normal values and both signed zeros are preserved bitwise. Only IEEE exponent
zero/nonzero-fraction values become same-sign zero. There is no caller epsilon,
object identity, time, case or parameter tuning. The immutable bytes-backed
receipt retains raw/delivered arrays, the exact bit mask, exact binary64 loss
decoded from integer significands, and the fixed 2^-126 bound at converted
elements (zero bound elsewhere). This is recorded loss, not lossless physics.

D3 applies this at its generated complete nodal-force and rigid-wrench upload
boundaries, before either candidate is staged. External nodal input still uses
the original strict `as_fp32` guard. Nonfinite generated loads still reject.
The published world receipt includes `node_force_delivery` and
`rigid_wrench_delivery`; original per-site laws and unnormalized contact
transpose remain visible. Fixed reactions use the actual delivered node force.
No accepted position/velocity is patched, no force is repaired to manufacture
conservation, and no numerical loss becomes an energy credit.

Run67 passed 26 fixed IEEE/schema/ownership cases and five actual joint-boundary
cases in interpret (31 total). Run68 passed the five joint-boundary cases in
CUDA. The latter exercise node/wrench subnormal delivery, nonfinite rejection
before joint publication and unchanged external-input rejection. First style
failures and their repairs are retained as runs61–66. All runs are in
`artifacts/p65-feature-until-0730-20261113-01`.

The original full-window steady-support failure remains open. This boundary
does not add material damping, relax the support window or qualify all D4
cases. The affected CUDA long-window and installed consumers require their
own actual outcomes after this source correction.
