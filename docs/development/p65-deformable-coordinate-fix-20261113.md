# D3 public witness coordinates: finite correction

The public triangle-distance rows already include each pair-local frame. D3
added that frame a second time when constructing contact points. It now adds
only the packet origin. Raw kernel witnesses retain their original pair-local
semantics. No contact law, integration, material parameter or tolerance changed.

Run20 first exposed the error at step57 of the new normal-load window. The new
regression uses sphere/capsule, original and translated placement, two real
successive steps, nonplanar material and nonzero angular velocity. Independent
geometry, material scatter, COM moments, work pairing, rigid response and the
complete stored FP32 recurrence are checked. Runs54/55 passed four cases per
backend; affected D5 asset short windows passed runs56/57, two per backend.

This accepts that coordinate/load scope. The extended real-arithmetic material
velocity qualification remains separate: run45 failed its unchanged 2e-5 gate
by a 2.64984691e-5 m/s discrepancy. The coordinate regression records this gate
and all original arrays as a separate qualification, while an independent
primitive-by-primitive arithmetic audit checks the actual stored trajectory.
The old test helper remains strict by default; only the new coordinate caller
provides the diagnostic callback. Position, multipliers, all other mechanical
checks and stored-arithmetic bounds remain hard gates. No measured error is
fed back into product state or contact force.

Evidence: `artifacts/p65-feature-until-0730-20261113-01`, original runs20/30/45,
qualified runs54–57. D4 steady-support failure and wheel acceptance are separate
from this correction. This does not establish arbitrary cloth/robot coupling.
