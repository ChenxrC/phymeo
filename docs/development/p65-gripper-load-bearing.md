# G2-P finite-force gripper

Status: CODE_READY_TESTS_PENDING. All new source and tests are authored; no import, collection, lint, format, pytest, native execution, build or profile has yet run for these files. This is a code checkpoint, not load-bearing acceptance.

The additive runner configures the public finite-gripper model and unchanged runtime for A high-friction hold / low-friction slip, B open-gap capture and upward hoist, and C one active distance-material step. It stores each complete actual result, state/history and raw timings with typed lossless array bytes. The CLI requires explicit case/mode/h/duration/output and records source provenance. The independent rational A/B/C equations live only in tests.

Authored host coverage includes full T=1 windows at h=1/8 and 1/16 with two fresh rebuilds, exact event/state/load equations, temporal halving or event-grid bound, material h=1/4 and 1/8 controls, mass/effort scales 0.5 and 2, finite patch size/vertical translation/declaration order, physical budgets, original work/momentum, complete history, serialization and immutable input controls. Actual interpret/CUDA tests retain T=1/2 windows, both grids/rebuilds, C, transformations and explicit FP32/launch checks. Fault tests use real operation wrappers for late prediction/path/terminal/mapping/CCD/close, partial construction, full-frame rollback and retry equal to a fresh frame; capture exercises nonzero terminal friction.

Independent physical authority: p65-g2-gripper-proposal.md A/B/C and the frozen integration spec. G2 uses existing native cache behavior because participant APIs expose no cache-root argument (R5). No old solver/runtime/law/participant/pin/history file is edited. Source/test formatting has not yet been executed; initial check results must be retained rather than invented as historical TDD RED.

The C case is DistanceConstraints only. Full B2 material integration, bending/folding/self-contact, general robot support, mesh calibration and complete P6.5 are not established by this checkpoint. Remote CI stays user-deferred and unpassed.
