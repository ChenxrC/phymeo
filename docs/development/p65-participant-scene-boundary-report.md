# S3 public compiled participant boundary query

Status: `CODE_READY_TESTS_PENDING`; child authoring released after the commits below.
This is the Batch B S3 wiring increment over accepted Batch A `c45a0174`.
No project import, collection, test, numerical probe, lint, formatter, compilation,
build, installation or profile was run by this author. Parent owns serial validation.
The parent has imported the first test and compiler checkpoints and begun its
separate validation queue; this report does not assign an outcome to that queue.

## Change and public contract

The previous compiler always required a complete response-law table and passed
all original material geometry through HOST patch admission. A mesh declaration
therefore could not reach the already accepted S1/S2 boundary query source.

`ParticipantSceneDeclaration(contact_response="none")` now explicitly selects
the existing scene free plan. Its material table must be empty, including no
entries with value `None`. The default `"host_patch"` still requires the complete
old table; `None` retains its original explicit exclusion meaning on that route.
The selected mode is part of declaration identity and source-change checking.
Checkpoints and query packets keep the original owner/registry binding, so another
compiled owner, including one compiled with the other mode, cannot consume them.

`compiled.prepare_boundary_query(*, limits=None, memory_limit_bytes)` automatically
registers the original prepared model and bindings through the accepted scene API.
No caller supplies a private bank, reconstructed current positions, or substitute
topology. The optional limits default is the existing `BoundaryDistanceLimits`.
The explicit positive memory budget is still required.

Multiple plans are supported. Each is independently owned by the real scene.
`last_boundary_result` and `require_boundary_result(packet)` refer only to the
most recently **successfully prepared** plan. A failed preparation does not change
that pointer. Preparing a new plan does not close previous returned plans.
A last result can be historical after a physical step; currentness must be checked
with `require_boundary_result`. All plans close through the existing batch owner.

The none route retains original masks and excluded pairs. The declaration's
`intra_participant_pairs` option remains a HOST response option; it does not
rewrite boundary material filtering. Existing static shape and asset domains are
unchanged. Unsupported boundary shapes still fail at the original query admission.

```python
declaration = ph.ParticipantSceneDeclaration(
    original_articulations, contact_response="none"
)
with ph.compile_participant_scene(declaration, mode="cuda") as scene:
    query = scene.prepare_boundary_query(
        limits=ph.BoundaryDistanceLimits(),
        memory_limit_bytes=64 * 1024 * 1024,
    )
    scene.step(1 / 1024)
    result = query.query(margin=0.5)
    scene.require_boundary_result(result)
```

This example is authored documentation, not an executed result.

## Exact source scope

| File | Change |
| --- | --- |
| `src/phymeo/builders/participant_scene_compiler.py` | Only product change: explicit response mode, original source retention, public query/current forwarding, existing free-step routing |
| `tests/test_participant_scene_boundary.py` | New test-first declaration, geometry, source and transaction matrix |
| `tests/installed/check_participant_scene_boundary.py` | New standalone installed consumer |
| this report | New authoring handoff |

No private builder helper was necessary. No backend query, source adapter, bank,
four-tree binding, narrow kernel, BVH, resource ledger, existing test or package
export was edited by this author.

The compiler creates no additional device arrays. Existing S1/S2 accounting and
admission remain authoritative: persistent `68S + 36P`, source-query scratch
`112S + 36P`, existing BVH/group costs, and `1060C` narrow scratch. Retaining a
model, bindings and the current plan is HOST object ownership, not a new device
state or an omitted device allocation. Late query failure preserves previous
query publication and does not roll back already accepted physics.
Cleanup delegates to the accepted resource/lease owner. A delegated close failure
does not mark the compiled facade closed.

## Predeclared source matrix

Manual enumeration only: **20 TYX nodes per selected backend plus 7 HOST nodes**.
No collection was run by the author.

| Fixed test group | Nodes | Obligations |
| --- | ---: | --- |
| Four declared asset families | 4 | Original ordinary/affine/floating/floating-affine asset materials; different anchors; three actual steps; complete finite geometry oracle |
| Heterogeneous sparse worlds and reversed declaration order | 2 | Four families plus free object in each world; reused local IDs; logical membership and full registry |
| Fixed tool plus root-only free object, original/cycle | 2 | Public declaration, original folded tool mesh, actual velocities and all source fields |
| Original masks | 1 | Mask filtering preserved without response-law defaults |
| Partial reset, restore and real next step | 1 | Selected world isolation; old query stale; subsequent accepted geometry |
| Multiple plans and failed prepare | 1 | Explicit current-plan semantics, old returned plan remains valid, failure does not replace pointer |
| Late packet/read/native-source failure | 3 | Complete attempt saved; previous result retained; accepted physics unchanged; real retry |
| Group/candidate/memory refusal | 3 | Whole-query rejection and retained independent plan; no partial packet |
| Empty original geometry | 1 | Full logical tick membership still read from accepted state |
| Mode identity and cross-owner checkpoint/query | 1 | Real owner/registration rejection, no foreign restore |
| Delegated close failure | 1 | Facade does not pretend closed; owner remains reachable until normal close |
| Invalid response mode | 5 HOST | Rejection before runtime allocation |
| Nonempty none-mode law table | 2 HOST | Both None entry and real law rejected before allocation |

The tests reuse only independent reference material/FK and finite-triangle
oracle code. Current device source output is never fed back as expected geometry.
Distance, material witness and normal gates remain `1e-4 L`, `2e-4 L`, and
`2e-3`, with the original local pair scale. Source-coordinate checks retain the
accepted local material-scale gate. No global translation is used to widen it.
Source evidence saves original declaration, returned raw packet, each attempted
physical/query operation, actual receipts and failure prefixes.

Suggested parent order: new HOST admission checks; all new interpret nodes;
all new CUDA nodes; two affected default-route neighbors
`test_explicit_none_pair_and_empty_geometry_are_real_no_contact` and
`test_local_urdf_root_static_installation_and_old_declaration_unchanged`.
The unchanged S1/S2 matrices are reused, not requested wholesale again.

## Standalone installed consumer

CLI: `--mode interpret|cuda --installed-root ABS --output freshABS`.

This is a distinct second installation scenario: a public URDF affine-driven
fixed tool carrying the original OBJ mesh, plus a public URDF root-only free
sphere. It is not the previous ordinary/floating-affine two-arm consumer.
Logical worlds are 17 and 83; local material IDs are reused across worlds and
physical anchors differ. Inputs retain nonzero velocities, explicit feedforward
and a real free-body external force.

The authored route has four physical steps (eight logical worldsteps, sixteen
participant steps), five successful queries, three stale-result rejections and
one zero-candidate-capacity rejection followed by a fresh-plan retry. Counts are
expectations, not measurements. Row counts are recorded from the complete
independently checked finite candidate sets, not preclaimed.

Motion checks independently use the tool's reduced mass `1 + 2 = 3 kg` and free
object mass `2 kg`; no response wrench is allowed in none mode. Geometry uses
authored material coordinates, accepted public q, complete finite candidate
enumeration and independent segment/triangle closest points. Original asset
provenance, logical ownership, all facets, native poses, anchor shifts, full raw
BVH/COL2 packets and per-query counters are saved to a lossless JSONL gzip stream.
Physical attempt/return/snapshot are distinct events, preserving an actual
returned result even if later capture or assertion fails. The final close path
does not mask a primary failure.

The consumer imports no repository tests/examples or private source-bank
helpers. All loaded Phymeo module origins must lie beneath the installed root;
the script, cwd and installation must be outside a checkout. Original XML/OBJ
inputs are written only under the fresh output directory.

## Checkpoints and limits

- `e9960491`: test-first matrix.
- `52cdc090`: compiler wiring.
- `07892444`: standalone installed consumer.
- Final report commit follows these source checkpoints.

Remaining scope is unchanged: this is a finite current-snapshot **query** and a
public assembly route. It adds no mesh response law, CCD, deformable producer
bank, MuJoCo-TYX cross-backend routing, or implicit coupled contact solver.
Passing this slice must not be presented as full Newton collision parity.

Handoff: `AUTHORING_SLOT_RELEASED`. Product/tests/consumer are ready for the
parent's actual serial checks; there is no child execution session to resume.
