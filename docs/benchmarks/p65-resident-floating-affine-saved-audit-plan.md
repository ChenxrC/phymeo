# C11 saved floating-affine audit: fixed authoring plan

Status: **AUTHORING_UNEXECUTED**. Parent contract `2966aa9`, plan section 11.
This observer is not a producer, numerical verifier, or a new physics test.
No output payload was read to choose these identities; no project/audit command
has been executed. Authoring owns exactly this document and
`tests/scale/resident_floating_affine_saved_audit.py`.

The original test/source basis is RD `be4f1a052a13c2d3227fc4a4ab8599c044e64a42`.
Its wheel change only renames the shadowing local to `selected_drive`; the
ordinary/full/reduced inputs and engineering gates stay frozen. Native source is
robot `5b34f120519afec9ea4772b04c237c3a14be0ed6`. Actual execution must bind its own
complete before/after source snapshots, not infer source equality from these IDs.

## Membership fixed before evidence

The code's `membership()` lists full function/parameter identity, collection
order, relative gzip paths, h, step counts and replay scope. One mode comes from
the actual `--tyx-mode` argument, not a pytest parameter suffix.

| Authored group/backend | Nodes | Gzip | Save contract |
|---|---:|---:|---|
| step public/cycle/A03/A04/PD/counts | 15 | 15 | 1/4/4/1/40/4/1; A04 window 40+21 |
| finite contact A02 and A05 | 5 | 5 | each 2 steps, original reverse/cycle identity |
| mixed core W1x4/W4 | 1 | 5 | each 4 steps; original root-only and FreeSphere |
| mixed contact W1x4/W4 | 1 | 5 | each 40+19-to-21; lifecycle load/open/reentry |
| X01 and X06 preflight | 14 | 0 | original same-source pytest result only |
| X02 restore | 3 | 3 | complete first/replay and retained state |
| X03 capacity | 2 | 2 | bank0/world503, bank1/world91, original retry |
| X04 late fault | 6 | 6 | upload/candidate/receipt x banks0/1 |
| X05 owner | 2 | 2 | stale restore; close/rebuild |
| X07 actual diagnostic corruption | 3 | 3 | warm, real constructor refusal, retry |
| **Core** | **52** | **46** | 38 nodes save; 14 do not |
| optional W32 | **1** | **1** | separate 4-step result or explicit NOT_RUN |

No standalone HOST identity exists. A01's HOST premise belongs to its device
node. The two modes have separate coverage; repeated successful nodes require an
explicit repair/retest qualification pointing to the prior run. A failed node's
partial or opaque artifact never fills successful coverage.
Completed nodes earlier in the same RAW1 run may contribute only after the
original fail-fast log/order and all their complete payloads have been audited.
Exact CLI targets, literal function-name
selection and deselections determine order before payload inspection; pytest's
sanitized 30-character prefix and ordinal bind each node's original tmp directory.

Native membership is A01-motor **1**, A03-fan **4**, A05-native-contact **4** steps:
three gzip files, **nine steps total per device**, CPU and cuda:0 separately.
Wheel membership is one complete gzip/backend: W4 40+21, A04-p1 four, two capacity
refusals and two retries, at most two actual alignment steps; 262..270 accepted
world steps and two rejected world attempts. No new native or wheel run is added.

## Bounded interface and scope

`python -I -B <auditor> --manifest ABS_JSON --output ABS_NEW_DIR` writes
`audit.json` in a fresh directory. Manifest explicitly lists ended original runs,
original job/config paths, head, raw exit, full source SHA set, selected/completed
nodes, first failure if any, payload roots, and external output roots. Original
CLI output paths and relative source/basetemp paths resolve against recorded cwd.
The observer does not follow arbitrary imports, invoke collection, or glob to
choose successful samples. Inventory is only a completeness/extra-file check.

Limits: 64 MiB compressed/256 MiB decoded per payload, 2 GiB total decoded,
16 MiB metadata, 8192 files and depth 96. Duplicate JSON keys and bare nonfinite
numbers fail. X07's deliberate NaN tags are allowed only in its declared modified
array and must agree with saved uint32 bits; original arrays remain finite.
No directory is decompressed to disk and no old evidence is removed.

All original input/source/graph/P/c/C and compiled fields remain hash-bound.
The observer checks N/K/B/D shapes; root/local/full/independent checkpoint,
held/history, full/reduced mechanics, saved independent comparisons, original-row
reaction/power/root balance, actual material/application records, exact replay,
world maps, ticks, status and resources. It never recomputes FK, J, M, bias,
forces, inertia, constraints, or integration. A saved independent expected value
is compared to its saved actual; it is not generated from a product matrix.

Read-only flags, injected callback timing, the 14 no-save preflights, and some
assert-only intermediate checks cannot be reconstructed from JSON. Their bound
original test result is reported separately, never invented as saved telemetry.
Historical incomplete outputs preserve raw failure context without a new pass.
The original source snapshots (including URDF, all three tests, independent
references, numeric helpers, actual products/shared hooks) and raw results are
required; missing source coverage fails explicitly rather than borrowing current
checkout state. Wheel archives may be binary snapshots and keep hash binding
without UTF-8 decoding. Installed package membership/loaded origins and native
required checks remain distinct from product physics acceptance.

Implementation is complete as a text-authored candidate; parent performs all
static/runtime validation after handoff. **UNEXECUTED**.

## Exact manifest contract

Top level is `schema: "c11-saved-affine-v1"`, ordered `pytest_runs`,
`external_runs`, `historical_runs`, and `w32`. Each run has `id`, `kind`, `mode`,
absolute `directory` and original `config`, exact `head`, integer `raw_exit`, and
`source_sha256` mapping every original absolute source path to its digest. The
original config contains `owner`, `exe`, `args`, `workdir`, and `files`; all must
agree with the run's `start.json`/`final.json`, raw exit and before/after source
snapshots. The script's `REQUIRED_SOURCE` declares the device group's minimum
source set, including C11, inherited scalar/fixed/affine/floating mechanisms,
contact bridge, numeric helper, original tests/references and both URDFs.
Additional actual source snapshots are preserved, never discarded.

A pytest run additionally supplies:

- `selected`: exact expanded original node IDs, in the actual CLI collection order;
  `completed`: its successful first prefix; optional `failed`: the next original
  failed node. A failed group must be the original `-x` or `--maxfail=1` run.
- `basetemp`: original path relative to the run directory; `payload_roots`: the
  exact node-to-original-pytest-directory mapping for saving nodes only. The
  reader independently derives each prefix/ordinal from all selected nodes.
- `failed_payloads`: original decodable relative files actually present for the
  first failure, a subset of that node's fixed paths; these do not fill coverage.
- Optional `failed_opaque_artifacts`: exact records with only `path`, `bytes`,
  `sha256`, `reason`, and `log_contains`. Each path belongs to that same first
  failed node's fixed membership and its original pytest prefix/ordinal root.
  It cannot overlap `failed_payloads`, duplicate another artifact, or represent
  any completed node. The byte count is a nonnegative integer within the
  original compressed-file bound; SHA is the complete original raw-file digest.
  `log_contains` names a nonempty original exception line (e.g. the full
  `TypeError: ...` line) present in the bound pytest stdout. Unknown artifact
  fields, including any assertion of completeness, are rejected. The failed
  node must be in `payload_roots` when either kind of failure file is declared.
- `qualifies_completed`: only repeated successful identities, each with its
  actual `prior_run` and concrete repair/retest `reason`.

Exact file/function/node targets and `--deselect` are accepted. `-k` uses a bounded
closed grammar of **complete authored function names**, `not`, `and`, `or`, and
parentheses; no `eval`, arbitrary substring matching or discovery from output.
Unsupported CLI spelling fails explicitly. All 52 identities must be covered in
each mode. Original stdout counts must agree with the selected first prefix.
There are no standalone HOST nodes in this contract.

`w32` explicitly contains `interpret` and `cuda`, each with `status` equal to
`NOT_RUN`, `PASSED`, or `FAILED`; the latter two name their original `run`.
A successful optional run needs `w4_reference: {run, path, sha256}` to the already
audited same-mode contact W4 gzip. Its original config `env` must bind
`PHYMEO_C11_W4_TRACE` to that exact path. The observer checks all four original
worlds' first four records against that complete 40+21 source; it never chooses a
W4 file by output similarity. Missing optional evidence remains separate from
core completeness.

External run `kind` is `native` (`mode: cpu|cuda:0`) or `wheel`
(`mode: interpret|cuda`), with absolute `output_directory` matching the original
CLI `--output`. Four complete external identities are required. Native source
snapshots include all 16 pin files plus the driver, independent affine reference
and its floating base. The native required check table, whole raw state/force,
solve-origin decode, coordinate transforms, full balance and original-row
reaction remain separate observations. Optional metadata is permitted only where
the original producer explicitly permits it (e.g. unused force/parent-wrench
buffers); mandatory mechanics or model fields cannot be replaced with `None`.

Wheel observation verifies its explicit archive, original file URL, optional
installer hash, all archive/installed product members, metadata and actual loaded
module origins. It compares the exact full replay fields and every accepted/
refused world's checkpoint, geometry, diagnostic, held input, material receipt,
energy, status and counters. Original `energy` is the producer's name for
`batch.summary()`; this is a validation alias, not a rewritten physical record.
Generated wheel URDFs carry `text` rather than pytest's `utf8`; both retain their
complete original bytes and hashes. Full ordinary controls use the actual
`SceneInputs.domains` packet; ordinary FK has no invented velocity fields.

`historical_runs` bind their original command/raw/source context and explicit
`reason`; optional `saved_prefix_files` are hash-read and structurally decoded
without granting successful coverage. External failed attempts may instead be
listed with `historical: true` and their fixed original prefix file names.
The reader does not create traces for unsaved cases or upgrade an archived
failed prefix merely because its gzip is complete.

## Handoff and remaining validation

Core auditing reads one bounded gzip at a time, retaining audit summaries,
file identities and small per-world comparison fingerprints. Optional W32 also
holds its one explicit prior W4 payload for the fixed four-step comparison.
It does not decompress evidence directories to disk. A 256 MiB decoded file can
still require substantially more process memory as Python objects; the 2 GiB
limit includes each distinct expanded payload and uncompressed source/metadata
file once, not a claim of peak-RSS performance. Metadata,
archive members and directory inventories are independently bounded. Existing
evidence is read only; only the fresh `ABS_NEW_DIR/audit.json` is written.

The source schema was checked by text against RD `be4f1a0` and native `5b34f120`.
Corrections during authoring were reader-only: compiled delay, ordinary versus
floating FK fields, complete nested ordinary control, actual native decoder
fields, generated-source text encoding and explicit output-root provenance.
No test, native producer, wheel consumer, physical source, h, cap or tolerance
was changed. No evidence payload was opened; no import, collection, numerical
probe, audit, syntax compiler, formatter, linter or test was run. Parent owns
all executable qualification. **CODE_READY / UNEXECUTED / AUTHORING_RELEASED**.

## Pre-execution review: pre-plan contact initial record

RD's text review identified a reader-schema mismatch at the standalone contact
window's initial record. Its producer calls `_record(batch)` before
`prepare_contacts`, so the five authored contact payloads have no `receipts` key
at that one initial position. The reader now requires the key to be absent there
and still validates the complete state, FK, diagnostics and summary. Mixed
contact initial records are produced after plan creation and continue to require
an explicit empty receipt mapping. Every accepted, replay and retained contact
record retains its original complete bound-world receipt coverage check.

This is a pre-execution observer correction, not a physical failure. Membership,
inputs, thresholds and producer bytes are unchanged. No evidence payload was
opened and no project or audit command was executed. **UNEXECUTED**.

The same text review found that `from_asset` model provenance was only checked
for presence. For floating-affine materials it is now compared exactly with
`entry.control[0][1]`, the full original `floating_model_token` retained inside
the graph token. The independent complete `_source_shape` comparison remains;
no geometry is regenerated and direct-binding topology checks are unchanged.

## Serialization failure artifacts and counts-only owner markers

The parent reported run 553 as RAW1 after eleven completed nodes, with the next
`n16-k1` node failing during JSON serialization of the complete collision result.
Its later literal assertion was not reached and `n16-k15` was not run. This
observer change was authored from that report and producer source only; the
original log and truncated gzip have not been opened by the author.

Counts payloads must now retain `owner` and `registration` as the exact marker
`{"opaque_process_local_type":"builtins.object","identity_serialized":false}`.
Both fields are required only in the counts result, with no additional marker
keys. This records that process-local identity was deliberately not serialized;
it neither reconstructs object identity nor replaces the producer's actual
`plan.require_current(result) is result` assertion. The wheel broad-phase subset
and other producer schemas acquire no new owner/registration requirement.

An incomplete serialization gzip can be declared in `failed_opaque_artifacts`
under that failed node. It is read only as bounded raw bytes, checked against its
explicit length and SHA, included in the original gzip inventory, and recorded
as `decoded=false`, `payload_pass=false`, `coverage_contribution=0`. No gzip or
JSON decoder runs for that artifact. This observes the declared historical
failure; it does not assert that a partial file contains valid physical data.
A wrong path, unknown artifact, hash/length mismatch, decoded/opaque overlap,
missing original exception, or attempt to mark the failed node complete refuses
the audit. The original RAW1, source snapshots and failure log remain intact.

Successful earlier nodes require the original CLI's fixed authored collection
order, original fail-fast option, exact first `FAILED` identity, complete ordered
verbose `PASSED`/`FAILED` node lines, pytest summary counts, unchanged
before/after source bytes, and each completed node's complete
schema-valid saved payload. The report records this `prefix_evidence`; only
those completed identities enter coverage after the whole run audit succeeds.
A later run covers the disjoint failed/unreached complement, with repeated green
nodes still requiring the original explicit retest qualification. The failing
counts node gets no pass even if some physical assertions ran before `_save`.

This is an observer-only manifest extension approved before execution. Original
52/46/14 membership, physical data, thresholds, fixture source and test count are
unchanged. No audit, test, import, numerical probe, static check, or evidence
read was performed. **CODE_READY / UNEXECUTED / AUTHORING_RELEASED**.

## Final text review: explicit per-node outcome evidence

RD identified that prefix identity plus aggregate pytest counts alone did not
prove each completed node's explicit log outcome. The reader now parses complete
verbose node lines and compares the whole ordered `(full node ID, outcome)` list
exactly with `completed` as `PASSED`, followed by the declared first `FAILED`
node when present. Duplicate, missing, reordered or different parameterized
identities refuse coverage. Matching is whole-line/whole-ID, not substring or
parameter-prefix matching; an optional standard pytest percentage is permitted.
No quiet-log fallback or inferred missing `PASSED` is supported. The original
RAW1, summary, fail-fast, unchanged-source and complete-payload gates remain.
`prefix_evidence.verbose_outcomes` retains the matching original stdout line
numbers. Failed opaque artifacts still never contribute a payload or node pass.

This correction is limited to the two new observer files. No physical check,
input, scope or threshold changed; no evidence was opened and no audit, project,
import, static check or numerical command ran. **UNEXECUTED**.

## 父静态准备（2026-11-08）

已导入最终0e044ef并完成原RD全文/增量交叉审阅。571首Ruff的49项均为E702分号；572首format-check为1文件。573仅formatter，574 Ruff/575 format-check/576原始AST逐树相等均RAW0，未改任何审计语义。原首失败与源码快照保留；保存审计本身仍待实际运行，不能从静态通过称证据已完整验收。