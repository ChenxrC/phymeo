# C11 floating-affine installed-wheel consumer

Status: **AUTHORING_UNEXECUTED**. This is the two-NEW consumer handoff authorized
by parent `5907e10`, plan §9; it is not C11 product or wheel acceptance. C10 formal
acceptance, parent C11 integration and the core matrix precede any execution.
No project import, collection, numerical probe, test, static check, formatter,
build, install, or native command was run by this author.

## Fixed public task

The single script uses `import phymeo as ph` for declarations, load/import,
controls, steps, restore, queries and owned public readouts. It imports no checkout
test/reference and no private Phymeo integration kernel. Its independent reference
embeds the accepted C9 transform time jets and C11 full original-row KKT, including
A04's four authored inertial components. The two old fixed-root Px worlds have
literal mass/J and finite-face formulas, not product mass/J as their oracle.

| Section | Frozen inputs and calls |
|---|---|
| W4 | World 503 lifecycle, 91 A05 same-tree two spheres, 1207 original ordinary P box, 6 original fixed-affine A03 chain; h=1/1024; 40 accepted steps, checkpoint at tick19, restore, 21 replay steps |
| Capacity | On the same W4, bank0/world503 and bank1/world91 each use a new zero-site-capacity plan once, then the unchanged sufficient-capacity plan and controls once; at most two real bank-alignment steps, fully recorded |
| A04-p1 | Original floating nonparallel R/R affine chain plus offset positive-mass fixed tool; h=1/4096, four steps, original nonzero root/joint velocity and full COM force/torque packet |

All these declarations keep their **original delay0**. Parent confirmed that §9's
old delay1 wording was a reminder for applicable core cases, not authorization to
alter these consumer inputs. The independent core delay1 tests remain unchanged.
Full held controls and history are still compared and saved at every step.

The protocol has 69–71 batch step calls/backend: W4 40+21+2 capacity attempts+2
retries+0–2 alignment, and A04 four. Expected accepted world steps are 262–270,
with exactly two capacity-rejected world attempts. Replay counts actual work but
restores the clock; final W4 ticks are 43+a for worlds503/91 and 44+a for1207/6,
where a is the actual alignment count. Summary counts returned acceptances from
real rows; it never manufactures these expected totals after a failure.

## Same-input verification and records

- The full root+joint mechanics, reduced mechanics, S/P/c/C, graph tokens,
  independent motors, full applied effort, original-row multipliers and powers,
  root constraint balance, full/reduced q/v, COM pose/twist, energy and same-entry
  response use the original gates (5e-6 absolute/5e-5 relative, 2e-5 mass/reaction
  and equation bounds, 2e-4 accumulated-window absolute). Root quaternion
  hemisphere continuity and full body/velocity/actuator counts remain distinct.
- Literal A05 root/independent acceleration `(2/7,-10/7)` and lambda `-2/7`, and
  fixed-A03 acceleration `4/7` / original lambdas `(41/14,-25/7)`, precede device
  comparisons. Actual uploaded FP32 wrench may supply the *same-entry response*
  reference only after its independent finite-geometry FP64 law is compared.
- Every actual material ID/weight/skin point, normalized patch weight and original
  law is retained. Stored entry/end poses reconstruct the original two surface
  anchors and common load point; that is record consistency, not a new dynamics
  oracle. Their separate works and static COM force/torque are saved. The original
  1e-10 m record-construction and 1e-12 J/1e-10 relative work gates are unchanged.
- Full forward/replay state, geometry, diagnostics, receipt, control/history,
  summary and broad physical fields compare exactly. Epoch and cumulative device
  scheduling statistics are retained separately and are not called physical
  differences. No physical field is deleted or normalized to get replay equality.
- Zero capacity retains the failed world's complete checkpoint/root/full/K state,
  held command/history/FK/energy and the original plan's accepted receipts. Other
  worlds actually advance. Sufficient-capacity retry starts at the retained state.
- Lifecycle must load, open and reenter within its original 40 steps. A05 retains
  both dynamic reaction sides and true root response. A04 retains all four inertial
  parts and authored frame maps. This is HOST finite mu=0 sampling plus TYX response,
  not full-device narrow phase, CCD, continuous friction, or a performance result.

## Installation/source protocol

Required CLI (each path absolute; run only by the parent after prerequisites):

```text
python -I -B <fresh-outside-cwd>/resident_floating_affine_wheel_consumer.py --mode interpret|cuda --output <new-absolute-output> --wheel-site <installed-site> --expected-package-root <installed-site>/phymeo --expected-wheel <actual-new.whl> --forbidden-checkout <checkout>
```

The cwd must equal the copied consumer's directory. Cwd/script/package/output must
be outside the forbidden checkout and every discovered `.git` checkout. Output
must be new and outside the package. The actual wheel path/SHA, local direct URL,
installer hash when present, every installed `phymeo/` member's exact bytes and
membership, dist-info bytes, and every loaded Phymeo module's origin/bytes are
checked against the wheel. Package/wheel and copied script bytes must remain
unchanged before/after the run. Interpreter and actual dependency versions are
recorded. No editable or checkout fallback is accepted.

Each backend writes one `resident-floating-affine-wheel.trace.json.gz` and a
`summary.json`. The trace contains embedded original URDF text/bytes/SHA, model
and graph declarations, inputs, all public outputs, independent values, real
resource counters and failure prefix. It is limited to 256 MiB decoded and 64 MiB
compressed. An over-limit temporary archive remains evidence and causes failure;
no truncated success or deletion of old evidence occurs. Archive IO errors are
separate from a physical rejection. A complete *prefix archive* is explicitly
labelled `prefix_only`, while window completion is separately reported. All created
plans/batches close through nested finally paths and must report live owned bytes
zero. No private compilation table is invented when absent from the public API.

## Exact authoring scope

1. `tests/scale/resident_floating_affine_wheel_consumer.py`
2. `docs/benchmarks/p65-resident-floating-affine-wheel-consumer.md`

The original five C11 test/reference/report files, product modules and shared hooks
are unchanged. Parent owns all subsequent static, installed-wheel execution and
saved-output auditing. **AUTHORING_RELEASED; UNEXECUTED.**

## Pre-execution review correction

Read-only cross-review found that `floating_audit` assigned to the local name
`selected` while calling the helper with that same name, causing an unbound-local
error. The local value is now `selected_drive`, including its `advance` argument.
Only these two references changed in the consumer. Original inputs, windows,
delay0, gates and saved records are unchanged. This was a source review finding,
not an executed failure. **UNEXECUTED / AUTHORING_RELEASED** remains in effect;
no import, test, numerical or static command was run for this correction.

## 父580源码外首轮观察器修复（2026-11-08）

新wheel578/离线安装579完成。580 interpret在W4第4实际接受步观察失败（前三完整、16 accepted world-steps），A04未开始；完整prefix、原RAW1与原copy保持。原record显示world503此步sites为空，static_reactions仍保原登记静态key，符合产品对全部注册静态形状发布零反力的合同。

consumer旧material_audit从空dict只累计active site，却用该集合要求完整static_reactions相等。仅将expected初始化为独立sources声明中全部group=None的零六维反力，再累加sites；严格集合相等、逐key参考点/力/矩与全部材料/application功及J转置检查保留。零site的登记形状现在必须明确给零反力，缺/多key仍失败。无产品、fixture或阈值修改。重试沿同一已构建wheel与全新consumer目录，不重建同字节产品或重跑已绿源码组。