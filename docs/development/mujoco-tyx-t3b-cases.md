# MuJoCo-TYX T3b finite-pair preregistration

## 父最终验证（2026-11-10）

产品/测试 `de623915` 已由父单一串行槽完成实际验证：interpret36联合覆盖、CUDA34、相邻33、新wheel源码外两backend均通过。完整结果及原失败见[实际验收报告](../benchmarks/p65-mujoco-tyx-finite-contacts.md)。本文件下文的UNEXECUTED/PENDING指原作者交接快照，不能覆盖当前结果，也不把新结果倒灌到旧commit。

最终三pair的局部ordinal独立于跨精度对应；完整几何双射保重复数和全部物理检查。原.512s四条胶囊的支撑门未过，每后端另外四条盒子通过；全部八条保持同一初态/参数/pulse与误差门连续至1s，恢复检查通过。Kahan设备时钟、reset/clone/rollback和受影响旧积分路径已验，保留高位每步严格增加的范围限制。原作者无执行owner，父本批执行已全部结束。


2026-11-10. Baseline `275872beca81767bdc2ee8a1d0ca52fb5ae9c100`. **AUTHORING_ONLY / UNEXECUTED**. Parent owns all project execution. This file fixes inputs and gates before new product output. Only NEW tests/test_mujoco_tyx_finite_contacts.py and this file belong to the test author.

## Native semantics and scope

CPU MuJoCo3.3.7 is the independent oracle. The existing condim1, source options, physical-M coupled QP, native Euler damping/integration and atomic publication remain unchanged. The added families are capsule-capsule, capsule-box and box-box; native geometry, regularization and source control are compared at the actual stored input, then an independent CPU trajectory advances continuously.

Read the fixed [capsule primitive source](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_primitive.c), [capsule/box and box/box source](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_box.c) and [native collision driver](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_driver.c). Parallel capsule axes can produce two endpoint contacts, while the general closest-segment branch produces one. Capsule-box may select a second contact. Box-box has face and edge branches; its CPU contact distance convention, including the face branch half-depth, is retained exactly.

Native driver filtering and stable compaction are part of the resulting contact set. New feature_index is the final per-pair native output ordinal; it is not a persistent material address. A complete (pair, ordinal) bijection preserves all final native contacts, including any equal/near-equal positions that remain. No sorting by position, output deduplication, or clipping CPU contacts to Warp's eight-point layout is allowed.

Static slots are capsule-capsule2, capsule-box2, box-box24; total static ceiling32 remains. The box24 bound is conservative scratch/output allocation, not an expected typical contact count. One box-box pair is admitted; two need48 and reject at construction. Runtime detected and active-row capacities remain separate and are tested at exact native count and count-1.

## Fixed inputs

Units are SI and native q/v coordinates. Defaults: h=.001 s, Euler, Newton source iterations128/tolerance1e-10, dense J, condim1, solref(.02,1), solimp(.8,.9,.002,.5,2), nonnegative margin. No parameter is fitted to TYX output.

| Group | Inputs and independent checks |
|---|---|
| capsule-capsule | parallel overlapping axes with two endpoint contacts; skew axes at right angle; endpoint-only; separation and exact positive-margin detection |
| capsule-box | horizontal side over a finite top face; vertical endpoint over top; side near a finite edge; endpoint near a corner; same-input full geometry, law, J, force |
| box-box | flat face; yawed face polygon; tilted edge; general nonidentity rotation/offset COM; retain complete native manifold and native distance convention |
| free eccentric rotation | m2kg, COM(.03,-.02,.01), inertia(.08,.10,.12,.002,-.001,.003), nonzero incoming translation/local omega; 24 steps capsule-box and 24 steps box-box with world COM wrench |
| two dynamic branches | fixed common ancestor, two hinge-slide branches with opposing moving capsules; both real joint groups receive nonzero equal/opposite contact force and nonzero joint torque; 32 steps |
| full unload/reentry | each of horizontal capsule-box and box-box on a real vertical slider, m in {.25,2}kg and h in {.001,.002}s, full Cartesian 8 windows, all from original contact initial state to .512s |
| capacities | same new layout exact detected/active native counts pass; count-1 fails atomically; detected inactive row counts; box24slots plus second pair48 rejects explicitly |
| batch/protocol | new finite-pair W4 sparse IDs, selected reset then clone next real step, actual late readback failure/all-world retry, real device step with CPU/Warp/HOST/NumPy numerical fallback disabled |
| public ENV | finite box-box contact force enters observation and control; real accepted action held through decimation; selected reset and next-step comparison, without changing contact law or rerunning old T3a cases |

The full unload input uses a finite static box halfsizes(.24,.18,.05). A horizontal capsule (radius.1, halfaxis.12) or box halfsizes(.12,.08,.06) starts .001m inside its top, incoming vz=-.02m/s, armature0/damping0. The unit-gear vertical motor applies zero until .096s, then upward 3mg during [.096,.160)s, then zero to .512s. Gravity is -9.81m/s^2 throughout. No q/v reset, position prescription, force injection after integration, or hidden support is used. All eight windows must exhibit initial real support, a positive-duration contact-free/zero-force interval, then positive force on reentry; final total support is checked against weight, not every contact being positive.

## Unchanged engineering gates and independent oracle

Reuse T3a's registered gates: point/distance/body pose rtol3e-5 atol4e-5; normal atol1e-4; J rtol2e-4 atol5e-5; M rtol3e-4 atol5e-5; smooth forces rtol4e-4 atol4e-4; R/aref/Jv rtol5e-4 atol4e-4; dual H rtol7e-4 atol5e-4; force/qfrc rtol3e-3 atol3e-3; qacc/integration rtol3e-3 atol1e-2; continuous q atol2e-4, v atol3e-3, time2e-6.

KKT independently reconstructs H=J M^-1 J^T+diag(R), b=J a0-aref from native same-input fields. With g=Hf+b, residual=max_i(f_i>0?abs(g_i):max(0,-g_i)), scale=max(1,max_i(abs(b_i)+sum_j abs(H_ij*f_j))); residual/scale<=5e-5 and f>=0. Runtime's own converged/iteration/budget/stored tolerance checks also remain. For <=4 active rows the existing independent exhaustive active-set oracle is retained.

T3a serialization/native mechanics helpers may be reused. New test-only wrappers supply final native per-pair ordinals and h-dependent integration acceleration; they do not use product geometry to manufacture expected values. The new complete-ordinal matching replaces only test-side matching when calling the old validation helper, never production calculations. Same-input and continuous CPU results are both checked, so neither a CPU endpoint restart nor matching only final q can hide an incorrect trajectory.

Each full window saves every q/v/control/time/contact-count/normal-force and selected complete input/native/receipt at first, pulse boundaries, first separation, first reentry and final. Failure adds its complete evidence. Compact JSONL only; no oversized proof snapshots. Phase behavior gates: support before pulse; at least .02s with no contacts and zero force after lift; reentry after pulse and before .48s; final .02s average force in [0.7,1.3]mg, |vz|<.1m/s and height within .01m of initial support. These are behavior gates alongside unchanged per-step native engineering comparisons.

No execution result is claimed. Old T3a suites are reused by acceptance and not rerun. Friction, generic mesh/CCD, multiple free roots, equality/tendon and G1 standing remain outside T3b.

## Author handoff, before execution

The independent test file is authored with 11 static positive geometry fixtures, 3 dynamic fixtures (24/24/32 steps), all 8 full .512s unload/reentry windows, 3 exact/overflow capacity fixtures, inactive-capacity and 48-slot rejection, W4 selected reset, genuine late readback failure/retry, no numerical fallback and 3 separated finite-pair controls with zero runtime capacities. Every window retains the original incoming velocity and a contiguous no-contact interval of at least .02s. The public finite-contact ENV case is owned by the parent in its separate test file; this lane does not duplicate it.

The h override recomputes only the independent Euler integration oracle with the actual native model timestep. The native contact ordinal wrapper retains a complete per-pair output bijection, including near-coincident positions. Existing helper product inputs and old tests are unchanged. Fixed fixtures, thresholds and pulse times are unchanged from the preregistration; no implementation output has been consulted. No project import, collection, numerical probe, test, lint, formatter or build has run in this lane. Validation remains **PENDING_PARENT_SERIAL_EXECUTION**.

## 09 后的跨精度局部编号裁定

09 首次 yaw-face 对照保留 RAW1：双方完整8点和对应力一致，TYX ordinal到CPU ordinal为[5,1,2,0,4,6,3,7]。原保存的CPU Rzz=1.0000000000000004，TYX纯yaw公式Rzz=1；独立源码给 c2-c1=(s-1)(.05-z-.06)，使CPU选择另一等深参考面。这是数学平局的舍入破局，不是漏点或窄相算法错误。产品几何不改、不加epsilon或fixture分支。

按用户接受工程精度、以通用流程为优先的要求，撤销“跨精度相同ordinal”这一过约束；feature_index明确为各后端各pair本步紧凑局部序号。测试先独立核双方各pair数量和完整不重复0..n-1，再用原位置/法向容差构造确定性增广路完整双射。每个重复点是独立节点，不能去重、截断、按力或J择配。保存原native编号；仅旧helper的局部拷贝适配编号，所有physical字段、rawslot、J/R/aref/force、H双轴、KKT及连续q/v检查原样保留。整个原参数网格不变。

补一个纯合成负控节点，覆盖置换、重复数、坏ordinal/位置/法向、贪心陷阱、Hall失败、力字段不参与选择。现共68节点（解释35、CUDA33）。由于公共对应规则发生变化，最终完整解释重验所有35节点有新依据，不是重跑同源已绿集；真实CUDA33仍首次。原08/09失败作为历史，不改为通过。两个原作者独立纯文本复核已完成，原产品几何保持。

## 父首次执行后的稳定性修正

保留08/09/12/13原始失败。12倾斜box出现4点对2点及深度/力翻倍，不能当容差差异接受。原因是精确共轴产生重复SAT轴，FP32使它错误替换已有face为原生edge深度分支。修复只在penetration>=0时阻止精确单位行/列证明冗余的cross轴替换feature，仍保全部15轴分离检查；A_i=±B_k使A_i×B_j为第三B面法向，反向同理。零epsilon、无fixture/坐标/时间分支。原比较等价重排为c3-penetration < -penetration*1e-12，保负penetration原语义，另加正margin分离盒的两backend回归。

13双活动胶囊第1步：native已为1点，而FP32 Gram相减det丢失近共轴差异，误留2点并改变共同力。改为同一解析公式的稳定表达：det=|a×b|²，x1=-[(d×b)·(a×b)]/det，x2=-[(d×a)·(a×b)]/det，后续原顺序夹紧、mjMINVAL、重复规则不变。不扩大平行阈值或改物理模型。

补充后的完整测试节点数为70：interpret36（含两个无后端合成/准入节点），CUDA34。最终以实际collection/日志为准。8原连续0.512s轨迹、所有原误差门仍保持。

## 连续窗口追加观察（原 .512 s 门不改写为通过）

16 第一条 capsule/.25kg/h.001 从原入步速度完整跑至.512s，每步原native几何/力/动力学/q/v和时间门全部通过，入层、离地、重入均成立；末.02s平均支撑1.61804796N，小于原.7mg=1.71675N，故原末态静支撑门RAW1保留。物理仍处回弹恢复，不能修改力律让它人为更早站稳。

后续所有8条原case保持原初态、pulse、mass、h、参数和逐步工程门，延长同一连续状态至1s（无中途reset）。.512s原支撑门继续计算并保存true/false，不把它改写为通过；1s新增恢复检查沿用同一[.7,1.3]mg/末速度/高度标准。原reentry必须在.160..48s与至少.02s完全离地仍必需。原始前缀有失败，增加真实后续状态后才复验；不是无理由重复已绿全套。

14时间首次失败已保留；15确认当前TYX无f64类型，随后用两FP32 high/comp缓冲在同一设备积分kernel内Kahan累加。发布high-comp和原始前后pair一起核对，q/v/M/J/接触精度不变，不是HOST修补运动。16完整.512s时间门已通过，原clock型别与reset/clone/rollback仍需最终CUDA/相邻验证。
