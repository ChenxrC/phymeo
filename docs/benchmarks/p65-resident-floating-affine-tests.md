# C11 浮根 affine：独立案例作者交接

**AUTHORING_RELEASED / UNEXECUTED**。本页是测试与独立参考交付，不是 C11 验收结果。依据父 `d8bbf40` 的[冻结计划](../superpowers/plans/2026-11-08-resident-floating-affine.md)及[原五锚清单](../development/p65-resident-floating-affine-case-manifest.md)。C10 正式接受仍是父 OLD 接入与 C11 执行的前置；作者没有运行 import、collection、数值脚本、pytest、静态检查、构建或原生引擎，也没有合并父或改旧文件。

## 交付范围与先验

恰五 NEW：

- `tests/scale/test_resident_floating_affine_step.py`：A01 有效公开 RED 前置、A01/A03/A04/PD/大 N 短档、A04-p1 的 40+21、完整记录辅助。
- `tests/scale/test_resident_floating_affine_contacts.py`：A02 有限原 right_pad–静态球与 A05 同树双动态，反序/协变、法向与原材料点、同次 full/reduced 响应。
- `tests/scale/test_resident_floating_affine_transactions.py`：原三类 mixed 世界、graph/full/reduced/local_origin 边界、两 bank 容量与实际晚异常、stale/close-rebuild；W32 单独后置。
- `tests/reference_newton/robot/resident_floating_affine_reference.py`：NumPy/stdlib 独立参考，没有 Phymeo/TYX/Newton import，也没有 FD。复用 C9 已接受的独立齐次变换 time jets；这不是 native Newton 运行器。
- 本页。

首 checkpoint `9109931` 在已有真实 jaws HOST 资产上先确认完整 8×8 质量、literal X 块、`a=(1/14,5/14,-5/14)`、原行 `lambda=-4/7` 与根约束零，然后才在函数内查询 `ph.FloatingAffineRobotTree`。`8a98c29` 把手推恒等式先乘 14 用整数核对，避免二进制有理数括号舍入造成假的严格前提失败；不是物理值或设备误差门变化。首 node：`tests/scale/test_resident_floating_affine_step.py::test_a01_public_affine_floating_root_reaction`。有效 RED 尚待父串行执行。

A03 fan/chain 保持各自原始约束行、非零 offset、独立入速 .02 和四 body 原外力；不因相同 P 合并 graph identity。A04 按原 elbow 的四个 authored 惯性部件逐个累计 M/b/J，再按质量汇总完整 physical-group COM/J；没有读取产品折叠结果生成期望。完整未约化 KKT 给出 full acceleration 与每条原关系乘子；独立 S/E/P/c 来自原输入声明，真实电机仅 E 注入。A04 观察将实际 principal group frame 通过已有 authored site 还原，与原部件 frame 比；不把 principal 轴当原 URDF 轴。

A02 首步额外以已有几何和原 law 核真实 −2/+2 N；其后独立有限 box–sphere 最近点公式与完整 KKT 对照。A05 使用独立球–球几何，首根/独立/从动加速度为 `(2/7,-10/7,10/7)`，原行乘子 −2/7；共同点矩和两侧材料位移由完整原 receipt 重建。HOST patch 只是真实入口前提，不能冒 device narrowphase。

## 公开合同及保存

公开声明 `ph.FloatingAffineRobotTree(asset,state,*,kp,kd,effort_limit,delay_steps)`；输入 `.inputs(Kx3, external=FloatingRobotInputs)`，full joint external effort 全零，COM 力矩即时。root 无电机。checkpoint 的 `control` 为 `(graph_token, drive, full_zero_effort, body_force, body_torque)` 五块；root/local、full/reduced q/v、applied/held/history 不省略。

直接 `ResidentFloatingAffineTreeBinding(key,world,group,points,topology=checkpoint.topology,graph_token=command.graph_token)`；A02 原有限指面使用 `from_asset`。诊断明确 `coordinate_kind='floating_affine'`，`floating_affine_diagnostic` 与 batch 同次实际读数一致；full 四 counts 及 `full_dof_count=6+n`、`reduced_dof_count=6+K`、`independent_dof_count=K`、完整 S、关节 c、原 C、graph token 均断言。`row_powers_w`、`constraint_power_w` 与六维 `root_constraint_residual` 分别保留。

每次 short/window 保存完整原输入、URDF 文本/SHA、原 FP64 模型/图、实际编译 FP32 表、即时外载/选择 drive、entry/accepted、完整 FK/J/M/b、full/reduced 外力/加速度/原行反力功、energy、材料/application 位移/功及状态。未把 independent predicted state 标成设备 candidate。晚故障在实际原方法完成后注入，分别保已上传数组、实际 candidate packets/group arrays 或未发布 receipts；失败 prefix、retained 全状态及实际错误仍保存。

原 batch 在失败后会使 transient `tree_diagnostics()` 无效；测试要求它为 None，保持 accepted receipt 中原诊断不变。合法 restore 自身推进 epoch/切 bank，stale 测试只禁止随后过期 attempt 发布，不错误要求恢复本身无 epoch/bank 变化。所有 full/reduced/history/FK/held 和完整原 receipt 仍必须保持或按合法重放恢复。

每份直接无损 gzip，上限 64 MiB 压缩/256 MiB 解码；超限独立报 IO/resource 错误，不删帧、不清旧证据、不伪装物理拒绝。后续 saved audit 总解码上限仍为原 2 GiB。保存辅助序列化普通 DrivenRobotProgram 的完整 `problem_signature()`，而不是无法复算的对象 repr。来源或实际数据不一致不拼接成同源结论。

## 固定矩阵与运行次序

以下是手工预声明，**不是 collection 或通过数量**；实际父收集如不一致应先查作者错误。

| 组 | 核心实例/backend | 完整 gzip/backend | 物理窗口 |
|---|---:|---:|---|
| 短档 A01/A02/A03/A04/A05/PD/n16 | 19 | 19 | 原 1/2/4 步；h 沿清单 |
| A04-p1 代表窗口 | 1 | 1 | h=1/4096，40+19→21 |
| core W1×4 与 W4 | 1 | 5 | h=1/1024，每配置仅4步 |
| contact W1×4 与 W4 | 1 | 5 | h=1/1024，每配置40+19→21 |
| X01–X06 新状态/图/事务 | 27 | 13 | 14个0步负控无gzip，其余原<=2/3调用 |
| X07 同次读回/回执校验 | 3 | 3 | A05 W1，暖步1+拒步1+同命令重试1 |
| **核心** | **52** | **46** | 原49/43均保留，不铺全排列长窗 |
| 后置 W32 | 1 | 1 | 4步，另授权/单独调度 |

准确 function、variant IDs、保存相对名沿原清单。核心三文件实际手写节点为 S 15、C 5、T 32；T 另有 optional W32 1。没有额外 HOST fixture 测试计数，A01 HOST 前提属于同一个 node。全部 JSON 保实际 world keys，不删 owner 或重写 world 标签做同值比较。

core 四 world `503/91/1207/6` 分别为 A01、A03-chain、**原 C9 `reference.case('P05',n=0)`**、原 FreeSphere(2 kg,r1/8,z2,g−8)。前两及旧 C9 delay1；这是父确认的 root-only 身份补足。contact 四 world 为原声明的 jaws root lifecycle、A05、原 C7 ordinary P、原 C7 fixed-affine A03。旧兼容 world 只用于同批回归，不计新增浮根 affine 能力。lifecycle 必须在原40步中实际正载→零载→正载；未发生则保存失败，不加长/改 g/m/k/h。

所有 delay1 初始 target 来自 **FP32 独立 initialq**，rate/ff 为零；A01 的 u=0 只是该例的数值，不能推广为全零历史。PD 首帧即使 target=0 仍响应当步 COM 外载；用原 Meff=14/5、beff=−1/5 的手推 Schur 结果核实际 motor。保留父 C10 原 nonzero-initialq 揭示的延迟约定，不调产品去匹配错误零假设。

父可先执行最小 node（示意，必须由父 fresh guard/唯一队列运行）：

```text
python -m pytest tests/scale/test_resident_floating_affine_step.py::test_a01_public_affine_floating_root_reaction --require-tyx --tyx-mode=interpret --basetemp=<new absolute path>
```

CUDA 同 node 只切 `--tyx-mode=cuda`。核心三文件使用 `-k 'not test_w32_bounded_affine_contact_batch'`，父按首绿 identity 排除已完成节点，再按原短档→事务→代表窗调度，不能一次绕过短档。后置 W32 必须显式设置 `PHYMEO_C11_W4_TRACE` 指向同 backend、同源码完整 `contact/w4/trace.json.gz`；保存源码 SHA、40原行/21重放均核后，只比较原前四帧。它不会扫描输出挑成功样本，也不会重跑 W1/W4。其余28world保完整自身身份及独立物理审计。

## 边界及父后续

沿原 same-entry atol5e−6/rtol5e−5、M/方程/归一原图余额2e−5、PD5e−5、窗口2e−4；材料1e−10只核 stored 数据重建。n16/K1和15只一真实步，并以实际 current bounds 核 body16，不以 K 作 body 上限。角动量/有限步 penalty 功不冒精确能量守恒；两方材料功与 application 功独立保存。

本批没有 native、wheel consumer 或 saved auditor 新代码授权；这些仍在原后置计划。C11 未执行、未确认通过、不改变 C9/C10既有验收状态。父先接受 C10，合并 NEW 和其唯一 OLD 接缝后才运行；成本超原排程门可将整代表节点明确 NOT_RUN 排后，不能缩窗后保同名“通过”。

## X07 只读发现与定向负控补充

固定产品作者快照 `6eb43b8` 的五 NEW 已完成纯文本交叉审阅；仅发现 `diagnostics_from_arrays` 缺少 selected full/reduced 数值有限性及一次推进 tick 校验。`4ad81d9` 的 NEW storage/report 增量已文本闭环：核实际 B/Dfull/Dreduced/R 切片和 assembled world q，保原物理/门及 padding、未选世界语义。该结论没有执行证据，也不代表 C11 验收。

按父 `a610add` 冻结 X07：`test_x07_candidate_diagnostics_reject_corrupted_readback[full_nan|reduced_nan|stale_tick]`，保存 `diagnostic-{variant}.json.gz`。三例各复用原 A05 W1/delay1/h=1/1024：一次真实合法暖步建立 accepted receipt，在下一次正常 `storage.diagnostics_from_arrays(worlds,full_arrays,reduced_arrays)` 入口注入，实际 IO 已完成、原构造未开始；分别改 selected full mass[0,0]、reduced mass[0,0] 为 NaN，或 info exit=entry，然后必 call-through 原构造器。无手工抛错冒校验、无新候选 schema、无再次构造替代父正常发布路径。

断言失败保全 root/full/reduced/held/history/FK/summary/accepted receipt/status/epoch/bank；transient diagnostics 仍按旧 API 无效。退出补丁后同 plan、同 command 只重试一个真实步，入口须为原 retained checkpoint，并复用完整独立同态与材料审计。每例因此是两次接受、一拒绝 attempt，不新增长窗。

完整原读回和修改后的 full/reduced mappings 都保存；另存 modified selected array 的 dtype、shape、前后完整值，两个 NaN 例追加前后 `.view(np.uint32)` 的全部原始字，tick 例保 int32 原值。原 encoder 对非有限值的语义标记没有改动，原始字不会由该标记代替。核心新增仅3节点/3gzip，原49/43不改，后置 W32 仍1/1。全部 **UNEXECUTED / AUTHORING_RELEASED**；只读审阅与纯作者未运行任何项目命令。

## 父 542 首次前置失败及精确算术同步

父报告 542 首 A01 实际尚未触及新公开 API：旧 HOST 完整 M/b/a 与反力锚已过，随后所运行 tests-first 快照中的 `5 * (1 / 14) - 5 / 14 == 0` 因两种 FP64 运算次序的舍入不同而失败。这是测试前置断言错误，**不是有效能力 RED，也不是数学或产品失败**；原始失败收据保留。

作者树此前 `8a98c29` 已用乘 14 后的整数恒等式避免该问题。本次按父裁定统一为 `5 * Fraction(1, 14) - Fraction(5, 14) == 0` 并仅增加标准库导入，以原有理数直接表达独立动量恒等式。原 literal 物理值、HOST/device 数值门、资产、控制和窗口均不变。

该增量仍为纯文本作者，未 import、运行测试或静态工具；父负责唯一 A01 重试及结果归档。**UNEXECUTED / AUTHORING_RELEASED**。

## 父短档实际结果与保存修复（2026-11-08）

551 A01 interpret 1 passed/1.32s、552 CUDA 1 passed/3.19s，均 RAW0。553 interpret 短档首11节点通过，第12 n16-k1 在最后保存 ResidentCollisionResult 时因 owner/registration 两个进程局部 object 无 JSON 编码失败，RAW1；n16-k15未执行。失败节点的最后 literal 断言在 _run 返回后，未执行，不能称该节点通过。原部分 gzip、完整日志和源码快照保留。

仅 _json 精确 type(value) is object 输出显式不可恢复标记，两个能力字段仍保留；其余 dataclass 完整字段不删，未知类型仍失败。标记不表达不同 token 相等，不保存身份或提供重放权限；实际 plan.require_current(result) is result 继续在运行时执行。物理实现、输入、工程阈值和窗口不变。后续只重试计数函数两项；此前完整绿节点不重复。