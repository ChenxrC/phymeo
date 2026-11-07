# C7 固定根 affine-mimic 常驻接触：正式限域接受

当前状态：**LOCALLY_ACCEPTED / C7_COMPLETE**，产品876ae848。2026-11-08。下文作者交接是历史记录，实际验收见末节。原作者状态 **MATRIX_UNEXECUTED / AUTHORING_RELEASED**：作者仅编写四个 NEW 文件，未 import、collection、执行数值、pytest、静态检查、构建或设备探测；未修改旧产品或 merge 父。所有运行由父唯一串行队列持有。

权威范围见[冻结计划](../superpowers/plans/2026-11-08-resident-affine-contact-feedback.md)与[原提案](../development/p65-resident-affine-contact-proposal.md)。本片是有限 static、mu=0、discrete-entry 的 **HOST 几何/法向律 + TYX full-N/K 真实受力响应**，不是设备窄相、CCD、floating、moving/moving 或持续摩擦。

## 已有执行与本次作者范围

父保存 359 有效 RED：1 failed / 1.48 s，真实 3 N、旧绑定/FK/状态前提均成立，准确停在 ordinary-only binding guard。随后父产品 37897e38742becb4eee5fbdbe456454e16614434 下，368 首 C7-01 interpret 1 passed / 1.32 s：reduced a=1、full a=(1,2)、原行 lambda=-1 均通过。369 同一首节点实际 CUDA 1 passed / 3.97 s。370 新接缝曾在旧 C6 W1/W4 复现局部 group 索引遮蔽导致 TypeError；父仅改名 group_index 后，371 同节点 1 passed / 14.96 s。原失败保留，无数学/阈值修改。这些是父通知的首节点和受影响回归结果，不是本作者已执行整个矩阵；新两 affine 的 W4 仍独立待验。

C6 历史验收 bc0aba5 及原 C5 reference 仅复用，不重新运行。C7 两完整文件及其余 CUDA 矩阵、源码外新 consumer、最终格式/资源审计均等待父实际执行；这里不预写通过数或性能收益。

四个文件：

- `tests/scale/test_resident_affine_contacts.py`：独立全组件/KKT oracle、真实有限几何与逐步材料/作用点/功审计。
- `tests/scale/test_resident_affine_contact_transactions.py`：混合世界、容量、原图/owner、两 bank 故障与恢复。
- `tests/wheel_consumers/resident_affine_contacts.py`：无需 checkout 夹具的公开 wheel 消费者。
- 本报告。所有 OLD 产品接缝归父。

## 冻结数值与独立真值

测试复用 `tests/reference_newton/robot/resident_affine_reference.py` 的独立 full-component 几何、能量导数与原约束鞍点解。当前产品 M/J/acceleration 不生成预期；`tree_diagnostics()` 只核对同次真实读数与 receipt 中 `affine_diagnostic` 的一致性。消费脚本不 import 该 reference 或任何仓库测试，而直接写入下面 A03/A05 的经典平面闭式公式与声明 URDF。

| 案例 | 输入与首步独立预期 | 新覆盖 |
| --- | --- | --- |
| C7-01 / A01F | full m=(2,1)、P=(1,2)、offset=(0,1/8)，follower 真 3 N 替代原外加 3 N；Mr=6、a=1、lambda=-1、full reaction=(2,-1) | sphere / capsule / hull 的总站点权重只计一次；40 步 |
| C7-02 / A02 | fanout P=(1,2,-1/2)、m=(2,1,4)，独立 motor 7 N + 第三 body 接触 3 N；a=11/14，lambda=(11/7,-32/7) | 负比例、非零 offset、40 步、反序 URDF/轴循环与平移 |
| C7-03 / A03 | chain P=(1,2,-1)、m=(2,1,1)，motor 7 N + 3 N；a=4/7，原行 lambda=(41/14,-25/7) | flatten 保同运动，但原行 lambda=(8/7,-25/7)、graph 身份不互换 |
| C7-04 / A05/G | 真实 R/P 分支、全交叉 M；a=(-512,935)/741、lambda=53/247；原 gravity=(0,-2,0) 留出 a=(-1360,469)/741、lambda=25/247 | full N COM wrench/Jᵀ，再 Pᵀ；40 步和变换 |
| C7-05 / A04 | 两 R、gain=-2、offset=pi/2、入步 u=.5、motor1，实际2N；a=27/20、lambda=27/40 | 原角速度与 COM 偏心、实际 stored R/p 的两类材料运输/有限步功 |
| C7-06 | A01F 固定 u0=.25、ff=-6，h=1/1024 与 1/2048，最多7/8 s | 同轨迹首次卸载后真实正载重入即停止；无 reset/调控制/延长窗口 |

C7-06 独立递推为 `N=max(0,64*(3/64-2*s))`、`u+=h*(-6+2*N)/6`、`s+=h*u`。原有限面的内部域逐步检查，不能脱离有限几何后继续用无限平面 oracle。各首步手推常数先同独立 oracle 比较，再核产品真实响应。

同次 receipt 保存 full N input/contact、K drive、P/c/C/graph、完整 full/reduced mechanics、原行反力及功。每个 site 用原 vertex IDs/weights + world skin offset 重建两侧表面；参考点用真实 COM，速度为实际 Jv + omega×(共同点−COM)。原 entry R/p 逆映射后，用实际 candidate R+/p+运输两侧表面及共同作用点，分别重算两种 work。KE 用独立全 M 核对；entry power、实际有限步 work、KE 变化分列，不强加能量守恒。

状态/力学门保原 atol=5e-6、rtol=5e-5；40步 atol=2e-4；约束残差原2e-5。1e-10 m 仅材料记录构造一致性，1e-12 J 仅逐项同记录点积再合成的一致性；都不是提升物理精度或能量通过门。R 不重正交化，既有5e-5存储域由产品保持。PD 延迟0/1、低cap保原实际 bound solve；接触不混入 held drive 或随 drive 延迟。

## 混合/事务与保存

W4 keys=(503,91,1207,6)：普通 C6 2 kg 竖直 P 盒、A03 chain、A05 R/P、2 kg 自由球 z=2/gz=-8。W1 同输入对照、逆序 world 登记、重复 local IDs；40 步后恢复 tick19 并准确重放余21步，保完整 N/K/FK/held/history/receipt。形状局部 ID/坐标与 world 身份分离。

某 affine 容量0分别从两个 active bank 拒绝，完整 N/K/FK/**实际 energy bank**/held/history/tick 与旧 receipt 保留，邻 world 仍推进；新完整帽 plan 按保留状态重试。零力几何站点也计帽。映射强核 affine 原7/8不变、9为容量，普通7不变；不会用数字标签充当物理通过。

额外边界：全批 candidate 帽、空绑定、图/group/world/law/采样/动动配对先拒绝；实际第二 scratch 分配后故障、实际上传/候选读回/receipt 构造后的晚故障，各两 bank；同tick restore 的 epoch、跨batch plan、chain/flatten checkpoint、只读返回/读观察不推进、close幂等与从公开 checkpoint 重建。未重复原 C5/C6 全套故障矩阵。

窗口全部完整 gzip，保原输入、帧、实际诊断、oracle 与计数，不用 hash 代替输出。失败时也保最后已发布状态/receipt，consumer 在 step 前先建 attempt 行，记录实际完成检查数、原异常和 retained 状态。所有 counter 按实际 backend 保存；不编造 CUDA字节或 launch。

## 父执行入口（作者未执行）

父先短组，再连续/混合/事务，按既有 fresh guard 严格串行。两个文件的完整入口如下；`--basetemp` 必须本批新绝对路径。

```powershell
python -m pytest tests/scale/test_resident_affine_contacts.py tests/scale/test_resident_affine_contact_transactions.py --tyx-mode interpret --require-tyx --basetemp <new-absolute-interpret-dir> -q
python -m pytest tests/scale/test_resident_affine_contacts.py tests/scale/test_resident_affine_contact_transactions.py --tyx-mode cuda --require-tyx --basetemp <new-absolute-cuda-dir> -q
```

测试实际 collection 数以父收据为准。独立 reference 必须取父当前已接受的 C5 文件，不能在落后 child 用缺依赖替代有效 RED；此作者未 merge 父。

源码外 consumer 复制至 checkout 外新目录，cwd 必须为该脚本目录；从实际本地 wheel 安装到新 wheel-site，命令使用绝对路径：

```powershell
python -I -B <outside>/resident_affine_contacts.py --mode cuda --output-dir <fresh-absolute-output> --wheel-site <absolute-installed-site> --expected-package-root <absolute-site>/phymeo --expected-wheel <absolute-actual-wheel.whl> --checkout <absolute-parent-checkout>
```

interpret 同命令仅换 mode 和新输出目录。严格核 `direct_url.json` 的实际本地 wheel 路径、可用安装器hash、实际wheel SHA256、所有安装产品成员的完整字节、所有已加载 Phymeo 模块的 `__file__/spec.origin/__path__`；不靠仅包版本或同名路径判断来源。`-I -B`、外部 cwd、新输出不可覆盖为前置条件。

consumer 四世界40步+21步重放，再单次局部容量拒绝及一次恢复；仅公开 API 推进，所有负例有真实错误记录。完整 `resident-affine-contacts.trace.json.gz` 与短 `summary.json` 同时保存。若只通过此烟测，不等于完整 C7/新 Newton 对照或性能验收；未声明速度目标。

## 2026-11-08 父执行与正式限域接受

最终产品 **876ae848f313ed76bbff1bb6533a69f7041bb50d**。本片已完成原冻结功能、工程门和本机验收；没有远程CI或跨引擎速度结论。全部项目执行由父fresh guard串行完成，原作者仅NEW源码/测试/消费者与保存审计脚本作者，现均释放。

| 证据 | 实际结果 |
| --- | --- |
| 首公开节点368/369 | interpret 1 / CUDA 1通过；各后续完整组只排除此已绿节点 |
| 380/381两文件剩余完整矩阵 | interpret 49 passed/145.94s；CUDA 49 passed/224.49s；每backend联合50，未以deselection掩盖未测节点 |
| 原接口受影响检查387 | CUDA 3 passed/9.33s：无接触affine、PD拒绝后正常solve、接触world与未绑定affine/scalar共存 |
| 原C6 W1/W4接缝371 | 修复真实group遮蔽后1 passed/14.96s，370原失败保留 |
| 摘要口径382→383/384 | A02原RED为generic0而nested完整max=6.811959707420101e-8；仅generic摘要修正，各backend单节点通过 |
| 388同次保存审计 | 90主gzip完整payload、两个独立A02补件、382原失败部分记录；固定身份精确membership通过 |
| 389–392新wheel | 构建/离线新target安装与源码外interpret/CUDA均RAW0 |
| 393包与消费者审计 | 423产品文件源码/快照/archive/安装字节全同；两backend全21恢复重放的state/geometry/receipt/diagnostic均同 |

保存审计逐字段保留原N/K、完整输入/graph/P/c/C、owner/material、全部site及能量/功记录。W1/W4与tick19后21步完整重放通过。93个归档包含4782次receipt出现和3052次site出现（含重复保存引用，不冒唯一物理步数）；13个无保存调用的测试实例每backend仍只按真实pytest证据计。主集合固定为368/369+380/381，未将370/371或387混入。

原完整矩阵在8fe797c执行；876ae848只改generic摘要与对应断言，真实full/K门、数值推进和原nested诊断未改。90主记录中3906处旧generic摘要与nested max的字面差异完整保留；两A02补件与各自原记录全字段比较，唯一差异正是receipt.diagnostics.equation_residual，且等于原nested max。不能将新摘要倒灌旧输出。新wheel使用最终876ae848，完整受载与恢复实测补齐实际发布路径。

每backend源码外四混合world完成40主步+21恢复步+局部容量拒绝/重试，共63 batch、251 accepted world-steps和1 rejected world-step；原图身份拒绝、owned只读记录、所有owned device buffers关闭释放均通过。284实际加载Phymeo模块来自新安装target。真实CUDA统计1707 submitted kernels、1036 synchronizations；这是功能观察，未作吞吐比较。

新wheel 1,245,222 bytes，SHA256 **799dde87987d0f6dfe2df5bf66f9a6599f573b25f576763978f3e0818812a6c1**。完整证据在 artifacts/p65-generic-collision-20261107-01；索引为 c7-final-tested-source-v1.json、c7-saved-audit-v1.json、c7-wheel-parity-v1.json。7组最终测试before/after及实际snapshot同源已核，最终current产品复用393的423成员核，无新物理重跑。

原359、370、382功能失败，374/375首style失败，360预执行guard错误与361/362无效只读scope均按原始日志保留。最终专项Ruff/format已过，未放宽预算/更换fixture。原robot只读审阅指出的遮蔽问题已闭环。

**限域仍是固定根ordinary/affine树对有限static S/C/H，mu=0、discrete_entry、HOST窄相/法向律+TYX完整受力响应。** 动态双方、浮根设备树、持续摩擦/CCD与完整设备窄相仍需后续实现；C7接受不等于整个P6.5完成。下一片从[动态双侧提案](../development/p65-resident-dynamic-contact-next-proposal.md)冻结多案例与薄接口。
