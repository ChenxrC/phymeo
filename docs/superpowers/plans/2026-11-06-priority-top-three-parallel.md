# 按用户优先级补齐前三方向：子 agent 实施计划

> **For agentic workers:** 实施时使用 superpowers:subagent-driven-development；沿用本文三个原作者。所有项目执行归父唯一串行槽，不能按技能默认重开助手或并行测试。复选框只记录实际完成。

**Goal:** 依次优先补场景与机器人、碰撞检测、GPU与多场景的通用能力，以多案例证明可复用性。
**Architecture:** 三个独立新增模块隔离编码，父维护编译、碰撞分派、设备入口和原子发布等旧共享接口。先核公开缺口与独立案例，再实现、逐线验收和联合使用。
**Tech Stack:** Python/NumPy、既有 TYX interpret/CUDA、现有精度/材料合同，Newton 1.7.0 为参照。
**Spec:** [当前较大功能差异](../../development/phymeo-newton-functional-overview-2026-11-06.md)及下述用户优先级。基线 clean `2c15bde69617d1202a43d3f979aac0a122dc1e77`，产品 `8c4a7db`。
**状态:** 2026-11-06 本文 A/B/C 首片已实施、串行验证并本地集成，产品 `dddb612`，源码外 wheel 通过。C0 实测触发 C1，18组配对只在选择snapshot显示重复收益，旧默认均保持。[实际结果与边界](../../benchmarks/p65-top-three-batch5.md)。后续 A2/B2/C2 等未在本批启动。

## 1. 用户顺序与工作方式

总顺序为：

1. 场景与机器人。
2. 碰撞检测。
3. GPU 与多场景。
4. 可微仿真与训练工具。
5. 传感与交互。
6. 跨域求解编排。
7. 布与体软体。
8. 杆缆、MPM/流体。

按上下文将原文“可谓仿真”“快于求解编排”理解为第4、6项。该顺序覆盖上一份总览的 P0/P1 建议。持续摩擦是相关场景的基础能力，按具体阻塞归入第1/2项，不再单独排在整个路线之前；不以旧100g抓持失败或软布递布牵引前三线全部工作。

前三项可并行做无依赖编码；资源或共享接口发生冲突时，优先 A→B→C。某线等待验证时其他线仍可编码，不要求第一大模块穷尽后才开始第二模块。其余五方向本轮保留队列，不启动。

| 工作线/原作者 | 首批交付 | 难度 | 实现工作树 |
|---|---|---|---|
| A 场景与机器人：newton_robot_reference | 同一 HOST world 的多固定根机器人共同作用于自由物体，统一命令/观测/事务 | 高，4/5 | p65-robot-substep/phymeo |
| B 碰撞检测：newton_rr_reference | 胶囊自身转弧 × 固定或平移闭网格，补真实分派空洞 | 中高，4/5 | p65-tyx-rays/phymeo |
| C GPU与多场景：newton_rd_reference | 选择世界观测的成本实测；满足门槛后做可复用批量读回计划 | 初测中2/5；实现中高3/5 | p65-world-scale/phymeo |
| 父 | 旧共享接缝、队列、必要审阅、集成示例与源码外 wheel | 集成高 | 当前集成树 |

工作树均位于 `C:/Users/38347/.codex/worktrees/`。启动编码前再次核 clean/WIP，正常 fast-forward 到父批准基线；不能 reset。三作者当前旧提交均已被父接受，不重复实施上一批。

## 2. A：多机器人场景

### 依据与首批范围

构建器已可重复 add_urdf，但 [build_robot_scene](../../../src/phymeo/runtime/robot_scene.py) 目前明确要求一台固定根机器人、一个 world、HOST，动作集合也只收一个注册 ID。浮根动力学和多行 mimic 已存在，不重新开发。

首批支持一 world 内两个及以上普通或已支持 mimic 固定根机器人、多个已有刚体和有限台。共享自由体只积分一次。机器人直接互碰须显式 pair(A,B,None)；不能暗中排除。直接互碰、浮根、柔体、CUDA及多机器人新历史摩擦留给后续节点，并在入口明确拒绝不支持组合。该首片不是全部多机器人能力完成。

### 文件和接口

作者新增：
- `src/phymeo/runtime/_robot_participant.py`：每台资产、树、独立命令、状态和几何键映射。
- `src/phymeo/runtime/multi_robot_scene.py`：同入态组装、一次推进、完整候选与公共观测。
- `tests/test_multi_robot_scene.py`、`examples/multi_robot_push.py`、`docs/benchmarks/p65-multi-robot-scene.md`。

沿用 `ContactSceneBuilder.add_urdf(...)`、`build()`、`scene.step(dt, actions={id: RobotCommand(...)}, external=...)`、`state()`、`observe_scene()`。观测按 (world,id) 分属，全部物理 link 保留原始地址；不得用一台虚拟大机器人掩盖 owner。

父独占 `runtime/contact_scene.py`、`robot_scene.py` 编译路由，`_robot_scene_contacts.py` 全场载荷拆分，`contact_events.py` 多 owner 事件及根导出。刚体–刚体载荷必须全场采样一次，机器人–刚体载荷逐独立配对一次；不得将原单机器人 sampler 完整调用 N 遍从而重复桌面力。

### 多案例与步骤

- [x] 先 author 独立案例与首个公开缺口测试，父保存有效 RED；再实现上述两个模块及父薄路由。
- [x] **A01 对称推载：** 两个2kg滑轨、1kg自由球，零重力/PD，实际相反法向各2N。核球加速度0、滑轨沿同一世界轴为−1/+1 m/s²；法向由真实几何与既有材料律产生，不能直接填载荷。
- [x] **A02 非对称与协变：** 2/4kg滑轨给球+3/−1N，球+2、滑轨−1.5/+0.25 m/s²；登记换序、整体刚体变换仍同物理结果。
- [x] **A03 不重复计力：** 两机器人远离，球–有限台2N只出现一次；台边外留出无伪载荷。
- [x] **A04 拓扑/持续窗口：** 普通 URDF + 多行 mimic，共同推载32步并撤退8步，记录两侧反力、各轴冲量、分离归零。公开示例复用该入口。
- [x] **A05/A06 事务与隔离：** 第二机器人非法命令/越限时两树、刚体、时间、loads/events均不发布；合法重试等同重建。重复局部 shape ID 不冲突，返回快照不能污染内部；只跑受改动影响的旧单机器人兼容组。
- [x] 每项阈值及真实几何先写测试；父串行验证、作者小提交、父正常集成。首步解析控制与长窗物理分开判定。

后续 A2：双机器人直接接触及完整双侧反力（高，依赖双方几何/材料见证）；A3：浮根进入同一事务（高，复用全质量响应和 COM 语义）。球铰/D6/更多闭环随后独立排期，不能混进本首片。

## 3. B：胶囊转动与闭网格

### 依据、接口与范围

[collision_pipeline.py](../../../src/phymeo/geometry/collision_pipeline.py) 的 mesh–round 分支明确将 round_rotates 拒为 unsupported。反向的网格转动/胶囊朝向固定已经有实现；通用 convex-support 路径不接受 mesh。本任务是新增空洞，不重跑旧旋转组合。

作者新增：
- `src/phymeo/geometry/_rotating_capsule_mesh.py`；
- `tests/collision/rotating_capsule_mesh_cases.py`、`test_rotating_capsule_mesh.py`、`test_tyx_rotating_capsule_mesh.py`；
- `docs/benchmarks/p65-rotating-capsule-mesh.md`。

接口 `RotatingCapsuleMeshQuery(prepared_mesh, capsule, mesh_path, capsule_path, mesh_first).result(moving)` 沿用原五项结果和 MeshRoundTrace。复用原闭体分类、有限轴段距离、Rodrigues 包围与有界首触搜索；父只接旧 mesh–round 分派及必要导出。首片网格无转动，胶囊单轴转弧，枢轴可仿射移动。不得以无限直线代替有限杆身，也不得用凸包代替凹闭网格。

### 多案例与步骤

- [x] 首先冻结输入表和独立 oracle，再保公共入口 unsupported 的有效 RED，随后实现新增模块。
- [x] **B01 杆身首触：** 固定盒网格 [1,2]×[−1/8,1/8]×[−1/4,1/4]；胶囊轴 (0,3/2,−1)→(0,3/2,1)，r=1/8，枢轴0、轴+z、总角−π。原端点ID41/93。首触 t=acos(1/6)/abs(实际存储angle)，杆身参数全区间[3/8,5/8]，两端球皮肤净距至少5/8；首次点x=√35/4。此例轴方向保持z，只验证转弧位置，不能独自证明朝向变化。
- [x] **B02/B03 分离与凹性：** 上例轨道半径9/4全程分离；真凹U槽绕行不产生凸包假碰撞。输入/oracle在执行前落 cases 文件，不按运行输出选夹区。
- [x] **B04 真朝向变化：** 轴端点差的单位方向在入末态不平行；另含斜轴、移动枢轴、端帽命中。每例都由独立有限段–面/边距离给首触参考，不能由新 producer 自验。
- [x] **B05/B06 留出：** 初始包含/开离、孤立擦触、预算耗尽、非法闭体；反序、镜像、整体变换、重复局部ID和跨world。预算未决与确证分离不可混淆。
- [x] HOST 验最早括区、实际占用样本、双侧材料权重/offset重建；interpret/CUDA验真实设备宽相与公开结果一致。检测不冒称冲量响应或稳定摩擦流形。
- [x] 父串行执行和父 hook 集成，报告全部支持范围及首错，作者按模块提交。

B01 独立推导：令α=abs(angle)t，轴点=(3sinα/2,3cosα/2,−1+2u)。上面首次距离r需cosα=1/6；此时x在盒内，z面内给u区间。案例代码需单独核盒面朝向、闭合和实际储存输入，不借该手推免测试。

后续 B2：双方独立转轴（高）；B3：逐面 TYX 保守候选（中高，依赖本批 HOST oracle，可交C线）。Newton当前姿态的mesh–convex可作离散参照，其 speculative contacts 不充当TOI真值。

## 4. C：按世界选择观测的 GPU 读回

### 依据与成本门

[TyxWorldBatch.snapshot](../../../src/phymeo/backends/tyx_world_batch.py) 子集逐 key 启核/同步；[read_kinematics](../../../src/phymeo/backends/_world_kinematics.py) 下载全部世界后HOST选行，状态及ticks为52W字节。源码证明冗余，不证明速度收益。此前48→12次提交没有稳定墙钟改善，故不直接加新内核。

作者先新增 `scripts/profile_selected_world_readback.py`、`tests/scale/test_selected_world_readback.py`、`docs/benchmarks/p65-selected-world-readback.md`。若下面门槛通过，再新增 `backends/_world_readback_plan.py` 与 `_tyx_world_readback_kernels.py`。父独占 `tyx_world_batch.py` 的方法/计数、`_world_kinematics.py` 旧适配入口；不改物理步进、机器人或碰撞算法。

### 步骤与验收

- [x] 预先固定 W=1/32/128；K=1/8/全量（去重复且只用K≤W），自由球与延迟滑轨混合。每接口分别测冷启动、5预热窗口和20个正式32步窗口，拆推进、同步、下载与HOST构造。snapshot 与六量 kinematics 不是同一任务，不能互相比速度。
- [x] 父唯一串行槽运行最小基线，保原物理资格与全部原始观测，不重跑旧尺度矩阵。若子集下载或逐 key 提交占该观测任务墙钟≥15%，进入下一步；否则记录无充分优化依据并结束此节点，不制造加核任务。
- [x] 通过门槛后实现 `batch.prepare_readback(keys, *, kind="kinematics"|"snapshot")` 返回与实例/布局绑定的计划；`plan.read()` 从当次接受缓冲读拥有的HOST记录；`plan.close()`释放资源。固定选择索引一次上传、两套接受状态轮换均正确绑定、预分配gather输出、每次至多一次显式同步。kinematics下载28K字节，snapshot下载52K字节，必要设备状态标志额外单列，不藏计数。
- [x] 非连续键/请求顺序、空选择零设备动作、重复/未知键、跨实例计划、关闭后拒绝、局部拒步/恢复、读回故障重试、旧帧所有权均有测试；32暖步零新增设备分配/静态索引上传。持有旧计划跨step不得读旧active缓冲。
- [x] 若实现，选择发生瓶颈的W32/W128子集与W1退化控制，三轮独立AB/BA/AB比较。每轮用相同物理输入和完整同种输出；记录p50/p95、字节、launch/sync及推进+观测总成本。无重复可见端到端收益则不切默认；功能支持与性能结果分别签收。

这里只优化公开观测，不把拥有HOST记录称作全设备驻留。后续 C2：合并材料/逐站力读取（中）；C3：已有原语设备状态直接供宽相（高，依赖设备状态版本和碰撞候选合同）。C3再与B3整合，不能提前相互依赖。

## 5. 公共审阅、执行与合入

主要风险分别钉在A05/A06、B04/B05/B06与C功能项：重复载荷、跨owner碰撞键、真实方向变化、未知误判分离、双缓冲/旧帧污染。默认能力与首片显式限制均必须出现在公开入口和错误中。

- [x] 父核三条新路径不冲突、旧共享接口唯一owner，工作树更新且WIP可解释。本轮计划不授agent项目执行。
- [x] 作者先提交新测试；父对指定新节点保有效首次失败，随后作者并行写限定新模块，父接窄共享hook。
- [x] 父按就绪队列A→B→C执行；每次fresh entrypoint-aware guard。所有import/pytest/静态/构建/profile仍严格串行。遇外来真实执行等待自然结束，不kill。
- [x] 功能合入后，复用三个原作者做不同模块的边界审阅；不重启旧builder/reviewer，不另派重复实现。
- [x] 父完成有新增价值的组合示例与一次源码外wheel。共享回归只跑实际受影响组；此前同源green复用，不以全仓重跑代替定位。
- [x] 每线报告新增公共能力、未覆盖边界、实际数值与错误、性能条件。仅文档/手推不得勾功能完成。每个逻辑交付本地小commit，禁止mainmerge/发包/远程CI。

**止损与完成：** 同一节点连续两轮没有新可运行能力或更小可复现失败时，收束当前结论、调整接口/算法/范围，不增长证明、放宽断言或反复重跑。前三首片分别验收，不等同三个大模块已全部追平Newton。

## 6. 调研与原有证据

当前父源码的单机器人限制、mesh–round旋转拒绝、选择观测路径是本计划的直接依据。Newton提供多world模型和关节表示，但异构求解器覆盖仍有限；参照它的模块边界而不照搬后端假设：[Worlds](https://newton-physics.github.io/newton/1.7.0/concepts/worlds.html) · [Articulations](https://newton-physics.github.io/newton/1.7.0/concepts/articulations.html)。

[第四批已验结果](../../benchmarks/p65-scene-tangent-batch4.md)、[多行mimic](../../benchmarks/p65-urdf-transmission-graphs.md)、[设备阶段成本](../../benchmarks/p65-multiface-phase-cost.md)均保留。无需再开发已存在的相机、mimic图、单机器人交互或固定hull复用。

## 7. 串行验证入口（本批已执行）

父使用现有已固定环境、child/src、禁止插件自动加载和字节码；每个命令给全新绝对 basetemp，并绑定此次源快照。预期完整组为全部通过；首次 RED 仅取预声明的新功能节点，不能把环境/导入路径故障冒作产品缺口。

| 组 | 命令主体 | 成功证据 |
|---|---|---|
| A | `python -m pytest tests/test_multi_robot_scene.py -q -x -p no:cacheprovider --basetemp <fresh-absolute>` | A01–A06全通过、真实反力/事务记录 |
| B HOST | `python -m pytest tests/collision/test_rotating_capsule_mesh.py -q -x -p no:cacheprovider --basetemp <fresh-absolute>` | B01–B06的独立真值与材料重建 |
| B TYX | 相同pytest形式选择 `tests/collision/test_tyx_rotating_capsule_mesh.py`，复用现有 `tyx_mode` fixture分两次interpret/CUDA | 实际完成launch与完整公开结果一致 |
| C | `python -m pytest tests/scale/test_selected_world_readback.py -q -x -p no:cacheprovider --basetemp <fresh-absolute>` | 选择/所有权/轮换/故障各门独立通过 |

C新脚本命令契约：`python scripts/profile_selected_world_readback.py --backend cuda --worlds <1|32|128> --select <1|8|all> --kind <kinematics|snapshot> --expected-head <full-head> --out <fresh-absolute>`。初测不改变旧实现；通过15%门槛后，同脚本显式增加 `--compare-plan` 才运行新旧配对。不支持组合在启动前拒绝。完整命令、环境、原退出码、无损公开记录和实测计数入本轮新artifact；不会覆盖第四批证据。
