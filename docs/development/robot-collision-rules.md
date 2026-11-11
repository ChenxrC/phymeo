# 机器人碰撞规则补齐方案

2026-11-11，设计节点，尚未实现。适用于独立实验分支
`feature/wh/phymeo-physolve-integration`。审查基线 PHYMEO `50a50bfd`、
codedev `6bb483ee`、TYX `7272d3df`、P2PE `1edba631`。
本方案补齐 PHYMEO 自身的模型声明和原生执行适配，不复制 Newton 源码、测试或默认策略。
六个求解 CUBIN、现有数值容差及原环境保持不变。

## 1. 问题与目标

[FANUC 审计](physolve-runtime.md#fanuc-碰撞模型审计b5)表明，原资产底座和第一节
碰撞代理存在约 67 mm 重叠；PHYMEO 原生导入没有增加几何偏移。固定根并入 world 后，
仅凭运行期 body 父子关系不足以表达“机器人连接处的两个形状不生成接触”。
这不意味着所有邻接碰撞都无效，也不意味着需要改写求解算法。

目标是让用户明确声明形状对排除，在编译时得到可审计、可拒绝的结果，并在选定后端
准确执行。FANUC 首个生产候选仅显式排除已审计的底座—第一节碰撞形状对；
原始未过滤模型作为对照保留，其他自碰撞、地面及障碍物接触保持。
过滤后的运动恢复不能代替全部惯性、驱动和机械几何的物理验收。

## 2. 已有结构与实际缺口

| 已有机制 | 复用边界 / 缺口 |
| --- | --- |
| [CollisionKey / CollisionModel.excluded](../../src/phymeo/constraints/collision_shapes.py) | 已有 world/body/shape 身份、规范化形状对与引用校验；不建立第二套排除表协议 |
| [LinkCollision / RobotAsset](../../src/phymeo/builders/_robot_asset.py) | 已保留 link 和形状身份；原生仅编译路径不能假装一定存在可 lowering 的 RobotAsset |
| [collision_pipeline](../../src/phymeo/geometry/collision_pipeline.py) | 已有不同 world、同刚体、mask、excluded 等诊断原因和编译候选；保留其原有职责 |
| [scene_compiler](../../src/phymeo/builders/scene_compiler.py) | 已有 world 身份重映射；当前 M1 不支持 self-collision，补声明不等于开放求解能力 |
| [mujoco_compiler](../../src/phymeo/builders/mujoco_compiler.py) | 保留完整原生模型与 geom 映射；固定根融合前的 URDF link/collision 来源和规则到 native geom 的映射仍需补齐 |
| codedev 的 physolve_tyx 前端 io / collision_driver | 已有完整和压缩候选对表，适合冷准备接入；目前没有本方案的显式输入合同 |

`CollisionPolicy` 当前控制容量、margin 等检测参数，不把机器人连接规则塞入该类型。
PHYMEO 的选择仍是“精简物理描述 + 可组合算子 + 专用求解器”，不新增通用碰撞管理器、
全局 context 或完整 SceneDeclaration 到 MjSpec 导出器。

两条路线必须分别资格化：PHYMEO resident 场景编译与完整 native MJB → PhySolve。
详见[现有统一场景边界](unified-scene-handoff.md)。native-only mesh 可以被原生后端执行，
不因此声称 resident lowering 支持 mesh/self-collision。

## 3. 默认语义与首批合同

1. 未提供新规则时，各路径的行为、模型参数、候选集和 capability 不变。
   README 中 RobotContactSystem 的显式形状排除约定保留；不将它扩大成所有后端的统一默认。
2. 首批只接受显式形状对 exclude。按实例限定来源身份，引用必须唯一；自配对、缺失、
   歧义、跨 world 引用和不支持的规则明确失败。pair 顺序规范化，重复规则按既有合同拒绝。
3. 不默认排除所有父子 link，也不默认关闭机器人自碰撞。不按 FANUC 文件名硬编码。
   后续可提供显式开启的邻接规则展开器，但须单独审查 joint 类型、固定连接、每 link
   多形状及保留接触的例外；它只生成普通形状对声明，不成为新的热路径机制。
4. 首批不提供强制 allow。原生显式 contact pair 与新增 exclude 冲突时拒绝编译，
   不静默覆盖 authored pair 或丢失其摩擦、维度、margin 等参数。
5. 保持原生源的 mask、body exclude 和显式 pair 语义。PHYMEO 的双向 mask 是 AND，
   MuJoCo 的 contype/conaffinity 是 OR；不能直接复制位域并声称语义相同。
   本批增加排除关系，不做全局 mask 语义统一；已有不等价 lowering 仍拒绝。
6. contact response 的排除不自动解释成射线不可见或所有 sensor 不可见。
   原生 collision sensor 当前不在 PhySolve 支持域；未来接入须明确独立的查询语义。

## 4. 身份与编译方式

在已有 builder/compiler 层保留来源记录：资产内容身份、实例、原始 link、collision
名称或该 link 内的稳定序号。序号随源变更可失效，必须与内容身份一起校验，不能跨模型
版本默认为稳定 ID。最终解析为现有 CollisionKey 或 native geom ID 的冻结映射。

原生来源记录须在固定根融合前取得。对 URDF 的冷解析只补来源关系，不另做几何转换、
动力学编译或 runtime 模型。采用唯一内部标识追踪到 MjSpec/compiled geom，并验证
命名处理没有改变几何和动力学字段；保留 authored name。若导入器无法可靠追踪无名
geom，就拒绝该引用并报告原因，不能按 mesh 文件名、包围盒或输出数组顺序猜配。
MJCF include/default 等来源暂不能恢复时，也须公开资格边界。

`body_id=0` 不是 robot base 的身份：同一 world 还可能有地面、桌面和障碍物。
禁止用 world—link 的 body-wide exclude 代替 base shape—link shape 排除。
不存在 resident shape_key 的 native geom 保留原生映射身份，不伪造 lowering 成功。

编译产物记录：源身份、规范化规则、解析结果、后端语义/版本、有效排除和失败原因。
仅候选编译拥有可变工作区，完成校验才冻结发布；调用方修改输入不能影响已提交实例。
规则变化属于模型重编译，不在 step/forward/reset 时解析或临时改表。

## 5. 各仓库改动与执行边界

| 仓库 | 实施内容 | 明确边界 |
| --- | --- | --- |
| PHYMEO | 在已有模型编译入口接收显式规则；补来源映射、冲突预检、诊断和 fingerprint；resident 复用 excluded | 不新增大 context，不开放未资格 self-collision，不改物理参数 |
| codedev | 在 physolve_tyx 前端 put_model 冷准备入口增加明确的 geom-pair 排除输入及版本能力；统一生成候选和派生表 | 碰撞规则解释留在 PHYMEO；保持 Host/CUBIN ABI，六 CUBIN 不变 |
| TYX | 本批无预定修改；若公共数组/编译/运行时出现可复现缺口，再独立修复 | 不承载机器人碰撞政策，不引入 Warp |
| P2PE | PHYMEO direct 资格通过后，将 Stage 的明确规则传入编译候选，披露能力与身份 | 不复制编译器；仍由现有 Runtime 持有模型、状态、solver 与资源；暂不新增 C ABI |

codedev 当前源码位置为 `mjwKernelTest/cudaHostPython/physolve_tyx/_src/io.py`：
`nxn_pairid_contact` 用 -2 表示排除、-1 表示动态接触、非负数表示显式 contact pair。
显式 pair 当前在基础过滤之后写入。新增排除应先验证冲突，再在生成 `nxn_include`
及派生结构之前施加；完整/压缩 pair 表、geom pair 表、类型计数、容量/支持检查和
NXN/SAP 分支必须来自同一结果。禁止 runtime 外挂 monkey patch 或只改一个 device 表。
`collision_driver.py` 的 SAP 分支消费完整表；不能只验证 NXN 压缩表。

这些候选准备位于求解 CUBIN 之前，源码支持“不改 CUBIN”的实现方向；实际有效性
仍需 CUDA 测试证明，不能仅凭字段名认定已经打通。保留 native source MJB 与应用规则
后的有效执行身份，避免对外声称两者的候选集仍相同。

CPU MuJoCo 的 body exclude 不能普遍精确表达任意 geom-pair exclude，有限 mask 位域
也不是通用编码。CPU 路线仅在能证明等价时采用 lowering，否则显式拒绝此规则组合。
B5 的定向 mask 可保留为已审计场景的独立参照，不能推广成所有机器人的产品方案。
不加入全局 Python contact callback，不改变六 CUBIN 来绕过前端缺口。

## 6. 生命周期、复杂度和缓存

模型/runtime 持有最终候选表及 device buffers；规则适配只做冷准备或借用。
双 bank 使用同一不可变模型规则，各自 state 绑定不变。重编译创建新 owner，失败保留
旧 runtime 和待清理 candidate；P2PE revision/stale view/epoch gate 规则照旧。
Graph 捕获前完成表上传和容量验证，固定重放不改指针；规则变更使旧 Graph 失效并按
已有释放/失败重试流程重建，不能以 direct 回退冒充 Graph 通过。

设每个同构模板 G 个 geom、E 个声明排除。身份索引与引用解析目标 O(G+E)，
规范化排序另需 O(E log E)，不逐 world 重建。现有 native 前端全 pair 枚举本来为
O(G²) 冷成本，复用其 pass，禁止对 E 个
排除重复扫描全 pair 导致 O(EG²)。不声称本批已经消除既有平方级存储。
热路径增加的规则解析、CPU→GPU 拷贝、同步和分配次数均应为零。
fingerprint 纳入源内容、规则、映射/layout、后端版本与编译选项；缓存不能跨语义复用。

## 7. 分批实施与验收

| 批次 | 交付 | 必须获得的证据 |
| --- | --- | --- |
| C0 身份与合同 | 来源映射、显式规则预检、冻结编译记录 | 无规则模型字段/候选不变；有名/无名、多 geom、多实例、固定根融合、错误引用/冲突测试 |
| C1 原生 direct | codedev 冷准备接口 + PHYMEO 适配 | 全 pair eligibility 只有声明项变化；NXN/SAP 派生一致；FANUC N=1/8/64 step/forward/full/masked reset |
| C2 物理与兼容 | 外部接触/非邻接自碰撞反例及来源资格说明 | GPU 上保留 link_1 对地面/障碍物接触与合理响应；非邻接应碰撞对仍产生接触；Panda 原失败保留并独立追踪 |
| C3 生命周期与接入 | 固定 Graph、重编译和 P2PE Stage 适配 | 实际 cuda_graph、GC/双 bank/失败重试、stale view、双 handle、原环境/无规则回归及安装身份 |

复用正式测试体系：`tests/test_mujoco_compiler.py`、`tests/test_scene_compiler.py`、
`tests/collision/test_collision_pipeline.py`、`tests/test_fanuc_collision_model.py`、
`tests/test_physolve_models.py` 和 `tests/test_physolve_runtime.py`。
codedev 在原前端测试入口登记候选表合同；P2PE 继续使用已有 Stage/C ABI/Graph 门禁。
每个真实 GPU 进程记录实际模块路径，禁止安装/导入 Newton、Warp、MuJoCo-Warp，
并比对六个 CUBIN 与 Host 身份。若仅 Python 前端变化，Host 二进制也保持原散列。

数值沿用原测试步长、精度、驱动和容差；新增接触场景在候选对照前固定 SI/frame/
方向、参考点及判据，不要求不同约束顺序 bit-exact，也不放宽已有 1e-6 门限。
CPU 语义参照、CUDA direct、Graph、完整 C ABI、GUI、安装发行分别报告，互不代替。
原始 FANUC 静止失败和 Panda 数值失败不删除、不改 xfail，也不因新增规则宣称已修复。

## 8. 本节点结论与后续决策

设计选择为：保留各路径默认语义，优先补原生来源身份和显式 shape-pair 排除，
沿既有编译/候选表接入，不移植 Newton 框架。邻接自动展开是后续可选需求，
不作为解决当前 FANUC 问题的必要条件。

本节点只新增方案与索引记录，没有修改生产代码、模型、环境或载荷。
下一实施节点是 C0；进入 C1 前必须证明来源映射可靠、无规则行为保持且前端所有
候选分支可一致降低规则。无法表达的组合按 capability 拒绝，不能静默缩减验收范围。

文档验证：P2PE 文档/政策单测 18/18，通过 change-policy；两仓 diff 空白检查通过。
PHYMEO 全库文档检查仍失败于 479 条缺失目标，日志与 B5 修改前基线的 SHA256 相同，
没有新增缺失目标。证据在独立实验根 `evidence/collision-plan-*.log`。
已按默认兼容、来源身份、显式 pair 优先级、sensor 独立语义、冷准备复杂度和资源所有权
完成自审及第二轮复核；均由同一执行者完成，不记作独立审查人签字。
本批未运行物理、GPU、Graph、C ABI 或 GUI 测试，不产生新的运行时资格结论。
