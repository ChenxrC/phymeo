# 2026-11-13 功能验收矩阵：D5-A / D4-N / R3-N / G2a

本页是本轮执行中的**状态截点**，不是统一包验收报告。依据父 `/root`
确认的实际结果、[当前功能计划](../superpowers/plans/2026-11-13-feature-completion-until-0730.md)、
[既有全景](phymeo-newton-panorama-2026-11-12.md)及当时已落盘的消费者源码。
读取时父 HEAD 为 `80204636`，仍有明确 WIP 和运行中任务；这些 WIP
不能作为最终包版本。本页作者仅文本/file IO/Git，没有执行项目或重审旧保存档案。
之后新结果由父按实际收据更新，不把 PENDING 自动改为通过。

状态分开记录：**实现**说明入口存在；**源码验证**说明指定来源和后端已验；
**安装验证**要求新包、源码外调用及模块来源；**连续物理**要求完整预登记轨迹；
**性能**要求独立计时/资源口径与公平基线。这五层不能互相代替，
也不按测试数、world 数或清单数给总体完成百分比。

## 一、当前状态和能交付的功能

| 节点 | 用户可调用的功能 / 路径 | 源码状态 | 安装状态 | 连续物理状态 | 性能状态 |
| --- | --- | --- | --- | --- | --- |
| **D5-A 公共声明接线** | resident 路径的 `ParticipantSceneDeclaration(contact_response="deformable_normal")` + `ParticipantDeformable` → `compile_participant_scene` → 唯一 D3 双 owner → step/query/reset/restore | **指定源码矩阵已通过**，父确认；作者旧报告的 TESTS_PENDING 不覆盖父结果 | **PENDING**，新消费者已编写，未把源码通过当包外通过 | 两类不同短窗：W2 资产球和 W1 程序资产胶囊，真实变形/双方受力/恢复；**不等于 D4 的持续支承** | 未做性能比较；仍含 HOST 载荷/诊断工作 |
| **D4-N 持续法向支承** | 复用 D3，在有固定边界的五点膜上运行入层→支承→主动卸载→离层→再接触；新增公共 runner 和独立连续参考 | runner/审计路径已实现；**严格资格 FAIL**：run46 返回第 301 步后超原 same-entry 速度门，绝对差 2.27667386e-5 m/s | **PENDING**；不存在已通过的完整 D4 安装窗 | **诊断长窗运行中，结果未定**。`--diagnose-short-state` 只可产出 DIAGNOSTIC_COMPLETED，不能 PASS；其余三固定窗未获本页通过结论 | 完整窗成本未形成最终结果；无 Newton 对比 |
| **R3-N 浮根球树法向接触** | MuJoCo-TYX 路径：原 FREE 根 + BALL/R/P 后代、NONE JOINT motor 与 primitive condim1 接触共用完整 M/J/积分 | **SOURCE_VERIFIED**：两新文件完整 55 节点、相关 7 兼容节点实际通过；两后端包含在固定矩阵 | **PENDING**，统一包与新消费者未验 | 双足环境接触及同树活动分支互碰各 24 步、非零入步速度/正反控制/卸载；这是短轨迹组合资格，**非长时稳态或人形舞蹈资格** | 未做公平性能比较；不宣称最大 DOF/最大 contact 的笛卡尔积都已验 |
| **G2a 原容量内异构规模** | MuJoCoTyxHeterogeneousBatch 的真实按源分桶；同一次 whole-batch 发布所有 world 的控制、力和运动 | **部分通过**：三个 W1 来源 × interpret/CUDA 共六格通过；W8/B3、W16/B3、W16/B8 及新规模故障/上限节点待队列 | **PENDING**，按完整格选择运行，不能截短每格或抽样 world | 已验 W1 原 24 步及边界真实后继；更大规模待验。它不等于 D4 的承载稳定性 | 仅准备分段 walltime；live/peak device memory 均 **UNKNOWN**，累计 allocation/upload 非峰值；未比较 Newton |

D5-A 与 D4-N 共享原 D3 机制但解决不同问题：前者减少用户手工装配，
后者检验连续功能是否达到固定目标。D4 的严格失败不能抹去 D5 的公共接线增量；
D5 的短窗通过也不能关闭 D4。R3-N/G2a 是另一条 MuJoCo-TYX 路径，
不能自动获得 resident 场景的刚柔耦合能力。

## 二、每片的具体接受条件与剩余缺口

| 节点 | 必须观察到的行为 | 当前需要补齐 | 本片不承诺 |
| --- | --- | --- | --- |
| D5-A | 原 asset/model/shape/point/facet 身份经公共声明完整保留；真实 accepted 双源→同入步层力→材料/刚体候选→一次共同提交；控制后初态 checkpoint/reset 保 held loads；current query 与 last_step 入步 query 明确不同 | 新包两后端消费者的完整原始返回、独立几何/材料/刚体参考、strict module origins 和关闭结果；父最终产品 hash | 任意 multibody/affine/多刚体每 world、摩擦/CCD、所有 SceneDeclaration 入口统一、无 HOST 往返 |
| D4-N | 四角固定自由中心；球/法向胶囊/柔度×2/h减半；连续真实速度和外功；四 facet 以完整(surface,round,facet)身份施力；支承、离层和再接触由实际几何及力判定 | 先读当前诊断窗结果：功能事件、支承窗、侧别、峰值、长参考分别通过还是失败；严格短状态门继续单列 FAIL。其余完整固定格按时限决定，不拿前缀代全窗 | 零穿入、精确连续能量守恒、隐式耦合、面积压力、摩擦/换面历史、CCD |
| R3-N | 每个 world 完整 FK/COM/J/M/bias/passive/applied、CSR motor、法向几何/J/R/aref/力、qacc 与积分后 q/v；两侧 COM 力矩及同树共同祖先广义反应；same-entry 与独立连续 native 参考同时成立 | 统一包解释/CUDA的 feet/siblings 和恢复/partial reset 后继；保准确 capability/domain | pyramidal3、FILTEREXACT、limits/dry、BALL affine bias/spring、多 FREE 根或任意组合 |
| G2a | W8=3/3/2、W16=6/5/5；B8 是 8 份真实独立编译身份、每桶 2 world，非只改 model key；全 world/全 24 步保原参数及门；末桶故障不部分发布并真实 retry | 尚未执行的完整格及规模事务边界；再运行最终包对应完整格。原 B8 仅更改/补充 XML documentary model name，逆变换须恢复原 XML 全字节 | 扩容到 W32/128/256、单 fused kernel、跨 world 接触、完整异构训练或性能超过 Newton |

D4 固定功能门仍按[原协议](p65-deformable-normal-load-protocol.md)：
两支承窗 [0.75,1) 和 [2.75,3)，平均法向支承力为 16N 的 0.8–1.2，
刚体 COM 和自由中心各自法向速度 RMS≤0.1m/s；
峰值≤128N、gap≥−r/2并独立保持膜原侧；事件至少持续 1/128s。
长参考 position≤2e-3m、velocity≤1e-2m/s、orientation≤2e-3rad。
主动上提包含真实外功；不改时间窗、阻尼、质量或步长来追绿。
FP32 冲量审计单列有先验界的 predictor/ordered-position/reconstruction 项，
保原力、材料 impulse、固定反力和原始缺陷；**审计解释不是物理状态修复**。

## 三、最终包外消费者必须保留的观察

| 消费者 / 固定工作量 | 不能省略的观察 | 包外当前已写负控 / 生命周期 |
| --- | --- | --- |
| [D5 consumer](../../tests/installed/check_deformable_scene.py)：每后端两类 16 步短窗及 A/restore/B/reset 后继，预声明 38 batch calls / 57 worldsteps / 6 显式 accepted queries | 原声明与 source_map；完整 query raw/材点身份；同入步两侧载荷、COM 力矩、ordered 材料 impulse/固定反力和候选状态；每次 step 返回先记，再 observation/assert；独立有限几何与连续材料/刚体参考 | 旧 query 在真实步后拒绝；control-only revision 后旧 query 拒绝；恢复 A/B；只 reset 选中 world，未选完整状态保留，声明初始 held controls 恢复后再真实推进 |
| [R3 consumer](../../tests/installed/check_free_ball_normal.py)：每后端 feet/siblings 各 W2×24 步及 W4 生命周期；预声明 56 calls / 112 worldsteps | 全部 native 模型/输入、FP32 stored controls、所有 contacts/约束原行、双侧 wrench、JOINT CSR、完整 q/v/act/time；独立连续 native；实际 TYX 步内禁止 CPU step/step1/step2/forward | checkpoint 字节往返、clone/恢复完整后继、stale CAS 拒绝且 capsule 不变、选中 reset/未选原始 clock 与 history 保留，真实后继 |
| [G2a consumer](../../tests/installed/check_mujoco_tyx_heterogeneous_scale.py)：每格固定 24+3 calls；三个 W1、W8、两个 W16 共 1161 worldsteps/后端（只是完整计划量） | 原参考消费者完整 SHA 固定副本；每源 XML/archive/MJB 身份、实际 layout/bucket/row；每 world 完整 same-entry 和连续参考；全 force/motion publication；construction/physics/full observation/reference/save/boundary 分段时间 | A/restore/B 全记录相同、selected reset 后真实 successor、未选 history 保留；旧 helper SHA 或安装模块来源不符必须在使用前拒绝。不能把仅 W1 的结果外推到 W16 |
| [D4 runner](../../examples/run_deformable_normal_loads.py)：仅在父决定新增完整安装窗时执行；随附 cases + impulse auditor 完整字节 | 首返回前后、全部 force/query/material/rigid/state、短门失败明细与未修 raw、长期独立参考、完整事件和固定统计窗；每步流式前缀及结尾完整状态 | 原 post-return observer failure 明确为“已提交物理＋观察失败”，不能称回滚；diagnostic-only 标签必须保留，不能被退出码或 DIAGNOSTIC_COMPLETED 冒充严格 PASS |

上述计数是消费者当前固定合同，用于发现漏步、漏 world 和不完整保存，
不是新实际执行事实。最终摘要必须填 actual returned / checked / accepted
worldsteps / query counts，并与完整流式记录一致。

所有包外工作还须共同满足：

- 同一最终产品构建与安装，完整 source/wheel/installed 产品字节对应；
  消费者复制到源码外，脚本/cwd/导入路径不来自 checkout；
  所有已载入 Phymeo 模块锁定安装根，原 native 版本/模式明确。
- 原数组保存 dtype/shape/字节，含完整 source/control/input/output/revision/
  world/order/bucket 和 raw 字段。manifest/hash 绑定的是本次实际前缀，
  不能用 summary 的统计数字替代原记录。
- 构造、reference、实际 step、返回、observation、assert 和 close 的失败分别记录；
  主失败不能被清理失败覆盖。没有观察到正常 free 的 UNKNOWN_RELEASE
  不能写成“已释放/零 live”，不能重复释放或重建成功掩盖旧 owner。
- 完成原声明负控与生命周期后才作有限包外接受；新包不是全部历史功能重跑。
  未运行的消费者/后端/规模格逐项保留 PENDING。

## 四、不能在包外报告里偷换的源码负控

这些节点已在新源码测试中存在或排队。**当前消费者并不重演它们全部**；
最终验收应引用准确同源源码结果，不能把引用写成“包外故障注入已验”。
若安装暴露实际来源/ABI差异，父再指定最小受影响负控，不全量重复旧测试。

| 源码责任范围 | 必须保留的 negative cases | 与包外状态的分界 |
| --- | --- | --- |
| D5-A | source/model/install/revision 篡改或缺失、缺材料、world 不符；原 host_patch/none 准入不变；真实 query 后及共同提交前晚失败；wrong owner checkpoint、闭合生命周期；构造清理异常保 owner | 当前 consumer 已覆盖 stale query/control 与恢复/重置，**未声称重新注入所有晚失败或 UNKNOWN** |
| D4-N | synthetic event/统计缺失不得通过；真实物理返回后 observer fault 保完整提交前缀；CONDITIONING/缺法向近候选不能丢行施力；严格门失败记录不能转 PASS | 诊断模式是获取功能证据，不是负控或严格精度通过。长期功能失败若出现，仍保原输入 |
| R3-N | 不支持 condim/activation/limits/dry/transmission/多根等在设备前拒；zero/nonunit root/child quaternion；原 pairs/slots exact/overflow；真实末 world status/readback/transmission 与 restore 晚失败、retry；坏控制/外来 capsule | consumer 的 stale CAS 和 lifecycle 不代替源码领域/容量/late-fault 矩阵 |
| G2a | W16/B3 和 W16/B8 的 last-status / last-download / late-aggregate：accepted banks/ticks/checkpoint/history/旧 receipt 全保，恢复 seam 后真实同输入重试；W17/model9 在 preflight/allocation 前拒 | 此截点这些规模节点待验；包外 consumer 只做公共正格和边界，**未实现包外私有 seam 注入** |

## 五、收尾裁定与下一步

1. 先完成 D5-A、R3-N 的统一包闭环；它们有可复用的公共功能增量，
   不必等 D4 的长期结果才收口，但范围严格保持。
2. D4 分别报告严格数值资格和完整功能诊断。若诊断证明功能可达，只能按
   用户接受有限精度的方向提出**明确后续工程合同**，不能反写旧门 PASS；
   若支承/离层/侧别/漂移失败，保存最小实际阶段并停止本假设，不调 fixture。
3. G2a 完成整格才新增相应规模资格；到点未运行的格直接 PENDING。
   没有独立设备 live/peak observer 和公平 Newton 同模型/同质量/同输出基线，
   性能栏继续 UNKNOWN/NOT_COMPARED。G2b 扩容、G3 性能另立节点。
4. D4 摩擦/换面/CCD、更广 D5 多刚体/关节刚柔、R3 pyramidal/activation 等
   仍是后继功能缺口；旧 G1 长舞、J/R 学习和 B-point 表示失败不随本表关闭。

最终表应绑定父本轮 tested/product/wheel commits 和实际日志收据，再更新
[全景](phymeo-newton-panorama-2026-11-12.md)与续作入口。这里没有新增物理支持、
修改旧报告或给整个 P6.5/功能对齐作完成结论。
