# Phymeo / Newton 功能差异全景（2026-11-12）

本页更新至 **2026-11-12 本批源码与安装收尾**。[最新批次](p65-parity-motor-asset-20261112.md)完成直接BALL电机与原接触组合（BALL-MC）、四类原资产设备边界来源（M1-A）；四个源码外解释/CUDA消费者通过，产品 `d362799f` 的516成员source/wheel/installed同源。M2只完成真实producer设计。[上一批](p65-parity-m1-assembly-20261112.md)的BALL-M1、多参与者装配、direct M1及515成员/六消费者保持原资格。旧[07:20结果](p65-parity-until-0720-20261112.md)的首包 `85caed4e`、增量包 `1ce57489`、N7与N8按原版本复用，不倒灌为新包全量回归。J/R学习门和其他保留失败没有因此关闭。[旧11-11全景](phymeo-newton-panorama-2026-11-11.md)及07:20架构HTML保留历史；HTML尚未画入近期新增模块，当前以本页为准。

比较对象固定为 **Newton 1.7.0** 官方文档及本地已固定源码，不声称调查了后续最新版本。这里按公共能力、适用范围和可复用案例判断差距，不按测试数或清单数计算总体完成百分比。

## 总体判断

Phymeo 已连接模型编译、受限机器人动力学、设备接触与约束、批量环境、真实 CUDA PPO 和恢复流程。当前主要差距集中在**能力的组合范围与案例覆盖**：更广关节和传动、设备 mesh/变形接触到载荷的连接、复杂机器人训练及批量传感；可微物理、跨域通用编排、密集软体和杆缆/MPM/流体仍各有独立模块缺口。

此前已解决四个圆柱Y查询法向失败，增加BALL1/A/C、COL2/COL3和紧凑native镜像，随后连接JOINT moment表、显式多参与者装配和真实accepted设备FK。本批将直接球电机接入原normal/pyramidal接触，并让四类原资产来源进入设备边界查询。随机重置、D步窗口与PPO的原N7组合资格保持。曲面触点表示、J/R学习门、G1长轨迹、变形来源及查询到载荷的缺口仍开放。

## 两条路径必须分别看

| 路径 | 已连接的主要流程 | 当前边界 |
| --- | --- | --- |
| Phymeo resident | 程序/资产模型 → ordinary、affine、floating、floating-affine 树 → TYX FK/宽相/设备凸体查询 → 默认 HOST patch 或限定 S/C/OBB/H 设备 patch → 原材料点、完整 J/M、双方力与 COM 力矩 → 共同发布 | 已验受载分页和混合参与者；材料点转换、载荷装配与诊断仍有 HOST 工作。曲面查询的新能力不会自动成为曲面 patch 或载荷能力 |
| MuJoCo-TYX | 统一模型入口 → CPU MuJoCo 3.3.7 编译模型 → 明确的数据映射 → TYX 树动力学、primitive 接触和混合约束 → Runtime/VecEnv → CUDA RSL PPO | CPU 编译不代替设备物理步。CUDA 默认 MuJoCo-TYX，显式 MuJoCo-Warp 是另一实现；各关节/执行器/接触组合有独立准入，不能把固定球树和自由根接触域任意拼接 |
| 公共 CollisionWorkspace 边界查询 | 完整 HOST 当前快照 → 原过滤/注册和 TYX BVH 候选 → COL2 有限线段–三角距离 → typed aggregate 与事务发布 | 新 COL3 是独立查询接口；不是 resident 设备顶点借用、实心内外判定、CCD 或接触力通道。此公共接口也不会自动扩展 MuJoCo-TYX primitive 求解器 |
| 新 resident M1 边界查询 | 四类direct完整拓扑或from_asset原资产树的真实accepted设备FK → gather/bounds → BVH/pair pack → 原有限边界核 → typed事务发布 | 保留HOST读回/校验和整数候选控制；ParticipantScene多bank映射、M2变形、CCD、实体内外和接触载荷仍未连接。与COL3输入路径及MuJoCo-TYX求解器分别资格 |

## 本轮实质更新与保留失败

| 节点 | 当前实际状态 | 不应据此推导的能力 |
| --- | --- | --- |
| COL1a 通用凸体查询 | 同候选发布加完整固定点停滞出口，两后端各新 10、目标/候选 12、受影响矩阵 282 项通过；原四 Y 法向及九资源回归闭环。原门和容量未放宽 | 不是 GJK/EPA 全输入完备性证明，也没有完成新曲面接触 patch |
| B-point 曲面零维触点 | **FAILED / 停止当前候选**。解释首轮 11 通过、touch-SE 失败；法向满足查询门，但该法向的椭球单点支撑与旧见证不满足原位置门。其余未访问节点和 CUDA 未运行，候选未合入 | 不能把 query RESOLVED 改称 point/完整 patch 成功；C-point 载荷仍未实施 |
| COL2 有限边界核 | 源码 93 节点：4 HOST 资源、44 interpret、45 CUDA 通过。完整候选、两见证、参数、来源转换和资源失败有资格；首包两后端各 2 次 launch/8 rows 通过 | 距离是有限 segment–triangle 边界距离；球为零轴、胶囊为有限轴加半径。没有实体穿透符号、连续碰撞或载荷 |
| COL3 公共分派 | 源码 75 新项、11 旧发布兼容项通过；公共类型导出 HOST 1 项通过。支持 sphere/capsule 对 triangle/mesh 当前快照、自动 BVH 候选、完整原 facet 映射与整体发布 | HOST 快照上传、邻近带有限查询；空候选不证明闭 mesh 内部物体分离，未提供带外全局最近面。首包两后端各 4 次 aggregate 查询通过，仍不扩展实体/载荷语义 |
| BALL1 / BALL-A | 固定根 BALL/HINGE/SLIDE 完整动力学及合法标量执行器组合已通过 interpret/CUDA。BALL-A 最多 32 nu / 32 na，保 nq/nv 分离、完整耦合与恢复；首包 BALL1 两后端各 40 worldsteps、BALL-A 各 58 通过 | BALL-A 的执行器只驱动真实 hinge/slide，球被动耦合运动；**不是直接球电机**。BALL-A原验收不含接触，后继BALL-C已单独验收；仍不含自由根、限位/干摩擦等任意组合 |
| ROB2b | 单自由根/原法向或混合约束与 NONE/FILTEREXACT 标量执行器复用组合，解释 22、CUDA 22、兼容 48 通过；首包两后端各 40 worldsteps 通过 | 不是任意 transmission、闭环或完整人形动作资格；与固定球树分别受限 |
| N5 / N6 | N5 按 stable world/reset index 的随机初态、恢复已过源码；N6 D=1..8 控制窗口、完整物理子步、endpoint 任务/部分 reset、GRU 续接及跨 D 拒绝已过源码；首包 N5 CUDA 32、N6 CUDA 192 worldsteps 通过 | 当前 N5 是 q/v 状态随机化，不是质量/摩擦/资产随机化；N6 有真实失败前缀，物理已推进后的任务失败不称整窗回滚 |
| S256 紧凑镜像 | full/integration 各 48 个实际步与三次 PPO 更新通过，共同物理、learner、Adam、RNG 逐字节相同。合计 96 步 native 构造/copy 为 0，原 arena 预算不变 | 不是全流水线零 HOST 工作，也不是 256 台 G1。进程 commit 在匹配观察阶段由旧约 4.886 GB 降至 1.268 GB；**不是峰值内存或 Newton 对比** |
| N7 组合规模 | run265 J/W8/D2/GRU、266 S/W128/D4、267 S/W256/D4 的 full/integration 两模式均通过；每模式 24 policy windows，分别 384/12,288/24,576 个实际物理 worldsteps，共同物理/动作/reset/Adam/RNG（及 GRU hidden）一致 | 这是更新与规模连接资格，不是 J 学会、S 新学习实验或超越 Newton。后续评价器收紧跨窗口 action/input/output 绑定，275 HOST 32 与 276 J8 实际 pair 通过；原 S 大格不冒作新评价器下重跑 |
| 首包安装 | run235 的 509 个产品成员在 source/wheel/installed 逐字节相同；七消费者所有声明后端均 RAW_EXIT=0、strict origins 通过，无 source tests/examples 导入或 CPU 物理代跑 | 首包绑定 `85caed4e`。run240 只暴露消费者 FP32/FP64 编码错误，修复后保存原失败；后续评价器增量与新消费者须新包另验 |
| BALL-C 固定球树接触 | 源码52新节点与6受影响入口通过；最终包解释/CUDA各64worldsteps通过。完整M/J、双方COM力矩、标量驱动/activation、力与运动共同发布、连续接触与生命周期 | 限fixed BALL/HINGE/SLIDE、scalar NONE/FILTEREXACT、primitive condim1/pyramidal3、nv8/nq10；无直接球motor、FREE+BALL、limits/dry。安装stale-CAS与源码设备晚失败资格分开 |
| BALL-M1 直接球电机 | 统一native JOINT CSR映射、同次实际scalar/广义力与运动、checkpoint恢复；47新源码节点+9兼容节点；安装解释/CUDA各56worldsteps | fixed BALL/HINGE/SLIDE、直接BALL只JOINT/NONE/FIXED/BIAS_NONE、na0；无直接球电机的activation/contact、FREE+BALL、limits/dry |
| Participant scene | 显式资产/参与者/控制/完整材料对表连接原四类resident树；HOST19+interpret19+CUDA19；安装各24worldsteps | 有HOST离散接触采样；不是新求解器、MuJoCo-TYX BALL导入、跨树隐式KKT或R03通过 |
| M1 resident boundary | 实际accepted设备来源与runtime lease已实现；4HOST+67interpret+68CUDA独立节点、20兼容节点；安装各6worldsteps/3queries/12rows | 限刚性完整拓扑direct constructor。不是零HOST、M2、CCD、实体内外、带外全局最近面或接触力 |
| BALL-MC 直接球电机接触 | 源码51新节点+6兼容通过；统一包解释/CUDA各112worldsteps。原normal/pyramidal接触与真实JOINT motor的力、广义力、运动共同发布 | 限fixed BALL/HINGE/SLIDE、JOINT/NONE/FIXED/BIAS_NONE、na0、nq10/nv8/nu32；无direct BALL activation、FREE+BALL、limits/dry或任意组合 |
| M1-A 原资产设备来源 | 四类from_asset完整model/shape/graph/group/原材点进入M1；2HOST+37interpret+37CUDA、旧direct/static各8兼容；安装各6worldsteps/3queries/24rows/2stale拒绝 | 是TyxWorldBatch原资产接线；ParticipantScene另一个多bank owner未接通。不扩展闭实体、CCD、接触力或变形 |
| M2 后继 | [真实producer设计](p65-m2-producer-bank-design.md)已核现有TYX生命周期及固定Newton 1.7 State，拆为M2-P持久节点owner与M2-Q查询；尚未实现 | accepted/candidate bank、提交所有权、lease及数组预算需落实；复用原ordered节点核，不能以HOST两帧替代 |
| J/R 学习与 N4e | 原 J/R 固定预算学习门仍失败。J 偏置居中三种子完成但未过门，已停止该因素；N4e 独立控制器在固定窗口可达 | 控制可达不是 PPO 学习成功；不追加种子、预算或改原门制造改善 |
| N8 信用分配诊断 | 原final J14011/R14211/S14111各32步W8，实际768转移/0更新及297条保存审计通过。J/R actor范数裁剪活跃、S未给区分原因；三者ratio clipping均0 | 只是本次固定采样的证据。裁剪因子不等于带原Adam历史的参数步缩放，不证明弱学习原因，不授权自动改clip/lr或新增训练 |
| G1 / R03 | G1 原 46.29 s 参考 XY 门及完整 131.48 s 动作未通过；R03 两项小净加速度门仍开放 | 已有短窗、驱动/传感修正和两侧响应不能替代这些原验收 |

表中测试数只标出证据范围。早期作者报告中的 TESTS_PENDING 不覆盖父后来完成的执行结果；当前事实以本轮父记录为准。

## 八个方向的大颗粒度差异

| 用户方向 | Phymeo 已有基础与本轮变化 | 与固定 Newton 能力版图相比仍需补齐的模块 |
| --- | --- | --- |
| **01 场景与机器人** | URDF/MJCF/USD/程序场景、统一编译、PD/IK、四类resident树及新增显式多参与者装配；MuJoCo-TYX自由根标量树、BALL1/A/C、ROB2b与直接BALL JOINT电机及其原接触组合 | 更广关节/闭环/传动；直接球电机activation、FREE+BALL与limits/dry组合；ParticipantScene多bank到新设备边界源的接缝、更广多机器人/工具和抓取释放库。单个Panda/G1不替代组合矩阵 |
| **02 碰撞检测与响应** | 既有RR/RD/DD有限域、S/C/OBB/H设备patch与载荷、七形状support-map查询；COL2/COL3有限边界核，新M1/M1-A连接direct与原资产的刚性真实设备FK来源 | ParticipantScene逻辑参与者/多bank到M1、M2真实变形bank、mesh/变形查询到持续载荷、曲面patch表示、跨特征身份/摩擦历史、通用CCD响应。预计算SDF、SDF接触缩减和hydroelastic仍独立缺失 |
| **03 GPU 与多场景** | TYX批量动力学/约束、W129/W256受载分页、设备action、真实GPU PPO；紧凑镜像、N7 S128/S256两模式；新M1直接使用accepted设备FK | 更广异构资产/约束规模、减少现存HOST装配/观察/候选控制往返、变形bank及端到端规模；峰值资源与吞吐基线。尚无Newton同模型/精度/输出性能比较 |
| **04 可微仿真与训练工具** | 实际 CUDA RSL PPO、MLP/GRU、组合 checkpoint、S 持续支撑学习成功、N5 随机状态重置、N6 控制频率窗口与 N7 组合规模资格 | J/R 仍未达学习门；更广机器人训练、模型/环境随机化、可复用任务库。多步物理梯度、辨识与梯度对照仍是独立能力，不能由 PPO 能运行推导已可微 |
| **05 传感与交互** | 深度/法线/ID、TYX 射线、IMU 基础、真实轨迹 Three.js、G1 传感坐标绑定 | 批量、多传感器时序/坐标协议，更广 RGB/材质/光照和交互编辑工具。单个 gyro 修正不代表完整传感生态 |
| **06 跨域求解编排** | 事务、点端口、域图与指定刚–布–软体耦合；同 world 多参与者共同发布 | 通用区域/后端选择、异构界面力/约束、耦合迭代、能量与共同失败恢复。resident 多树一次载荷组合不等于跨树隐式 KKT，也不等于通用多求解器耦合 |
| **07 布与体软体** | 拉伸/剪切/弯曲、自接触、StVK 四面体、双向载荷和递布基础 | 密集折叠、持续静摩擦/换特征、软块抓捏、广泛材料/网格和软布放台。需完整多案例资格，原有限几何/响应范围不扩大为全部柔性碰撞 |
| **08 杆缆、MPM/流体** | 刚性连杆、粒子与弹性体基础可复用 | 完整弯扭杆缆、粒子–网格转移、塑性/流动、流固边界仍需分别立项；连杆不是柔杆，粒子不是 MPM/流体求解器 |

Newton 本身也按 solver/backend 配置提供能力；上表不把所有 Newton 模块描述为可以任意一起使用。Phymeo 的旧 HOST 曲面几何和部分载荷资格仍可复用，但不自动升级为新 device 支持或完整多场景接口。

## 下一批优先级与可交付节点

按用户的八方向保持模块顺序，同时保留已提高优先级的训练与规模主线。先收束已有公共连接，再扩域；三支可并行作者开发，父仍独占共享接缝和串行项目执行。

| 优先级 / 难度 | 下一可交付节点 | 最小多案例与成功条件 | 停止边界 / 依赖 |
| --- | --- | --- | --- |
| **最新批已关闭 / 高** | BALL-MC、M1-A统一wheel；M2设计 | 产品d362799f、516成员parity；四个源码外消费者两后端通过，分别112/6worldsteps每后端 | 两项实现有限接受，M2只设计；前批三模块资格复用，不冒任意跨路径组合 |
| **P1-机器人 / 高** | 从十组合同选择direct BALL activation或FREE+BALL等下一有限组合 | 复用已有moment/contact路径；多模型同入步native/独立连续轨迹、完整力矩与运动、事务、安装消费者 | BALL-MC已关闭，不重派。先冻结下一组合，limits/dry等不直接删除guard |
| **P1-场景/碰撞连接 / 中高** | ParticipantScene→M1多bank来源 | 明确逻辑world/participant与内部机械row/group、多bank lease/共同epoch；四家族、多机器人/工具、反注册次序、过滤、恢复与晚失败 | 原资产from_asset已接通；ResidentSceneBatch和TyxWorldBatch是不同owner，不能复用单bank假设。禁止HOST当前坐标回填，CCD/载荷另接 |
| **P2-训练 / 高** | 对N8同次saved梯度与原Adam历史做两候选虚拟更新比较，再决定是否值得新单因素实验 | 完整参数ID/原moments/group绑定，pre/post两条虚拟129的范数、方向与历史/eps贡献；不执行optimizer/物理或修改模型 | N8已完成，不能重复采样；没有可检验预测就停止当前假设。J/R原三种子门保持，不直接更改clip/lr或加预算 |
| **P1-变形来源 / 高** | **M2-P持久节点producer**，随后M2-Q复用M1查询 | 实际物理推进导致边长改变、稳定 point/facet IDs、完整失败不换 accepted bank、来源/epoch/同步与 UNKNOWN_RELEASE | 调研已完成；下一步冻结M2-P数组预算/API并实现accepted/candidate owner。首片不迁移完整复合布/R7/coupled，不靠HOST手写快照声称设备变形支持 |
| **P2-表示 / 高** | 重新冻结曲面 selected-normal 点离散与查询见证之间的表示合同 | 原 touch-SE 最小反例加九族/四来源；明确哪些法向和表面点共同发布，原材料地址、力/COM 力矩与拒绝语义 | 先定工程表示取舍与独立验收，不继续本夜 B-point 原候选；未完成几何资格不接 C-point 载荷，不新写九个形状专用 solver |
| **P2-规模 / 中高** | 基于 N7 与已知热点选择一个端到端开销节点，再建立等输出比较 | J/S/R 与混合约束/异构场景，记录 step、观察、reset、策略、保存分别的耗时和资源；共同结果先过再比较 | 不用单个 kernel、不同输出模式或观察阶段内存推导超越 Newton；性能优化不得覆盖用户现有环境或降精度域 |
| **P3 / 分别立项** | 传感/交互 → 通用跨域 → 布/体软体 → 杆缆/MPM/流体 | 每族先多案例库、现存差异模块、独立失败/恢复与实际运行；传感选同场景时序，软体选支撑/抓捏/折叠，杆缆/流体分开 | 不借机器人或单个递布通过替代这些独立模块；物理梯度另立对照，不混入 PPO 成功判据 |

每个开发片应有一份短合同、明确公共 API/文件所有权、预先登记的正负案例、实际执行结果和安装消费者。通用算法先用矩阵覆盖，不能以不断追加单个 shape pair 代替统一模块。

[原后继设计](../superpowers/plans/2026-11-12-newton-parity-next.md)中的直接BALL、M1和资产装配已经按[本批执行计划](../superpowers/plans/2026-11-12-parity-m1-assembly.md)完成有限域开发与安装资格，不再重派原实施任务。[十组机器人案例](p65-robot-next-compositions-20261112.md)和[M1的17组来源案例](p65-m1-case-library-20261112.md)继续作为回归设计依据。本批BALL-MC与M1-A已按[新计划](../superpowers/plans/2026-11-12-parity-motor-contact-asset.md)完成；下一片按上表处理ParticipantScene多bank接缝与M2-P，不重做原资产准入。[N8诊断](p65-credit-diagnostic-report.md)保持768转移/0更新，后继只沿[保存数据与Adam历史裁定](p65-credit-diagnostic-next-decision.md)补辨别量，不自动调clip/lr或再训练。

## 如何识别成功与无效循环

- **成功**：公共能力能在不同形状、来源、world 和连续步上复用；独立 oracle、原数值门、原子发布/容量负控和安装调用均有实际证据。学习成功额外需要原持出门，不只训练脚本退出成功。
- **已推进但未完成**：能明确缩小缺口。例如 COL2/COL3 完成当前边界查询，但尚无载荷；N4e证明控制可达，但学习仍失败。单列边界即可继续独立模块。
- **应停止当前假设**：同一最小反例在一次有界通用修复后引入新回归，或两次针对尝试未缩小障碍；保存原输入、真实中间量和失败，回到合同/表示或另一个独立模块。不能不断改 fixture、门、容量、预算来使案例消失。
- **避免重复劳动**：同源已绿范围按原证据复用；算法/来源发生实质变化才跑受影响回归。先诊断实际数据再选修复，不重新做无限精确证明或无假设训练。

## 当前关闭与待补结果

| 收尾项 | 当前状态 | 证据边界 |
| --- | --- | --- |
| 首包 `85caed4e` + 七消费者 | **PASSED** | run235 509 成员 parity；236/237 ROB2b、238/239 BALL1、245/246 BALL-A、247/248 COL2、249/250 COL3、251 N5、252 N6 全部 RAW0/strict origins。240 消费者编码首错保留 |
| N7 三格 | **PASSED，限定组合与更新资格** | 265/266/267，两 mode/格，每 mode 24 policy windows；J8 384、S128 12,288、S256 24,576 worldsteps。共同完整状态/更新一致，不称学习或 Newton 性能胜出 |
| 新评价器 `4118f2ec` | **源码及最终包 PASSED** | 275 HOST32、276 J8实际pair；第二包290 CUDA192worldsteps。完整action/input/output跨窗口绑定，不倒灌首包 |
| BALL-C | **SOURCE_AND_INSTALLED_PASSED** | 284/285共52新节点，286兼容6；291/292源码外各64worldsteps。范围见[BALL-C报告](p65-ball-contact-composition-report.md) |
| BALL-M1 / Participant scene / M1 | **SOURCE_AND_INSTALLED_QUALIFIED** | 515成员同源；061–066六消费者通过。新增72文件安装归档及installed-close.json；详细源码矩阵、修复与首失败见本批报告 |
| BALL-MC / M1-A | **SOURCE_AND_INSTALLED_QUALIFIED** | d362799f的516成员parity；043–046四消费者通过。61文件安装归档与installed-close.json保留完整压缩表示 |
| M2 | **RESEARCH_AND_DESIGN_ONLY / NOT_IMPLEMENTED** | M2-P/M2-Q分片已完成调研；真实producer状态、提交/lease及查询尚未实现，不以M1刚性查询替代 |
| 本批关闭与架构图 | **BALL-MC与M1-A功能关闭，M2仅设计** | 当前Markdown已更新；07:20HTML仍是历史快照。G1、J/R、R03、B-point及其余模块仍开放 |

## 固定依据与入口

- [最新实际批次与保存证据](p65-parity-motor-asset-20261112.md)、[本批计划](../superpowers/plans/2026-11-12-parity-motor-contact-asset.md)、[上一批](p65-parity-m1-assembly-20261112.md)、[旧07:20记录](p65-parity-until-0720-20261112.md)；父执行事实优先于旧作者TESTS_PENDING标签。
- [公共三角边界查询](../architecture/triangle-boundary-query.md)、[COL2 报告](p65-device-triangle-distance-report.md)、[COL3 报告](p65-triangle-distance-dispatch-report.md)。
- [BALL-C 后继合同](p65-ball-next-composition-contract.md)、[M1/M2 设备来源合同](p65-device-mesh-source-next-contract.md)、[N7 组合规模](p65-window-scale-report.md)。
- [ROB2b 组合](p65-rob2b-actuator-composition-report.md)、[BALL1](p65-native-ball-joint-report.md)、[BALL-A](p65-ball-scalar-actuation-report.md)。
- [N4a 学习](p65-mujoco-tyx-n4a-report.md)、[N4d 固定消融](p65-n4d-initialization-protocol.md)、[native 紧凑镜像](p65-mujoco-tyx-native-mirror-report.md)、[观察内存诊断](p65-compact-mirror-memory-diagnostic.md)、[G1 保存诊断](g1-drift-diagnosis-20261111.md)。
- Newton 1.7.0 官方固定参照：[solver 矩阵](https://newton-physics.github.io/newton/1.7.0/solvers/index.html)、[collision](https://newton-physics.github.io/newton/1.7.0/concepts/collisions.html)、[worlds](https://newton-physics.github.io/newton/1.7.0/concepts/worlds.html)、[sensors](https://newton-physics.github.io/newton/1.7.0/concepts/sensors.html)、[coupling](https://newton-physics.github.io/newton/1.7.0/concepts/coupling.html)、[deformable objects](https://newton-physics.github.io/newton/1.7.0/concepts/deformable_objects.html)。仅作功能/架构参照，没有新增 Newton 性能实测。

本页收尾只依据本批真实源码/安装结果和固定Newton参照，没有新增Newton调查、性能实测或未执行的通过声明。三条原作者已释放，后继开发按当前授权与具体范围另行登记，执行仍严格串行。
