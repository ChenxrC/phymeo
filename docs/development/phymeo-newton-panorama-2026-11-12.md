# Phymeo / Newton 功能差异全景（2026-11-12）

本页是 2026-11-12 05:00 批次中的功能快照，依据父集成源码 `85caed4e`、[本轮实际记录](p65-parity-until-0720-20261112.md)及[本轮计划](../superpowers/plans/2026-11-12-newton-parity-until-0720.md)。新安装包和 N7 组合规模仍在进行中，下表显式保留 PENDING；不能用已有源码测试替它们通过。旧[11-11 全景](phymeo-newton-panorama-2026-11-11.md)保留历史，本页取代其中已过时的 COL、执行器、随机重置与控制窗口状态。

比较对象固定为 **Newton 1.7.0** 官方文档及本地已固定源码，不声称调查了后续最新版本。这里按公共能力、适用范围和可复用案例判断差距，不按测试数或清单数计算总体完成百分比。

## 总体判断

Phymeo 已连接模型编译、受限机器人动力学、设备接触与约束、批量环境、真实 CUDA PPO 和恢复流程。当前主要差距集中在**能力的组合范围与案例覆盖**：更广关节和传动、设备 mesh/变形接触到载荷的连接、复杂机器人训练及批量传感；可微物理、跨域通用编排、密集软体和杆缆/MPM/流体仍各有独立模块缺口。

本轮已经解决四个圆柱 Y 查询法向失败，增加球关节动力学和合法标量执行器组合、当前三角边界距离公共接口，并减少普通 MuJoCo-TYX 步进的 native 状态复制。尚未解决的曲面触点表示、J/R 学习门和 G1 长轨迹不能被这些增量覆盖。

## 两条路径必须分别看

| 路径 | 已连接的主要流程 | 当前边界 |
| --- | --- | --- |
| Phymeo resident | 程序/资产模型 → ordinary、affine、floating、floating-affine 树 → TYX FK/宽相/设备凸体查询 → 默认 HOST patch 或限定 S/C/OBB/H 设备 patch → 原材料点、完整 J/M、双方力与 COM 力矩 → 共同发布 | 已验受载分页和混合参与者；材料点转换、载荷装配与诊断仍有 HOST 工作。曲面查询的新能力不会自动成为曲面 patch 或载荷能力 |
| MuJoCo-TYX | 统一模型入口 → CPU MuJoCo 3.3.7 编译模型 → 明确的数据映射 → TYX 树动力学、primitive 接触和混合约束 → Runtime/VecEnv → CUDA RSL PPO | CPU 编译不代替设备物理步。CUDA 默认 MuJoCo-TYX，显式 MuJoCo-Warp 是另一实现；各关节/执行器/接触组合有独立准入，不能把固定球树和自由根接触域任意拼接 |
| 公共 CollisionWorkspace 边界查询 | 完整 HOST 当前快照 → 原过滤/注册和 TYX BVH 候选 → COL2 有限线段–三角距离 → typed aggregate 与事务发布 | 新 COL3 是独立查询接口；不是 resident 设备顶点借用、实心内外判定、CCD 或接触力通道。此公共接口也不会自动扩展 MuJoCo-TYX primitive 求解器 |

## 本轮实质更新与保留失败

| 节点 | 当前实际状态 | 不应据此推导的能力 |
| --- | --- | --- |
| COL1a 通用凸体查询 | 同候选发布加完整固定点停滞出口，两后端各新 10、目标/候选 12、受影响矩阵 282 项通过；原四 Y 法向及九资源回归闭环。原门和容量未放宽 | 不是 GJK/EPA 全输入完备性证明，也没有完成新曲面接触 patch |
| B-point 曲面零维触点 | **FAILED / 停止当前候选**。解释首轮 11 通过、touch-SE 失败；法向满足查询门，但该法向的椭球单点支撑与旧见证不满足原位置门。其余未访问节点和 CUDA 未运行，候选未合入 | 不能把 query RESOLVED 改称 point/完整 patch 成功；C-point 载荷仍未实施 |
| COL2 有限边界核 | 源码 93 节点：4 HOST 资源、44 interpret、45 CUDA 通过。完整候选、两见证、参数、来源转换和资源失败有资格 | 距离是有限 segment–triangle 边界距离；球为零轴、胶囊为有限轴加半径。没有实体穿透符号、连续碰撞或载荷 |
| COL3 公共分派 | 源码 75 新项、11 旧发布兼容项通过；公共类型导出 HOST 1 项通过。支持 sphere/capsule 对 triangle/mesh 当前快照、自动 BVH 候选、完整原 facet 映射与整体发布 | HOST 快照上传、邻近带有限查询；空候选不证明闭 mesh 内部物体分离，未提供带外全局最近面。包外资格仍 PENDING |
| BALL1 / BALL-A | 固定根 BALL/HINGE/SLIDE 完整动力学及合法标量执行器组合已通过 interpret/CUDA。BALL-A 最多 32 nu / 32 na，保 nq/nv 分离、完整耦合与恢复 | BALL-A 的执行器只驱动真实 hinge/slide，球被动耦合运动；**不是直接球电机**。固定球树仍不含自由根、接触、限位/干摩擦等任意组合 |
| ROB2b | 单自由根/原法向或混合约束与 NONE/FILTEREXACT 标量执行器复用组合，解释 22、CUDA 22、兼容 48 通过 | 不是任意 transmission、闭环或完整人形动作资格；与固定球树分别受限 |
| N5 / N6 | N5 按 stable world/reset index 的随机初态、恢复已过源码；N6 D=1..8 控制窗口、完整物理子步、endpoint 任务/部分 reset、GRU 续接及跨 D 拒绝已过源码 | 当前 N5 是 q/v 状态随机化，不是质量/摩擦/资产随机化；N6 有真实失败前缀，物理已推进后的任务失败不称整窗回滚 |
| S256 紧凑镜像 | full/integration 各 48 个实际步与三次 PPO 更新通过，共同物理、learner、Adam、RNG 逐字节相同。合计 96 步 native 构造/copy 为 0，原 arena 预算不变 | 不是全流水线零 HOST 工作，也不是 256 台 G1。进程 commit 在匹配观察阶段由旧约 4.886 GB 降至 1.268 GB；**不是峰值内存或 Newton 对比** |
| J/R 学习与 N4e | 原 J/R 固定预算学习门仍失败。J 偏置居中三种子完成但未过门，已停止该因素；N4e 独立控制器在固定窗口可达 | 控制可达不是 PPO 学习成功；不追加种子、预算或改原门制造改善 |
| G1 / R03 | G1 原 46.29 s 参考 XY 门及完整 131.48 s 动作未通过；R03 两项小净加速度门仍开放 | 已有短窗、驱动/传感修正和两侧响应不能替代这些原验收 |

表中测试数只标出证据范围。早期作者报告中的 TESTS_PENDING 不覆盖父后来完成的执行结果；当前事实以本轮父记录为准。

## 八个方向的大颗粒度差异

| 用户方向 | Phymeo 已有基础与本轮变化 | 与固定 Newton 能力版图相比仍需补齐的模块 |
| --- | --- | --- |
| **01 场景与机器人** | URDF/MJCF/USD 与程序场景、统一编译、PD/IK、四类 resident 树、混合参与者反作用；MuJoCo-TYX 自由根标量树、BALL1、BALL-A、ROB2b | 更广关节/闭环/传动、直接球驱动及已有球树与自由根/接触组合；更广多机器人/工具交互与完整抓取释放库。Panda 和 G1 各是案例，不能替代组合矩阵 |
| **02 碰撞检测与响应** | RR/RD/DD 的既有有限域；S/C/OBB/H 设备 patch 与真实载荷；七形状 support-map 查询的 18 新族已过；COL2/COL3 打通 sphere/capsule→有限面/mesh 邻近边界查询 | 新曲面 patch/载荷表示；真正设备来源的 mesh/变形窄相与持续接触、跨特征身份/摩擦历史、通用 CCD 到响应。预计算 SDF、SDF 接触缩减和 hydroelastic 是另列缺口，不能因“已有 mesh”而算作具备 |
| **03 GPU 与多场景** | TYX 批量动力学/混合约束、W129/W256 resident 受载分页、设备 action、真实 GPU PPO 小任务规模；紧凑镜像消除普通步 native copy | 更复杂异构资产/约束规模、减少 HOST 装配/观察往返、设备 mesh 输入与端到端规模；峰值资源与稳定吞吐基线。缺少与 Newton 同模型/精度/输出的直接性能比较 |
| **04 可微仿真与训练工具** | 实际 CUDA RSL PPO、MLP/GRU、组合 checkpoint、S 持续支撑学习成功、N5 随机状态重置、N6 控制频率窗口 | J/R 仍未达学习门；更广机器人训练、模型/环境随机化、可复用任务库。多步物理梯度、辨识与梯度对照仍是独立能力，不能由 PPO 能运行推导已可微 |
| **05 传感与交互** | 深度/法线/ID、TYX 射线、IMU 基础、真实轨迹 Three.js、G1 传感坐标绑定 | 批量、多传感器时序/坐标协议，更广 RGB/材质/光照和交互编辑工具。单个 gyro 修正不代表完整传感生态 |
| **06 跨域求解编排** | 事务、点端口、域图与指定刚–布–软体耦合；同 world 多参与者共同发布 | 通用区域/后端选择、异构界面力/约束、耦合迭代、能量与共同失败恢复。resident 多树一次载荷组合不等于跨树隐式 KKT，也不等于通用多求解器耦合 |
| **07 布与体软体** | 拉伸/剪切/弯曲、自接触、StVK 四面体、双向载荷和递布基础 | 密集折叠、持续静摩擦/换特征、软块抓捏、广泛材料/网格和软布放台。需完整多案例资格，原有限几何/响应范围不扩大为全部柔性碰撞 |
| **08 杆缆、MPM/流体** | 刚性连杆、粒子与弹性体基础可复用 | 完整弯扭杆缆、粒子–网格转移、塑性/流动、流固边界仍需分别立项；连杆不是柔杆，粒子不是 MPM/流体求解器 |

Newton 本身也按 solver/backend 配置提供能力；上表不把所有 Newton 模块描述为可以任意一起使用。Phymeo 的旧 HOST 曲面几何和部分载荷资格仍可复用，但不自动升级为新 device 支持或完整多场景接口。

## 下一批优先级与可交付节点

按用户的八方向保持模块顺序，同时保留已提高优先级的训练与规模主线。先收束已有公共连接，再扩域；三支可并行作者开发，父仍独占共享接缝和串行项目执行。

| 优先级 / 难度 | 下一可交付节点 | 最小多案例与成功条件 | 停止边界 / 依赖 |
| --- | --- | --- | --- |
| **P0 / 中高** | 完成本轮新 wheel 与七个公共消费者；N7 只完成已登记组合资格 | 包外来源、真实 step、完整失败/恢复；J/W8/D2/GRU、S/W128/D4、S/W256/D4，两观察模式共同状态/更新相同 | 当前 PENDING；任一格实际不符就保失败，不降 world 或改 D 追绿。没有安装证据的能力仍只算源码资格 |
| **P1-训练 / 高** | 用可达性和原保存诊断冻结 J/R 下一次单因素学习实验；先补任务/优化的可观察性，再执行固定预算 | 至少 J、耦合 R 与已学会 S 对照；固定初态/持出集/三种子，区分 untrained、控制器、PPO 的绝对成本及动作/优化证据 | 不重复已否定的均值居中，不增加预算或挑种子；没有新可区分假设则停止训练调参，先补诊断接口 |
| **P1-机器人 / 高** | 从 BALL-A/ROB2b 的真实准入差异选一个组合片，优先球关节与驱动/接触的公共动力学接缝 | 先列 nq/nv、传动/activation、两侧 J/M/力矩矩阵；固定球链、偏心多关节、反向/卸载和恢复负控。明确本片选直接球驱动还是球树接触，不能一批泛化全部 | 先独立 native/解析参考再实现；原 G1/R03 未过门单列，不以一个舞蹈替代组合验收 |
| **P1-碰撞 / 高** | 在已验 COL3 之后，先冻结设备 mesh/变形来源适配及同帧有限候选的合同；保查询与载荷分层 | 稀疏多 world、刚性 mesh/变形快照、共享 facet/凹体、sphere/capsule front/back/edge；source/epoch/容量/晚失败有统一公共包 | 当前 COL3 的 HOST snapshot 不能冒 resident device loan；先交公共查询，再另立载荷/CCD。此支与 B-point 表示失败独立 |
| **P2-表示 / 高** | 重新冻结曲面 selected-normal 点离散与查询见证之间的表示合同 | 原 touch-SE 最小反例加九族/四来源；明确哪些法向和表面点共同发布，原材料地址、力/COM 力矩与拒绝语义 | 先定工程表示取舍与独立验收，不继续本夜 B-point 原候选；未完成几何资格不接 C-point 载荷，不新写九个形状专用 solver |
| **P2-规模 / 中高** | 基于 N7 与已知热点选择一个端到端开销节点，再建立等输出比较 | J/S/R 与混合约束/异构场景，记录 step、观察、reset、策略、保存分别的耗时和资源；共同结果先过再比较 | 不用单个 kernel、不同输出模式或观察阶段内存推导超越 Newton；性能优化不得覆盖用户现有环境或降精度域 |
| **P3 / 分别立项** | 传感/交互 → 通用跨域 → 布/体软体 → 杆缆/MPM/流体 | 每族先多案例库、现存差异模块、独立失败/恢复与实际运行；传感选同场景时序，软体选支撑/抓捏/折叠，杆缆/流体分开 | 不借机器人或单个递布通过替代这些独立模块；物理梯度另立对照，不混入 PPO 成功判据 |

每个开发片应有一份短合同、明确公共 API/文件所有权、预先登记的正负案例、实际执行结果和安装消费者。通用算法先用矩阵覆盖，不能以不断追加单个 shape pair 代替统一模块。

## 如何识别成功与无效循环

- **成功**：公共能力能在不同形状、来源、world 和连续步上复用；独立 oracle、原数值门、原子发布/容量负控和安装调用均有实际证据。学习成功额外需要原持出门，不只训练脚本退出成功。
- **已推进但未完成**：能明确缩小缺口。例如 COL2/COL3 完成当前边界查询，但尚无载荷；N4e证明控制可达，但学习仍失败。单列边界即可继续独立模块。
- **应停止当前假设**：同一最小反例在一次有界通用修复后引入新回归，或两次针对尝试未缩小障碍；保存原输入、真实中间量和失败，回到合同/表示或另一个独立模块。不能不断改 fixture、门、容量、预算来使案例消失。
- **避免重复劳动**：同源已绿范围按原证据复用；算法/来源发生实质变化才跑受影响回归。先诊断实际数据再选修复，不重新做无限精确证明或无假设训练。

## 本页待父补齐的结果

| 收尾项 | 本页当前状态 | 完成后应补的事实 |
| --- | --- | --- |
| 新 stage wheel + 七消费者 | **PENDING**，源码资格已有，安装验证正在父串行队列 | 最终产品 revision、包/源码/安装成员一致性、各消费者实际后端与结果、保留失败 |
| N7 J/W8/D2/GRU | **PENDING**，纯编码/待实际资格 | full/integration 共同流、更新/恢复、reset 与 D 的实际计数 |
| N7 S/W128/D4、S/W256/D4 | **PENDING**，未据编写结果宣称通过 | 前置格通过后各实际结果；若未启动明确写未启动，失败不改规模补跑 |
| 最终架构与全批关闭 | **PENDING** | 公共入口对应、剩余失败和下一授权范围；不把本页文字提交当整批完成 |

## 固定依据与入口

- [实际批次及保存证据入口](p65-parity-until-0720-20261112.md)、[当前计划](../superpowers/plans/2026-11-12-newton-parity-until-0720.md)；父执行事实优先于旧作者 TESTS_PENDING 标签。
- [公共三角边界查询](../architecture/triangle-boundary-query.md)、[COL2 报告](p65-device-triangle-distance-report.md)、[COL3 报告](p65-triangle-distance-dispatch-report.md)。
- [ROB2b 组合](p65-rob2b-actuator-composition-report.md)、[BALL1](p65-native-ball-joint-report.md)、[BALL-A](p65-ball-scalar-actuation-report.md)。
- [N4a 学习](p65-mujoco-tyx-n4a-report.md)、[N4d 固定消融](p65-n4d-initialization-protocol.md)、[native 紧凑镜像](p65-mujoco-tyx-native-mirror-report.md)、[观察内存诊断](p65-compact-mirror-memory-diagnostic.md)、[G1 保存诊断](g1-drift-diagnosis-20261111.md)。
- Newton 1.7.0 官方固定参照：[solver 矩阵](https://newton-physics.github.io/newton/1.7.0/solvers/index.html)、[collision](https://newton-physics.github.io/newton/1.7.0/concepts/collisions.html)、[worlds](https://newton-physics.github.io/newton/1.7.0/concepts/worlds.html)、[sensors](https://newton-physics.github.io/newton/1.7.0/concepts/sensors.html)、[coupling](https://newton-physics.github.io/newton/1.7.0/concepts/coupling.html)、[deformable objects](https://newton-physics.github.io/newton/1.7.0/concepts/deformable_objects.html)。仅作功能/架构参照，没有新增 Newton 性能实测。

文档 authoring 只读取源码、计划和父已完成记录，未执行项目 import、测试、数值、静态检查或构建；未修改旧入口/报告，也未授予后继实施。
