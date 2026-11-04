# Phymeo 与 Newton 1.7.0 功能对照更新

本表沿用[完整盘点的 19 个功能族](phymeo-newton-1.7-functional-inventory.md)。同一功能族内的通用性、求解精度、设备路径和独立对照各有范围。表内保留上一阶段已经验证的部分，并注明本轮新增的实际应用边界。各族的工程量差别很大，读数不换算成整体百分比。

| 功能族 | 当前可用的入口或本轮增量 | 下一处有工程意义的差距 |
| --- | --- | --- |
| 01 场景与步进 | 公开构建器可编排多个自由刚体和可变形片，按世界指定材料接触。它与 URDF 操作会话均可在一个锁内读到同一步的载荷和完整几何。[构建与回滚](../benchmarks/p65-declared-contact-scenes.md) · [机器人会话](../benchmarks/p65-interactive-robot-payload.md) | 多机械臂、多自由物体和软体仍需显式选择组合程序；合并模型、生命周期和跨域控制的工作量高于单一场景。 |
| 02 资产导入 | 普通 URDF 的固定工具链会保留原链接地址，将真实部件质量、COM 和完整惯量纳入所属运动链。在两种工具上测了受力步。[固定工具链](../benchmarks/p65-urdf-fixed-tools.md) | 单驱动夹具已抓放 100 g 自由箱；无环分叉和串联投影均有全坐标独立反力对照。仍需复杂网格、闭环约束和跨格式语义。[夹具](../benchmarks/p65-urdf-mechanical-mimic.md) · [多行传动](../benchmarks/p65-urdf-transmission-graphs.md) |
| 03 刚体与关节动力学 | 固定基、浮动树、广义质量和关节反作用有实际测量；新多球岛同时解多个动态刚体的法向冲量。[球岛](../benchmarks/p65-dynamic-sphere-islands.md) | 多动态球可分步处理斜向初触和高速反弹，失败时按世界回滚。其他形状和关节的持续接触、闭环约束与设备驻留范围仍有限。[时间、冲量与离散误差](../benchmarks/p65-substepped-sphere-islands.md) |
| 04 IK 与控制 | Panda 实时更改关节目标/努力上限；已接受状态反馈改变同一轨迹的放置点。第二 R–P 机构使用相同命令接口。[公开命令](../benchmarks/p65-interactive-robot-payload.md) | 任意机械臂规划、障碍环境中的同时双臂控制和带接触反馈的复杂轨迹仍有留出。 |
| 05 刚体检测与地形 | 既有形状目录、分组、凸形及有限地形可复用；一般射线能直接读取解析曲面与有限地形。[组合目录](../benchmarks/p65-collision-public-catalog.md) · [射线](../benchmarks/p65-ray-queries.md) | 保持复杂凹网格、体积 SDF、稳定降维接触和异形稠密堆叠是更大的检测任务。 |
| 06 连续检测 | 偏心球按真实刚体转弧检查，复合转轴与下游伸缩可沿真实关节路径检查有限凸工具对厚布的净空。[多关节](../benchmarks/p65-compound-tool-clearance.md) | 部分拓扑、解析表面交线、擦边和多刚体运动仍可明确返回未决；普通交互会话本身采用采样接触。 |
| 07 刚–刚施力 | 箱体–有限胶囊、箱面–有限台、凸面滑动与自转、两体反弹和球岛多接触均有实测。[材料反弹](../benchmarks/p65-rigid-material-restitution.md) · [凸面牵引](../benchmarks/p65-rigid-friction-stack.md) | 通用长期堆叠、独立的扭转/滚动阻力材料、稠密多凸体设备求解和双侧体积压力仍有限。 |
| 08 刚–柔施力 | 圆体、凸工具和真实关节端可驱动可变形有限布面并得到反力/力矩，托板有完整拖动和撤板轨迹。[凸工具](../benchmarks/p65-convex-tool-cloth.md) · [完整软布操作](../benchmarks/p65-convex-tool-soft-transfer.md) | 任意机器人/复杂网格同时作用于折叠布、多个工具的连续加载与碰撞覆盖尚需扩展。 |
| 09 布、柔–柔及自接触 | 弹性布步进和有限折角材料持续接触可运行。布料可切换软硬，受力、释放与独立飞行由真实运动验证。[折角站点](../benchmarks/p65-continuous-folded-sites.md) | 参考布双臂转移早已有安装包验收；两种目标软布的独立物理窗口均未达到双臂共享承载门槛。新方案可更换真实材料边并清开腕杆，2° 在 0.92 s 最接近捕捉时，共同指面足印的最小有符号余量为 −5.791 mm，距 +3.5 mm 的捕捉门差 9.291 mm。[选边后的连续受力与资格](../benchmarks/p65-material-boundary-handoff.md)复杂自碰及大网格长期运行仍需检验。 |
| 10 体软体 | 既有四面体变形、StVK 及有限接触入口。本轮重点在三角布和刚体，不计体软体新增验收。[原盘点](phymeo-newton-1.7-functional-inventory.md) | 大变形、复杂材料、体积边界以及多软体与机器人联合的大规模工作流。 |
| 11 GPU、多世界 | TYX 驻留 W128、动态球岛及多个世界真布面有验证。W32 持续受载具有严格相同物理输出的直接对标。[受载实测](../benchmarks/p65-loaded-sphere-planes.md) | 逐束世界编号把 W32 的 800 束光线从 32 次 CUDA 提交合为 1 次。很多精细碰撞窄相仍在 HOST，复杂形状整体耗时有待独立对标。[世界路由](../benchmarks/p65-device-raycast.md) |
| 12 力学传感 | 力与切力可按世界和具体对端组成矩阵；力权重位置、扳手、关节读数与 IMU 可用于反馈。[原生 SensorContact 对照](../benchmarks/p65-world-contact-matrix.md) | 力的生效时刻、求解器原生力和从动量重建的净冲量仍须区别，应用生态尚待扩展。 |
| 13 多求解器组合 | 多域图、点端口、实际关节/布同时受力均能运行。[关节工具](../benchmarks/p65-articulated-convex-cloth.md) | 用户在同一场景按区域自由更换和组合求解器的通用入口尚缺。 |
| 14 在线交互与交换 | Python 客户端逐步发指令并读回，同一会话可在带载运动中换目标，深度与力有实际快照和离线回放。[控制闭环](../benchmarks/p65-metric-depth-camera.md) | 在线交互式 3D 操作、通用 USD 时间序列交换、多种外部诊断/可视化工具尚待补齐。 |
| 15 柔性杆缆 | 现有布–刚性工具接触已可用。[原盘点](phymeo-newton-1.7-functional-inventory.md) | 柔性杆/绳的扭转、弯曲与外接触共同推进仍缺端到端入口。 |
| 16 颗粒/塑性介质与流体 | 现有自由粒子、动态刚球、弹性固体各有自身模型。[原盘点](phymeo-newton-1.7-functional-inventory.md) | MPM 粒–网、塑性流动、雪与流体仍属独立能力族。 |
| 17 视觉 | 米制深度、前向深度、世界法线、分割编号与反投影已接入实际抓放。Newton 原生 CUDA 相机与多起点光束对照均有实际图像。[相机](../benchmarks/p65-metric-depth-camera.md) · [曲面](../benchmarks/p65-ray-queries.md) | 真实 TYX 相机已进入 Panda 载荷回放，387,072 像素中 8 个轮廓标签不同。尚缺材质 RGB、复杂光学和大规模可微传感。[设备测距](../benchmarks/p65-device-raycast.md) |
| 18 可微物理 | 已有局部导数/学习型组件；本轮未验证多步反向物理求解。[原盘点](phymeo-newton-1.7-functional-inventory.md) | 选一个可微控制或参数辨识轨迹，建立从损失到控制/材料参数的实际反传和独立校验。 |
| 19 机器人学习界面 | 多世界可读状态、发动作并局部复位；视觉、载荷、夹持轨迹已有可用信号。[动态世界](../benchmarks/p65-dynamic-sphere-islands.md) | 完整的批量 episode、奖励、终止、随机化与策略训练适配仍未验收。Newton 的 Isaac Lab 对接也是外部生态的一部分。 |

基准系统为本机固定版本。Newton 各求解器对恢复、摩擦、关节、可微性和材料的支持不同。[Newton 1.7 求解器矩阵](https://newton-physics.github.io/newton/1.7.0/solvers/index.html) · [Newton 1.7 碰撞与接触](https://newton-physics.github.io/newton/1.7.0/concepts/collisions.html) · [Newton 1.7 传感器](https://newton-physics.github.io/newton/1.7.0/concepts/sensors.html)
