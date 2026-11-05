# Phymeo 与 Newton 的较大功能差异（2026-11-06）

盘点到第四批产品提交 `8c4a7db568265a0d587ff38d9c6fb9a2499e3f5a`。Newton 参照为本机比较所用 1.7.0；本轮查阅同版本官方文档，未重新跑旧仿真。本文是功能能力与工程完整度盘点，不是精度或速度排名，也不按测试数给完成百分比。

## 核心判断

Phymeo 已有机器人、刚体、布、体软体、RR/RD/DD 检测、传感和多世界的可用入口。当前最大差距是这些部件组合成通用场景后，能否在更换物体、关节、材料和接触特征时持续运行，而不需要再写一个案例专用适配器。刚体抓放通过是必要案例，不能代替堆叠、推滑、滚转、软体抓捏和软布双臂递交。

“支持有限”表示已有模块和实测，但未覆盖该族所有输入；“缺完整实现”表示在当前产品入口和验收中未见相应端到端链路。Newton 本身也分后端：VBD/Kamino 及跨求解器耦合为实验性，XPBD 布没有自碰撞，基础可微性仅部分后端具备，不能把所有勾选当成一套万能求解器。[官方后端矩阵](https://newton-physics.github.io/newton/1.7.0/solvers/index.html)

## 功能族概览

| 大模块 | Phymeo 当前能力 | 还实际缺什么 | 判断/用户顺序 |
|---|---|---|---|
| 场景与机器人资产 | URDF、受限 MJCF/USD、Panda、自由物体/有限台、固定/浮根标量关节树、mimic 传动图、PD 与 IK；公共 Python 接口可在线改变输入 | 通用多机器人/多参与域统一装配；更多球铰/D6/闭环与接触组合；资产和驱动模型的完整兼容 | 支持有限，顺序1 |
| 刚–刚碰撞检测 | 原语/凸体/凹闭网格、有限面/边/背面、地形/SDF 多组检测，BVH/TYX 宽相及部分旋转连续路径 | 形状×表示×运动的完整组合覆盖；一般旋转复杂网格与多关节非球工具的连续检测；更多设备窄相 | 支持有限，顺序2 |
| GPU 与规模 | TYX 世界隔离、驻留状态、异构合批、选择性复位、受载多世界实测；固定几何复用有局部收益 | 复杂窄相、寻址、新持续切向及若干审核仍在 HOST；缺通用机器人–刚柔全流程设备驻留和充分跨引擎受载吞吐对照 | 支持有限，顺序3 |
| 可微仿真与训练工作流 | 有解析 Jacobian、TYX 底层能力；有多世界状态与复位基础 | 引擎公开多步反传/参数辨识；稳定的向量化 observation/action/reset 接口与训练生态适配，不等于只有自动微分算子 | 缺端到端工作流，顺序4 |
| 传感、交互与工具 | 米制深度/法线/形状 ID、TYX 射线、IMU、接触扳手；在线 Python 控制和同源 Three.js 回放 | 通用交互查看器与场景编辑、RGB/光照/材质和更广相机模型、统一外部生态适配 | 基础已用，顺序5 |
| 统一求解器/跨域编排 | 能力注册、事务、点端口、多域响应，指定场景可共同推进 | 同一模型按区域选择后端、自动组合和统一处理接口约束/失败的公共流程；不能把现有 scene 类集合当成已完整实现 | 缺通用组合层，顺序6 |
| 刚–柔、柔–柔与布 | 有真实双向受载、边/面冲击、换面/滑出、拉伸剪切弯曲与自接触；已有单爪与参考布递交 | 持续静摩擦在任意可变形接触上的接入、密集多层折叠、材料地址变化、换网格后的稳健性；原目标软布两初态双臂放台仍未通过 | 支持有限，顺序7 |
| 体软体与材料 | 四面体 StVK、质量、内力、积分与点载荷耦合 | 更广材料库、大变形及一般表面接触/抓捏的组合工作流与独立多案例验收 | 支持有限，顺序7 |
| 柔性杆与缆绳 | 现有“布–杆”的杆主要是刚性连杆 | 弯扭杆/缆本构、截面取向、自/外接触及动力学共同推进 | 缺完整物理族，顺序8 |
| MPM/颗粒塑性/流体 | 自由粒子与弹性体不是该能力 | 粒子–网格步进、塑性/流动材料、双向边界载荷和对应案例库 | 缺完整物理族，顺序8 |
| 持续接触与摩擦响应 | 有法向层、冲量、双向反力、Panda 抓放；本批完整跨站点 W、弹簧历史、实际功与运动共同发布 | 新模式仍需显式站点，只接 HOST 刚体/普通移动关节；旋转/铰接/mimic、自动跨特征继承、滚动/扭转摩擦、密集堆叠和通用面压力尚需补齐 | 跨项基础能力，随相关场景补齐 |

以上顺序采用用户2026-11-06的新安排，覆盖此前P0/P1建议。持续摩擦保留为相关场景/检测的基础能力，不单独置于整个路线之前。前三项按 [子 agent 分工计划](../superpowers/plans/2026-11-06-priority-top-three-parallel.md) 优先推进；后五项不在本轮启动。

## 参照依据与容易混淆的边界

Newton 的 GPU 碰撞体系按原语/凸体、网格 BVH/SDF 分流，并有有条件的 hydroelastic 面压力和约减；它的“全配对”仍取决于实际表示和配置。Phymeo 目前已有碰撞管线，缺的是完整覆盖和跨域复用。Newton speculative contact 也不是一般首碰时间求解，因此 Phymeo 受限 CCD 的时间括区应单列保留。[Newton 碰撞](https://newton-physics.github.io/newton/1.7.0/concepts/collisions.html)

Newton 的 rod/cloth/tet 有统一对象构建标识；MPM 是另一求解路线。Phymeo 的柔性三角面和四面体不是空白，但杆缆和连续介质不能由“已有粒子”推定已实现。[变形对象](https://newton-physics.github.io/newton/1.7.0/concepts/deformable_objects.html) · [求解器 API](https://newton-physics.github.io/newton/1.7.0/api/newton_solvers.html)

Newton 可在一个模型内按参与域划分后端并交换约束/力，相关组合接口为实验性；这与 Phymeo 已有共享点端口是不同完成层次。[跨求解器框架](https://newton-physics.github.io/newton/1.7.0/concepts/coupling.html)

视觉差距已从“没有物理相机”缩小到观测广度和工具生态。Newton 相机还提供颜色/材质及多种标定射线，查看器有实时、记录和外部集成等选项。[相机与传感](https://newton-physics.github.io/newton/1.7.0/concepts/sensors.html) · [查看器](https://newton-physics.github.io/newton/1.7.0/guide/visualization.html)

### 当前仓库证据入口

- [第四批场景摩擦与几何复用](../benchmarks/p65-scene-tangent-batch4.md)：本批真实推进、失败及 wheel；[第三批核心](../benchmarks/p65-coupled-tangent-loaded-batch3.md)保留旧 mimic 失败。
- [机器人交互](../benchmarks/p65-interactive-robot-payload.md)、[机械 mimic](../benchmarks/p65-urdf-mechanical-mimic.md)、[传动图](../benchmarks/p65-urdf-transmission-graphs.md)、[浮动树](../benchmarks/p65-floating-tree-dynamics.md)、[点闭环](../benchmarks/p65-robot-point-closures.md)。
- [碰撞案例库](../benchmarks/p65-collision-public-catalog.md)、[原独立 Newton 基线](../benchmarks/newton-1.7-reference-baseline.md)；旧案例结果仅在原范围复用。
- [软布原目标](../benchmarks/p65-soft-handoff-acquisition.md)、[连通布](../benchmarks/p65-connected-cloth-contact.md)、[体软体](../benchmarks/p4-softbody.md)。
- [米制相机](../benchmarks/p65-metric-depth-camera.md)、[TYX 射线](../benchmarks/p65-device-raycast.md)、[实际 IMU](../benchmarks/p65-robot-site-imu.md)。
- [GPU 驻留](../benchmarks/p65-g2-resident-world-step.md)、[混合受载](../benchmarks/p65-mixed-physical-w32.md)。

旧 [11月4日逐项盘点](phymeo-newton-1.7-functional-inventory.md)作为历史保留；其中相机、在线控制、mimic 图和平面–凸包等缺项已被后续工作部分补齐，不能作为当前缺失清单直接使用。

## 后续顺序与成功判据

1. **按用户顺序推进。** 场景与机器人 → 碰撞检测 → GPU与多场景 → 可微仿真与训练工具 → 传感与交互 → 跨域求解编排 → 布与体软体 → 杆缆、MPM/流体。
2. **前三项先做独立切片。** 多机器人同场共同推动自由物体；胶囊自身转动与闭网格；选择世界观测的成本与批量读回。分工、文件、案例、难度和执行门见 [实施计划](../superpowers/plans/2026-11-06-priority-top-three-parallel.md)。
3. **编码并行、验证串行。** 父维护共享接口，原三作者分别隔离实现；就绪任务发生资源冲突按上述顺序排队。每条线保留独立多案例，组合验收不只看Panda或递布。
4. **后续方向保留真实缺口。** 不因排后而将软布、持续摩擦或通用耦合写成完成；可微训练提前到第4项，杆缆/MPM放在第8项。测试工具开源仍按用户要求留待后续优化。

每批至少交付一项可复用能力或一个缩小问题范围的可复现失败。若连续两轮只增加报告/证明，没有新增可运行案例或定位更小的失败，应调整模型/接口/任务范围；不能加长原窗口、放宽断言或反复重跑已绿组冒进展。性能优先做同输出本引擎对照，跨Newton优越性必须另有同条件实测。
