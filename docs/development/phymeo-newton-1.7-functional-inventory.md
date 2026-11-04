# Phymeo 与 Newton 1.7.0 功能盘点

2026-11-04。Phymeo 基线为集成提交 `c686676`，包含经典 Panda 抓放及 W32 受载验收。对照基线为 Newton 1.7.0 的本机安装源码与同版本官方文档。本次复用已归档实测，未重跑仿真。下文“直接对照”指两边使用对应输入并实际读回指定物理量。只在 Phymeo 跑通、Newton 有相应接口、两边读取不同量，三种情况分别注明。

## 整体判断

按使用者要完成的任务合并为 19 个功能族。前 12 项有可用入口和具体案例，但每项都保留适用范围；第 13、14 项有基础组件，通用工作流程仍需补齐；最后 5 项在当前 Phymeo 产品树与验收记录中未找到对应的端到端入口。功能族大小和工程成本不同，不折算总体完成率。测试工具的开源整理单列后续工程待办。

对机器人操作，最大的功能差距是换机器人、物体、接触材料或网格以后，能否从同一套公共接口连续完成检测、施力、持载、放开和观测。对通用多物理平台，还要计入柔性杆缆、连续介质、设备端覆盖、感知和训练工具链。Newton 的多个后端也各自有边界。官方矩阵列出 Featherstone、SemiImplicit、MuJoCo、XPBD、VBD、Kamino、Style3D 和 ImplicitMPM；VBD 与 Kamino 标为实验性。[Newton 求解器和材料矩阵](https://newton-physics.github.io/newton/1.7.0/solvers/index.html)

### 已有可用入口和实测的 12 项

| 功能族 | Phymeo 已能做什么 | 对照 Newton 后还差什么 |
|---|---|---|
| 01 场景建模与推进 | 形状、刚体、粒子、关节树、状态、控制与材料可构造；SceneStepper 支持候选验证和失败回滚。Panda、布与异构多世界都有公共入口。 | Newton 用 ModelBuilder 组合资产，统一分配 Model、State、Control、Contacts，支持世界复制和全局实体。Phymeo 的参与域、接触绑定和支持的组合仍常由不同程序分别装配。下一关是同一个场景描述换几何或求解配置后，控制和观测代码仍可复用。[Phymeo 运行时](../../src/phymeo/runtime/scene_stepper.py) · [Newton 世界构建](https://newton-physics.github.io/newton/1.7.0/concepts/worlds.html) |
| 02 机器人与场景资产 | 有 URDF、MJCF、选装 OpenUSD 的受限导入，验证过局部闭合 OBJ、单位、惯量、锚点和碰撞体。经典 Panda 专项保留固定手掌、双指和厂家惯量。 | 普通导入器与 Panda 专用兼容路径范围不同。其他机器人的固定链、联动、驱动、闭环、更多网格、材质和 USD schema 仍须逐项打通。Panda 使用厂家碰撞代理，显示 CAD 与物理形状也是不同数据。[Panda 来源与裁剪](../../src/phymeo/builders/panda_assets/SOURCES.md) · [Phymeo USD/资产](../benchmarks/p65-independent-rigid-usd-20261103.md) · [Newton USD](https://newton-physics.github.io/newton/1.7.0/concepts/usd_parsing.html) |
| 03 刚体与关节动力学 | 有自由刚体、固定根和浮根标量树、完整质量耦合、偏心冲量、负载反力；固定方向的点闭环已走通四杆带载。 | Newton 有广义坐标和最大坐标后端，关节与闭环支持因后端而异。Phymeo 的一般球铰、D6、多种等式约束、浮根闭环与接触联合持续承载，广度仍不足。[浮动树](../benchmarks/p65-floating-tree-dynamics.md) · [四杆](../benchmarks/p65-robot-point-closures.md) · [Newton 关节支持](https://newton-physics.github.io/newton/1.7.0/solvers/index.html#joint-support) |
| 04 运动学、IK 和执行器 | 有多站点、多目标、有界 IK，关节力矩、有限推力、柔顺止挡、0/1/2 步延迟 PD 和真实跟踪。 | Newton 有批量多种子 IK、LM/L-BFGS、可组合驱动和限幅，执行器接口仍属实验性。Phymeo 缺同等的批量选解和执行器模型库。关节限位语义也会改变结果，现有留出中硬界停在 0.25 m，Newton 软罚解在 0.40 m。[多目标 IK](../benchmarks/p65-multi-site-ik.md) · [驱动子步](../benchmarks/p65-driven-robot-substeps.md) · [Newton IK](https://newton-physics.github.io/newton/1.7.0/api/newton_ik.html) · [Newton 执行器](https://newton-physics.github.io/newton/1.7.0/concepts/actuators.html) |
| 05 刚–刚离散检测与地形 | 有原语、凸体、凹闭网格、有限面、地形和 SDF 的多组查询，含真实材料点和双侧法向；有 TYX 宽相和 BVH。 | Newton 有更宽的 GPU 形状配对矩阵及网格 BVH/SDF 管线。具体缺口包括普通无限平面×导入箱体（凸包）及更多地形配对。目前能让方块落在有限桌面，不能据此推出所有平面、地形与刚体的组合都已支持。[案例库](../benchmarks/p65-collision-public-catalog.md) · [地形](../benchmarks/p65-rounded-particle-heightfield.md) · [Newton 配对表](https://newton-physics.github.io/newton/1.7.0/concepts/collisions.html#shape-compatibility) |
| 06 连续运动与首碰时间 | 旋转闭网格撞有限胶囊杆身、双铰链球形工具沿椭圆首触已有时间括区，并有两端分离的反例。 | 胶囊本身独立转动、多活动铰链上的非球工具、孤立擦边仍可能不支持或返回未决。Newton 的推测接触按速度扩大当前姿态候选带，官方明确它不求首次撞击时刻。此项按轨迹和几何分别比，Phymeo 的限域时间证据值得保留。[网格–杆身](../benchmarks/p65-rotating-mesh-finite-capsule.md) · [双铰链](../benchmarks/p65-two-hinge-ellipse.md) · [Newton 推测接触](https://newton-physics.github.io/newton/1.7.0/concepts/collisions.html#speculative-contacts-velocity-adapted-gaps) |
| 07 刚–刚接触响应与抓取 | 有点库仑摩擦、完整凸面角点冲量与刚体力矩；Panda 另用有标定的柔顺指面采样进行连续抓放。 | 通用硬接触仍以零恢复为主。持续接触身份、滚动/扭转摩擦、弹跳、双 SDF 分布压力和密集接触约减都影响堆叠和抓持。Newton 的这些能力各有条件，例如双侧体积 SDF 与显式启用，摩擦材料也随求解器变化。[凸面冲量](../benchmarks/p65-rigid-polygon-impulses.md) · [Panda](../benchmarks/p65-panda-rigid-grasp.md) · [Newton 面压力](https://newton-physics.github.io/newton/1.7.0/concepts/collisions.html#hydroelastic-contacts) · [材料支持](https://newton-physics.github.io/newton/1.7.0/solvers/index.html#contact-material-support) |
| 08 刚–柔接触 | 有有限材料三角面与刚体网格、连杆、SDF 的查询，也有布–刚性杆两侧同时受力、受载和分离的物理窗。 | 可检测的形状组合多于能把力、摩擦历史与材料状态共同推进的组合。换复杂工具、跨边换面、折叠布长期承载时还缺统一闭环。Newton 的 VBD/代理耦合提供另一条组合路线，也须指定后端与耦合条件。[网格–布](../benchmarks/p65-closed-mesh-cloth-triangle.md) · [布–刚性杆](../benchmarks/p65-g3-world-contact-loads.md) · [Newton 耦合](https://newton-physics.github.io/newton/1.7.0/concepts/coupling.html) |
| 09 布料、柔–柔及自碰撞 | 拉伸、剪切、弯曲、厚度与折回自接触有案例。顶点–面和边–边已测冲击卸载、动摩擦、滑出有限边界及状态回滚。 | 还缺密集多层折叠中持续静摩擦、任意材料特征切换和分辨率变化后的可靠性。Newton 的布自碰撞对照宜选 VBD；其 XPBD 布没有自碰撞。Style3D 另提供布料路线。[连通布](../benchmarks/p65-connected-cloth-contact.md) · [冲击](../benchmarks/p65-newton-vbd-impact-window.md) · [滑动](../benchmarks/p65-dd-material-sliding.md) · [Newton 矩阵](https://newton-physics.github.io/newton/1.7.0/solvers/index.html#supported-features) |
| 10 四面体体软体 | 有参考构型、集总质量、完整 StVK 内力、材料有效域、积分和点载荷耦合；HOST 与 TYX 要按具体算子区分。 | 与 Newton 的体网格构建及 VBD 等路径相比，材料选择、大变形、表面接触和持续抓捏的组合能力较窄。尚缺同一四面体网格、相同加载卸载的跨引擎轨迹，无法给出体软体精度排名。[Phymeo 体软体验收](../benchmarks/p4-softbody.md) · [跨域实测](../benchmarks/p5-coupling.md) · [Newton 可变形对象](https://newton-physics.github.io/newton/1.7.0/concepts/deformable_objects.html) |
| 11 GPU、多世界与设备驻留 | 有世界隔离、分页、设备状态、异构合批、选择性复位。W32 受载混合世界已验证完整输出和实际 GPU 启动。 | Newton 的 Warp 模型支持同构复制、异构世界和全局实体。Phymeo 中复杂窄相、部分材料寻址、切向求力及验收仍走 HOST，整条 Panda 抓放为 HOST。缺同条件的受载 Newton 吞吐、尾延迟、显存和读回对照。[W32](../benchmarks/p65-mixed-physical-w32.md) · [直接对比](../benchmarks/p65-g2-typed-newton-direct.md) · [Newton 世界](https://newton-physics.github.io/newton/1.7.0/concepts/worlds.html) |
| 12 力学传感和状态观测 | 相对站点、姿态与速度、指定站点 IMU、接触合力和参考点力矩均已有读回。 | 跨求解器需要统一时刻、坐标系和原生/重建含义。Phymeo 的瞬时加速度与另一个后端的步平均量可以不同；Newton 的原生接触线力也不能直接当作参考点力矩。场景级传感器选择和批量观测仍需扩展。[IMU](../benchmarks/p65-robot-site-imu.md) · [扳手](../benchmarks/p65-contact-wrench.md) · [Newton 传感器](https://newton-physics.github.io/newton/1.7.0/concepts/sensors.html) |

### 已有部件，但仍需搭建通用工作流的 2 项

| 功能族 | Phymeo 当前边界 | Newton 对应能力与实际差距 |
|---|---|---|
| 13 多求解器选型和跨域耦合 | 有能力注册、状态事务、点端口、多域响应及指定场景的刚–布–软体交互。耦合时由程序选择参与者、绑定与时序。 | Newton 提供统一求解器接口和按模型划分的实验性 Proxy/ADMM 组合，ADMM 也只支持部分跨域关节行。Phymeo 尚缺让使用者在同一模型上按区域选后端、统一迭代、追踪接口力与失败的通用路径。[Phymeo 耦合图](../../src/phymeo/coupling/domain_graph.py) · [Newton 跨求解器耦合](https://newton-physics.github.io/newton/1.7.0/concepts/coupling.html) |
| 14 实时交互、检查与交换 | 有可移植的 Three.js 离线回放、真实子步状态、力读数和 Python 控制。Panda 页面能切轨迹和视角。 | 缺常规场景通用的在线操控闭环，例如运行时拖动目标、改夹持力、下一物理步生效并读回。Newton 有 ViewerGL、File、USD、Viser、Rerun、可选 RTX 等接口，外部依赖和观测内容各不相同。[Phymeo 回放](../benchmarks/p65-panda-rigid-grasp.md) · [Newton 查看与调试](https://newton-physics.github.io/newton/1.7.0/guide/visualization.html) |

### 当前未见同类端到端入口的 5 项

| 功能族 | Newton 1.7.0 的参照 | Phymeo 所缺的可交付能力 |
|---|---|---|
| 15 柔性杆、缆绳 | 杆单元、缆绳构建与扭转例程，ROD 关节随 VBD 支持。 | 杆中心线、截面取向、弯扭本构和自/外接触共同推进。现有“布–杆”的杆为刚性连杆。验收可用端部扭转传播及绕柱回弹。[Newton 杆与曲线](https://newton-physics.github.io/newton/1.7.0/concepts/deformable_objects.html) · [Phymeo 布–刚性杆](../benchmarks/p65-g3-world-contact-loads.md) |
| 16 颗粒、塑性连续介质与流体 | ImplicitMPM 有颗粒、雪、黏性材料及类水溃坝示例，属于粒子–网格求解路线。 | 未见对应的粒–网步进、材料屈服/流动及双向边界载荷链。现有自由粒子或 StVK 固体不能覆盖倒砂、注液。[Newton MPM 算例](https://github.com/newton-physics/newton/tree/v1.7.0/newton/examples/mpm) |
| 17 相机与视觉观测 | SensorCamera 可产生颜色、深度、法线、形状编号等图像。 | 未见与当前物理状态同步的相机模型及同一子步可读取的图像观测接口。Three.js 截图不能承担深度与标签传感器的合同。[Newton 相机](https://newton-physics.github.io/newton/1.7.0/concepts/sensors.html) |
| 18 可微时间展开 | Newton 的 SemiImplicit、Featherstone 有基础可微示例，用 Warp Tape 对多步损失回传；多种其他求解器不支持。 | TYX 底层自动微分、解析雅可比和自由粒子的学习型候选验证已有基础，未见 Phymeo 公开的多步物理轨迹反向传播和参数辨识验收。首个可比目标是无接触系统的控制或材料参数梯度，非光滑接触单列。[Phymeo 求解器边界](../../src/phymeo/runtime/stepper.py) · [Newton 可微性](https://newton-physics.github.io/newton/1.7.0/solvers/index.html#differentiability) |
| 19 机器人学习环境 | Newton 指向 Isaac Lab 的集成，环境、策略接口和训练框架属于外部生态。 | Phymeo 有物理步和局部复位，但未见面向训练用户的批量观测、动作、奖励/终止、独立重置、域随机化和策略调用的完整适配。与第 11 项分开，世界吞吐通过并不自动提供训练环境。[Newton 与 Isaac Lab](https://newton-physics.github.io/newton/1.7.0/lab/isaac-lab.html) |

## 三组会影响工程决策的读数

- **真实抓放与同姿态几何。** 100 g 方块在 Phymeo 指面摩擦 0.65 时，搬运读数为离台 59.86 mm、横移 77.35 mm，卸载后桌面承担 0.981 N；零指面摩擦时横移 0.057 mm。Newton CPU/CUDA 的 17 个同状态指垫位置最大差 0.250/0.200 µm。相同输入下的抓力、滑落和落点仍需 Newton 独立推进。双臂参考布已真实完成九阶段递交，两种目标软布仍未放台。[Panda 两条轨迹](../benchmarks/p65-panda-rigid-grasp.md) · [布料完整与局部窗口](../benchmarks/p65-newton-20261104-0800.md)
- **布料滑动与强冲击。** 在 2 N 正压的 32 步窗口，Phymeo 和 Newton CUDA 累计滑移为 0.435546875 m 与 0.435546428 m。另一组 112 步冲击中，边–边峰值分别为 5.6571 N 与 6.1447 N。Newton 软自接触力用同版本力核在已接受状态重算，实际动量另由速度读取。这些证据支持对应窗口，不能推断任意网格和材料响应一致。[滑动](../benchmarks/p65-dd-material-sliding.md) · [冲击和测力方法](../benchmarks/p65-newton-vbd-impact-window.md)
- **吞吐的两个分母。** W32 混合受载时，Phymeo 合批为自身逐世界的 1.327、1.477、1.385 倍，绝对吞吐 309.61–322.74 世界子步/s。直接同机、同输出的 W32 自由落体中，Newton/Phymeo 三轮为 0.903、1.305、1.457。复杂受载的跨引擎速度尚未测出。[受载合批](../benchmarks/p65-mixed-physical-w32.md) · [自由球直接对照](../benchmarks/p65-g2-typed-newton-direct.md)

## 按当前机器人与柔体目标排期

1. **先做换场景仍可用的刚体抓持。** 用同一 Panda 和另一种物体，改变质量、起姿、夹持力、步长和材质，两个引擎独立推进。比较离台、最小持载、滑移、失抓阈值、桌面卸载和落点。保留低摩擦失败；单例微米级 FK 继续收紧不会回答受载问题。
2. **并行补接触复用与设备链。** 刚–刚用箱–地面、转动非球工具和稠密堆叠检验配对及持续流形；刚–柔与柔–柔用边/面切换、静止持载、滑出和释放检验接触历史及功；设备线对同输出 W1/4/16/32 的受载窗口分列窄相、D2H、验收及 p95。各线先验独立案例，再接回软布递交。
3. **再扩大资产和操作工作流。** 用第二种机器人和不同工具检验配置复用，再接运行时目标/夹持力编辑、相机和环境适配。杆缆、MPM、可微轨迹分别按新应用需求立项。
4. **将测试工具开源列入后续待优化工作。** 统一对照驱动、公开夹具与许可、精简依赖和外部复现文档。当前保留内部验证及证据作为功能放行依据，工程开源整理后排。[执行计划](../superpowers/plans/2026-11-03-newton-reference-development.md)
