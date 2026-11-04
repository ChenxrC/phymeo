# P6.5 人工续作入口

## 当前入口（2026-11-05）

自由球已能向非指垫机器人连杆施加反力，50 mm 球撞臂的真实响应已在 Phymeo 和 Newton 独立推进。小折角材料片可带压换面，实际离散角动量偏差及同窗细分已记录。W4×8 的重复材料记录重建已消除，合批物理窗较上一版缩短约 13.3%，与同几何的普通逐世界调用耗时接近。[三线结果与下一批判据](../superpowers/plans/2026-11-05-structural-impact-fold-and-host-cost.md) · [首碰与峰值受力回放](../../artifacts/p65-fold-impact-and-batch-20261105-01/panda-link-impact.html)

### 同日上一批

多面接触现可跨世界共用 TYX 设备提交，W4×8 核数从 48 降至 12，墙钟尚未加速。独立布节点已在共面应变下完成受压换面。Panda 在两引擎中都搬运了 40 mm 实心球，50 mm 球的失持和首次未求解结构接触按真实时刻记录。[集成结果与下一批判据](../superpowers/plans/2026-11-05-ragged-contact-and-payloads.md) · [真实球形回放](../../artifacts/p65-ragged-seam-payload-20261105-01/panda-grasp-replay.html)

## 2026-11-04 续作记录

已验证共面平移材料片的带压跨边、旋转偏心长方体抓放，以及 2/8 个真实受载面在 W1/W4 上的成本。当前多面法向层可在 TYX 运行，跨世界接触批入口仍限定每世界单面。下一批优先验证自由节点换面与多面变长站点合批。[验收、数据和下一批判据](../superpowers/plans/2026-11-04-shared-edge-offset-and-density.md) · [偏心抓取与滑落的交互回放](../../artifacts/p65-crossing-rotated-density-20261104-01/panda-grasp-replay.html)

上一批已合入。Newton Featherstone CPU 采用同一材料接触律后，Panda 强夹完成抓放，弱夹仍留台；可动布面支持带预载共转和转滑；32 个混合世界的三个独立同场 CUDA 测量分别加速 4.64×、4.97×、4.45×，逐帧输出一致。下一批从受压跨面、旋转偏心抓取和更多受载面下的剖析继续。[本批证据与下批入口](../superpowers/plans/2026-11-04-contact-model-and-geometry.md)

[受载接触第一批分工与结果](../superpowers/plans/2026-11-04-contact-load-parallel.md)已集成平面–凸包支撑、有限布面静持/转滑、长方体强弱夹持与同姿态几何复核。Phymeo 的 120 g 强夹已放台，Newton Featherstone 同输入原生接触尚未复现整个受载过程。该批 W32 优化前的计量中单姿态最近特征查询占插桩墙钟约 86%。

[与 Newton 1.7.0 的逐项功能盘点](phymeo-newton-1.7-functional-inventory.md)覆盖 19 个功能族，并区分物理实测、同引擎加速与跨引擎对照。测试工具开源已列入后续待优化工作。

**经典 Panda 和自由方块（11月4日，08:00 后的集成批次）。** 打开[抓放例程与实测](../benchmarks/p65-panda-rigid-grasp.md)：从官方经典 Panda URDF 和真实关节惯量出发，100 g 方块在指面摩擦 0.65 时离台 59.9 mm、横移 77.4 mm、放下后桌面承重 0.981 N；零指面摩擦时留在桌面。相同资产的 17 个真实入步姿态已由 Newton CPU/CUDA 核实两侧指面位置。三条独立物理工作线已补[凸面多点冲量](../benchmarks/p65-rigid-polygon-impulses.md)、[旋转网格与有限杆身](../benchmarks/p65-rotating-mesh-finite-capsule.md)、[双铰链连续球路径](../benchmarks/p65-two-hinge-ellipse.md)和[有限布面离边动摩擦](../benchmarks/p65-dd-material-sliding.md)。[同输出 W32 混合受载合批](../benchmarks/p65-mixed-physical-w32.md)已达三轮 1.33–1.48 倍的本引擎逐世界吞吐。

下一批优先复用同一 Panda 资产与两条保存轨迹，扩展抓取起始姿态、质量和夹持力，在 Newton 做相同初态与控制定义的受载对照。通用碰撞线从胶囊自身独立旋转、跨步抓持面、双铰链非球工具和柔性材料的静止持载继续；GPU 线按 W32 的读回、复杂窄相与 HOST 审计分别定位时间。软布递交保留为另一条集成验收。

**08:00 阶段概览。** 刚体单转轴闭网格–球 CCD、柔体顶点–面与边–边完整冲击、浮动树持续承载和真实双端线支撑，以及 W1/W32 与 Newton、W4/W16 真实混合世界的吞吐对照已有实测。源码外 wheel 完成参考布九阶段递交，目标软布两种倾角仍未完成放台。下一批从旋转闭网格–胶囊、跨步材料摩擦、多活动关节链和带载 GPU 的分项计时继续。[可放大概览图与力曲线](../benchmarks/p65-newton-20261104-overview.html) · [数据、开放项与下一批验收](../benchmarks/p65-newton-20261104-0800.md)


- 机器人已从移动参考系观测推进到浮动基座与子树的动力学、固定基座的双边点闭环及带实际加速度的站点 IMU。闭环反力、两端功率和离散投影分别记录。IMU 的瞬时样本和 Newton 步平均加速度按各自时刻对照。[浮动树](../benchmarks/p65-floating-tree-dynamics.md) · [四杆反力](../benchmarks/p65-robot-point-closures.md) · [站点 IMU](../benchmarks/p65-robot-site-imu.md)
- 刚体几何已接通闭网格与球/有限胶囊的实体查询、受限单相对轴的闭网格旋转首触，以及有界曲面之间和曲面与有限布面的材料见证。首触时刻、末态边界净距和整体退让深度为不同观测。[闭网格与圆体](../benchmarks/p65-mesh-round-contact.md) · [转动闭体](../benchmarks/p65-rotating-mesh-solid.md) · [曲面与布](../benchmarks/p65-convex-support-pairs.md)
- GPU 已有独立的世界内配对、常驻自由球/滑轨推进和多世界布–连杆双向受力。相同局部编号的 RR、RD、DD 和空世界也经过同批查询。GPU 启动、必要读回与 HOST 几何复核分别计数。[世界内配对](../benchmarks/p65-world-partition.md) · [G2 常驻推进](../benchmarks/p65-g2-resident-world-step.md) · [G3 双向受力](../benchmarks/p65-g3-world-contact-loads.md) · [混合世界回归](../../tests/scale/test_mixed_world_queries.py)
- 本轮继续检验多目标 IK、凹闭网格与完整柔性三角面、稠密单世界规模及曲面几何产生的真实物理载荷。阶段签收使用多个独立材料/运动反例。两种软布初态递布沿用同一条真实轨迹的接管、桌面支撑和撤爪条件。[应用基线](../benchmarks/p65-soft-handoff-acquisition.md) · [规模计划](../superpowers/plans/2026-11-03-tyx-gpu-multiworld.md)

## 2026-11-03 进展和入口

- 当前可并行的三项是移动参考系传感、非零转动刚体首触试片和 TYX 世界内配对。固定基座止挡、延迟驱动及设备地形候选已经组合验证。一台滑轨因限位在半步时触地，另一台共址滑轨保持净空。G1 已测到 64 个世界，单世界扰动没有串扰。在该档位，完整配对表保留了 32,256 个跨世界过滤行，暖态合批查询慢于逐世界调用。[组合验收](../benchmarks/p65-driven-terrain-worlds.md) · [G1 成本](../benchmarks/p65-g1-world-isolation.md) · [下一批子任务分派与验收](../superpowers/plans/2026-11-03-p65-next-parallel-dispatch.md)

- 前一批接通了固定基座 URDF、MJCF 与可选 USD、双闭合凹网格的平移首触及三类静态曲面，并验证了 USD 工具球与圆柱的组合。[资产与刚体实测](../benchmarks/p65-independent-rigid-usd-20261103.md) · [地形与资产](../benchmarks/p65-robot-terrain-20261103.md)

- Newton 1.7.0 的独立 CPU/CUDA 参考环境和刚–刚 13、刚–柔 11、机器人与资产 11 个案例已实测。首轮记录作为差异基线，新增入口的通过结果单列在上方本批实测中。[逐项物理对照](../benchmarks/newton-1.7-reference-baseline.md) · [三方向后续开发计划](../superpowers/plans/2026-11-03-newton-reference-development.md) · [复现入口](../../tests/reference_newton/README.md)

- 公共碰撞检测已覆盖 15 个案例族、74 个登记构型。RR、RD、DD 各有独立受力运动窗口，三端共 636 个物理步。实际 TYX 与 HOST 的职责和各形状支持范围分别记录。[碰撞验收表](../benchmarks/p65-collision-public-catalog.md) · [三个动力学案例](../benchmarks/p65-collision-m4-loads.md)
- 单爪软布已真实承载、离台并弯曲。原双臂装配的两种初态都未放台。材料点跟踪使 2° 布在 B 单独持有时横移最多 9.13 cm；竖直布的 A/B 交接重叠仍不足。[单爪结果](../benchmarks/p65-single-gripper-bend.md) · [双臂对照](../benchmarks/p65-soft-handoff-acquisition.md)
- 已补通用胶囊–布面载荷、铰接关节反力及同源 Three.js 记录。原装配在初态有布–腕重叠，新掌部与两根移动指杆保留全部结构形状，两种软布初态各 240 对配对均分离。两种原协同轨迹已记录。竖直旧轨迹还检出了同一张布的远处材料面横穿；连通折叠试片已在 HOST、tyx interpret 和 CUDA 实现双向受力，并接入真实弹性布子步。完整放台、最终集成审阅和源码外 wheel 分开验收。[结构接触证据](../benchmarks/p65-cloth-link-response.md) · [当前实施计划](../superpowers/plans/2026-11-02-p65-tyx-collision-pipeline.md)
- 两种原始软布初态均已完成实测，尚未共同承载和放台。常用 `cooperative` 保留了竖直布 B 的双侧夹持及同轨迹 1,639 个子步的布自接触力。可达地址策略单列为 `aperture-guided` 实验模式，它在 2° 接取窗较早丢布。下一项是用已定位的 B 指杆–布面和 B 腕–布面两个入口先验证提前接近的路径。[双臂实测与后续验收](../benchmarks/p65-soft-handoff-acquisition.md#对照结果与下一项验收) · [竖直与 2° 回放](../benchmarks/p65-soft-handoff-acquisition.md#倾斜软布与两种布边路径)


## 2026-11-02 实施记录

- **碰撞组合持续任务已完成** RR04、RR05、RD03、RD05 已提交。全部 15 族、74 个登记构型已经在一个 164 形状模型中分别运行 HOST、tyx interpret 和实际 CUDA。82 条目标配对记录的完整内容逐字节一致。[验收总表与复现](../benchmarks/p65-collision-public-catalog.md)
- **框架目标** 建立基于 tyx 的通用碰撞检测，分别验收刚体–刚体（RR）、刚体–柔体（RD）、柔体–柔体（DD，含不同物体与自碰撞）。递布是必过的最终系统回归。可记录的小穿透和数值误差沿工程预算处理。
- **M0 首批已运行** RR04 转动扫掠、RD01 刚球撞布面、DD02 两边首触已完成独立 HOST 回归及现有接口对照，12 条逐构型观测。现有接口均未返回这三对物体的首次接触区间。[15 个案例的差异表](../benchmarks/p65-general-collision-baseline.md) · [可运行回归](../../tests/collision/test_case_catalog.py)
- **M0 第二批已运行** RR03 已知轴的裁剪可复用，RD02 沿静态胶囊轴找到穿面见证，DD03 确认旧共享顶点过滤会排除异处相交的面。5 个构型/族，共 15 条观测。[新增回归](../../tests/collision/test_contact_regions.py) · [证据与范围](../benchmarks/p65-general-collision-baseline.md)
- **M0 第三批已运行** RR05、RD04、DD05 有 16 条几何观测。表面正距离同时可能出现在凹槽空区和实体内部；已有拓扑函数可消去两个四面体的共享面，材料所有者需另行绑定。初始包含、退化输入和变形体外部首次接触分别记录。[第三批回归](../../tests/collision/test_solid_boundaries.py) · [实测差异](../benchmarks/p65-general-collision-baseline.md)
- **M0 已完成** 收尾批 RR01、RR02、RD03、RD05、DD01、DD04 有 31 条几何观测。四批共覆盖 15 个基础案例、74 个构型。DD04 保留了先触外面、后触斜面的原始失败和修正记录。[逐案例结果](../benchmarks/p65-general-collision-baseline.md) · [收尾批回归](../../tests/collision/test_detection_matrix.py)
- **M1 公共入口已验收** `phymeo.detect` 可在一个混合模型里筛选 RR/RD/DD。11 个形状的 55 对在 HOST、tyx interpret、实际 CUDA 上逐对一致。静态球、胶囊及球–零厚三角面已接入。尚无扫掠窄相的候选公开标为 `unsupported`。[用法和实测](../benchmarks/p65-collision-m1.md)
- **M2-RR 首批已验收** RR01 球–平面/球–球与 RR02 胶囊轴部/端帽已接到公开入口。运动时独立保存首触与末态几何。两端分离的胶囊穿越、擦边与一刻度近失均有回归。[支持范围与实测](../benchmarks/p65-collision-m2-round.md)
- **M2-RD 的 RD01 已验收** 平移球与双侧零厚三角面可返回整步首触和材料权重。支持三个顶点各自仿射运动，五个登记构型和变形留出通过三端核对。[用法和实测](../benchmarks/p65-collision-m2-sphere-face.md)
- **M2-DD 的 DD01/DD02 已验收** 异体零厚三角面对的点–面、边–边首触可返回两侧材料权重和取样时的交集。九个登记构型、双端分离的穿越和孤立擦碰已有三端实测。[范围、结果与复现](../benchmarks/p65-collision-m2-triangle-pairs.md)
- **DD03 同体邻接已验收** 同一布体按材料地址区别合法重合与异处相交。原五个构型和三条运动路径已通过 HOST、tyx interpret 及实际 CUDA 对照。[用法、范围与实测](../benchmarks/p65-collision-m2-cloth-adjacency.md)
- **RD04/DD04 外表面已集成** 原四面体单元共用同一份定向外面。平移球与零厚布可扫过变形体；厚布在当前恒定有理单位法线范围内返回完整层截面、首次及随后各面事件。十个构型、形状对调与独立负控已通过三端验收。[用法与结果](../benchmarks/p65-collision-m2-tetrahedral-surfaces.md)
- **DD05 闭体占据已验收** 不同物体的正向四面体网格按单元及整物体分别返回包含、交体积、内部材料点与外边界首触。跨接缝和纯边–边首触等独立对照通过三端验收。[用法与实测](../benchmarks/p65-collision-m2-closed-volumes.md)
- **RD02 已验收** 平移胶囊的端帽、完整轴身与非退化仿射零厚布面可共用公开查询。五个原构型、轴身–边首触、平行接触段与采样间孤立擦碰已三端核对。[用法与实测](../benchmarks/p65-collision-m2-capsule-face.md)
- **RR03 已验收** 从无序顶点集构造满维凸刚体，静止或平移查询可返回接触区间、并列支撑区域及双方材料点。滑动箱的面–线–分离、纯边–边首触、孤立擦碰和设备分页通过三端验证。[用法与实测](../benchmarks/p65-collision-m2-convex-hulls.md)
- **RR04 已验收** 旋转胶囊与球在原案例、留出、孤立相切和独立球位移上通过三端验证，两端分离时仍能返回途中首触。[结果与范围](../benchmarks/p65-collision-m2-rotating-round.md)
- **RR05 已验收** 凸体与闭合凹网格可区分空槽、内壁首触、交叠以及谁包含谁。左右内壁与两种三角化已通过三端回归。[结果与支持范围](../benchmarks/p65-collision-m2-closed-mesh.md)
- **RD03 已验收** 有限圆边板与布点、整张仿射三角面能返回正面、背面、侧边和角部的见证；整面覆盖时可返回面积，边接触可保留线段。[结果与支持范围](../benchmarks/p65-collision-m2-rounded-plates.md)
- **RD05 已验收** 变形布面进入满维凸刚体时，查询给出首触、进入实体的布面区域、真实边界面与有效法向。箱顶正例、留出及无理数首触均已通过三端回归。[结果与支持范围](../benchmarks/p65-collision-m2-hull-surfaces.md)
- **单爪承载与弯曲已验收** 唯一夹爪在 0.52 s 将原 48 g 软布抬离桌面 46.1 mm，零抓取摩擦时布落回桌面。两种软布初态的完整双臂轨迹另行验收。[实测和回放入口](../benchmarks/p65-single-gripper-bend.md)
- **夹爪已接入有限指垫反馈** 2° 软布的 0.36–0.37 s、竖直软布的 1.39–1.40 s 均在原输入和 10 mm 门槛下通过。两段共有 256 个真实关节及布料子步，仍需单爪及完整双臂验收。[同次力与运动](../benchmarks/p65-collision-m4-pads.md#铰接机器人与软布的实际反馈)
- **M4 有限指垫** 正背面、边角与滑动反力已联测，原软布失败步的 5 个材料观测均在指垫背后。公开接触力已能作用在同一个布面空间点上。[原失败位置与正面载荷对照](../benchmarks/p65-collision-m4-pads.md)
- **M4 三族动力学窗口已通过** 箱堆静载与卸载、球撞布后的反力和反弹、两张布在 2 秒内两次相遇与分离已用 HOST、tyx interpret 和实际 CUDA 验证，九条轨迹共 636 步。后续应用仍需单爪受载折叠及两种软布递布。[实测与边界](../benchmarks/p65-collision-m4-loads.md)
- **M3 已按声明范围验收** SAP、形状 BVH 与网格面级 BVH 保留全部 74 个构型的公开结果；另验收真实转弧与变形面的首触、持续、材料特征切换及拒绝后重试。下一阶段是 M4 的三族独立受力与运动回归。[筛选及成本](../benchmarks/p65-collision-m3-spatial.md) · [连续几何](../benchmarks/p65-collision-m3-curved-faces.md)
- **可复用基础** 球–平面、连杆胶囊、有限棱柱、线性特征扫掠和限定的壳几何可作起点。已有布内力、机器人树、双向点载荷、统一子步及回滚负责消费检测输出。
- **已完成的参考布案例** 带关节限位与六个连杆碰撞体的双臂交接。独立碰撞/限位/回滚检查已通过。3.62 s 的同次轨迹完成递布，B 独立横移 10.36 cm，末帧桌面支撑 0.47104 N；源码目录外的 wheel 逐位复现 B 接管步。[结果](../benchmarks/p65-robot-contact-results.json)及[本节点计划](../superpowers/plans/2026-11-02-robot-limits-contact.md)。
- **布料软硬实测（2026-11-02）** 同一 21 节点网格、48 g、重力和支承下，铰链柔量 20、200、2000 在 80 ms 的下垂分别为 0.72、8.22、18.49 mm。竖直软布已由 A 递给 B，在 1.39 s 的搬运步被 B 右指垫拒绝。2° 的试验在 A 左指垫产生另一条可重复的边缘反例。[参数与证据](../benchmarks/p65-soft-fabric-results.json)
- **应用回归** 上述 0.36 s 和 1.39 s 失败步保留在软布回归集中。完整软布递布需用原控制、两种初态、同次状态与力的 Three.js 回放验收。通用模块的放行先看各自的检测案例。
- **判停** 同一分支连续两轮没有新增能力、最小失败或可复现对照，先定位候选、特征、时间路径或响应中的具体差异，再决定改哪个模块。
- **工作纪律**：父分支独占共享接口；执行串行且每次 fresh guard；不重跑同源已绿组，不恢复周期任务，不修改依赖 pin，不碰已归档证据及 Temp junction。

[公共接口与运行示例](../../README.md#布料接触与搬运案例) · [Newton 对标与交付计划](../superpowers/plans/2026-10-31-p65-delivery-replan.md#2026-11-02-与-newton-的差距和执行顺序) · [双臂案例要求](../superpowers/plans/2026-10-27-dual-arm-cloth-handoff-milestone.md)

## 历史研究快照（保留追溯，不作为当前任务队列）

状态（2026-11-01）　用户已选定可标定接触层研究路线，已授权的独立算术验证已完成。07:00 限时续作已删除。N1 完成，当前 N2，普通质量联合物理尚未冻结。

- **交付物**　A 取布、B 接稳、A 松开、B 放台并撤离。同次仿真驱动可暂停、逐帧查看接触力的 Three.js 回放。[原始验收](../superpowers/plans/2026-10-27-dual-arm-cloth-handoff-milestone.md)
- **可复用基线**　有限夹爪、独立 B2/R7 材料、限域的自接触响应与几何已集成。严格二进端点的倾斜相触限制及旧 R7 范围反例按原版本保留。[框架范围](../superpowers/plans/2026-10-30-p65-material-self-gripper-integration.md) · [端点反例](p65-b2-sc2-first-touch.md)
- **新实证**　独立固定十向量在解释器及 CUDA 各 10/10，实际中间字和返回字一致，CUDA 启动一次。次正规、假零、上溢各有原始记录。没有联合回放或完整材料行的通过结论。[逐字收据](../benchmarks/p65-fp32-reduction-verification.json)
- **当前研究**　保留原几何半厚度，以单独 δ、面积刚度 k 和阻尼 d 描述有限接触层。零间隙有有限初储能。SC-L1 设计试片五个留出点和六个一维对照均通过。支撑载荷全窗至少为相对外载的 93.71%，冲击硬核余量大于 0.5760 mm。[候选定律与标定](p65-contact-layer-calibration.md) · [材料结果](../benchmarks/p65-contact-layer-screen-results.json)
- **步内实证**　SC-L1 的两种工况、三质量、三步长共 336 步满足峰值、冲量、时间矩、硬核间隙和功的数值判据。0.5 kg 冲击的卸载尾段有 8 个原时间步，发布的速度与间隙无法同时由非负压力产生，其中 7 步冲量恰为零。这排除了给当前端点单换插值的修复。[完整数据和反例](../benchmarks/p65-contact-layer-intrastep-results.json) · [逐步观测](../benchmarks/p65-contact-layer-intrastep-steps.csv)
- **共同发布**　固定的 8 个卸载反例各自从原入口重新推进，再由新进程接续一整步。16 段、822 个内部小段的非负力均同时给出读回速度和间隙，核对使用 FP64 主字加有理余量，运动包最大 217 字节。去掉余量时，同一压力在全部 16 段失去这种一致性。[逐例读回与负控](../benchmarks/p65-contact-layer-publication-results.json)
- **入层与峰值**　0.5 kg、1/64 s 的线性压力在峰值幅值上过关，但材料冲量与功超限。预登记的七阶对照通过前两个原时间步及跨进程读回，7.4375 ms 入层，11.15957–11.16007 ms 达到 17.13159482–17.13159489 N。续步为 504/512 个系数计数，算术开销单列。[两种更新的完整结果](../benchmarks/p65-contact-layer-entry-results.json)
- **持续支撑**　0.5 kg、1/64 s、0.125 N 外载的原支撑窗口已连续验证。八步七次跨进程续接，全窗最低支撑力约 0.124939464 N，末间隙 0.108072 mm。初始储能 0.001 J、外载有符号功 −13.509 µJ 和耗散独立核对。[逐步与能量结果](../benchmarks/p65-contact-layer-support-results.json)
- **连续冲击**　0.5 kg、1/64 s 的冲击从同一层外初态贯通 0.125 s，八步七次真实读回。7.4375 ms 入层，11.160 ms 左右达到 17.131595 N，25.1281 ms 转为开离。末压力约 2.89×10⁻¹⁵ N，仍有 0.596312 mm 层压缩。卸载曲线段界有微小上跳，最大 4.20×10⁻¹⁵ N。[全窗相位、能量和状态链](../benchmarks/p65-contact-layer-impact-window-results.json)
- **四节点实测**　原弯曲布片和 2 kg 平台完成一个 HOST 给定载荷步。布与平台的竖直净冲量误差为 4.34×10⁻¹⁹ N·s。载荷的时间矩在现有公开重放入口丢失，平台位移误差为 8.064 µm，限额为 0.010 µm。厚接触构造器拒绝弯曲材料，平台空间角冲量尚无实测通道。两面通用扫掠均未决，已对本例返回的仿射路径另证硬核安全。[诊断及下一次判别](p65-contact-layer-calibration.md#四节点布片和有限质量平台的一次载荷读回) · [原始量与独立核对](../benchmarks/p65-b2-joint-readback-results.json)
- **同冲量对照**　同一原始压力和倒序压力给平台相同末速度，末位移分别为 −14.0251624 和 −8.0637115 µm。移点使已交付 y 角冲量相差 0.000235614806 N·m·s，导轨独立约束反力仍不可读。三组候选的六个扫掠全部证实穿透，三组均拒绝续步。[三组审计](../benchmarks/p65-paired-load-history-results.json)
- **重力与六点联立**　三组原重力时间矩对照的质心均通过。倒序的质心间隙为 +0.869 µm，局部仍穿透 2.289 µm。已转入六点联立求力并送回真实参与者。最小末端硬间隙为 +11.229 µm，实际仿射弦段有独立安全证明。通用扫掠仍未决。至少一处定律冲量不一致，末间隙相对参考最多差 0.472205 µm。四个候选均未续步。[逐点结果](../benchmarks/p65-gravity-six-owner-results.json)
- **共同路径**　独立 HOST 研究程序从静止入口及真实磁盘非零速度分别接受一个 1/64 s 步，累计至 31.25 ms。在实际曲线上证明硬核安全，每点的冲量与时间矩、逐节点材料作用、原始能量和角动量均通过。首步末间隙与独立参考相差至多 4.09×10⁻¹² m。三个实际事务反例回滚，同一积分核心还通过了另一种双质量系统的解析对照。[通用框架](p65-common-motion-framework.md) · [两步证书](../benchmarks/p65-common-motion-results.json)
- **四节点 SC1 全窗**　从已验检查点完成余下六步，八步连续至 125 ms，七处衔接逐字节一致。全窗法向合力至少 0.12493478 N，最终六点硬间隙为 0.099112–0.112960 mm。非抵消正能量缺口 ≤6.82×10⁻¹⁴ J，成对角动量误差上界和 ≤1.91×10⁻⁹ N·m·s。[八步结果与五个负控](../benchmarks/p65-common-motion-window-results.json)
- **下一动作**　在 N2 处理原四节点 SC2 的层外入射，隔离最早入层的材料地址及整段硬核，绑定零载、出生、峰值与卸载。生产 B2/R7、原生与一般机器人仍开放。[框架与后续判据](p65-common-motion-framework.md)
- **执行边界**　独立算术及本次一维筛选均已结束，不重复同源已绿组。N3/N4、远程 CI 均未启动。
