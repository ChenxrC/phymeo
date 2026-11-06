# 通用碰撞、转动逐面候选与设备状态直供：2026-11-07 08:00 前开发测试计划

> **For agentic workers:** 统一启动后按 writing-plans 对应的 subagent-driven-development 或 executing-plans 工作流逐任务执行；用户规定的原 agents、共享文件所有权及串行验证优先。**本文不是启动指令。**
>
> **状态：PLANNED_NOT_STARTED。用户还会补充其他工作，等待用户统一启动。当前没有开发任务或项目执行槽。**
>
> **截止：北京时间 2026-11-07 08:00（今天），不是 11 月 8 日。** 用户确认启动时重新读取时间；截止不随启动推迟而顺延。

**Goal:** 三线补齐通用凸碰撞/双侧响应、转动路径逐面候选和设备接受态直供宽相；以多案例组合矩阵验收，结束按每个场景复制一条碰撞和施力链的推进方式。

**Architecture:** 保留已有解析几何、凸支持映射、凸包接触多边形、网格与柔性有限特征查询。几何层产生有材料身份的接触点/线/面，接触律逐站点评估，所有者适配层将同一双侧载荷传至真实刚体、关节或浮动基座。形状组合决定算法路由，不能决定一套独立积分器。

**Tech Stack:** 现有 Python/NumPy HOST 几何与场景、TYX interpret/CUDA 的已有职责、现有 pytest 与 Newton 1.7.0 参考环境；不新增依赖或改变 pin。

**Spec:** 本文“统一设计与边界”和“验收矩阵”为这次新增合同草案；沿用[公共碰撞目录](../../benchmarks/p65-collision-public-catalog.md)、[凸支持查询](../../benchmarks/p65-convex-support-pairs.md)、[有限盒–杆响应](../../benchmarks/p65-box-capsule-response.md)、[凸包多点冲量](../../benchmarks/p65-rigid-polygon-impulses.md)。用户统一启动前允许继续合并其他工作要求。

## 1. 基线和本轮范围

父分支 codex/p65-parallel-modules，计划编写基线 0c6e9af276000e650815644fd4ca6fd92d436742；[第七批](../../benchmarks/p65-top-three-batch7.md)已经结束。现状不是“只有球碰撞”：公共几何已有更广组合，但机器人直接互碰适配器仍要求至少一端为球。

| 已有模块 | 这次如何复用 | 不能据此宣称 |
|---|---|---|
| geometry/_convex_support.py | 现有凸支持映射/GJK及有界穿透结果；核查实际公开路由 | 内部支持某 shape 就有完整多点接触或动态响应 |
| geometry/collision_witness.py、collision_pipeline.py | CollisionWitness、ConvexContactPatch、CollisionPair；材料点、原 owner/key、运动观测 | 任意负 gap 或一个支持点就能代表平面承重 |
| solvers/box_link_contact.py | box_capsule_witnesses 的有限轴–OBB最近集与归一权重 | 已覆盖任意凸包–胶囊、轴深入盒内或全部 CCD |
| 凸包多点冲量模块 | 完整接触多边形及双方材料地址 | 硬冲量已自动等同当前可标定的柔顺接触层 |
| runtime/_robot_robot_contacts.py | 双侧 COM wrench、真实 J 转置、原子场景发布 | 非球组合已能通过构造/采样并得到真实响应 |

本夜A线交付是 **一套凸实体接触适配链**。球 S、有限胶囊 C、OBB B、一般凸多面体 H 共 10 个无序组合是整族目标；OBB 在通用几何中是凸包的一种，单列只是为了验证解析快路与通路一致。一般 H 必须含非长方体留出，不能把 hull 全部重新解释成 box。

凹网格、地形、刚–柔/柔–柔要进入同一全局能力表，复用其已有检测记录；这一个夜间模块不承诺同时重写它们的检测和响应。圆柱/圆锥/椭球保留已有支持映射与检测范围，本夜记录它们缺少的 manifold/响应能力，不以退化代理或隐式凸化冒充支持。

## 2. 统一设计与边界

### 2.1 算法路由大表

此表是**开发路由**，不是当前全绿能力表。“—”是对称项，不是缺失。

| 第一类/第二类 | 球 | 胶囊 | 盒子 | 其他凸体 | 凹三角网格 | 平面/地形 | 柔性有限表面 |
|---|---|---|---|---|---|---|---|
| 球 | C | C | C | C | M | P | D |
| 胶囊 | — | C | C | C | M | P | D |
| 盒子 | — | — | C | C | M | P | D |
| 其他凸体 | — | — | — | C | M | P | D |
| 凹三角网格 | — | — | — | — | M | P | D |
| 平面/地形 | — | — | — | — | — | 静态对过滤/显式策略 | D |
| 柔性有限表面 | — | — | — | — | — | — | D |

- **C：凸算法族。** 优先复用已验解析快路，通用 support-map 路径覆盖其余凸组合；多面体接触 patch 复用现有有限特征/裁剪。GJK 负责距离/相交，穿透模块与 manifold 是另外的职责。
- **M：网格族。** BVH 候选、有限面/边窄相与闭实体包含；不能把凹体当整个凸包。网格–网格还须区分曲面交触与闭体占据。
- **P：环境族。** 无限平面解析、有限地形单元/BVH或已支持距离表示；不能用无限平面掩盖有限边缘。
- **D：变形材料族。** 有限面、边、厚度、材料点及 Jacobian/节点散回；整张布不是一个凸体。
- SDF 是可选几何表示，不额外伪装成一种互斥形状。粒子/体软体也按各自表示另登记。
- Newton 参照依据为[碰撞概念与兼容表](https://newton-physics.github.io/newton/1.7.0/concepts/collisions.html)：凸体 GJK/MPR、网格 BVH/SDF、几何与求解器分工。借鉴模块边界；不为了算法名一致替换本仓库已通过实现，也不把 Newton 当连续首触的唯一真值。

### 2.2 每个组合的能力必须分列

能力记录包含：shape pair、算法族、公开入口、静态距离/包含、点线面 manifold、平移 CCD、旋转 CCD、变形路径、双侧响应、刚体/固定根/浮根/柔体所有者、HOST/interpret/CUDA、误差/资源边界、测试及证据链接。

状态只用 verified、implemented_unverified、missing、unsupported_domain、not_applicable；每一个 verified 必须对应特定 source revision 与证据。候选在 GPU 上生成、窄相仍 HOST 的路线标 hybrid，不能把整条链标 GPU。登记旧验收可引用原记录，不重新跑整库取一次绿灯。

### 2.3 接触结果与载荷合同

1. 复用 CollisionPair / CollisionWitness / ConvexContactPatch。新增薄适配层统一 point、segment、polygon；保留两侧 owner/key、材料地址、世界表面点、法向、局部 gap、观测时刻、来源和有效域。
2. gap/normal 约定保持原 first→second；区分边界净距、整体退让深度和用于弹性层的局部法向间隙。只有穿透界而没有可绑定局部见证时，返回具名 unresolved，不能编造接触点。
3. 一个共同载荷点施加 +F/-F，各体用真实 COM 力臂；站点速度使用同一载荷点的真实线/角速度，广义力仍由完整 J 转置生成。两侧表面点用于材料绑定和 gap，不能与载荷点混称。
4. 复用 RobotContactMaterial / sample_loads。首片法向参数继续是每个 shape-pair 的 N/m、Ns/m；每个 patch 的站点权重非负且总和为 1，不能因多取几个点将整对刚度倍增。线最近集端点各 1/2；多边形用面积加权重心三角分解得到顶点权重，保持合力一阶矩。不是连续体压力 Pa 模型。
5. 每站点用自己的 gap、速度和原接触律；接触力的正负、阻尼、摩擦截断不改。用面积权重只是离散接触层选择，不能由此保证面积细分后的所有非线性摩擦完全相同。
6. 所有者映射与形状无关：自由刚体→COM wrench；固定/浮动机器人→真实 body wrench、完整广义载荷及固定约束反力；柔体保留材料/节点映射边界。柔体响应扩展单列后续任务，本夜不把刚体适配器声称为柔体求解器。
7. 一步真实入态采样、一个实际积分、一次共同发布；不重置速度、不复制 scene、不二次加力、不用对象名称/坐标/时间分支识别测试。
8. 新 manifold 路线先显式启用；旧球快路/原默认结果保持。只有受影响兼容项与误差检查通过才允许扩大公开支持；不删 guard 代替补能力。

## 3. A线开发节点、难度与所有权

全部复选框在统一启动后才可勾选。下面的 NEW 路径是计划创建，不表示文件已经存在；执行前先确认不存在，避免覆盖其他工作。

| 节点 | 难度 | 交付 | 依赖 |
|---|---|---|---|
| K0 能力登记与案例合同 | 中，约 35 分钟 | 全局路由表 + 10 组合/各层现状 | 启动后的新基线检查 |
| K1 通用接触 patch 适配 | 高，约 100–140 分钟 | 几何查询转为统一有限站点 | K0 |
| K2 所有者载荷映射 | 高，约 90–120 分钟 | 相同接触输出驱动真实双方 | K0 的接口，可与 K1 编码并行 |
| K3 场景接缝与事务 | 高，约 60 分钟 | 非球普通/浮根/自由刚体共用链 | K1、K2 |
| K4 分层验收与参考对照 | 中高，约 70–100 分钟 | HOST + 实际 TYX 范围 + Newton 留出 | 按依赖串行穿插 |
| K5 收尾、wheel、差异报告 | 中，预留至少 45 分钟 | 可用版本/未完成范围/08:00 概览 | 已验证产品冻结 |

以上保留A线工作量估计，不是三线同时全量完成的承诺。新的B/C线已纳入下文；全夜共享验证预算见第6节，代码就绪不等于已通过。

### K0：一张可查询的能力表和预声明多案例库

**文件：** NEW src/phymeo/geometry/collision_capabilities.py、tests/collision/test_collision_capabilities.py、tests/collision/convex_family_cases.py、docs/development/collision-capability-matrix.md。读取原 case_catalog.py 和 accepted benchmarks；不改旧结果。

**接口：** frozen CollisionCapability(first_kind: str, second_kind: str, operation: str, owner_mode: str, backend: str, status: str, route: str, domain: str, evidence: tuple[str, ...])；collision_capabilities() -> tuple[CollisionCapability, ...]。这是一份事实登记，不直接替换 runtime dispatch。两边共享 canonical pair 排序，不能靠两个独立方向的人工布尔表长期维持一致。

- [ ] 登记实际公开入口、内部 kernel、manifold 和响应的区别；旧 verified 绑定已有 commit/测试记录。
- [ ] 先写登记一致性测试：反序等价、无重复、verified 必须有证据、hybrid 不可标全设备；未知组合给显式状态。
- [ ] 编写 10 组合案例生成器，输入只有形状/姿态/材料/所有者，不含场景名称识别逻辑。每对至少分离、相触、浅入层三个构型。
- [ ] 冻结下节的独立物理 oracle、h 和容差，再在父串行槽做一次有意义的 RED；缺少公开行为的失败和坏测试本身分别记录。
- [ ] 实现登记、通过其专项并小提交。登记工作超过 35 分钟时停止扩写历史，转入当前 10 组合的代码缺口。

### K1：点/线/面共用接触 patch

**文件：** NEW src/phymeo/geometry/contact_patch_adapter.py、tests/collision/test_contact_patch_adapter.py。父必要修改 collision_pipeline.py / collision_witness.py；复用 box_capsule_witnesses、现有 HullPairQuery 和 support-map 路由。

**接口草案：** frozen ContactSite(witness: CollisionWitness, load_point: tuple[float, float, float], weight: float)；frozen ContactPatchSample(sites: tuple[ContactSite, ...], dimension: int, source: str)；contact_patch_sample(pair: CollisionPair, *, time: float) -> ContactPatchSample。pair 的材料/shape身份必须保留；time 必须绑定 pair 实际保存的姿态/见证时刻，不允许任意取 end witness 冒作首碰。

- [ ] 先实现点、有限最近线段、多边形的归一权重测试与坏材料地址/负权重/错误时刻负控。
- [ ] 直接适配已有解析结果与完整 patch，不重写胶囊–盒子算法；通用 H 与 C–H 真正缺失的部分落到共同 support/feature 层。
- [ ] 以共同采样结构返回局部 gap 与双方材料点；多法向并列必须显式处理或 unresolved，不任挑第一法向。
- [ ] 深嵌入/重合/退化不能拿球或 AABB 代替成功；保存最小复现和真实适用域。一般非盒凸包是必需留出。
- [ ] 通过点/线/面的独立 oracle、换序/刚体变换/细分不增刚度专项后小提交。不能以只返回 boolean 的通路关闭此任务。

### K2：与形状解耦的共同双侧载荷

**文件：** NEW src/phymeo/solvers/contact_patch_loads.py、tests/test_contact_patch_loads.py。读取 sampled_box_contact.py、RobotContactContribution；既有共享文件由父接入。

**接口草案：** sample_patch_loads(sample: ContactPatchSample, law: RobotContactMaterial, first_site_velocity: np.ndarray, second_site_velocity: np.ndarray) -> dict[str, np.ndarray]。两速度均为 (n,3)；返回每站点 first_force、second_force，形状也是 (n,3)，均包含 weight。几何 adapter 不调用接触律；此函数不积分、不发布、不管理 robot ID。

- [ ] 使用 K0 的固定解析向量先写 RED，包含 1 点、2 点线、4 点面的相同总刚度。
- [ ] 按现有 sample_loads 原法向/切向语义逐站点求力，权重只乘一次；保留原真实速度阻尼和摩擦上限。
- [ ] 检查共点双侧合力/关于任意原点总力矩、实际功率以及全部有限值；坏输入在任何发布前拒绝。
- [ ] 力→真实 COM wrench/J 转置的映射复用旧代码。由单独测试将相同 sample 分别作用于自由体、固定树、浮树，证明几何层不认识所有者。
- [ ] 通过新核测试后小提交；不借这一层另引入时间积分/新的摩擦历史系统。

### K3：接入原共同场景

**文件：** 父修改 runtime/_robot_robot_contacts.py、_robot_scene_contacts.py，以及确有需要的 multi_robot_scene.py / contact_scene.py；NEW tests/test_convex_robot_contacts.py、examples/convex_contact_family.py。原 participant 的积分和发布协议不复制。

- [ ] 删除“至少一端球”门之前，替换 SphereLinkContacts 专属编译/采样依赖；一般 hull 保留原顶点及材料身份，不能强制 box。
- [ ] 保留显式 owner-pair 材料/排除、world隔离、shape key映射与原 body/广义速度维度；仅对已验能力扩大支持。
- [ ] 将新 patch 载荷接入真实双侧 COM/J；固定–固定（两棵有关节树）、固定–浮、浮–浮、机器人–自由刚体分别运行。
- [ ] 原同入态事务继续覆盖第二方求解/坏几何/外载异常：所有 q/v/接触事件/历史/力记录一致回滚，修复输入重试等于独立重建。
- [ ] 公开例程展示分离→入层→承载/转动→卸载的真实运动，保留入步速度；使用非 Panda 的小树和一般 H 留出。Panda 仅是后续集成回归之一。
- [ ] 通过受影响旧球路线和新场景专项后提交，才开放对应 capability 的响应状态。

## 4. 多案例验收与独立 oracle（K4）

### 4.1 几何层

10 组合为 SS、SC、SB、SH、CC、CB、CH、BB、BH、HH。每对登记静态、manifold、平移与旋转的**各自状态**。本夜响应按每步实际姿态采样；既有 CCD 能复用则复用，新旋转 CCD 未通过时不得写成旋转路径安全。

必测留出：有限杆端外不接触、平行杆身最近线、斜交胶囊、盒边/角、共面面支撑、偏心面支撑、一般凸多面体、完全包含、两端分离中途命中、换序、共同平移/旋转、顶点/owner重排与重建。深包含暂 unresolved 是有效负控，不算该域支持。

### 4.2 固定数值 oracle

以下是在启动前预声明的测试，不依靠被测算法计算期望值。无阻尼/摩擦正控使用 k=100 N/m；计入权重后的合力幅值由 -k*gap 得到。

| 案例 | 固定几何 | 必须得到的观测 |
|---|---|---|
| CC 线支撑 | 两轴沿 x，端点 x=±0.5；半径均0.05，轴距0.09 | gap=-0.01m；完整最近线；两端各1/2，总力1N |
| CB 线支撑 | 盒半长(0.6,0.1,0.1)；轴x=±0.5、z=0.14、y=0，r=0.05 | gap=-0.01m；杆身而非无限延长线；总力1N |
| BB 面支撑 | 两盒半长均(0.1,0.1,0.1)，中心沿z相距0.19 | gap=-0.01m；4个材料角点，均匀对称时各1/4，总力1N，中心法向力矩0 |
| 有限端负控 | 上述CB轴平移至x∈[0.7,1.7]，其余同 | 最近端到盒角距离>r；不能沿无限轴误加力 |
| 偏心/质量 | 在合法有限patch上，载荷相对COM有0.02m垂直于F的偏移；双方质量1kg与2kg | 1N对应0.02Nm；首个自由受力瞬间加速度幅值1及0.5m/s²，双方反向；关节版由独立小矩阵/J核对 |
| 速度阻尼/摩擦 | 给双方不同线速度及非零角速度，站点速度按v+ω×r独立算 | 阻尼来自真实相对速度；原摩擦上限与功率符号保持；不能用中心速度代替 |

HOST解析小例的几何绝对误差门为1e-8m，力/力矩为1e-8（对应N/Nm）；通用 support 路径沿用其已有2e-7m界，不为新解析例改全局容差。interpret/CUDA仅对实际设备执行的法向/散回模块使用现有测试门；若无同量既有门，预声明 rtol=2e-5、atol=2e-6（对应SI量），失败先归因，不在结果后调门。坐标尺度限制写入案例，不能拿这些绝对门泛化任意尺度。

### 4.3 动态、事务与模型差异

- [ ] 三个真实动态窗口：偏心两自由体、两普通关节树、固定–浮/浮–浮组合。连续128步，h=1/1024s；从同一初态以h/2独立重建，物理总时长相同。姿态、速度、真实力/力矩、材料点、接触出生/结束逐步保存；禁止把几何姿态脚本当自由响应。
- [ ] 无外载/零摩擦留出比较双方总动量和总角动量；已有积分器误差单列。步平均量与瞬时量不得混比。对阻尼层记录接触势能、耗散和外功，保留实际离散能量差。
- [ ] 预声明动态工程门：无NaN、无穿越整个有限实体、无load丢失/重复、无跨world串扰、无失败后部分发布；h与h/2终态平移差<=1e-3m、转角差<=1e-2rad，总接触冲量差<=5%（冲量接近零时改比绝对1e-5Ns）。这些是本批约0.1m尺度的工程门，不回写旧物理预算。
- [ ] 新路径能跑通但工程门未过时，标 implemented_unverified 并保留轨迹；先区分建模、离散误差、接口错误。小穿透如实记录深度/特征尺度比，不能无限追求浮点精确相触，也不能把整物体穿越叫精度问题。
- [ ] Newton只做3个预声明留出：CC平行/斜交、BB面接触、非盒HH。相同几何/姿态下核相交、法向/见证与同类gap语义；动态只有力律、阻尼、摩擦、质量/惯量、步长都对应时才作数值对照。原生不同力律仅作定性对照。
- [ ] 复用现有 TYX 法向层/散回接口者跑真实 interpret/CUDA，记录实际kernel和数据路径；HOST-only robot scene 明确保持HOST，不以一个CUDA几何查询宣称机器人全GPU。
- [ ] 每次改动只补其依赖相关测试；不重新运行15族旧目录或重跑同源已绿组补数字。

### 4.4 计划中的测试入口

统一启动后父为每条命令设置既有Python3.11环境、child/src、禁止pytest第三方autoload、单线程及独立basetemp，保存command/source/raw exit；下列是测试选择，不是本轮执行记录。

~~~text
python -m pytest tests/collision/test_collision_capabilities.py -q -x
python -m pytest tests/collision/test_contact_patch_adapter.py -q -x
python -m pytest tests/test_contact_patch_loads.py -q -x
python -m pytest tests/test_convex_robot_contacts.py -q -x
~~~

每组先运行真实缺失行为的RED，再实现；已通过同源完整组不复跑。设备测试由同一文件内明确backend标记选出，不能以HOST fallback代替。K3旧接缝变更的回归从 test_multi_robot_scene.py、test_floating_robot_scene.py、test_box_link_contact.py 中选择真实受影响项，冻结选择后执行。

新增核的关键测试骨架（类型与K1/K2一致；fixture来自K0固定CC几何，不调用被测力律造期望）：

~~~python
def test_parallel_capsules_preserve_pair_stiffness(parallel_capsule_pair, zero_damping_law):
    sample = contact_patch_sample(parallel_capsule_pair, time=0.0)
    assert len(sample.sites) == 2
    assert sum(site.weight for site in sample.sites) == pytest.approx(1.0)
    v = np.zeros((2, 3))
    loads = sample_patch_loads(sample, zero_damping_law, v, v)
    np.testing.assert_allclose(loads["first_force"] + loads["second_force"], 0, atol=1e-8)
    assert np.linalg.norm(loads["first_force"].sum(axis=0)) == pytest.approx(1.0, abs=1e-8)
~~~

## 5. 新增B/C线：独立开发与明确验收

两条线现已正式纳入这份**待统一启动**计划。父与两个原作者仅完成源码只读规划核对，无新项目执行。两个能力独立于A的接触力语义；本夜组合时只接已验证接口，不强行把三条线拼成未证的全设备物理引擎。

### B：转动路径的真实TYX逐面候选（中高）

**依据：** geometry/collision_pipeline.py 当前显式跳过任何带转动的 mesh–round selector；_dual_turn_mesh_round.py 的 _minimum 仍全量枚举面。原单/双独立转轴、移动枢轴及 first_bounded_contact 的时间搜索/速度界已存在，不重新推导或更换数学路线。

**首片范围：** 对每一个真实旋转CCD采样点，以保守AABB在TYX上筛面，精确距离/包含与连续时间判断保持HOST。支持现有单转胶囊、单转网格–球及双转轴mesh–球/胶囊的原有效域。显式 mesh_midphase='bvh'；默认 all_facets 保留。全时间cell一次性面集合与全设备窄相不包含在本首片。

**文件：** 原RD作者NEW geometry/_rotating_mesh_candidates.py（位于src/phymeo/）、tests/collision/test_rotating_mesh_candidates.py、tests/collision/test_tyx_rotating_mesh_candidates.py、docs/benchmarks/p65-rotating-mesh-candidates.md。父旧hooks：geometry/collision_pipeline.py、_rotating_capsule_mesh.py、_rotating_mesh_sphere.py、_dual_turn_mesh_round.py；不改 _bounded_contact.py 的既有判据/资源上限。

**接口：** MeshRoundArcCandidates(mesh, selector).sample_axis(ends, *, radius) 返回稳定原facet索引；复用 MeshRoundFaceSelector.sample 及现有 query_boxes callback。三个转动query接可选selector，最小距离与最终pose中的 MeshRoundQuery 都沿同一接口。新类位于上述NEW helper，不能把本地采样索引冒作跨时刻常驻安全集。

- [ ] 先写真实缺口RED：旋转分支应实际发起设备面级查询；仅把参数放入构造器不算实现。
- [ ] 当前有限轴到完整mesh顶点距离给最近距离上界，再构造外向FP32候选盒；保留真正最近面和全部并列，不能仅查半径内面。
- [ ] 连续段排除仍用原距离误差、复合旋转速度上界和时间宽度。两端盒的弦、采样命中面的并集都不能替代全程界。
- [ ] inside、全部顶点最远支持、完全包含、穿面判断保留原全量语义；孤立擦触/容量未决仍未决。设备坏回读必须逃出普通unknown处理并中止发布。
- [ ] 固定正负案例：单转杆身命中但两端球分离；双独立转轴及移动枢轴；两端分离中途命中；凹槽近失；初始inside/大球包含；孤立擦触；108面细分盒真实剪枝。
- [ ] 换序/镜像/材料重排、同ID异world、同key新姿态、坏回读/容量/close后重建：完整结果及旧last/layout事务与all_facets一致。
- [ ] 父串行HOST、interpret、CUDA各执行实际职责；记录真实面候选数、核、上传/读回/同步。不能HOST fallback伪装设备成功。
- [ ] 通过后小提交并更新能力表。每采样一次设备提交可能更慢；功能正确且无速度收益也可结束，不为收益改CCD门。

计划测试选择：
~~~text
python -m pytest tests/collision/test_rotating_mesh_candidates.py -q -x
python -m pytest tests/collision/test_tyx_rotating_mesh_candidates.py -q -x
~~~
后者在原环境分别选择interpret/CUDA；同一未改绿组不重复。公共结果字段、首触括区与材料记录全比较，不只比较boolean。

### C：设备接受态直接供碰撞宽相（高）

**依据：** backends/tyx_world_batch.py 已有双缓冲、world行映射和锁，接受标志成功读回后才切换active；拒步世界保留旧态。restore可回退tick，目前没有独立单调state epoch。geometry/batched_collision.py仍HOST建包络，backends/tyx_collision.py上传envelopes。

**首片C1：** 每world一个既有真实FreeSphere，与同world显式登记的有限静态几何产生当前接受姿态候选。半径/shape revision/排除/mask来自实际模型，不能凭状态行猜shape。静态有限几何可为盒/网格等既有可绑定包络，不把不同resident world拼成同一物理world。

**后续C2条件扩展：** 只有C1通过且有时间才加入已有滑轨状态；必须显式提供几何原点、固定朝向、轴及body/shape绑定，从q重建真实几何。原binding仅有轴与标量q并不足够。旋转机器人、变形面、候选积分态和CCD另列后续，不在C1完成时自动勾选。

**文件：** 原robot作者NEW src/phymeo/backends/_resident_collision_plan.py、_tyx_resident_bounds.py、tests/scale/test_resident_collision.py、docs/benchmarks/p65-resident-collision.md。父旧hooks：backends/tyx_world_batch.py、tyx_collision.py、_tyx_collision_bvh.py；沿用_world_readback_plan.py的owner与close模式，不借用私有buffer给调用者长期持有。

**接口草案：** ResidentCollisionPlan(batch, model, bindings).query() -> ResidentCollisionResult；结果含owner、topology_generation、单调state_epoch、各world tick、bounds和候选shape keys。plan和result类型位于_resident_collision_plan.py。query在同锁/同runtime借用当前接受buffer，CPU仅持元数据；不先snapshot完整状态。动态AABB在设备产生并直接给既有BVH设备入口。bounds/候选的必要读回仍逐项计数。

- [ ] 先写RED：现有暖态路径会读状态/上传envelopes，且同tick restore不能可靠识别陈旧引用。不得通过mock计数制造成功。
- [ ] 引入单调epoch：成功状态变更、reset/restore与拓扑换代分别处理；query和返回metadata绑定同一接受版本。数值拒绝的world状态/tick保持，批次epoch可保守失效所有旧引用。
- [ ] 同runtime、同lock、明确借用生命周期；另一个owner、已经close、旧topology或错误world映射在读写设备前拒绝。失败不发布半张候选表。
- [ ] 设备AABB外向包围FP32球心和真实半径，实际BVH消费该buffer；暖态无动态状态snapshot、无动态envelopes上传、无新增设备分配。静态初始上传、status/bounds/candidates读回分别记录，不能声称零读回。
- [ ] 六类独立测试：真实推进改变AABB；FP32恰触/一侧近失；共址多world重复局部ID；同tick restore到不同位置；单world拒步/设备异常；双缓冲交替/empty/close/容量。
- [ ] 候选必须包含独立HOST oracle全部可能碰撞对，允许保守false positive；不得把宽相候选当窄相接触。普通FP64几何参考还须计输入FP32表示差。
- [ ] HOST协议验证、真实interpret、CUDA串行通过；W1/W4/W32及空/受载候选混合，用同任务旧路径配对核完整结果与实际传输计数。性能计时仅在功能全绿且预算允许时做一次预声明的小型AB/BA留出，无收益就记录关闭。
- [ ] C2若未启动或未验收，单列开放，不把C1的球范围写成全部刚体/机器人驻留。新report说明直接数据流与仍HOST的部分。

计划测试选择：
~~~text
python -m pytest tests/scale/test_resident_collision.py -q -x
~~~
按协议/interpret/CUDA分别选择，在真实设备调用中观察传输与分配；不执行完整旧多世界suite取新计数。

### B与C的接缝边界

B使用当前几何和时间样本，C1只提供接受姿态球宽相；**二者相互不阻塞，也不自动接成旋转全设备管线**。未来组合需同版本运动描述和整段保守包络。B保持原query_boxes签名，C在backend新增设备bounds入口，避免同时修改一条调用协议；父审定旧文件改动后顺序合入。

## 6. 三线分工、串行预算与08:00收尾

统一启动后才派发。仍只沿用原agents；本轮的两个agent调用是只读规划核对，已经结束，不持执行槽。

| 角色 | 独立开发任务 | 共享边界 |
|---|---|---|
| 原RR agent | A的K1/K2新模块与专项 | 父拥有所有旧geometry/runtime hooks |
| 原RD agent | B旋转采样面级候选与测试 | 保持原时间搜索合同 |
| 原robot agent | C设备接受态/包络计划与测试 | 父拥有旧buffer/锁/BVH hooks |
| 父 | A的K0/K3、所有共享接缝、证据审核与唯一项目执行 | 三线按依赖顺序集成；不覆盖WIP |

原RR此前额度不可用，统一启动时核实际可用性；若仍不可用，由父推进A核心，其余两作者分别B/C，不新建替代团队、不许诺重置额度。A内部的K1/K2不再占两个额外并行槽。

所有 import/collection/pytest/数值/probe/lint/formatter/build/install/profile仍严格串行。代码/测试可以并行写，读审可以并行；每条项目命令fresh entrypoint-aware guard，外来真实执行自然结束、不kill。首RED、首失败、source/command/raw exit必须保留。

**项目执行总预算先按180分钟预留：** A约70分钟，B约40分钟，C约40分钟，共享回归/必要静态约30分钟；另预留45分钟收尾。只是排队上限估计，不是删测试的授权。某组实测超出预算时，推迟尚未验证的扩域；不重跑已绿同源组、不因deadline降低门。

**队列规则：** 各线接口/测试可准备后，父按A→B→C的用户优先级选择已就绪任务；A未就绪时可先验B或C，不空等，也不预empt运行中的测试。共享接缝稳定后只做一轮必要集成回归和一个新stage wheel。先功能，性能对照排在后；C2和新热点优化最后。

| 时间/硬截点 | 三线安排 | 交付门 |
|---|---|---|
| t0后0–35分钟 | 冻结A接口/案例，B selector、C状态epoch与scope；RED按槽排队 | 新基线、所有权和测试选择清楚，开始代码 |
| 随后约2小时 | A patch/载荷、B采样筛面、C state→bounds并行写新模块 | 每线有独立可运行最小能力 |
| 随后约1–1.5小时 | 父接旧hooks；按就绪度串行专项 | 不仅有参数/注册表，而有实际调用和输出 |
| 随后剩余验证窗 | A整族/真实响应、B真实TYX与全输出、C版本/设备链；只做必要对照 | 各线独立签收，未过范围如实保留 |
| **07:15** | 停止新算法和扩域；C2/完整时间cell候选没就绪则留后续 | 冻结候选产品；只处理小而明确的阻断 |
| **最迟07:30** | 收尾审阅、一次build及源码外组合使用、差异报告 | 预计不能在08:00结束的长命令不启动 |
| **08:00** | 停止新项目执行并交付概览 | 完成/失败/未做/剩余差距分别列出，不静默顺延 |

t0是用户最终统一启动时刻。用户继续加任务则重新核总预算，不叠加无限任务；若启动晚，优先保A通用核心，B/C只交各自已通过片段。**代码写完但设备测试来不及的，标待验收，不合并成已支持功能。**

本夜完整目标是A/B/C三个明确限定域均通过；降级交付保留三个独立状态。A不足10组合时不能宣布凸族目标完成；B只有HOST时不能宣布TYX完成；C只有HOST协议或内部mock时不能宣布设备直供完成。功能通过但未提速可以如实签收功能。

K5文件：NEW docs/benchmarks/p65-generic-collision-20261107.md 汇总三线，链接各自专项；更新续作入口/能力矩阵及[差异全景](../../development/phymeo-newton-panorama-2026-11-07.md)。只为实际新增产品做一次源码外wheel验收；计划文档本身不触发构建。报告按模块与能力层展示差距，不按测试数计算完成率。无远程CI、push、main合并、新依赖或周期任务。

## 7. 怎样判断成功，怎样识别空转

**通用化成功：**

- [ ] 新形状/owner组合通过声明与同一查询/载荷入口接入，不需要复制scene或增加“某机器人/某场景”的施力分支。
- [ ] 10组合均有真实分层状态；要宣布“本夜凸族响应目标完成”，必须10组合在声明浅接触域的几何和双侧载荷都通过，包含非盒H、线/面多点及失败事务。
- [ ] 至少三种所有者组合和三个连续运动窗口通过；两方真实速度/力矩响应有独立oracle，Panda单例不足以签收。
- [ ] 测试能发现漏反力、错力臂、重复刚度、无限杆误触、错误world、伪GPU和部分发布。Newton差异按相同观测定义说明。
- [ ] 文档状态与实际源码/执行证据相符；剩余旋转CCD、凹体/软体响应、全设备窄相等差距没有被泛化措辞遮住。

**空转/调整触发器：**

- 连续两次修订没有新增可运行能力、没有缩小失败复现，也没有排除一个明确错误假设：暂停该节点扩写，记录最小失败并换实现/拆分交付。
- 一个节点60分钟还在追加数学条件、测试名称或文档，而没有公开入口/最小测试进展：回到现有复用接口，禁止再开同义证明分支。
- 每个pair都新增完整solver、同一状态采样/积分/发布复制多份：架构不达标；先共用接触表示和owner映射。
- 调用成功但总力随接触点数倍增、末端运动靠重设速度、只比较图像、CUDA标志实际HOST：视为功能错误，不当数值误差。
- 性能无收益可以结束优化节点；本夜不重复第七批已无稳定收益的局部复用计时。功能完成后才安排新有证据的热点优化。

## 8. 待统一启动与后续范围

用户已将B转动路径逐面候选、C设备状态直供宽相加入本计划，现与A通用碰撞共享上面的时间、人员和验证预算。这里只确定开发和测试安排；用户还可补充事项，之后统一启动。本轮没有开发、项目测试或周期任务。

更广机器人驻留、完整时间cell面候选、全设备窄相、浮mimic/闭环、可微训练和其他物理族见[差异全景](../../development/phymeo-newton-panorama-2026-11-07.md)。它们不因本夜三线首片通过而自动完成。
