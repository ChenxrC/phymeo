# 通用碰撞与双侧响应：2026-11-07 08:00 前开发测试计划

> **For agentic workers:** 统一启动后按 writing-plans 对应的 subagent-driven-development 或 executing-plans 工作流逐任务执行；用户规定的原 agents、共享文件所有权及串行验证优先。**本文不是启动指令。**
>
> **状态：PLANNED_NOT_STARTED。用户还会补充其他工作，等待用户统一启动。当前没有开发任务或项目执行槽。**
>
> **截止：北京时间 2026-11-07 08:00（今天），不是 11 月 8 日。** 用户确认启动时重新读取时间；截止不随启动推迟而顺延。

**Goal:** 用通用算法族、统一接触表示和共同力映射，扩展非球机器人/刚体互碰；以多案例组合矩阵验收，结束按每个场景复制一条碰撞和施力链的推进方式。

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

本夜主交付是 **一套凸实体接触适配链**。球 S、有限胶囊 C、OBB B、一般凸多面体 H 共 10 个无序组合是整族目标；OBB 在通用几何中是凸包的一种，单列只是为了验证解析快路与通路一致。一般 H 必须含非长方体留出，不能把 hull 全部重新解释成 box。

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

## 3. 开发节点、难度与所有权

全部复选框在统一启动后才可勾选。下面的 NEW 路径是计划创建，不表示文件已经存在；执行前先确认不存在，避免覆盖其他工作。

| 节点 | 难度 | 交付 | 依赖 |
|---|---|---|---|
| K0 能力登记与案例合同 | 中，约 35 分钟 | 全局路由表 + 10 组合/各层现状 | 启动后的新基线检查 |
| K1 通用接触 patch 适配 | 高，约 100–140 分钟 | 几何查询转为统一有限站点 | K0 |
| K2 所有者载荷映射 | 高，约 90–120 分钟 | 相同接触输出驱动真实双方 | K0 的接口，可与 K1 编码并行 |
| K3 场景接缝与事务 | 高，约 60 分钟 | 非球普通/浮根/自由刚体共用链 | K1、K2 |
| K4 分层验收与参考对照 | 中高，约 70–100 分钟 | HOST + 实际 TYX 范围 + Newton 留出 | 按依赖串行穿插 |
| K5 收尾、wheel、差异报告 | 中，预留至少 45 分钟 | 可用版本/未完成范围/08:00 概览 | 已验证产品冻结 |

时间是工程预算，不是通过承诺。K1/K2 可以并行写代码，所有执行相加后仍受截止约束；用户增加任务后从同一个预算重新分配。

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

## 5. 并行安排与唯一串行验证

统一启动后优先沿用三个原 agent；此刻**不派发**：

| 角色 | 可独立编码/编写测试 | 禁止写入 |
|---|---|---|
| 原 RR agent | K1的新几何适配与专项，补通用凸族缺口 | 父旧collision_pipeline/runtime文件 |
| 原 robot agent | K2新载荷核及所有者测试，协助K3接口 | 父旧scene/participant文件 |
| 原 RD agent | K0能力表、10组合/跨所有者案例库和留出审阅 | 另造solver、与前两人重写同一实现 |
| 父 | 合同定稿、所有旧共享接缝、集成和全部项目执行 | 覆盖子树WIP或重复派实现 |

先冻结 K1/K2 类型、法向/单位和权重，再并行编码；K3等两路通过后接入。测试文件可以并行写，任何 import/pytest/数值/离线probe/lint/formatter/build/install/profile都只由父单一槽运行，每命令 fresh entrypoint-aware guard；外来真实执行自然结束，不kill。

当前某原 agent 曾返回额度不可用，统一启动时核实际状态；不可用则父按优先级串行或收缩夜间范围，不承诺自动重置额度，不复制一支替代团队。代码审阅可与独占项目验证并行，审阅不等于运行通过。

## 6. 到08:00的时间与收尾

以用户统一启动时刻t0计，保留截止前至少45分钟；按下面阶段消耗实际剩余时间，其他任务加入时整体重排，不叠加一份无限夜班。

| 相对预算/硬截点 | 工作 | 退出条件 |
|---|---|---|
| t0后0–35分钟 | K0和接口冻结、有效RED排队 | 能力/案例/所有权清楚，开始代码而非继续扩写历史 |
| 随后约2小时 | K1/K2并行编码，父穿插串行focused测试 | 共同patch与共同力核可运行 |
| 随后约1小时 | K3共享scene接入 | 至少一条非球整链+故障回滚，其他组合进入同一驱动 |
| 随后约1–1.5小时 | K4整族矩阵、真实运动、必要设备/参考 | 分层记录10组合通过/未决/不支持，不以局部绿替全族 |
| **07:15起** | 停止新算法/新大模块；只修已明确的小阻断 | 冻结候选source和最终测试选择 |
| **最迟07:30** | K5代码审阅、一次stage build与源码外使用、报告 | 可用交付和开放项清楚；剩余工作不静默延长 |
| **08:00** | 停止新项目执行并给总体概览 | 成功/失败/未做/差距/下一步分别列出 |

若启动很晚：优先完成K0+K1/K2的可验收核心，推迟尚未进入验证的K3扩域；不得删除测试、偷偷降低门或把代码完成当通过。07:15后不启动预计无法在08:00前结束的长测/build；时间不足则明确“未构建/未验收”，下一批继续。

K5文件：NEW docs/benchmarks/p65-generic-collision-20261107.md，更新 docs/development/p65-continuation.md 与能力矩阵。只为真正新增产品做一次源码外wheel验收；只写了计划/登记文档则不构建。最终概览按“算法族×能力层”展示已补和剩余差距，避免按总测试数量算完成率。无远程CI、push、main合并、新依赖或周期任务。

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

## 8. 等待用户补充的夜间工作边界

本文件先纳入已讨论的通用碰撞建议。第七批剩余“转动mesh–round逐面候选”和“设备接受态直接供宽相”仍在总路线中，但**尚未占用本夜开发槽**；用户补充其他工作后再统一排序、核依赖和时间预算。新任务不能与本计划竞争同一共享文件owner或项目执行槽。

本轮仅保存计划并建立续作链接；没有启动上述开发、测试、代理或自动任务。用户统一启动前保持该状态。
