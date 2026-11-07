# C6 设备接受态到有限静态接触反馈

状态：**TEST_AUTHORING_ACTIVE / PRODUCT_PENDING**，2026-11-08，无固定截止。C4/C5已接受，父沿既有持续计划推进。本文冻结[首片提案](../../development/p65-resident-contact-stage-proposal.md)的R01–R05及下列补足；不是全设备窄相、CCD或整个P6.5接受。

## 行为与接口

每world一棵ordinary fixed-root R/P树，与有限static sphere/capsule/unrounded convex接触。真实accepted设备FK/J/速度→HOST旧contact_patch_sample/sample_patch_loads与原RobotContactMaterial(mu=0)→原TYX mechanics/有界PD/积分→完整candidate审核→同一batch唯一swap。原完整惯量/COM/full交叉项、delay0/1、512默认/4096上限、5e-5 PD和2e-5质量门不变。旧batch.step保持行为。

```python
plan = batch.prepare_contacts(
    collision_model, bindings, materials_by_pair,
    candidate_capacity=4096, site_capacity=128, sampling="discrete_entry",
)
accepted = batch.step_contacts(plan, h)  # 与step相同的world->bool
plan.last_accepted  # owned immutable Mapping[world, ResidentContactReceipt]
plan.last_status   # 本次逐world具名结果；诊断，不冒接受态
```

bindings沿现ResidentTreeBinding/ResidentStaticBinding。只允许ordinary tree的接触绑定，affine/floating绑定明确拒绝；batch中没有接触绑定的ordinary/affine/scalar world沿原step推进，不造零运动dummy。materials_by_pair必须覆盖所有声明允许的canonical shape-key pair，值为原RobotContactMaterial且friction=0；不猜默认law。dynamic/dynamic必须在原模型排除。不支持的几何或nonfinite不能当分离。

candidate_capacity沿原resident collision整数上限，为整plan帽，溢出或设备/IO/未知几何异常整批不发布。site_capacity为0..4096整数（每已绑定world同帽）或精确覆盖已绑定world的Mapping[int,int]；按完整几何patch sites计，不按力是否正计。一world站点超帽得到SITE_CAPACITY_EXCEEDED，其全部q/v/FK/held/history/tick保留，邻world正常推进。增容通过新建plan，旧plan不可变。所有配置与source/world/group关系在分配前检查。

## 同一次入步、暂态载荷与收据

父复用已接受的mechanics/FK，临时contact COM wrench每次全量清零、只加到本次mechanics的external后再原tree_solve，不能改held command或delay队列，不能PD后补kick。static侧等反力/关于明确reference point的矩单列，不冒固定root所有支撑反力。

entry packet包含扩宽至FP64的真实FP32 q/v、world FK/COM/anchor、设备full J/M/b、原input external和已选择延迟drive。FK是anchor-relative时只加一次anchor。HOST用真实stored变换得到全部有限材料点，不QR/极分解/吸附正交；不经要求1e-10正交的OwnerPose冒原生FP64。原shape radius/vertex IDs/source/局部材料不替换，量化差单列。末态原材料/作用点必须由同candidate几何运输；不可用entry切线代替转动路径。

ResidentContactReceipt至少有：world、entry_tick、exit_tick、dt_s、entry_state/accepted_state（完整原checkpoint语义）、entry_geometry/accepted_geometry（完整FK）、input_external、selected_drive、law_wrench_fp64和uploaded_wrench_fp32（full N×6 COM F/T）、contact_generalized（实际FP32散回扩宽值）、actual_drive、sites、static_reactions、diagnostics。site保存canonical pair、source IDs/weights、两表面点/共同点、normal/gap/weight、entry双侧速度、原law、实际force，以及末态两种点位移与work。quantization差、entry功率*h、实际work、动能差分别保留，不声称此半隐式接触能量严格守恒。

所有receipt的数组owned不可变。last_accepted逐world只在该world成功时替换，拒绝world旧收据保持；last_status可记录新失败。source/owner/epoch/entry tick绑定从stage持续到publish。candidate读取/结果构造/分配都在唯一swap前。公开query目前会改layouts/_last，父只能提取纯stage并在最后publish；不能直接先query冒同次成功。恢复同tick仍递增epoch，使旧attempt无效。所有实际launch/读回/上传计数保留。

## 多案例库和新增几何覆盖

原提案R01–R05的数值、完整独立公式和参数不变。补R01几何变体，k64/c0、动侧单Pz/m2、N2/a1：
- sphere–sphere：动态center=(0,0,3/32), r1/8；static center=(0,0,-1/8), r1/8，实际depth1/32。
- sphere–capsule：static轴端点(±1/4,0,-1/8),r1/8；动态球先在x0轴部，再x1/4端点正上方，其余同上。另x=1的越端分离必须零力，不能无限延长轴。
- 转动box–非box凸体：动态box半宽1/8、绕z=.37 rad、center z3/32；静态凸台top square半宽1/2,z0，bottom square半宽3/4,z=-1/4（八点截锥）。top完整有限patch和N2，原局部顶点与FK量化保留，不能替换AABB盒。

R02跨支、声明重排/整体轴循环和平移；R03实际偏心圆弧、入步omega×r、40步；R04原0..3/4s卸载重入及h/2；R05 W1/W4/重复局部ID、容量、全部晚故障/恢复/资源/owned矩阵。所有结果按原独立公式验证，不能用产品采样/响应生成预期。unsupported/deep/tie/错误shape-source也各有明确负控。

## 所有权、分批与成功判据

原robot唯一 **TEST_AUTHORING_ONLY**：NEW tests/scale/test_resident_tree_contacts.py、tests/scale/test_resident_tree_contact_transactions.py、docs/benchmarks/p65-resident-tree-contacts.md。可新增一个纯独立算术测试helper在同测试文件，不导入产品oracle。先提交R01真实旧几何/法向2N成立后调用缺失prepare_contacts的有效RED节点；余矩阵同任务继续。禁止作者全部项目执行、lint/format/数值probe、改旧文件、merge父或另加助手。

父唯一执行owner及旧batch/collision/tree storage/kernel接缝owner。首RED后NEW _resident_contact_plan.py/_resident_contact_bridge.py（其他新device scatter仅在确实需要时追加）复用原kernel与几何，不造另一个Scene/runtime/solver。顺序为首接口→R01/R03→R02/W4→全短窗/事务→interpret/CUDA→必要相邻兼容→同源保存/新wheel。原有同源绿组不重跑。首实际失败保存，再按原因修复；不修改物性、h、预算或复制状态让单例通过。

本片成功需完整上述多案例和两真实backend及源码外消费；只R01或Panda单例不足。性能只是按相同输入/输出公开成本，不设虚假超越结论。moving/moving、affine接触投影、持续摩擦、浮根、全GPU窄相/CCD是后继节点，不随本片自动标完成。

### 收据容器与构造一致性补足（2026-11-08，执行前）

sites是immutable tuple[ResidentContactSite]，每项pair为canonical CollisionKey二元组，geometry复用原owned ContactSite完整witness/IDs/weights/skin/world points/load_point/weight，law是原RobotContactMaterial。另保first_velocity_m_s/second_velocity_m_s（共同load_point入步速度）、first_force_n/second_force_n、first_reference_point_m/second_reference_point_m（动态真实COM，static声明顶点均值的固定world点）、first/second_material_displacement_m、first/second_application_displacement_m、application_work_j/material_work_j（双侧合计，可由各force/motion复算）。static_reactions是owned Mapping[CollisionKey,ResidentStaticReaction]，value为reference_point_m/force_n/torque_nm。

input_external为原mechanics external列加接触之前的full N广义载荷、未减bias；selected_drive为本次已选择delay后的N×3 target/rate/feedforward，actual_drive为N。原未投影held/full loads仍保在entry_state.control；总external可独立拆回input_external+contact_generalized。entry/accepted_geometry沿原TreeKinematics。

两种入口world点均由实际stored entry R/p作FP64 solve逆映射，再经实际stored candidate R+/p+运输；static位移为0。不正交化、不HOST重算FK。1e-10 m只是公开记录构造一致性门，不能冒FP32物理精度；独立q/v/FK和圆弧物理门仍5e-6/5e-5、窗口2e-4。晚故障边界方法名冻结plan._read_candidate（实际candidate读回）与plan._finish_attempt（完整owned receipt构造），均在唯一swap之前。

R04原提案的“前1/8s，随后最多再3/4s”明确为总7/8s。上文“原0..3/4s”为简写错误，本补足在任何执行前纠正，不是延长失败测试预算。h/h2在首个真实卸载后再出现正入步N的接受步停止。
