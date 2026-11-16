# R4 选型与 I0 三维面积压力端口

日期：2026-11-16。分支 codex/exp；稳定 dev 不变。本合同在新源码/测试运行前冻结。

## 1. 选型与证据边界

首个集成候选选 **刚性表面 B 对柔顺压力体 A**，后续 I1 表示为连续分片线性四面体压力场与刚性三角表面的交集。柔顺描述接触层，尚不增加体内变形自由度。首期 I0 接收已经裁出的三角接触面；不包含碰撞筛选、裁剪或体网格资产生成。

[V05](../../development/area-pressure-planar-dynamics-report-20261116.md)和[V06](../../development/area-pressure-impact-report-20261116.md)分别提供有限支撑俯仰/卸载、双方有限质量冲击的 NumPy 动力学证据。它们不证明一般 3D 或 MuJoCo-TYX 接线。[V03](../../development/area-pressure-newton-reference-report-20261115.md) 的 Newton 双软球面积/收敛 FAIL 保留；不能因换表示称其已修复。新 PL 前端仍须独立面积、压力中心、力矩与边界测试。

采用理由：真实面积积分、可保压力中心与力矩、已有速度级近似文献；先缩为一软一硬，避免同时开发双软等压面。FFT 保持小变形半空间标定旁路；Fourier 几何尚无明确误差下的整步收益，暂不移植。两者不是本路线前置。

资料于 2026-11-16 核对：
- [Drake Hydroelastic 用户指南](https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html)：刚性三角表面/柔顺体内四面体压力表示。
- [速度级压力场接触论文 IV/IV-A](https://arxiv.org/html/2110.04157v2)：冻结几何的局部压力变化、面积刚度与代理间隙。
- [ContactMechanics](https://contactengineering.github.io/ContactMechanics/)：小变形半空间范围。

这里是有边界的工程选型，不是宣称已通过完整 Hydroelastic 外部验证。

## 2. 本批可执行交付与职责

| 节点 | 难度 | 交付 | 完成条件 |
| --- | --- | --- | --- |
| R4 | 中 | 本选择与后续准入 | 保留失败和未支持域，不改旧门 |
| I0a，本批 | 中 | 三维真实面积、压力、梯度、身份/相位端口 | 独立解析测试，输入不可变，非法输入整体拒绝 |
| I0b，本批 | 中 | 全共享自由度 Jacobian 与双侧力/矩映射 | 虚功/共同坐标/两侧反作用、k/phi 单位正确 |
| I1，下一批 | 高 | tetra–triangle 裁剪与连续 PL 压力资产 | 斜面、跨 cell、有限边缘、无重复面积、旋转/材料交换；固定解析格 |
| I2，之后 | 高 | MuJoCo-TYX 独立接触律行装配与共同解 | 关节与自由体、多接触、同一步状态/力发布；同 pair 只一种接触模型 |
| I3–I5 | 高 | 设备生命周期/多场景/变形体 | 各自范围验收，不能由 HOST 端口替代 |

本批只新增：
- src/phymeo/geometry/area_pressure.py
- src/phymeo/solvers/area_pressure_rows.py
- tests/test_area_pressure_surface.py
- tests/test_area_pressure_rows.py

不修改旧 ContactPatchSample（其 weight 和为 1）、旧整体 patch 刚度、MuJoCo-TYX 运行时、导出、pin 或已有测试。父维护计划、报告与续作入口。原 scene 作者两 source，原 training 作者两 tests，互不读对方新实现；均只纯文本编码。root 唯一项目执行 owner。

## 3. 方向、单位与几何合同

A 固定为 compliant，B 固定为 rigid。单位 m / Pa / Pa/m / N / N m。三角 winding 给单位 n，方向 **B→A，力作用于 A**；相对法向速度 vn=n·(vA-vB)，分离为正。压力场固定在 A，首域有效梯度 g=grad(pA)·n 严格正，故 dp/dt=-g*vn。

g<=0 整体拒绝，不能取绝对值、夹小正值、静默丢面。例如 minimum-distance 压力场某些侧边单元在顶面会出现 p>0/g=0；这确实不在首个等效行域。I1 必须报告此情况，不能把缺失力算成功。双软的有效梯度通常是两个正方向梯度的调和组合，不在当前 API。

每个输入三角保存 world vertices (3,3)、3 个非负顶点压力与 constant world gradient (3,)。必须 finite、面积严格正、g>0，并验证每条边的压力差与 grad·edge 一致。残差门固定为 1e-10 Pa + 1e-12*max(abs(p_i),abs(p_j),abs(grad·edge))；这只是 FP64 输入一致性门，不是物理接受预算。叉积、面积、计算输出均须 finite，溢出明确拒绝。

采用三个二次精确重心点 (2/3,1/6,1/6) 的排列，每点 dA=area/3，p 为线性插值。可精确积分平面内线性 p 的合力与一阶力矩；不允许只取几何中心替代压力中心。不要求跨三角共同法向。所有面积权重之和等于实际三角面积之和，绝不归一化。该端口不独立证明输入三角互不重叠、体压力连续或合法碰撞；这些属于 I1 生产者义务。

## 4. 固定公开 API

所有 dataclass frozen，所有 NumPy 输出和持有数组通过 bytes-backed FP64 copy 真正不可写；不能只是 setflags(False)。接受 array-like，先界定 shape/cap 再复制，NaN/inf 拒绝；无项目外依赖。标识为非空 str 长度<=128；整数拒 bool。

geometry/area_pressure.py：
- AreaPressureIdentity(compliant_body: str, rigid_body: str, compliant_material: str, rigid_material: str)，body 不同。
- AreaPressureStamp(world_id: str, step_index: int, sample_time_s: float, configuration_id: str)。step>=0/time>=0 finite；configuration_id 是调用者绑定标签，不冒实际状态哈希证明。
- AreaPressureTriangle(triangle_id: str, compliant_cell_id: str, rigid_face_id: str, vertices, vertex_pressure, pressure_gradient)。
- AreaPressureSurface(identity, stamp, triangles: tuple[AreaPressureTriangle,...])。允许空接触；triangle_id 唯一，重复整体拒绝，单面 source cell/face 可相同。
- quadrature_area_pressure(surface) -> AreaPressureQuadrature。
- AreaPressureQuadrature 为只由上述函数构造的不可变结果，字段 identity, stamp, sample_ids: tuple[(triangle_id, local_slot),...], source_ids: tuple[(compliant_cell_id, rigid_face_id),...], points(Q,3), normals(Q,3), dA(Q,), pressure(Q,), gradient(Q,), total_area_m2: float。gradient 是 g 标量。顺序保输入 triangle 顺序及固定 slot 顺序。空结果 shapes 仍准确。不要求完全禁止 Python 手工实例化，但 assembler 必须再次校验全部数据与元数据，不能信任类型标记。

solvers/area_pressure_rows.py：
- AreaPressureKinematics(stamp, sample_ids, dof_ids: tuple[str,...], jacobian_a(Q,3,D), jacobian_b(Q,3,D))。同一共享 generalized coordinate basis，D>=1，dof IDs 唯一。双方即使耦合到同自由度也完整相减。
- assemble_area_pressure_rows(quadrature, kinematics) -> AreaPressureRows；stamp 和 sample IDs 顺序必须完全相同，不能自动重排或广播。保存 identity/stamp/sample_ids/source_ids/dof_ids、points/normals、两原 J，以及 normal_jacobian(Q,D), stiffness(Q,), surrogate_gap(Q,), elastic_force(Q,)。
- normal_jacobian = n^T(JA-JB)；k=dA*g，phi=-p/g，f0=dA*p=-k*phi。零压力合法且 f0=phi=0；这是局部冻结几何系数，不是物理间隙/动态已求解。
- map_area_pressure_normal_forces(rows, normal_forces(Q,), *, stamp) -> AreaPressureLoads。先完整校验非负 finite forces/stamp，再返 generalized_a(D,), generalized_b(D,), generalized_total(D,), wrench_a(6,), wrench_b(6,)。
- wrench 顺序 [Fx,Fy,Fz,Mx,My,Mz]，关于同一 world origin；A 每点 force=f*n，B 相反，同一 world 接触点。广义力分别 JA^T force 与 JB^T(-force)，两者在同共享 basis 相加，不能把总广义力当必然零。
- Loads 包含 identity/stamp/sample_ids/source_ids/dof_ids；没有冲量、状态更新、damping/friction 或时间步函数。输入 force 是调用者提供的力，不能标记为共同求解结果。

硬资源：surface<=1024 三角，Q<=3072，D<=256，任一 Jacobian 元素数<=1,000,000；超限先拒绝。所有 derived arrays finite。对结果对象 assembler/map 重新校验必要字段、shape、数值、元数据；不把 frozen 当作输入认证。非法输入统一 ValueError。

## 5. 独立验收

首轮只运行两个新完整 HOST 文件。test 作者按合同/解析积分写期望，不调用旧压力 producer 充 oracle。

固定正控：
1. xy 三角 (0,0,0),(2,0,0),(0,1,0)，p=(1,2,4) Pa，grad=(0.5,3,5) Pa/m，n=+z，area=1 m2；Fz=7/3 N，Mx=11/12 N m，My=-3/2 N m（压力一阶积分用 simplex 单项式独立核）。三点积分与细分不改合力/矩。
2. 相同压力/梯度的面积放大；曲面由多个非共面三角组成，各自法向/力矩；刚体旋转和平移 covariance。
3. 两自由刚体的 [vA,omegaA,vB,omegaB] 共 12DOF J；任意速度下虚功一致，COM 力矩与 origin wrench 对得上。
4. 双侧共享坐标的 cross terms 不可遗漏；D 不同于 12 的关节型 J；固定端 JB=0。
5. k/phi 与压力正、零压力、分离扰动方向的独立单位和数值检查；任意非负 force pullback。
6. 空 contact、真实不可写、caller 修改原数组无影响；phase/config/world/step/sample order 错误拒绝；重复ID、invalid gradient、压力不一致、degenerate/NaN/inf/overflow/cap 拒绝。手工构造坏结果也拒绝。

工程数值门：普通数量级解析/协变/虚功 allclose rtol=1e-12, atol=1e-12；无需把 HOST 端口精度宣称为 CUDA/native 精度。禁止改固定输入或门追绿；功能失败先定位，最多一次明确修正，再运行受影响及最终新套。同源全绿不重跑，V00–V06不跑。

## 6. MuJoCo-TYX 接线待决清单

I0 不填旧 solref/solimp 冒充新物理。I2 前必须逐项记录实际旧行的 acceleration/force/impulse、J 与正负约定、D/R 正则项、h 与 row activation。
已读真实代码：_constraint_rows 构造 frame·(JB−JA)，aref=−b·Jv−k_native·impedance·position，R 来自 impedance/invweight，D=1/R；_constraint_solver 解 Ma=tau_smooth+J^T f 与 Ja+Rf−aref，未知 f 单位是力，solver 本身不接 h。_constraint_device 再以 M+h*Cjoint 积分（此 Cjoint 不同于行 D）。这些不能由 I0 的 k/phi 直接替换。

未来需显式 pair 模型选择；area pair 不再同时进入原 native contact 发力，其他 pair 原路径保留。全关节与其他接触共享质量算子/解向量，摩擦额度来自同解法向力；输入几何相位、最终状态/接触力共同发布，failed trial/reset/history 不混用。
需要三维自由体和至少一个关节实际受力用例后才称 I2 完成；HOST pullback 与示意接口本身不构成集成通过。

## 7. 执行、证据与停止规则

保现有环境，无安装；root 每条 command fresh entrypoint guard，环境→首静态→必要 style 修复→新完整 HOST 串行。保存所有原始 exit/log/source before-after。原作者不执行。保持父三份无关 untracked 和作者 WIP。

本批成功是代码端口与独立测试通过、R4 与 I1/I2责任明确；不是再做一个视频。若压力梯度准入不能覆盖目标，明确 unsupported 并在 I1 调整表示选择，不靠漏面/调参修饰。源一旦验过，只在新变更或新失败需要时再验证。

## 8. 实际完成结果

测试源码777b7f12，首次两个完整新HOST文件59 passed/1.75s/RAW_EXIT0。I0a/I0b按上述有限范围接受；只有排版/import修正，qualified AST全同，无数值修复、旧组重跑或runtime/GPU执行。原始日志与13项最终source/config核对完成，父/两作者已释放。详[报告](../../development/area-pressure-3d-port-report-20261116.md)。后续I1真实裁剪、I2共同求解及I3–I5均未完成；V03 FAIL、dev基线及独立训练状态保持。
