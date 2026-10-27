# P3 固定基座机器人 CPU 参考设计

用户于 2026-10-27 授权继续 P3–P5。基线 `2993614dd7a8e5630cf3e7d53c8ac32f21c11ac7`，分支 `codex/p3-p5-dynamics`。本设计落实[已有大纲](../plans/2026-10-27-integrated-framework-outline.md)，不改变 P1/P2 已验收合同。下面算法和数值选择为实施者决定，不冒称用户指定了算法。

## 范围与路线

固定世界基座、拓扑有序树、每连杆一个旋转或移动关节；包含非单位零位旋转、非零 COM、分支、多作用点、外力和力矩。浮动基座、闭环、球关节、限位、腱、复杂执行器不支持，相关参数/类型必须明确拒绝。P3 只提供 NumPy FP64，FP32 与新算法设备执行留 P6；旧 tyx 路径仍需回归。

选择独立运动学、世界坐标牛顿—欧拉加速度/惯性载荷和稠密关节空间求解。质量矩阵以各 COM 雅可比的质量/惯量二次型组装，偏置由零广义加速度的惯性载荷投影计算。它是透明的小规模参考，不宣称实现了优化 CRBA/ABA。直接包装 MuJoCo 不能证明本项目算法成立；立即做稀疏/设备优化会扩大本轮范围。MuJoCo 仅为可选、隔离的验证依赖，不进入 phymeo 必需依赖。

数学参考：[MuJoCo 3.3.7 官方计算说明](https://mujoco.readthedocs.io/en/3.3.7/computation/index.html)明确区分质量矩阵、重力/科氏偏置和接触广义力。只借鉴公开数学，不复制上游求解源码；对照版本已有[公开发行包](https://pypi.org/project/mujoco/3.3.7/)。

## 数据及坐标

SI、Y-up；关节 q/v 使用连续角度 rad/rad·s⁻¹或 m/m·s⁻¹，不自动 wrap。模型和状态全部 FP64；模型持有不可写自有数组，状态/输入构造复制，绑定后模型及有序配置变化拒绝。

`RobotModel(parents, joint_types, joint_position, joint_orientation, joint_axis, mass, com, inertia_diagonal, gravity)`：

- n≥1；parents 为长度 n 的整数，父索引 -1 表示世界，否则必须 0≤parent<i；支持多个根连接同一固定世界基座，拒绝环、前向父引用、浮点/布尔索引。
- joint_types 为 n 个 `revolute` / `prismatic`。每个连杆恰一个自由度。
- joint_position (n,3) 为父连杆原点到零位关节原点的父坐标偏移；joint_orientation (n,4) 是零位子坐标到父坐标的 xyzw 四元数，单位模误差≤1e-10。
- joint_axis (n,3) 在零位关节坐标中，单位长度误差≤1e-10。旋转正向遵循右手规则。
- mass (n,) 严格正；com (n,3) 是连杆原点到质心的本体坐标；inertia_diagonal (n,3) 是同一本体坐标轴下的 COM 主惯量，严格正且满足三角不等式。gravity (3,) 为世界加速度。
- 不接受自动质量推断、缺失惯量、零质量连杆或隐式单位转换。

`RobotState(q, v)` 为两个 (n,) 数组。`RobotInputs(effort, force, torque)` 中 effort (n,) 是直接关节力矩/力，force/torque (n,3) 分别为施加在各 COM 的世界外力和绕 COM 的世界力矩；外力不含重力。zeros(n) 生成 FP64 零输入。

`robot_family()` 提供局部 StateFamily：kind=`robot`，字段 q/v、effort/force/torque，dtype_field=q。形状/精度/有限性必须匹配模型；不扩展全局注册表，也不修改 SceneStepper 来识别机器人。

## 运动学和动力学

世界基座 R=I、o=0。令 R0 为零位旋转，a=Rp R0 axis。旋转关节：o=op+Rp joint_position，R=Rp R0 Rot(axis,q)；移动关节：o=op+Rp joint_position+a q，R=Rp R0。COM 世界位置为 o+R com。

`forward_kinematics(model, q) -> RobotKinematics(position, rotation, axis, com_position)`，结果均为自有数值数组。`point_jacobians(model, q, indices, local_points)` 返回线/角雅可比 (k,3,n)。仅祖先关节贡献：旋转的线列 a×(point-o_joint)、角列 a；移动的线列 a、角列 0。重复索引及共享祖先保留完整交叉响应。

`mass_matrix(model, q)` 组装 Σ(m JvᵀJv + Jwᵀ Iworld Jw)。`inverse_dynamics(model, state, acceleration)` 通过世界坐标递推得到 COM 加速度及角加速度，计算 m(ac-g)、Iα+ω×Iω，再投影到各关节；移动关节必须包含 2ω_parent×(a v) 科氏项。`bias_forces` 为 acceleration=0 的结果。`generalized_forces` 将 COM 外力/力矩投影后加 effort。`forward_dynamics` 解 M a = applied-bias，不显式构造 M⁻¹。

矩阵/结果有限性、正定性和解残差有门禁；不得加未声明阻尼、惯量正则项或悄悄使用伪逆。质量与右端方程残差以 max(1,||rhs||∞,||M||∞||a||∞) 归一化，阈值 1e-10。求解失败是 SolverFailure，配置错误保留异常。

## 端口与运行器

`RobotPointPort(model, state, indices, local_points, domain='robot')` 遵守现有 PointPort：世界作用点、Jv、响应 J M⁻¹ Jᵀ p；apply_impulses 只更新 v，q 逐元素不变，返回拥有所有权的 RobotState。一个批次先汇总所有 Jᵀp 再求解，禁止忽略共享祖先交叉项。交换使用同世界作用点，保留输入及重复索引语义。

`RobotProgram` 创建单域阶段工作区：predict 恰一次半隐式 Euler（v'=v+h a，q'=q+h v'）；没有内部约束时位置/速度 sweep 不推进时间。诊断保留力/加速度、能量和方程残差。所有后段失败由原 SceneStepper 恢复 q/v 及显式历史；P5 的接触位置更新通过独立参与者适配器加入，不向 P3 添加未使用的空方法。

## 冻结验证

单杆：L=0.5 m，m=1 kg，COM=(0,-0.25,0)，惯量=(1/48,0.001,1/48)，关节 Z 轴。解析 Mzz=1/12，重力偏置=9.81×0.25 sin(q)。两杆按相同参数串联，第二关节偏移 (0,-0.5,0)。另建含移动/旋转、非零零位旋转的分支树防止只验证平面特例。

- FK 和解析质量/重力：绝对误差≤1e-12（尺度为上述 SI 量级）。点雅可比中心差分 ε=1e-6，误差≤2e-8；角雅可比用 Rdot Rᵀ 独立差分。
- 32 个固定种子样本 q∈[-0.75,0.75]、v∈[-0.5,0.5]、effort∈[-0.5,0.5]：M 对称误差≤1e-12、Cholesky 成功；inverse_dynamics(a)-bias 与 M a 一致 atol=1e-10、rtol=1e-10。
- 独立能量/力、端口功率对偶和多点响应；移动关节科氏项用运动学二阶差分/外部对照，不能只拿同一函数比较自身。
- MuJoCo 固定 3.3.7，单独环境和哈希锁；关闭接触、阻尼、armature、关节限位，固定直接力输入、惯量、重力、Euler/步长。通过公共 API 比较 FK、M、bias、qacc：atol=5e-10、rtol=1e-9。只读取文档和调用 API，不复制源码。
- S1 两杆 q=(0.35,-0.2)、v=(0.1,-0.05)，零驱动，g=(0,-9.81,0)，h=1/2000 s，1000 步（0.5 s），同积分 MuJoCo q/v 最大差≤1e-7；重复本实现逐元素相同。失败保留证据，不调宽阈值。
- 入口别名、非单位轴/四元数、非法惯量/拓扑/形状、未知关节、额外字段、非有限、作用点越界、后段子步失败和配置变化均拒绝或全局恢复。

本阶段性能只记录 FK/M/bias/step 冷热态及有限规模（1/2/8 自由度）；未测项明确列出，不优化。没有外部 MuJoCo 证据时不能标 P3 完成，可先推进独立 P4 工作。
