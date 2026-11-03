# Newton 1.7.0 机器人与资产工具链对照计划

## 本轮问题

两连杆的运动能否从程序建模和本地机器人资产得到同样的结果？在固定底座上，工具偏置、角度单位、移动轴或速度参考点只要有一处错误，非零姿态和速度就会偏离下列手算值。另用单轴滑块回答控制器输出的力是否真正推进了状态。两组测试均可脱离整机、桌面和布料运行。

[案例输入、独立预期和逐量绝对容差](../../../tests/reference_newton/robot/cases.json) · [执行入口](../../../tests/reference_newton/robot/baseline.py) · [矩阵格式与运行环境](../../../tests/reference_newton/README.md)

## 固定底座、工具点与传感读数

世界单位为米、千克、秒，Z 向上。角坐标 θ 绕 +Z，移动坐标 s 沿工具局部 +X。铰点 A = (1/4, −1/2, 1/8) m。第一连杆长度 1/2 m，滑轨安装时绕局部 Z 转 90°。工具上的球半径 1/32 m，其中心 e = (1/8, −1/16, 1/32) m 和工具质心重合。程序式定义保存在目录的 arm fixture。URDF 的关节姿态和转动限位按弧度声明，MJCF 显式选择度，USD 的转动限位也用度。USDA 的米制、质量单位和竖轴均显式给出。[URDF](../../../tests/reference_newton/robot/arm.urdf) · [MJCF](../../../tests/reference_newton/robot/arm.mjcf.xml) · [USDA](../../../tests/reference_newton/robot/arm.usda)

令 R 表示绕 Z 转 θ，工具原点为 A + R(1/2, s, 0)，球心 P = A + R(9/16, s + 1/8, 1/32)，滑轨方向 u = R(0, 1, 0)。球心速度为 θ̇ Z × (P − A) + ṡ u。预期坐标与速度已固定在各案例中。

| 构型 | 输入 | 工具原点 m | 球心 m | 球心速度 m/s |
| --- | --- | --- | --- | --- |
| 零位 | θ=s=θ̇=ṡ=0 | (0.75, −0.5, 0.125) | (0.8125, −0.375, 0.15625) | (0, 0, 0) |
| 四分之一转 | θ=π/2，s=1/8，θ̇=1/2，ṡ=−1/8 | (0.125, 0, 0.125) | (0, 0.0625, 0.15625) | (−0.15625, −0.125, 0) |
| 斜轴留出 | cos θ=3/5，sin θ=−4/5，s=−1/16，θ̇=−3/4，ṡ=1/16 | (0.5, −0.9375, 0.125) | (0.6375, −0.9125, 0.15625) | (−0.259375, −0.253125, 0) |

Newton 对两个滑轨坐标调用设备 FK，两次相隔 1/64 m 的工具原点给出 u。身份变换的静态世界 site 作为传感器参考，SensorFrameTransform 对模型中真实碰撞球读出 P。速度取自设备 body_qd 的世界质心分量，同时核对球心与导入的质心重合。资产案例还逐项核对转动/移动关节类型、范围、球形状、实际碰撞体数量及球的局部中心。URDF 含额外可视方盒，可检测把显示几何当作碰撞体的错误。[Newton 关节与速度约定](https://newton-physics.github.io/newton/1.7.0/concepts/articulations.html) · [传感器](https://newton-physics.github.io/newton/1.7.0/concepts/sensors.html)

Phymeo 程序式正控用 RobotModel。HOST 从 FK 与 RobotPointPort 读取位姿和速度。CUDA 运行 TyxRobotMath.geometry 和 jacobians，返回设备求得的工具原点、轴、质心与点雅可比，主机完成最后的雅可比乘速度。报告含已完成的 kernel 调用和下载次数。这个正控给出了相同物理标记的运动，格式导入单独记 gap。[模型定义](../../../src/phymeo/core/robot.py) · [点端口](../../../src/phymeo/coupling/robot_port.py) · [设备运动学](../../../src/phymeo/backends/tyx_robot_math.py)

## 运行与判据

| 难度 | 登记 ID（均以 ROBOT- 开头） | 依赖与预期 | 止损判据 |
| --- | --- | --- | --- |
| 易 | FK-ZERO | 程序式 FK。滑轨 +X 映到世界 +Y，校准零位。 | 原点、工具标记或方向超限时，先检查本地铰点和四元数。 |
| 中 | FK-TURN-SLIDE、FK-OBLIQUE-HOLDOUT | 同一机械臂，两种转角、滑移符号和非零速度。三维位置与速度绝对容差 2 µm、2 µm/s，差分方向容差 2×10⁻⁵。 | 零位正确、非零位错误时保留原样本，分开查看关节父子变换与速度参考点。 |
| 中、高 | URDF-TOOL、MJCF-DEGREES、USD-ZUP | 三个小型自包含资产使用完全相同的四分之一转输入。应得到 −π/2 至 2π/3 rad、−0.1 至 0.25 m 及唯一偏心碰撞球。 | 单格式失配时对比已解析模型字段与同输入传感器输出，再定位资产层和设备 FK。 |
| 易 | USD-CENTIMETER-WARNING | 在独立内存层将同一 USDA 的 metersPerUnit 改为 0.01。HOST 应读回 0.01 及一条刚体非单位米警告。 | 只登记导入诊断。厘米刚体的正确 SI 运动需要独立缩放案例。 |
| 高 | IK-REACHABLE、IK-OFFPLANE | 已知可达点 P=(0, 1/16, 5/32) m；把它在 Z 方向抬高 1/16 m 给出恰好 1/16 m 的最小残差。固定同一非解种子 (1.3 rad, 0.09 m)。 | 40 次迭代预算，读回所得 q 并重新做 FK。位置误差超 0.1 mm 或不可达项声称零残差时停止。 |
| 中 | DRIVE-REVERSE、DRIVE-NEGATIVE-HOLDOUT | 零重力 2 kg 滑块，h=1/16 s，kp=16 N/m、kd=1 N·s/m、限力 ±4 N。核对每一子步的力、速度、位移。 | 力正确而速度错误时，先核对 mass、被动阻尼及重复施力。保存最早失败步。 |

正向再反向的四步输入要求实际力为 (+4, +4, −4, −4) N，速度为 (1/8, 1/4, 1/8, 0) m/s，半隐式位置为 (1/128, 3/128, 1/32, 1/32) m。反向起步留出在第三步回到初始位置，末速度为 +1/8 m/s。Newton 将 DrivePD 和 ClampingMaxEffort 的输出散加到 Control.joint_f，交由 SolverFeatherstone 推进。Phymeo 把 solve_bounded_pd 的命令送进 make_robot_runtime，逐步读取实际接受的状态。两种控制律在本输入的每一步均达到饱和，因此可以逐步比较。非饱和时显式 PD 和隐式预测要分别设定离散参照。资产上的限位读回说明单位与上下界映射，物理撞停需独立动力学试验。[Newton 执行器](https://newton-physics.github.io/newton/1.7.0/concepts/actuators.html) · [Phymeo 限力控制](../../../src/phymeo/control/bounded_pd.py) · [机器人步进器](../../../src/phymeo/solvers/robot.py)

点 IK 使用位置目标及关节限位。目标的平面内偏移是 (−1/4, 9/16) m，θ=π/2、s=1/8 为界内解。对称根 s=−3/8 超出平移下限。离平面留出检查几何可达性的下界和投影位置。Newton 通过 IKSolver 的解析雅可比求目标，Phymeo 使用 solve_point_ik。各自重新测量偏心点的世界坐标。这两个案例还没有模拟执行器跟踪 IK 目标。[Newton 点目标 API](https://newton-physics.github.io/newton/1.7.0/api/newton_ik.html) · [Phymeo 点 IK](../../../src/phymeo/control/point_ik.py)

## 复现及记录

在集成工作树根目录使用串行执行槽。每次选择一个引擎、一个设备和新的结果路径，失败记录保留案例 ID 和原始异常。Newton 环境包含 Warp、本地基元导入所需的 Python 包、OpenUSD 与 Newton USD schema。案例未引用远程模型，MJCF 使用 Newton 的导入器。[依赖与命令](../../../tests/reference_newton/README.md) · [Newton USD 解析](https://newton-physics.github.io/newton/1.7.0/concepts/usd_parsing.html)

```powershell
.\.venv-newton-1.7.0\Scripts\python.exe tests/reference_newton/run_matrix.py --lane robot --list
.\.venv-newton-1.7.0\Scripts\python.exe tests/reference_newton/run_matrix.py --lane robot --engine newton --device cpu --out robot-newton-cpu.json
.\.venv-newton-1.7.0\Scripts\python.exe tests/reference_newton/run_matrix.py --lane robot --engine newton --device cuda:0 --out robot-newton-cuda.json
```

Phymeo 使用已装 tyx 的项目 Python，PYTHONPATH 指向本工作树 src，选择 --engine phymeo。HOST 的点 IK、限力命令与实际滑轨步进只登记 CPU。四组报告逐条检查 pass、fail、gap 和 device_not_covered，实际通过数量取自报告。源码静态核对、安装成功、CPU 烟测与 CUDA 烟测各自保留原有范围。

## 后续模块工作

| 顺序与难度 | 输入、独立预期和交付 | 依赖与止损 |
| --- | --- | --- |
| 1 · 易 | 为 Phymeo 增加最小 URDF 固定基座桥接。把本例的两个活动关节、具名碰撞球、质心和限位映射到公开模型，检查额外 visual 方盒的过滤。用同一角度、滑移与斜轴留出跑读回。 | 先确定固定根到 RobotModel 父索引 −1 的映射。任一已知字段丢失则记录具体元素和模型字段。 |
| 2 · 中 | 增补度/弧度、Y 向上舞台、非单位比例的 MJCF/USD 留出。对同一实长 0.5 m、同一世界轴核对位置、球半径与限位，校验每个格式一次建模一次转换。 | 先通过米制 Z 向上案例，再确认具体解析器的缩放能力。单位输入发出不支持告警时停止该物理对照。 |
| 3 · 中 | 用 2 kg 滑块测试限位受力。给下限外 1 mm 的位置、刚度 2000 N/m、零速度，恢复力应向上界方向且为 2 N。镜像上限、开离/闭合和卸载各设留出。再用含延迟的 PD 测阶跃生效的物理子步。 | 先写出两端求解器的限位定律和延迟索引。恢复力、冲量与位置钳制分开分类。 |
| 4 · 中 | 传感器增加非身份参考系。令参考 site 在原点绕 Z 转 +90°，测量世界点 (1/4, 0, 0) m，预期局部 (0, −1/4, 0) m。后续加入静止/随动 IMU、成对接触力与符号。 | 先核对 site、姿态时间戳和各 solver 的 observables。缺少所需加速度或力通道时登记具体传感器缺口。 |
| 5 · 高 | 浮动根设 q 为平移及 XYZW 单位四元数，质心局部 (1/8, 0, 0) m，世界质心速度 (0, 1/4, 0) m/s，角速度 (0, 0, 2) rad/s。身份姿态时原点速度必须为零。再观察雅可比和一段自由飞行。 | Phymeo 当前 RobotModel 只表达固定树。需要公开的 7 个坐标与 6 个速度接口。根状态或速度参照不清时停在转换试片。 |
| 6 · 高 | 四连杆平行四边形两固定铰相距 1 m、两侧杆各 0.5 m。两杆角度取 cos θ=3/5、sin θ=4/5，活动铰应在 (0.3, 0.4) 与 (1.3, 0.4) m。推进单个驱动并测闭合残差、反力和功。 | 从三条树边和独立闭合约束开始，逐个核对求解器的 loop joint 支持。闭合残差或约束力通道缺失时保留拓扑及失败位置。[Newton 求解器矩阵](https://newton-physics.github.io/newton/1.7.0/solvers/index.html) |

以上模块按每个公开入口、手算正控和留出独立验收，最后才进入双臂任务的整机轨迹及布料载荷回归。[项目续作入口](../../development/p65-continuation.md)
