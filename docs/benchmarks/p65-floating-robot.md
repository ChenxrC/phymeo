# 浮动根、分支运动学与单刚体自由飞行

根本体的世界原点、质心和任意安装点可以是三个不同的位置。设根质心在本体坐标为 c，根原点为 p、朝向为 R。世界质心在 p+Rc，原点速度为 vCOM−ω×Rc。给定 c=(1/8,0,0) m、单位朝向、vCOM=(0,1/4,0) m/s 和 ω=(0,0,2) rad/s，根原点瞬时静止。本体 X 坐标为 1/4 m 和 −1/8 m 的点分别以 +1/2 和 −1/4 m/s 沿世界 Y 运动。[Newton 1.7 的速度参照点](https://newton-physics.github.io/newton/1.7.0/concepts/conventions.html)

## 向树提供根状态

`FloatingRobotModel(root, root_com_m, children=None)` 接收单个具有正质量的 FP64 RigidBodyModel 和可选的 RobotModel 标量关节树。根惯量为根本体系中关于其质心的主惯量。世界线、角速度单位分别为 m/s、rad/s。q 以根原点 xyz、根到世界的 XYZW 单位四元数开始，后接各标量关节坐标。v 以世界质心线速度和世界角速度开始，后接关节速度。根占 body 0，子树原 link i 对应 body i+1，原 parent −1 的支链连在根本体系。位置、惯量和状态数组使用 FP64。[类型和运动学](../../src/phymeo/dynamics/floating_robot.py)

```python
import numpy as np

from phymeo.core.rigid import RigidBodyInputs, RigidBodyModel
from phymeo.dynamics.floating_robot import (
    FloatingRobotModel, FloatingRobotState, floating_kinematics, floating_point_jacobians,
)
from phymeo.solvers.free_root import step_free_root

root = RigidBodyModel(np.array([0.5]), np.array([[0.25, 0.375, 0.5]]), np.zeros(3))
model = FloatingRobotModel(root, np.array([0.125, 0, 0]))
state = FloatingRobotState(np.array([0., 0, 0, 0, 0, 0, 1]), np.array([0., 0.25, 0, 0, 0, 2]))
frames = floating_kinematics(model, state)
linear, angular = floating_point_jacobians(model, state, [0], np.array([[0.25, 0., 0.]]))
world_point_velocity = linear[0] @ state.v
step = step_free_root(model, state, RigidBodyInputs.zeros(1), 1 / 32)
```

`frames.position` 和 `frames.rotation` 给出每个本体原点与本体到世界的旋转矩阵。`com_position`、`origin_velocity`、`com_velocity` 和 `angular_velocity` 对应同一时刻、同一世界坐标系。`floating_point_jacobians` 为每个本体局部点返回两个 3×(6+n) 矩阵。每对矩阵的列次序等于 v，点力 F 与自由力偶 T 对应的广义载荷为 JvᵀF+JωᵀT。[全树解析与留出](../../tests/test_floating_robot.py)

| 构型 | 预期 |
| --- | --- |
| 根接 Z 转轴，转轴下接 X 滑轨，根另接 Y 滑轨 | 根原点 (1/4,−1/2,1/8) m。第一个关节转 π/2，两个滑轨分别取 1/8、1/4 m。转轴与两滑轨速度分别为 1/2 rad/s、−1/4、1/8 m/s。嵌套滑轨上偏心点在 (5/8,−1/8,3/16) m，其世界速度为 (−15/16,7/16,0) m/s。 |
| 对该点施加 F=(2,−3,4) N、T=(1/2,−1/4,3/4) N·m | 根绕质心的广义力矩为 (35/16,−9/8,−3/4) N·m。三个子关节广义载荷为 (3/8 N·m,−3 N,0 N)，全树瞬时功率为 −21/16 W。 |
| 斜轴、多个非身份姿态和并联支链重排 | 关节轴 (3/5,4/5,0)、(0,−4/5,3/5) 与非身份零位旋转组合。角速度各列通过绕根质心的世界旋转扰动及旋转矩阵导数核对。整体刚性变换与合法重排后比较所有世界场和列。 |

Newton 对照使用真实 FREE 关节、`eval_fk`、`eval_jacobian`。本体位姿、质心世界 twist、质心雅可比均由引擎读取。点速度及点线雅可比由所属本体的质心结果换点，广义力和功率在此基础上计算。两端对照同一组手算量。[对照程序](../../tests/reference_newton/robot/floating_root.py) · [Newton 1.7 运动学 API](https://newton-physics.github.io/newton/1.7.0/api/newton.html)

## 单个自由根的逐步运动

`step_free_root` 用现有刚体步进器推进质量、主惯量、重力和外部 COM 载荷。它返回 StepResult，每次接受后读回新的根原点、XYZW、质心速度和角速度。16 个 1/32 s 无载子步后，质心到达 (1/8,1/8,0) m，Phymeo 主轴自转角为 1 rad，原点在 ((1−cos 1)/8,(1−sin 1)/8,0) m。加载留出使用 2 kg、Izz=1/2 kg·m²、净力 (4,−2,−2) N、力矩 (0,0,1) N·m，再用非主轴角速度检查无载陀螺加速度。[单刚体推进](../../src/phymeo/solvers/free_root.py) · [解析逐步用例](../../tests/test_free_root_dynamics.py)

Newton 的 SolverFeatherstone 对各物理子步的末角速度使用归一化欧拉四元数更新。主轴自转的每步转角为 2 atan(hω/2)，Phymeo 指数映射的转角为 hω。两端分别核对各自的方向、偏心原点、质心轨迹与世界速度。Newton 加载试验用等效净力，Phymeo 将世界外力和重力分别输入。[Newton 实际逐步读回](../../tests/reference_newton/robot/free_root_flight.py)

## 本机实测

| 后端 | 分支工具点的世界速度 (m/s) | 广义力做功率 (W) | 16 步无载的末质心 (m) |
| --- | --- | --- | --- |
| Newton 1.7 CPU | (−0.937500019,0.4375,0) | −1.312500052 | (0.125,0.125,0) |
| Newton 1.7 CUDA，RTX 4060 Laptop | (−0.9375,0.4375,0) | −1.312499983 | (0.125,0.125,0) |

Phymeo HOST FP64 通过 5 项浮根运动学和 6 项单刚体动力学测试。加上原固定树运动学与刚体验证，同次共通过 84 项。16 步无载和 8 步加载均逐步对照独立预期。根原点速度为零的冻结构型在 Newton CPU、CUDA 上也都得到精确零。两个 Newton 后端各实际运行 3 个 FREE 运动学构型和 24 个 Featherstone 物理步，FP32 绝对容差为 2×10⁻⁵。加载末的 vCOM 为 (1,−1/2,−1/2) m/s，ω 为 (0,0,1) rad/s，CPU 与 CUDA 的最大分量误差均小于 1.3×10⁻⁷。Phymeo 本轮覆盖 NumPy HOST，CUDA 数据来自 Newton 的 Warp 核。

首轮收据分别保留缺少运动学模块与缺少动力学模块的原始退出码 4 和 2。项目专项、CPU/CUDA 四次 Newton 运行和最后的两项静态检查退出码均为 0。格式化前后六份 Python 文件逐份核对语法树与 SHA-256。原始收据目录为 `artifacts/p65-night-floating-root-20261104-01`。

## 复现范围

从仓库根目录运行。项目解释器的 PYTHONPATH 指向该工作树 src。后四项使用独立 Newton 1.7.0 和 Warp 1.17.0 环境。

```powershell
python -m pytest -q tests/test_floating_robot.py tests/test_free_root_dynamics.py
.\.venv-newton-1.7.0\Scripts\python.exe tests/reference_newton/robot/floating_root.py --device cpu
.\.venv-newton-1.7.0\Scripts\python.exe tests/reference_newton/robot/floating_root.py --device cuda:0
.\.venv-newton-1.7.0\Scripts\python.exe tests/reference_newton/robot/free_root_flight.py --device cpu
.\.venv-newton-1.7.0\Scripts\python.exe tests/reference_newton/robot/free_root_flight.py --device cuda:0
```

Phymeo 运动学适用于一个浮动根及任意现有标量关节树。本轮动力学只接受根单刚体及给定 COM 载荷。带活动子链的耦合惯量、关节约束反力、循环关节、接触和传感器加速度需要各自的动力学入口与验收。
