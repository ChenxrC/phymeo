# 固定树点闭环、四杆反力与驱动功

`RobotPointClosure(first, second, axes_world)` 连接两个 RobotSite 的位置。每个点可属于一个活动 link，也可在世界固定。`axes_world` 的一至三行是世界坐标中的正交单位方向，默认三行单位阵。对每个方向，两个点在该方向投影相同。两个方向可闭合平面四杆，同时保留其转动自由度。多个点约束可以在同一步联合施加。[公开入口](../../src/phymeo/solvers/robot_closure.py)

```python
import numpy as np

from phymeo.sensors import RobotSite
from phymeo.solvers.robot_closure import RobotPointClosure, step_robot_closures

pin = RobotPointClosure(RobotSite(1, (1, 0, 0)), RobotSite(2, (0.5, 0, 0)), np.eye(3)[:2])
step = step_robot_closures(model, state, inputs, [pin], 1 / 256)
if step.report.accepted:
    state = step.state
    force_on_first = step.reaction.force_first_world[0]
```

## 力与步进

用 B 表示方向投影后的两点线雅可比之差，g 表示相应的位置残差，其中第一点取正。初始状态须满足 g=0、Bv=0 至给定公差。求解器使用固定树的完整 M、重力和惯性偏置、当前关节驱动力及各本体的外部 COM 载荷。它沿祖先链解析求取 Ḃv，将约束加速度 Ba+Ḃv=0 与树的运动方程联立。约束空间矩阵为 B M⁻¹ Bᵀ。按各行的有效逆质量归一化后检查秩，并保留原有质量求解的残差检查。[树质量与惯性偏置](../../src/phymeo/dynamics/robot_dynamics.py) · [点雅可比](../../src/phymeo/dynamics/robot_kinematics.py)

`reaction.multipliers` 是步初每个投影方向的乘子，单位 N。`force_first_world`、`force_second_world` 是施加于各点的等大反向世界力，`generalized_force` 为 Bᵀλ。`power_first_w` 和 `power_second_w` 分别用各点步初速度计算。速度先更新，随后推进关节位置。非线性位置残差经质量加权迭代校正，末速度再投影到新姿态的切空间。步末的 `position_correction`、`velocity_projection_impulse` 和 `velocity_residual_m_s` 分开读回。输入状态失配、约束死点、重复约束失秩或投影无法收敛时拒步，可用同一原始状态重试。[解析、秩与重试用例](../../tests/test_robot_point_closure.py)

## 四杆完整窗口

两固定铰相距 1 m，两侧杆各长 0.5 m、质量各 1 kg、关于质心的转动惯量 1/16 kg·m²。水平耦合杆长 1 m、质量 2 kg、关于质心的转动惯量 1/8 kg·m²。三条树边采用左侧转铰、耦合杆相对转铰和右侧转铰。右侧活动闭口施加两个面内方向的点约束。平行四边形的关节坐标为 (θ,−θ,θ)，等效惯量为 3/4 kg·m²。解析参考用单一 θ，根据整机虚功和右侧杆、耦合杆各自的牛顿–欧拉方程求闭口反力。[独立四杆解析](../../tests/reference_newton/robot/four_bar_oracle.py)

水平初态 cosθ=3/5、sinθ=4/5，两个活动铰在 (0.3,0.4) 和 (1.3,0.4) m，θ̇=1/4 rad/s。左固定铰输入转矩 3/8 N·m，得到 θ̈=1/2 rad/s²。闭口施加于耦合杆的力为 (1/4,1/8,0) N，点功率为 −1/64 W，右侧杆上的力做功率为 +1/64 W。

倾斜留出将两根铰轴转到 (4/5,0,3/5)，初始角换为 cosθ=4/5、sinθ=3/5、θ̇=−3/8 rad/s。重力、耦合杆 COM 外力和自由力偶、右侧杆外力偶及左固定铰负向驱动共同参与。表中最大闭口是两个实际本体点之间的三维距离。外功以所有实际质心位移和本体转角计算，动能由接受的状态与完整 M 计算。

| Phymeo HOST FP64，均为 0.25 s | 末角速度 (rad/s) | 最大闭口 (m) | 最大闭口双点净功率 (W) | 外功−动能增量 (mJ) |
| --- | ---: | ---: | ---: | ---: |
| 水平，64 步 | 0.375000000 | 4.45×10⁻¹⁶ | 6.94×10⁻¹⁸ | 0.091553 |
| 倾斜，64 步 | −0.738552493 | 4.45×10⁻¹⁶ | 4.17×10⁻¹⁷ | 0.842512 |
| 倾斜，128 步 | −0.738607527 | 4.45×10⁻¹⁶ | 4.17×10⁻¹⁷ | 0.421209 |

倾斜粗步超过预设 0.8 mJ 检查，原失败保存在收据中。在相同 0.25 s 窗口把物理步长减半后通过原门槛。三个窗口逐物理步核对广义加速度、闭口力、两点各自的功率、实际 q/v 与末位姿。另一个转铰–滑块非线性约束完整推进 32 步，步初约束力为 25/88 N，两个端点功率为 ∓5/88 W；位置与速度都实际发生投影。空间转动–移动–转动支链另用独立速度导数检查曲率项。

## Newton 1.7 的原生闭环

Newton XPBD 直接推进三个刚体与四个真实关节。左、右固定铰和左侧活动铰使用转动关节，右侧活动铰使用球关节。平面内留有与上述四杆相同的一个自由度。将左侧活动铰登记为由耦合杆指向左侧杆，使耦合杆只接收来自闭口球铰的入关节载荷。每步直接从 BODY_PARENT_F 读取其世界 COM 处的传入力和力矩，点速度由同刻实际 body_q 与 COM twist 换点。参考程序保存每步状态、近似闭口反力、速度差和功。Newton 的反力是每物理步有限次迭代施加的平均载荷，PhyMeo 的 λ 是步初加速度约束的瞬时反力。[原生对照](../../tests/reference_newton/robot/four_bar.py) · [Newton 求解器](https://newton-physics.github.io/newton/1.7.0/solvers/index.html)

所有 Newton 窗口均为 64 个 1/256 s 物理步、每步 32 次迭代。下表的力误差是窗口平均力误差向量的模，角速度误差为全窗峰值。预设门槛分别为 0.08 N、0.015 rad/s；同时检查闭口、角度、约束功和功–能量误差。

| 线性/角松弛 | 设备 | 水平最大闭口 (µm) | 水平角速度误差 (rad/s) | 水平/倾斜平均力误差 (N) |
| --- | --- | ---: | ---: | ---: |
| 默认 0.7/0.4 | CPU | 0.361 | 0.06945 | 0.22935 / 0.58348 |
| 默认 0.7/0.4 | CUDA，RTX 4060 Laptop | 0.300 | 0.06937 | 0.22930 / 0.58357 |
| 共同 0.5/0.5 | CPU | 0.315 | 0.000195 | 0.000443 / 0.003181 |
| 共同 0.5/0.5 | CUDA，RTX 4060 Laptop | 0.248 | 0.000165 | 0.000290 / 0.003331 |

共同松弛的四个窗口均通过既定门槛，倾斜窗口闭口积分功在 CPU/CUDA 分别为 −0.02648/−0.03056 mJ。默认系数在两个设备上均超过水平角速度及平均力门槛。Newton 1.7 源码在同一位置约束乘子的线、角修正处分别乘这两个系数。该机制与观测差异相符；本实验同时改变两个系数，只能比较这两种设置的整体结果。

## 复现与边界

```powershell
python -m pytest -s -q tests/test_robot_point_closure.py
.\.venv-newton-1.7.0\Scripts\python.exe tests/reference_newton/robot/four_bar.py --device cpu
.\.venv-newton-1.7.0\Scripts\python.exe tests/reference_newton/robot/four_bar.py --device cuda:0
.\.venv-newton-1.7.0\Scripts\python.exe tests/reference_newton/robot/four_bar.py --device cpu --relaxation default
```

PhyMeo 通过 7 项专项与 63 项相邻回归。Newton 的两个默认配置运行均保留原始退出码 1 和全部窗口读数，共同系数的 CPU/CUDA 均为 0。证据根为 `artifacts/p65-night-robot-closure-20261104-01`。产品覆盖固定根 FP64 标量树、固定世界投影方向、独立且满行秩的理想双边点约束。浮动根闭环、随动约束方向、约束摩擦和设备侧 Phymeo 闭环推进仍需独立验收。
