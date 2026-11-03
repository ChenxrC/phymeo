# 浮动树的内部驱动、根反作用与惯性偏置

`step_floating_robot(model,state,inputs,dt)` 推进一个物理浮动根和零至多个标量转动或移动关节。输入 `FloatingRobotInputs(effort,force,torque)` 的 effort 按子关节排序，单位分别为 N·m 和 N。后两项按根、子体顺序给定，均以世界轴表达，torque 的参考点为各自质心。重力来自模型，根和子树须使用同一世界重力。`FloatingRobotInputs.zeros(n)` 为 n 个子关节创建零载荷。无子链时使用原有单刚体步进器。[公共实现](../../src/phymeo/solvers/floating_tree.py) · [浮动根状态与速度约定](p65-floating-robot.md)

```python
import numpy as np
from phymeo.solvers.floating_tree import FloatingRobotInputs, floating_dynamics_terms, step_floating_robot

# model 含一根一子，单位质量、坐标与初速见下表。
loads = FloatingRobotInputs(np.array([3.0]), np.zeros((2, 3)), np.zeros((2, 3)))
terms = floating_dynamics_terms(model, state, loads)
acceleration = terms.acceleration
result = step_floating_robot(model, state, loads, 1 / 8)
```

## 同一步的动力学方程

广义速度依次为根质心世界速度、根世界角速度、子关节速度。对每个本体使用关于其质心的线、角雅可比 Jv 和 Jω，世界惯量为 I=R diag(Ib) Rᵀ。完整质量矩阵累加 m JvᵀJv+Jωᵀ I Jω。外力与力偶按虚功投影到所有列，内部驱动仅加到对应的标量关节列。求解器复用固定树的稠密质量组装、正定性检查和归一化残差门槛。[共同质量与求解实现](../../src/phymeo/dynamics/robot_dynamics.py)

偏置从零广义加速度的实际世界运动递推。根质心加速度为零，偏心根原点仍有 −ω×(ω×Rc) 加速度。子体原点包含父角加速度和向心项，移动关节还带 2ω父×(轴·关节速度)，转动关节的角加速度带 ω父×(轴·关节速度)。将各本体的 m(aCOM−g) 与 Iα+ω×Iω 投影得 bias。诊断返回 mass_matrix、bias、applied 和 acceleration，满足 M a+bias=applied。内部驱动通过质量矩阵交叉块使根与其他活动关节共同响应。

每步先更新全部速度，再用末速度推进根质心、姿态和各关节。根姿态按世界角速度左乘指数映射，由质心位移及姿态变化还原偏心原点。数值失败时返回未推进的输入状态。该入口是 NumPy FP64 的稠密参考求解，覆盖具有正质量、物理主惯量和合法拓扑的标量活动树。[公开状态与质心雅可比](../../src/phymeo/dynamics/floating_robot.py)

## 独立构型

| 输入 | 解析预期 |
| --- | --- |
| 2 kg 根与 1 kg 子体沿 X 滑动，零重力，内部 +3 N | 根平移与相对滑移的质量子块为 [[3,1],[1,1]] kg。根加速 −1.5 m/s²，相对滑移 +4.5 m/s²，子世界加速 +3 m/s²。每步总线动量为零。输入 −3 N 并对整机施加共同初速时，相对反作用翻向，质心按初速漂移。 |
| 根受 +2 N、子受 −1 N，滑轨仍施内部 +3 N，g=(0,0,−1) | 根世界加速度 (−1/2,0,−1)、子世界加速度 (2,0,−1) m/s²。总外力 (1,0,−3) N，逐步总动量增量为该力乘以步长。 |
| 斜铰位于根质心，子质心到铰 1/2 m，相对驱动 +5/4 N·m | 世界轴 u=(0,3/5,4/5)，切向 t=(0,4/5,−3/5)。根、子世界角加速度分别 −5u/2、3u rad/s²，相对角加速度 11/2 rad/s²，根、子质心加速度 −t/2、t m/s²。用真实末姿态和 body twist 计算角动量和动能。 |
| 根以 2 rad/s 绕 Z 转动，滑轨伸长 1/2 m、向外 1/4 m/s，无外载 | 约化质量 2/3 kg。绕整机质心的转动惯量为 11/12 kg·m²。角动量平衡给 αz=−4/11 rad/s²，根横向加速度 −3/11 m/s²，相对加速度 2 m/s²。末动量和动能逐一对照每种积分器的离散姿态。 |
| 偏心根、斜轴转动/移动/转动分支、非主轴惯量和世界外载 | 对零广义加速度曲线做独立中心差分，得本体角加速度与质心加速度。对每个广义虚速度分别核对偏置、外力虚功和动能；产品递推使用解析导数。 |

Newton 1.7 的对照在一组 articulation 中建立真实 FREE 根与一个活动关节，以 `Control.joint_f` 施加内部驱动、`State.body_f` 施加世界 COM 外载。记录 SolverFeatherstone 同一物理末时刻的 joint_q、joint_qd、body_q、body_qd、质心、动能及世界线动量和关于世界原点的角动量，并比较引擎原生质量矩阵给出的动能。Newton 主轴四元数增量采用归一化欧拉步，旋转留出按该离散式推导。[冻结输入](../../tests/reference_newton/robot/floating_tree_cases.json) · [Newton 观察](../../tests/reference_newton/robot/floating_tree.py) · [Newton 1.7 空间力/速度约定](https://newton-physics.github.io/newton/1.7.0/concepts/conventions.html)

## 本机实测

| +3 N 滑轨 | 首步根 / 相对 / 子世界加速度 (m/s²) | 4 步末总 Px (kg·m/s) | 末动能 (J) |
| --- | --- | --- | --- |
| Phymeo NumPy HOST FP64 | −1.5 / 4.5 / 3 | −2.22×10⁻¹⁶ | 1.687500000 |
| Newton 1.7 CPU | −1.499999881 / 4.499999523 / 2.999999523 | −1.19×10⁻⁷ | 1.687499553 |
| Newton 1.7 CUDA，RTX 4060 Laptop | −1.499999881 / 4.499999523 / 2.999999523 | −1.19×10⁻⁷ | 1.687499553 |

反向加共同初速的 4 步后，Phymeo 实测整机总线动量 (1.5,−0.75,0.375) kg·m/s、动能 2.1796875 J。受外力与重力的 3 步后总线动量为 (0.375,0,−1.125) kg·m/s。静止偏心斜铰的一步相对角加速度为 5.5 rad/s²，末动能为 5.24520874×10⁻⁵ J。无载旋转滑轨从 2.0859375 J 出发，在 1/256 s 后，Phymeo 动能增加 4.24934×10⁻⁵ J，世界 Y 动量增加 8.86751×10⁻⁵ kg·m/s。Newton CPU/CUDA 的相同增量分别约为 4.23222×10⁻⁵ J 和 8.85725×10⁻⁵ kg·m/s。两个数值步进器的漂移均经过独立末态姿态及速度公式核对。

Phymeo HOST 共通过 11 项专项，另有 34 项相邻回归通过。Newton 1.7.0、Warp 1.17.0 在 CPU 与实际 CUDA 上各运行 5 个构型、13 个物理步。多轴三孩子分支的解析偏置、大载荷拒步与重试为 Phymeo HOST 专项。[HOST 独立验收](../../tests/test_floating_tree_dynamics.py)

各向同性留出找到了一个额外失败。对于 XYZW=(1,2,3,4)/√30 的同心自由铰，两个球惯量为 0.3I、0.7I，根的实际角惯量块应为 I。原来的逐元素纯相对对称检查误拒了接近零的非对角舍入项。共享组装器现在用与该元素单位相同的尺度 32ε√Mii√Mjj 判断消去误差，并在该分支对称化后再做 Cholesky 检查。单位尺度上刻意注入的 10⁻⁸ 非对称项仍被拒绝。证据目录 `artifacts/p65-night-floating-tree-20261104-01` 包含缺少新模块的首次退出码 2、该留出的首次退出码 1、修复后的全部收据及静态检查。最终专项、独立引擎对照、相邻回归和静态检查的退出码均为 0。

## 复现

```powershell
python -m pytest -s -q tests/test_floating_tree_dynamics.py
.\.venv-newton-1.7.0\Scripts\python.exe tests/reference_newton/robot/floating_tree.py --device cpu
.\.venv-newton-1.7.0\Scripts\python.exe tests/reference_newton/robot/floating_tree.py --device cuda:0
```

固定根与单自由刚体继续有独立验收。多自由关节、闭环、接触约束、浮动树驱动饱和/延迟和设备侧 Phymeo 耦合求解需各自的实现与测试。
