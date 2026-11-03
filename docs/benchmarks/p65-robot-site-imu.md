# 关节树站点 IMU 的比力、角速率和采样时刻

`measure_robot_imu(model, state, generalized_acceleration, site, *, time_s)` 在给定物理时刻读取一个理想 IMU。返回 `time_s`、`specific_force_m_s2` 和 `angular_velocity_rad_s`。后两项是站点局部坐标下的三分量输出。`RobotSite` 的安装位置从本体原点量起，XYZW 单位四元数将站点轴映到本体轴。固定树沿用 link 索引；浮动树的物理根为 body 0，后续索引对应各子 link。索引 None 表示安装于静止世界。[IMU 入口](../../src/phymeo/sensors/robot_imu.py) · [站点坐标](../../src/phymeo/sensors/robot_frame.py)

```python
from phymeo.sensors import RobotSite
from phymeo.sensors.robot_imu import measure_robot_imu
from phymeo.solvers.robot_closure import step_robot_closures

before = state
step = step_robot_closures(model, before, loads, closures, dt)
if step.report.accepted:
    imu = measure_robot_imu(model, before, step.reaction.acceleration, RobotSite(0), time_s=t)
    state = step.state
```

## 同一时刻的物理量

令 r=R(p站−p质心)，其中安装点和质心偏心都以本体坐标给出。安装点世界加速度为 aCOM+α×r+ω×(ω×r)。从中减去模型世界重力 g，再旋入站点坐标，得到比力；世界角速率 ω 也旋入同一站点坐标。自由落体读数为零，静止支撑读数为旋入传感器坐标的 −g。模型中的质量与主惯量通过实际求解的线、角加速度影响传感器。

固定树的广义加速度是各标量 q 的二阶导数；浮动树前六项是世界根质心加速度与世界角加速度。时间参数由调用者给出，q、v 和广义加速度须属于该时刻。闭环步给出的反力/加速度位于**步初**，与上例的 before 配对。浮动树也可对新接受状态与当时载荷重新求 `floating_dynamics_terms`，获得步末即时样本。[闭环反力时刻](p65-robot-point-closures.md) · [浮动树动力学](p65-floating-tree-dynamics.md)

## 独立试片

| 构型 | 可验收读数 |
| --- | --- |
| 固定基座竖直滑轨被世界点闭环支撑 | 质量 2 kg，g=(0,0,−4) m/s²。连续 4 个物理步逐步读取 +8 N 支撑反力、该步初零广义加速度。传感器绕本体 X 转 +90°，读数为 (0,4,0) m/s² 和零角速率。 |
| 无转动、带水平平移的自由落体 | 在同一重力场推进偏心根 8×1/64 s，读取下落速度和已接受状态。无载各时刻的三轴比力、三轴角速率均为零。 |
| 竖直固定转铰，端点离铰 1/2 m | 质量 2 kg，质心偏心 1/8 m，Izz=1/4 kg·m²，绕铰有效惯量 9/32 kg·m²。安装绕本体 Z 转 +90°。ω=±2、α=0 时读 (0,2,0) m/s²；ω=2、α=±3 时读 (±1.5,2,0) m/s²。角速率翻向只改变陀螺读数。 |
| 自由根世界外力与质心力矩 | 根质心在本体 +1/8 m，安装于 +3/8 m。2 kg 物体受世界 +2 N X 力和 +1.5 N·m Z 力矩，Izz=1/2 kg·m²，初速 ωz=2。首个瞬时比力为 (0.75,0,0) m/s²。后续按惯性世界力、变化的本体朝向、末角速率和固定安装轴逐项核对。 |
| 转动–移动–转动斜轴子链 | 固定根与运动根各一组。由真实 FK 给出的世界安装点速度作独立中心差分，检查角加速度、移动关节科氏项、偏心向心项及站点轴变换。 |

[Phymeo 专项](../../tests/test_robot_imu.py)

## Newton 对照

Newton 1.7 的 `SensorIMU` 从同一个传入 state 读取世界本体姿态与质心 twist，并通过 `BODY_QDD` 接收本体加速度。设备核读取安装点相对于质心的偏移、相应世界重力与安装姿态。Kamino 的 `BODY_QDD` 来自该物理步前后真实质心线/角速度差除以步长，传感器在步末姿态和角速率下使用该步平均加速度。这些试片采用恒定世界质心加速度及主轴角加速度以消除该时间口径对力学预期的歧义。世界站点与自由刚体、支撑体分别读取。[Newton 1.7 IMU](https://newton-physics.github.io/newton/1.7.0/concepts/sensors.html) · [本机原生对照](../../tests/reference_newton/robot/imu.py)

## 0.125 s 末态与全窗误差

| 后端 | 自由落体比力 (m/s²) | 驱动偏心站比力 (m/s²) | 驱动偏心站角速率 (rad/s) |
| --- | --- | --- | --- |
| Phymeo HOST FP64 | (0,0,0) | (0.477137502,0.448103207,0) | (0,0,2.375000000) |
| Newton Kamino / SensorIMU CPU | (0,0,0) | (0.477137685,0.448103666,0) | (0,0,2.375000477) |
| Newton Kamino / SensorIMU CUDA，RTX 4060 Laptop | (0,0,0) | (0.477137804,0.448103756,0) | (0,0,2.375000477) |

Newton 两端各实际推进 8 个 1/64 s 步长，每步读 6 个站点的三轴比力及三轴角速率。六站包括自由落体、支撑体、驱动体偏心点、驱动体质心、负向匀转偏心点和静止世界。每步原生 `BODY_QDD` 都与同一刚体前后两份真实质心 twist 的差除以步长交叉核对。CPU 全窗比力最大分量误差为 3.553×10⁻⁵ m/s²，CUDA 为 3.624×10⁻⁵ m/s²；角速率两端均小于 9.537×10⁻⁷ rad/s。Newton 的支撑体用竖直转铰承担重量，在此载荷下轴上转矩为零。+8 N 支撑反力的直接读回来自 Phymeo 的滑轨点约束；两个引擎在支撑体上的 IMU 比力另行比较。

专项通过 10 项，固定树站点、浮树动力学和闭环相邻回归通过 25 项。Newton 1.7.0 / Warp 1.17.0 的两个原生运行退出码为 0，格式与静态检查通过。首次缺少 IMU 模块的原始退出码 2 与最终源码/日志保存在 `artifacts/p65-night-robot-imu-20261104-01`。

## 复现

```powershell
python -m pytest -s -q tests/test_robot_imu.py
.\.venv-newton-1.7.0\Scripts\python.exe tests/reference_newton/robot/imu.py --device cpu
.\.venv-newton-1.7.0\Scripts\python.exe tests/reference_newton/robot/imu.py --device cuda:0
```

本产品覆盖 NumPy HOST、同一时刻给定真实广义加速度的固定/浮动标量活动树及世界站点。接触冲量的时间分摊、噪声、偏置、延迟与设备侧 Phymeo IMU 需要各自的采样模型及验证。
