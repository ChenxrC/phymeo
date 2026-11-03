# 固定基座机器人在移动参考系中的 site 读数

`measure_relative_site` 从同一个已发布的 RobotState 读取 q、v，给出目标在参考 site 中的位置、朝向以及相对线速度和角速度。两个 site 可以在世界、同一连杆或不同支链上。局部位置使用米，单位四元数按 XYZW 排列。[公开接口](../../src/phymeo/sensors/robot_frame.py)

```python
from phymeo.sensors import RobotSite, measure_relative_site

reference = RobotSite(0, (0.25, 0.0, 0.0))
target = RobotSite(None, (0.0, 0.5, 0.0))
measurement = measure_relative_site(model, step.state.domains["robot"], reference, target)
position = measurement.position_m
orientation = measurement.rotation
velocity = measurement.linear_velocity_m_s
spin = measurement.angular_velocity_rad_s
```

`rotation` 将目标坐标轴映到参考坐标轴。令 d=pT−pR，Rr、Rt 分别为两个 site 到世界的旋转。四个输出依次为 Rrᵀd、RrᵀRt、Rrᵀ(vT−vR−ωR×d)、Rrᵀ(ωT−ωR)。线速度是参考坐标内位置的时间导数。vR 和 vT 按各自的局部偏心点计算。世界 site 的线、角速度均为零。USD 入关节锚点有偏置时，先用资产的 authored_to_model_points 将源连杆局部点转换到 RobotModel 的 link 坐标。[点与角雅可比](../../src/phymeo/dynamics/robot_kinematics.py) · [USD 源坐标转换](../../src/phymeo/builders/usd.py)

## 解析输入与实测

| 构型 | 观测 |
| --- | --- |
| 静态参考旋转 +90° | 世界 +X 方向的 1/4 m 偏置读为参考系 −Y 方向的 1/4 m，目标相对朝向 Rz(−90°)。 |
| 两个 site 共转 | 局部点分别为 (1/8,0,0)、(3/8,1/16,0) m，目标另有 Rx(90°) 朝向。当连杆以 2 rad/s 转动，两 site 的世界线速度不同，相对线速度与角速度均为零。 |
| 参考转动，目标静止 | 参考位于 Z 轴心，q=π/2、v=2 rad/s，世界目标在 (1/4,0,0) m。相对线速度 (−0.5,0,0) m/s，相对角速度 (0,0,−2) rad/s。倒转 v，两项速度换号，位姿不变。 |
| 双支链斜轴留出 | 参考连杆在 cosθ=3/5、sinθ=4/5 处转动，目标支链一滑一转。点端口独立读得世界速度 (−0.1,0.075,0) 和 (0.125,−0.03125,0) m/s。相对位置 (−0.2,−0.275,0.375) m，线速度 (−0.0875,−0.14375,0) m/s，角速度 (0.15,−0.2,−0.5) rad/s。 |
| 接受与拒绝 | 2 kg 滑轨、延迟 2 子步、+2 N 前馈、h=1/16 s。前三个接受子步的 (x,vx) 读数为 (0,0)、(0,0)、(1/256,1/16)。独立验收器拒绝同帧第三步后，传感器读回发布的帧初状态 (0,0)。 |

四组几何的原始输入和期望独立存储。读取接口另外验证了点绑定、无效索引/四元数、只读输出和重复观测。Phymeo 的 7 项 HOST 测试及 1 项实际 CUDA 留出通过。设备留出禁用 HOST FK/雅可比回退，实际完成 3 次内核调用、33 次上传与 10 次下载。设备以 FP32 计算 FK 与点雅可比，HOST 用回读的几何和雅可比完成局部点投影、Jv 与参考系换算。[数值构型](../../tests/reference_newton/robot/site_frames.json) · [Phymeo 专项](../../tests/test_robot_frame_sensor.py)

### 各自与手算预期的最大分量绝对误差

| 实际运行 | 位置 m | 旋转矩阵 | 线速度 m/s | 角速度 rad/s |
| --- | ---: | ---: | ---: | ---: |
| Phymeo HOST，四个正向构型 | 5.56×10⁻¹⁷ | 2.23×10⁻¹⁶ | 3.07×10⁻¹⁷ | 2.78×10⁻¹⁷ |
| Phymeo TYX CUDA，斜轴留出 | 1.19×10⁻⁸ | 4.38×10⁻⁸ | 2.29×10⁻⁹ | 5.97×10⁻⁹ |
| Newton 1.7 CPU，四个构型与一条反向 | 5.97×10⁻⁸ | 3.58×10⁻⁷ | 2.99×10⁻⁸ | 2.39×10⁻⁷ |
| Newton 1.7 CUDA，同五条读数 | 5.97×10⁻⁸ | 1.44×10⁻⁷ | 2.99×10⁻⁸ | 5.97×10⁻⁸ |

表中误差向上取整至三位有效数字。HOST 每项绝对容差为 2×10⁻¹²，FP32 为 2×10⁻⁵。Newton 的 eval_fk 与 SensorFrameTransform 在指定设备运行，位置和旋转直接来自传感器。两项速度独立由同刻 body_qd 推导。对照程序在目标 site 的同一个世界点分别求两个刚体的速度场，相减后旋入参考坐标。每个速度场都从其所属刚体的真实质心换点。[独立 Newton 对照程序](../../tests/reference_newton/robot/site_frames.py) · [Newton 1.7 传感器约定](https://newton-physics.github.io/newton/1.7.0/concepts/sensors.html)

## 复现

从仓库根目录运行。前两项使用项目解释器及该工作树的 src，Newton 两项使用独立的 1.7.0 环境和 Warp 1.17.0。CUDA 项的临时目录每次选择新路径。

```powershell
python -m pytest -s -q tests/test_robot_frame_sensor.py -m "not tyx"
python -m pytest -s -q tests/test_robot_frame_sensor.py -m tyx --require-tyx --tyx-mode=cuda --basetemp artifacts/site-cuda-local
.\.venv-newton-1.7.0\Scripts\python.exe tests/reference_newton/robot/site_frames.py --device cpu
.\.venv-newton-1.7.0\Scripts\python.exe tests/reference_newton/robot/site_frames.py --device cuda:0
```

当前入口适用于固定基座、每 link 一个标量关节的 RobotModel。后续 IMU 验收还需同刻真实加速度和重力约定，浮动根需独立验证 7 个配置坐标、6 个速度坐标及质心速度换点。
