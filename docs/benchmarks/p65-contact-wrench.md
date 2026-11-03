# 从已发布接触记录测量力与力矩

`measure_contact_wrench` 读取 CollisionLoadSystem 已发布帧的步初力，或明确传入的 SurfaceLoads 状态样本。直接传 CollisionLoadStep 时，用 `published_by=system` 确认它是当前已发布帧。外层事务内的候选帧可能已完成局部计算，传感器会拒绝把它作为已发布值。`measure_cloth_link_wrench` 从已接受 SceneStepResult 中读取选定物理子步的中点布–连杆力。结果包含来源、采样相位、实际参与形状对和站点侧数；中点样本还包含子步编号与实际步长。[传感器实现](../../src/phymeo/sensors/contact_wrench.py)

```python
from phymeo.sensors.contact_wrench import ContactSelector, measure_contact_wrench

pad = ContactSelector(world=0, body=10)        # 该刚体的全部注册形状
sheet = ContactSelector(world=0, body=20, shape=5)
sample = measure_contact_wrench(system, pad, counterparts=[sheet],
                                reference_world_m=(0, 0, 0.25),
                                sensor_to_world_xyzw=(0.5, 0.5, 0.5, 0.5))
force_n = sample.force_n
moment_nm = sample.torque_nm
```

## 选侧、作用点和单位

数字世界、本体及形状沿用 CollisionKey；也可将 CollisionKey 直接用作精确形状选择器。省略 shape 汇总所属本体，省略 counterparts 计入所有对方，显式空集合的合力为零。两个不同世界的形状不会互为对方。结构布的选择器使用布域名及可选的面 key，机器人使用 (域名, link 索引) 及可选的胶囊 key。例如 `ContactSelector(0, ("arm", 0), "beam")`。该单场景报告自身没有数值世界索引，调用者的 world 参数绑定其世界，传感器核对选择器与该绑定。

对每个入选站点，使用感测侧实际的世界力 F 及该侧实际施力点 p，累加 F 与 (p−r)×F，r 为给定世界参考点。切向力由 F 在该站点切平面的投影累加。最后将三个向量旋入指定传感器轴，XYZW 为传感器到世界的主动旋转。力与切向力单位为 N，力矩为 N·m。本体级测量若命中同一记录的两侧，会分别计入两侧各自的力和力臂。返回的力矩是记录中点力的合矩。独立的滚动摩擦偶矩需要另外的物理来源。

CollisionLoadSystem 的瞬时层力随状态在步初求值，结构布力在物理子步中点求值。传感器直接读取发布的牛顿值。只有在验证动量时才用每个真实子步的已知时长乘该步中点力取得冲量。[碰撞层发布](../../src/phymeo/solvers/collision_loads.py) · [布与连杆站点力](../../src/phymeo/solvers/cloth_link_contact.py)

## 解析和真实窗口

| 构型 | 独立预期 |
| --- | --- |
| 球面与布的间隙 1/8 m，层宽 1/4 m，刚度 16 N/m | 两侧法向力为 ±2 N。球相对布沿 +X 滑动，摩擦系数 1/2，共同施力点 (1/4,−1/8,0) m。球侧总力 (−1,0,2) N、绕世界原点的力矩 (−1/4,−1/2,−1/8) N·m。调换几何登记顺序及转换测量坐标后分别核对。 |
| 横跨布面的胶囊绕 Z 转动 | X=±1/2 m 两个接触站的切向合力为零，绕 Z 的总扭矩为 −1/2 N·m。用单个平均点乘合力会遗漏该扭矩。 |
| 同一刚体上两形状分别向上/下承载 | X=±1/2 m 的形状分别受 ±2 N，刚体合力为零、绕 Y 力矩 −2 N·m。只选一个对方时读到该形状单独的 ±2 N。 |
| 两个世界复用相同本体、形状编号 | 不同间隙分别产生 2、1.5 N；空接触状态给出零读数，外层事务撤销仍读取此前已接受的帧。 |
| 布面与偏心转铰胶囊连续接触 | 16 个真实布–连杆中点子步分别测两侧力矩；关于世界固定铰的力矩投影应等于已发布关节努力。累计布冲量与真实布动量增量、累计铰角冲量与真实关节角动量增量分别对照。 |

[专项测试与持续接触窗口](../../tests/test_robot_contact_wrench.py)

## Newton 1.7 原生对照

Newton SensorContact 读取同一碰撞管线、真实求解步和 `CONTACT_F`。其 `total_force` 包含感测对象的全部接触；指定对方后的 `force_matrix` 逐对方给出世界力。切向分量另有输出。`position_matrix` 是按力幅加权的两侧中点。传感器忽略 `CONTACT_F` 的角分量，没有原生力矩输出。对照分别从每个原始接触读取力向量及感测侧的有效表面点，在世界参考点累加力臂力矩。表面点用该侧本体的真实变换作用于其本地支点加表面偏移。[Newton 传感器文档](https://newton-physics.github.io/newton/1.7.0/concepts/sensors.html) · [真实 XPBD 与原生读数](../../tests/reference_newton/robot/contact_wrench.py)

Newton XPBD 的 CONTACT_F 是实际物理子步内施加接触冲量所对应的步平均载荷。PhyMeo 两种来源均有明确的力学采样点。静载平衡、交换两侧、过滤、适用的时间积分和力臂可以分别比较，瞬态估计则保留各自的时序。

## 本机实测

PhyMeo HOST 9 项专项通过。实际布–转铰窗口使用 8 个外步、每外步两个物理子步。各站点力矩关于铰轴的投影与已发布关节努力逐步相等，实测最大差为 0。16 个子步累计的布冲量为 (−0.069089894, 1.90×10⁻²¹, −0.138118723) N·s，铰角冲量为 −0.120889638 N·m·s，峰值切向力 1.16203 N。两种冲量都与已接受状态的动量变化对上，绝对容差为 3×10⁻¹²。不同作用点和同一所有者命中两侧另有显式记录层留出。

| Newton 1.7 XPBD / SensorContact，CPU 与 RTX 4060 Laptop CUDA | 原生平均世界力 (N) | 逐点推导的参考点平均力矩 (N·m) |
| --- | --- | --- |
| 双球足、静止 2 kg 底座 | (0,0,1.999969482) | (0,−0.499992371,0) |
| 初始沿 +X 滑动 0.5 m/s，摩擦系数 0.25 | (−0.500003813,0,2.000015251) | (0,−0.689029181,0) |

每个设备实际运行两个 16×1/64 s 窗口，参考点始终为世界 (−0.25,0,0) m。每步分别核对 body 总力、指定地面的矩阵力、两个足端 shape 的力、地面反力和无接触对方的零读数。静载时两足各为 0.999984741 N。滑动窗口的累计重力及接触合冲量与真实平动动量变化之差，最大分量约 1.02×10⁻⁶ N·s；按每条表面作用点计算的质心角冲量与真实角动量变化之差约 1.51×10⁻⁵ N·m·s。传感器原生力与由原始 contact 逐行累加的差逐步检验。表中 CPU/CUDA 数字在所示精度相同。

原始证据根为 `artifacts/p65-night-contact-wrench-20261104-01`。缺少传感模块的首红退出码为 2；首次双世界发布试片的偏心动态球触发了当前连续碰撞路径的转动范围拒绝。事务试片改为过球质心的法向载荷，独立偏心力矩试片保留。相邻回归首次受系统默认 pytest 临时目录权限影响，在证据根使用独立临时目录后 20 项通过。最终专项、两个 Newton 设备及静态检查均退出 0。

## 复现

```powershell
python -m pytest -s -q tests/test_robot_contact_wrench.py
.\.venv-newton-1.7.0\Scripts\python.exe tests/reference_newton/robot/contact_wrench.py --device cpu
.\.venv-newton-1.7.0\Scripts\python.exe tests/reference_newton/robot/contact_wrench.py --device cuda:0
```

PhyMeo 汇总在 HOST 执行，输入须来自支持的物理采样记录。此处的传感器姿态决定表达轴，安装参考点由调用者从相应采样时刻的状态提供。
