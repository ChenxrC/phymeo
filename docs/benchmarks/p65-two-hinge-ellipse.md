# 两活动铰链的连续碰撞留出

两个平面转铰绕 +Z，轴间距 1 m。前臂上的工具球心距第二转轴 0.5 m。固定根各有一个球，球心分别在 (0, 0.5, 0) m 和 (0, −0.5, 0) m。三个注册球的半径均为 1/16 m。每次查询将工具球与同侧、对侧固定球分别配对。关节输入为 q₁ = σπt、q₂ = −2σπt，σ 取 +1 或 −1，t 是区间内的归一化时间。整段给定的是关节坐标轨迹。[MJCF 资产](../../tests/reference_newton/robot/two_hinge_ellipse.xml)

## 解析预期

前臂的世界转角为 −σπt。工具球心因此经过 (1.5 cos πt, σ·0.5 sin πt, 0) m，在 t = 0.5 与同侧球心重合。令 s = sin πt。到同侧球心的距离平方为 5/2 − s/2 − 2s²。首次表皮接触时该值为 1/64，故 s = (√322 − 2)/16，首次和末次接触分别为 **0.4734459540671194** 和 **0.5265540459328806**。两个端点的表皮间隙均约 +1.456139 m。

对侧球心距离平方为 5/2 + s/2 − 2s²。此式在 0 ≤ s ≤ 1 上为凹函数，最小值 1 出现在 s = 1，故全程真实表皮净空至少 **7/8 m**。若直接用端点坐标插值，工具球沿 y = 0 行进，两条端点弦对同侧球也都保有 **3/8 m** 净空。普通碰撞管线实际把这两条弦都判为整段分离。[解析、采样和连续入口](../../tests/reference_newton/robot/two_hinge_ellipse.py)

## 2026-11-04 的实际读回

每端读取正、反两个方向各 7 个姿态。首次阈值两侧的采样时刻为 0.4724693915671194、0.4744225165671194，邻近的实测间隙如下。外部运行的接触行来自 Newton 1.7.0 的原生 `CollisionPipeline.collide` 与 `eval_rigid_contact_kinematics`。形状 0、1 为固定球，形状 2 为工具球。设备 FK 给出真实刚体位姿，再以导入的局部球心读出世界坐标。

| 执行端 | 工具球心对解析最大误差 | 接触前采样的表皮间隙 | 接触后采样的碰撞见证间隙 | 中点碰撞见证间隙 |
| --- | ---: | ---: | ---: | ---: |
| Phymeo HOST | 5.56×10⁻¹⁷ m | +4.586727 mm | −4.587806 mm | −125.000000 mm |
| Newton CPU | 0.152 µm | +4.586763 mm | −4.587688 mm | −124.999858 mm |
| Newton CUDA 0 | 0.139 µm | +4.586853 mm | −4.587643 mm | −124.999948 mm |

表中接触前间隙由实际两球心和半径计算。Phymeo 以 4 m 宽相 margin 保留分离时的真实窄相间隙。Newton 两端同侧的原始接触行数均为 0、0、1、1、1、0、0，对侧始终 0。三个过程各自退出码为 0。离散观测只把首次接触夹在两个采样姿态之间。[HOST 原始读数](../../artifacts/p65-two-hinge-ellipse-20261104-01/01-two-hinge-host/stdout.log) · [Newton CPU](../../artifacts/p65-two-hinge-ellipse-20261104-01/02-two-hinge-newton-cpu/stdout.log) · [Newton CUDA](../../artifacts/p65-two-hinge-ellipse-20261104-01/03-two-hinge-newton-cuda/stdout.log)

对同一资产调用 `collision_motion(q_start, q_end)` 时，Phymeo 现行入口在正反两个方向均指出前臂工具球同时受 shoulder 和 elbow 两个活动转铰影响，当前路径编码只支持一个。此次 HOST 专项 1 项通过，覆盖姿态、解析式及端点弦负控。将来若连续入口能编码此运动，同一测试会要求同侧返回包含解析首次时刻、宽度至多 10⁻⁷ 的区间，并要求对侧整段分离。[公开入口回归](../../tests/test_robot_two_hinge_ellipse.py)

项目环境可运行 `python -m pytest -q -s tests/test_robot_two_hinge_ellipse.py`。独立 Newton 环境分别运行 `python tests/reference_newton/robot/two_hinge_ellipse.py --device cpu` 和 `python tests/reference_newton/robot/two_hinge_ellipse.py --device cuda:0`。
