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

07:46 首轮调用 `collision_motion(q_start, q_end)` 时，原入口指出前臂工具球同时受 shoulder 和 elbow 影响，当时只支持一个活动转铰。该轮的 HOST 专项 1 项通过，检查了姿态、解析式及端点弦负控。后续连续检测仍沿用相同输入和首次时刻，下面列出新增实测。[原冻结入口](../../tests/test_robot_two_hinge_ellipse.py) · [新增专项的旧实现首红](../../artifacts/p65-two-hinge-ccd-20261104-01/01b-public-red-basetemp/stdout.log)

项目环境可运行 `python -m pytest -q -s tests/test_robot_two_hinge_ellipse.py`。独立 Newton 环境分别运行 `python tests/reference_newton/robot/two_hinge_ellipse.py --device cpu` 和 `python tests/reference_newton/robot/two_hinge_ellipse.py --device cuda:0`。

## 双铰链连续检测

现在从 `asset.collision_motion(q_start, q_end)` 生成路径后可直接调用 `ph.detect`。两轴均可带符号，并可在空间取不同方向；上游有总平移的滑轨也可使用。通过路径的 `positions_at(t)` 可读取原始顶点的中间世界位置。[关节运动表示](../../src/phymeo/builders/_robot_joint_motion.py) · [球–球连续计算](../../src/phymeo/geometry/_two_hinge_spheres.py)

连续检测在同一对已注册球上读取轨迹。以 A、B 表示两个活动铰接点在初态的世界坐标，C 是初态球心，v 是最上游滑轨造成的总位移，R₁、R₂ 分别绕两个初始世界轴按其带符号角度转动。任意时刻 t 的球心为 A + tv + R₁(t)[B−A + R₂(t)(C−B)]。因此即使第二轴与第一轴不同，第二轴也会随第一连杆运动。单个中心的速度上界为 |v| + |Δq₁| (|B−A| + |C−B|) + |Δq₂| |C−B|。对运动的另一球独立求界并相加。

宽相采用覆盖整段的包围盒。HOST 窄相用带误差包围的 Rodrigues 旋转计算球心。只有整个时间单元的中心间距下界严格超过两个半径之和，才会排除该单元。已确认的首触区间宽至多 2⁻²⁷，时间宽度乘以相对速度上界至多 1 µm，上端有实际占用样本。结果保留两球各自的材料中心 ID、世界径向偏移及世界表面点。初始占用可以直接返回 t=0；重合球心的法向可未定。

| 物理构型 | 独立首次时间 | Phymeo 连续区间 |
| --- | ---: | --- |
| 原始椭圆及镜像、时间反转和共同刚体变换 | 0.4734459540671194 | [0.47344595193862915, 0.47344595938920975] |
| 两轴不平行，另一球独立平移 | 0.4448408016911062 | [0.44484079629182816, 0.44484080374240875] |
| 两个末端球分别沿正向、反向串联弧通过对方 | acos(1/24)/π ≈ 0.4867332474262846 | [0.48673324286937714, 0.48673325031995773] |

原椭圆两次在上界采样的实际表皮间隙均约 −25.0 nm。与对侧球的整段查询均返回 `separated`；前述解析最小净空是 0.875 m。斜轴和双运动球的上界占用样本间隙分别约 −4.67 nm 和 −27.25 nm。原生 Newton CPU/CUDA 的离散读数保留在此前表中。Phymeo 的实际 CUDA 宽相完成 3 次设备启动、2 次下载，得到相同的候选对，HOST 窄相给出与上表相同的首触区间。[HOST 连续读数](../../artifacts/p65-two-hinge-ccd-20261104-01/04-two-hinge-green/stdout.log) · [真实 CUDA 读数](../../artifacts/p65-two-hinge-ccd-20261104-01/06-two-hinge-tyx-cuda/stdout.log)

对孤立外切，512 个样本耗尽时返回 `unresolved`；单铰链的区间转角超过 32 rad 的候选也返回 `unresolved`。本路径的连续窄相只处理球–球，另一球可静止、仿射平移或沿关节弧运动。非球候选明确返回 `unsupported`。同一支链含第三个活动转铰、转铰下游的活动滑轨或失去世界坐标分辨率时，资产入口具名拒绝。HOST 专项与冻结回归共 8 项通过，原单铰链和滑轨邻近回归 7 项通过，CUDA 专项 1 项通过。[独立控制和能力边界](../../tests/test_robot_two_hinge_ccd.py)
