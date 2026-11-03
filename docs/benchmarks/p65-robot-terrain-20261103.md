# 固定基座机器人资产与刚球–地形增量

2026-11-03。以 Newton 1.7 首轮对照中的冻结输入、解析预期和容差验收 Phymeo 的新增能力。4 个刚–刚高度场案例现在能通过 Phymeo HOST 入口运行，URDF 和 MJCF 两个工具点案例现在各能在 HOST 与 tyx CUDA 运动学路径上运行。首轮 Newton 观测与原来的缺口统计仍可追溯。[首轮对照](newton-1.7-reference-baseline.md) · [刚–刚输入](../../tests/reference_newton/rr/cases.json) · [机器人输入](../../tests/reference_newton/robot/cases.json)

## 机器人与工具链

`phymeo.load_urdf(path)` 和 `phymeo.load_mjcf(path)` 都返回同一类机器人资产。`asset.model` 可直接交给原有 FK、点端口或机器人求解器，`asset.collision_model` 与 `asset.collision_motion(q)` 供公共几何查询使用。每个球或凸盒保留其名称、所属连杆、局部变换和物理尺寸。限位和驱动力界以资产数据返回，具体控制律另行选择。

- URDF 1.0 支持固定基座树、标量转动和移动关节、球与盒、显式质量和惯量。按惯性系姿态转到连杆系，只有现有 RobotModel 可表达的惯量才接受。额外 visual 方盒不产生碰撞体。[URDF 入口](../../src/phymeo/builders/urdf.py) · [姿态、拓扑、正反例](../../tests/test_robot_urdf.py)
- 本地 MJCF 支持显式惯量、一条有限 hinge 或 slide 对应一个活动 body、焊到世界的根及球与盒。解析 compiler 的度/弧度、Euler 轴顺序、WXYZ 四元数、盒半边长和父子过滤，未声明的 effort/velocity limit 保留为 `None`。不支持的元素会给出诊断。通用 defaults、网格资产、执行器、多自由度体、偏置关节枢轴另需扩展。[MJCF 入口](../../src/phymeo/builders/mjcf.py) · [不同旋转、单位、分支和反例](../../tests/test_robot_mjcf.py)
- 两份独立资产在 θ=π/2、滑移 1/8 m 时，HOST 均读到工具球心约 (0, 1/16, 5/32) m，速度 (−5/32, −1/8, 0) m/s。CUDA 的两份报告均记录了 3 次完成的运算、10 次下载，球心 x 约 −1.76×10⁻⁸ m。资产解析在 HOST，FK 与点雅可比在 tyx CUDA，最终的速度乘法在 HOST。[URDF 试片](../../tests/reference_newton/robot/arm.urdf) · [MJCF 试片](../../tests/reference_newton/robot/arm.mjcf.xml) · [两引擎测量入口](../../tests/reference_newton/robot/baseline.py)

## 刚体–刚体检测

`phymeo.Heightfield` 表示有界、双侧的规则三角高度图，行与列分别对应局部 Y、X，支持不同间距和固定刚体姿态。通过 `phymeo.detect` 查询两侧表面点、法向、实际单元、三角面及重心权重。可用非连续材料 ID 重建接触点。运动覆盖静止地形与球心直线平移，动地形和球心非零转弧有显式范围判断。[公共用法与边界](../development/p65-rigid-heightfield.md)

| 冻结案例 | Phymeo 实测 |
| --- | --- |
| RR-RIDGE-POSES | 两端分离，中间屋脊间隙 −1/16 m，地形点 (0, 1/8, 1) m。 |
| RR-RIDGE-MISS | 三个给定姿态均无接触。 |
| RR-SLOPE-FACE | 坡面样本间隙 −0.13572330470336313 m，斜面法向在登记预算内。 |
| RR-RIDGE-FIRST | 入步与步末净隙均约 0.394607 m，步内首次相触 t=3/8（2.9296875 ms），末次相触 t=5/8（4.8828125 ms）。两次触点均在 (0, 1/8, 1) m，地形指向球的法向 x 分量依次为 −0.6、+0.6。 |

连续求交还通过了孤立擦边与一 ULP 近失、两个分离接触窗、反向步和刚体变换试片。首末接触之间可能含有分离时段。穷举、SAP、BVH 三种实际 CUDA 宽相均检出了跨步屋脊候选并排除了远处球。高度场窄相及事件材料见证由 HOST 构造。[几何专项](../../tests/collision/test_heightfield_sphere.py) · [连续检测](../../tests/collision/test_heightfield_sphere_sweep.py) · [设备边界回归](../../tests/collision/test_tyx_heightfield.py)

## 两条入口组合与复现

用两种文件各建一次机器人，只取它们真实注册的工具球。将资产与另一物体的 2×3 地形合入同一 CollisionModel，球在零位时位于地形边界外。在转动加滑移的姿态，两种来源均返回 −1/64 m、同一法向和独立可重建的两侧点。该试片验证 FK 与通用碰撞检测之间的坐标及材料映射。[跨模块回归](../../tests/collision/test_imported_robot_heightfield.py)

本批完成的串行验证包括最终机器人 HOST 专项 54 项、导入 URDF 的三姿态 CUDA 专项 1 项、地形 HOST 静态与连续专项 26 项、地形三种 CUDA 宽相 3 项、既有宽相接口 6 项、两种文件的跨模块试片 2 项。另有上述 6 个冻结 ID 的 8 份 Phymeo 引擎/设备读数。第一次按测试驱动开发保存了缺少导入入口、缺少地形登记、连续查询尚未实现的红测。最终结果按各自阶段记录，各个阶段的重复验证不累加成独立案例数。本机原始日志与 JSON 在 `artifacts/p65-robot-rr-scale-20261103-01`。

```powershell
python -m pytest -q tests/test_robot_urdf.py tests/test_robot_mjcf.py -k "not imported_fk_runs_tyx"
python -m pytest -q tests/test_robot_urdf.py::test_imported_fk_runs_tyx --require-tyx --tyx-mode=cuda
python -m pytest -q tests/collision/test_heightfield_sphere.py tests/collision/test_heightfield_sphere_sweep.py tests/collision/test_imported_robot_heightfield.py
python -m pytest -q tests/collision/test_tyx_heightfield.py --require-tyx --tyx-mode=cuda
python tests/reference_newton/run_matrix.py --lane rr --engine phymeo --device cpu --case RR-RIDGE-FIRST
python tests/reference_newton/run_matrix.py --lane robot --engine phymeo --device cuda:0 --case ROBOT-MJCF-DEGREES
```

使用项目原有 Python，并按[案例运行说明](../../tests/reference_newton/README.md)将 PYTHONPATH 指向当前源码。后续机器人优先补 USD、单位及坐标轴。刚体检测优先补双凹网格的双侧真实表面见证与曲面原语。GPU 多场景另有分区、驻留和端到端成本的验收。[开发顺序](../superpowers/plans/2026-11-03-newton-reference-development.md) · [GPU 与多场景计划](../superpowers/plans/2026-11-03-tyx-gpu-multiworld.md)
