# USD 机器人、解析曲面与双凹网格

2026-11-03。三个独立工作树从同一提交 `d92e8fe` 分出。作者分别实现模块、编写专项测试、领取同一串行执行槽运行基础验证和修复首错。协调者维护公共注册、分派、选装依赖与最终组合测试。原 Newton 1.7 对照的形状、姿态、时间、预期物理量与容差继续沿用。[初始参考矩阵](newton-1.7-reference-baseline.md) · [前批机器人与高度场](p65-robot-terrain-20261103.md)

## 机器人 USD

`phymeo.load_usd(path_or_stage)` 使用可选 OpenUSD SDK。基础环境只需 NumPy。要读 USD，可在支持的 Python 环境执行 `python -m pip install ".[usd]"`。本机验证使用 usd-core 26.8、Python 3.12.14。加载本地路径或打开的 stage 后，同一 RobotModel、FK、RobotPointPort 和碰撞查询可使用导入结果。

支持一棵固定基座、有界标量转动/移动关节的静态铰接树，球和刚性定向盒。实际读取两侧关节帧、正反向关联、惯量、碰撞形状和形状归属。长度按 metres-per-unit、质量按 kilograms-per-unit、惯量按质量因子乘长度因子平方换算。源连杆的偏心点可用 `asset.authored_to_model_points(link_name, points_m)` 转入关节模型坐标，用 `asset.authored_link_pose(link_name, q)` 读回源帧的世界姿态。输入范围与具名拒绝详见[导入器](../../src/phymeo/builders/usd.py)和[测试](../../tests/test_robot_usd.py)。单位、父级变换和关节两侧的语义依据 [OpenUSD 长度单位](https://openusd.org/release/api/group___usd_geom_linear_units__group.html)与 [OpenUSD Joint API](https://openusd.org/release/api/class_usd_physics_joint.html)。

- 18 项 HOST 测试通过，另有 1 项厘米、Y-up、斜轴的真实 tyx CUDA 留出通过。米制 Z-up 和等物理尺寸的厘米 Y-up 两份资产对质量、惯量、限位、位置与速度给出相同的物理结果。冻结的 `ROBOT-USD-ZUP` 在 HOST 和 CUDA 各通过一次。两者原点、轴和碰撞球归属均从导入模型取得，CUDA 工具球 x 约 −1.76×10⁻⁸ m。[米制资产](../../tests/reference_newton/robot/arm.usda) · [厘米留出](../../tests/reference_newton/robot/arm_cm_y.usda)
- `ROBOT-USD-CENTIMETER-WARNING` 的原预期专门观察 Newton 对厘米刚体的告警。Phymeo 按 SI 换算，因此保留原预期，在 Phymeo 列标为 `not_comparable`。独立的物理单位试片承担转换验收。USD 解析在 HOST，CUDA 行验证 FK 和点雅可比。浮动根、闭环、驱动与通用材质仍有各自合同。

## 静态刚体曲面

用 `phymeo.CurvedSolid(kind, dimensions_m, orientation_xyzw)` 声明真实尺寸与姿态，再作为 `CollisionShape(..., curved=...)` 登记。该形状有一个中心材料 ID。见证分别保存双方的点和相对中心或平面锚点的世界偏移。`pair.curved` 保存曲面所属者、特征及并列支撑的几何类型。锥底盘、圆柱帽和并列法向可辨认。公共碰撞查询仍先做掩码、世界过滤和实际外包盒检查。[实体参数](../../src/phymeo/constraints/curved_solids.py) · [解析窄相](../../src/phymeo/geometry/_curved_solids.py)

| 冻结构型 | 实际末态间隙 | 独立量 |
| --- | ---: | --- |
| RR-CYLINDER-RIM | −0.03125 m | 球心到有限圆柱环边的截面给出法向 (0.6, 0, 0.8)，圆柱点为 (0.5, 0, 0.25) m。 |
| RR-CONE-TIP | 约 −0.0125 m | 倾斜有限圆锥的尖端与平面支撑分别发布。 |
| RR-ELLIPSOID-TILT | 约 −0.03125 m | 平面见证为 (0.21875, 0, 0) m，斜椭球见证在约 (0.21875, 0, −0.03125) m。 |

作者的最终 HOST 专项为 27 项通过，相邻接口回归 55 项通过，三个冻结构型各通过一次。平面见证由实际曲面点向平面投影求得。平移或转动整步的候选有显式 `unsupported` 范围，曲面 CCD 和双向受力另行验收。[专项测试](../../tests/collision/test_curved_solids.py) · [独立材料见证对照](../../tests/reference_newton/rr/curved_reference.py)

## 闭合刚体网格对

两个物体各自登记连通、外向、无自交的闭合三角网格。面级包围树筛候选后，窄相检查原始三角面的真实相交、包含与边界距离。`pair.mesh_mesh` 保存边界交点、所有并列最深的内部顶点支撑、两边的原面与权重。边界交叠和实体包含单列。无法确定的关系显式返回未解状态。[支持域和 API](../development/p65-double-mesh.md)

| 冻结构型 | 物理观测 |
| --- | --- |
| RR-U-NOTCH | 两个凹体的包围盒重叠，空槽三次采样均分离。边界净距依次为 0.5、0.4375、0.375 m。 |
| RR-U-RIGHT、RR-U-LEFT-HOLDOUT | 末态各有 11 个边界生成点，4/4 个并列顶点支撑均通过。接收体局部内壁坐标分别为 +1、−1 m。法向分别为 −x、+x，顶点侵入量分别为 0.125、0.1875 m。 |
| RR-U-FIRST | 两物体独立平移的首次相触为归一化 4/5，即 6.25 ms。初态分离、入口相触、末态边界交叠。精确入口的 6/6 个材料支撑均重建成功。 |

首轮静态专项 10 项、连续专项 5 项通过。补充同向共面部分重叠的留出后，最终两个文件合计 16 项通过。包围盒、真实交点、面间净距和 `vertex_intrusion_m` 可以回答不同的问题。后者取严格在另一实体内的表面顶点至真实边界的最大最近距离。需要全局最小推出位移或物理载荷时，须另立判据。当前连续能力为零圆角的闭合三角体各作平移，事件的精确分数及二进包围区间分别发布，非零转弧仍在后续范围。[静态及包含回归](../../tests/collision/test_mesh_mesh_contact.py) · [平移首触](../../tests/collision/test_mesh_mesh_sweep.py)

## 父集成及复现

在同一碰撞模型中放置共址世界、重复的局部 ID、一组圆柱–球及一组双网格。HOST 与实际 CUDA 宽相各通过一项组合测试。6 个配对中，4 个跨世界配对正确过滤，其余 2 个返回对应种类的材料见证。在另一试片中从 USD 导入带球工具，通过真实 FK 在有限圆柱上产生 −1/64 m 间隙。另用原有基础环境确认导入 Phymeo 无需加载 pxr。[跨模块试片](../../tests/collision/test_mesh_curved_asset_dispatch.py)

```powershell
python -m pytest -q tests/collision/test_curved_solids.py
python -m pytest -q tests/collision/test_mesh_mesh_contact.py tests/collision/test_mesh_mesh_sweep.py
python -m pytest -q tests/test_robot_usd.py -k "not cm_y_up_device_kinematics"
python -m pytest -q tests/test_robot_usd.py::test_cm_y_up_device_kinematics --tyx-mode=cuda --require-tyx
python tests/reference_newton/run_matrix.py --lane rr --engine phymeo --device cpu --case RR-U-FIRST
```

USD 物理测试需要安装选装依赖。TYX 命令使用项目固定环境，参考运行器的 PYTHONPATH 设为当前仓库的 src。本机各作者与父工作区的原始命令、退出码、对照 JSON 及首次失败记录位于 `artifacts/p65-independent-modules-20261103-01`。闭网格和解析曲面的窄相此次都在 HOST，CUDA 宽相、机器人设备 FK、以及未来的多世界动力学吞吐分别验收。[GPU 与多场景计划](../superpowers/plans/2026-11-03-tyx-gpu-multiworld.md)
