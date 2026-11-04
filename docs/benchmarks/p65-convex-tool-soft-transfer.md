# 关节凸托板抬布、横拖、撤出和自由下落

48 g 的 SOFT 布在实际关节驱动下抬高 **33.446 mm**，随后沿 x 横拖 **54.970 mm**。仅把托板摩擦系数从 0.75 变为零，抬高量相同，布在横拖阶段的位移变为 **−0.021 mm**。托板撤出后，两条轨迹各有 32 个连续零接触物理步，布质心在重力下独立下落。每条轨迹保留 288 步、289 个实际状态，可在侧视与俯视中核对布节点、托板、关节和有力的材料点。[软布回放](../../artifacts/p65-soft-convex-tool-transfer-20261105-01/soft.html) · [零摩擦回放](../../artifacts/p65-soft-convex-tool-transfer-20261105-01/control.html) · [梯形工具回放](../../artifacts/p65-soft-convex-tool-transfer-20261105-01/alternate.html)

## 操作与材料

水平和竖直棱柱关节各自有质量、速度反馈、重力补偿和有限电机。水平、竖直力上限分别为 22 和 20 N。控制器向关节下达平滑目标，实际关节载荷和布面载荷由共同中点步进器计算。初态中布与托板存在一个正间隙的接触层，布有预定的 0.004 横坡，四个自由质量节点通过五条距离约束和一条弯曲约束相连。[可更换的材料、工具和真实执行器](../../src/phymeo/experiments/convex_tool_transfer.py) · [参考及软布材料定义](../../src/phymeo/experiments/sheet_fabric.py)

横拖前至少三个连续物理步必须测得向上载荷超过半个布重，即 0.192 N，并且布质心已抬起至少 12 mm、托板已实际上升 20 mm。撤出由实际水平关节到位触发。横拖资格要求质心移动 15 mm。撤出后的自由飞行要求至少 32 个连续步全部材料站无载荷，且布质心至少降低 5 mm。不同阶段的驱动仅作用于关节，布的速度及位置完全来自布的实际步进。[操作门槛及逐步动量检查](../../tests/test_convex_tool_transfer.py)

| 布料及工具 | 摩擦系数 | 抬升 (mm) | 抬升门槛时支撑 (N) | 沿 x 横拖 (mm) | 独立落下 (mm) | 飞行末竖直速度 (m/s) |
|---|---:|---:|---:|---:|---:|---:|
| SOFT，矩形 | 0.75 | 33.446 | 0.21519 | 54.970 | 25.186 | −0.65297 |
| SOFT，矩形 | 0 | 33.446 | 0.21519 | −0.021 | 25.130 | −0.65208 |
| REFERENCE，六边梯形 | 0.75 | 33.445 | 0.22014 | 55.180 | 25.296 | −0.65473 |

三组的抬升门槛均出现在第 112 步（0.21875 s），第 240 步（0.46875 s）启动撤出，第 257–288 步连续无载，共 62.5 ms。SOFT 正控的最大边应变为 0.1252%，最大弯角变化为 0.04908 rad。REFERENCE 留出同时更换弯曲材料和实体托板轮廓，最大弯角变化为 0.00947 rad。零摩擦控制完成同一驱动和释放，横拖资格保持未通过。[三组原始记录与独立汇总](../../artifacts/p65-soft-convex-tool-transfer-20261105-01/summary.json) · [从实际归档拣取的关键帧](../../artifacts/p65-soft-convex-tool-transfer-20261105-01/keyframes.json)

独立审计直接从各节点质量、速度、受力材料点、实际电机和关节初末速度复算。三组中每步布线动量的最大分量残差小于 1.9×10⁻¹⁸ N·s，关节动量残差小于 1.3×10⁻¹⁶ N·s。在最后无载窗口内，全部质心位置与 `x₀+v₀t+gt²/2` 的最大分量偏差小于 1.1×10⁻¹⁶ m，速度偏差小于 2.3×10⁻¹⁶ m/s。动力学为 HOST FP64。[SOFT 原始逐步记录](../../artifacts/p65-soft-convex-tool-transfer-20261105-01/pytest-soft/test_actual_support_drag_relea0/trajectory.json.gz) · [零摩擦原始记录](../../artifacts/p65-soft-convex-tool-transfer-20261105-01/pytest-control/test_actual_support_drag_relea0/trajectory.json.gz) · [不同材料及几何原始记录](../../artifacts/p65-soft-convex-tool-transfer-20261105-01/pytest-alternate/test_actual_support_drag_relea0/trajectory.json.gz)

## 复现和回放

从仓库根运行。加 `--friction 0` 生成匹配控制，使用 `--fabric reference --footprint taper` 生成联合留出。Python 构建函数另接受自定义 `SheetFabric`、网格和凸多边形轮廓。[命令行入口](../../examples/convex_tool_transfer.py)

```powershell
python examples/convex_tool_transfer.py --fabric soft --footprint rectangle --output tray-soft.json.gz --html tray-soft.html
python examples/convex_tool_transfer.py --from-trace tray-soft.json.gz --html tray-soft.html
```

离线回放可拖动时间、播放、调整速度并跳到实测关键步。关节、材料节点和托板轮廓是步末姿态，红色标记表示该步实际施力的中点材料站。实际 Edge 浏览器核过三份嵌入数据中全部 289 个状态与原始 JSON 的逐值一致性，并验证两个视图、关键步跳转和播放。[浏览器记录](../../artifacts/p65-soft-convex-tool-transfer-20261105-01/browser-final/ui.json) · [横拖末帧](../../artifacts/p65-soft-convex-tool-transfer-20261105-01/browser-final/soft-drag.png) · [自由飞行末帧](../../artifacts/p65-soft-convex-tool-transfer-20261105-01/browser-final/soft-free.png)

本操作从单臂托板已经能够支承材料的初态出发。双臂软布交接仍需在供体承载条件下让接收臂的腕和指杆获得结构净空，并形成双侧共同承载。原竖直与 2° 布的已存阻挡状态、17 个参考方向以及提前出发后的有限搜索结果保持独立记录；这里的横拖与落体资格不计入交接成功数。[双臂实测障碍及后续判据](p65-soft-handoff-acquisition.md)
