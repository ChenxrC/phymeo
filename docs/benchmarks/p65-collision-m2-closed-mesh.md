# RR05 · 凸体与闭合凹网格

用 `CollisionShape(key, "mesh", vertex_ids, triangles=faces)` 登记刚体边界。每个有向三角形引用稳定材料 ID。登记的主体是一张连通、向外定向、无自交的闭合曲面，最多 128 个三角面。窄相同时使用这张曲面和它围成的实体，可与静止或平移的满维锐边凸体配对。网格及凸体均可平移，在公共运动输入中也可以使用零转角的移动枢轴。非零转角或膨胀半径在此配对返回 `unsupported`。[形状注册](../../src/phymeo/constraints/collision_shapes.py) · [几何查询](../../src/phymeo/geometry/_closed_mesh.py)

`pair.mesh` 保留起末态、首次相触的有理时间和该时刻的材料点。每个姿态分别给出实体关系和到**实际边界**的距离。一个物体完全在另一个内部且二者表面尚有净距时，`boundary_gap_m` 为正，实体见证的间隙为负。这个数表示两条边界的净距，没有最小推出位移的含义。切过壁面时可取得两侧材料权重、原网格三角面和边界交点。一个碰撞组合的结果与双方顺序一致。[结果结构](../../src/phymeo/geometry/collision_witness.py)

检测检查材料边、顶点邻域、正向体积和非邻接自交。凸体半空间裁剪每个真实三角面，封闭网格内外使用可重复的射线奇偶计数，遇到边或顶点会换方向。连续检测对各网格面求与凸体的平移相交区间。tyx 计算各分离轴上的投影和时间候选，HOST 用原始坐标复核。当前实现枚举网格面，保留后续网格加速的基线。[封闭三角面内外查询](https://doc.cgal.org/latest/Polygon_mesh_processing/classCGAL_1_1Side__of__triangle__mesh.html) · [移动凸集的分离轴方法](https://www.geometrictools.com/Documentation/MethodOfSeparatingAxes.pdf)

| 构型 | 公开入口结果 |
| --- | --- |
| 进入 U 形右臂，替换三角化对角线，打乱凸体顶点 | 三者首触均为 4803839602528529/6605279453476728，约 0.727273。末态凸体被包含，距边界 0.1 m |
| 留在凹槽 | 全段分离，末态还有 0.2 m |
| 左臂留出 | 首触 4/5，随后切过内壁并进入实体。末态距边界 0.05 m |
| 将整个 U 网格放进凸箱 | 返回相反的包含关系，双方表面净距 1 m |

HOST 集成回归 228 项通过，tyx interpret 与实际 CUDA 各 30 项通过。相同六对形状在两种 tyx 模式上均与 HOST 结果逐项一致，各完成 168 次投影核启动、1388 个轴区间。还检查了共同平移、初态包含、材料存储、错误网格、设备异常及容量拒绝后的发布历史。原始收据在 `artifacts/p65-collision-m2-rr05-local-20261102-01` 的 `08-host-pytest-200101555`、`09-interpret-pytest-200141959` 和 `10-cuda-pytest-200217488`。设备同次输出保存在各自 basetemp 的 `rr05.json`。[HOST 多案例](../../tests/collision/test_closed_mesh_contact.py) · [设备多案例](../../tests/collision/test_tyx_closed_mesh_contact.py)
