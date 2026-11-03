# 两个闭合刚体网格的真实边界

`CollisionShape(key, "mesh", vertex_ids, triangles=faces)` 可为两个不同刚体分别登记三角曲面。沿用已验收的闭合网格验证。每张曲面必须连通并正确外向定向，每条边恰有两个反向使用者，面不自交，最多 128 张。材料 ID 在各自刚体内解释。HOST 用面包围树筛候选，再在原始二进浮点输入所确定的有理坐标中核对三角面实际相交的点、线段和共面多边形。

`pair.mesh_mesh.end.boundary` 给出实际相交几何的生成点。每个 `MeshMeshSupport` 的 `first_faces`、`second_faces` 包括能够还原该材料地址的所属面；`witness` 在两个独立刚体上分别保存世界点和顶点权重。真实边界有交点时 `boundary_gap_m` 为零。两闭面分开时它是面到面的最小欧氏距离，严格包含也可能具有正净距。

`vertex_intrusion_m` 只度量严格位于对方实体内部的表面顶点。对每个这类顶点取到对方真实边界的最短距离，再取所有这些距离中的最大值。精确同距的所有顶点、所有最近边界地址均列入 `deepest`，同一条材料支撑去重。每个点对的有符号间隙等于第二点减第一点在 A→B 法向上的投影。不同支撑方向并列时代表见证的法向为空，各条支撑保留其方向。整物体最小平移推出量需要另一项计算。

离散关系分别记录 `separated`、`touching`、`boundary_overlap`、`coincident`、`first_contains_second` 或 `second_contains_first`。全表面共面覆盖按每个原三角形的交叠面积复核，允许另一种三角化。严格内部顶点、横穿两张面内部的交线，或同向开放共面区域均可证明内部相交。缺少内部顶点的边界穿越仍保留所有实际交点，此时 `vertex_intrusion_m` 为空。未能证明接触附近内部关系的边界会保留见证且返回 `unresolved / boundary_intersection_unclassified`。外部相贴可由分离平面证明。非零转动与表面膨胀单列超出范围。本批窄相在 HOST。

[解析盒体与双侧材料验证](../../tests/collision/test_mesh_mesh_contact.py) · [冻结的两个 U 形刚体输入](../../tests/reference_newton/rr/cases.json)

## 两个网格分别平移

两边可以采用不同的有限平移，包括零转角的移动枢轴。以两个真实网格的面级扫掠包围盒筛选，再对候选三角面的面法向、两侧边的叉积求闭合平移区间；平行三角面还检查各边的面内垂线。平移投影的分离与进入不等式直接使用原始 FP64 坐标确定的有理数。初态已经包含或相交时，闭实体首次接触就在 0。初态分离时从全部候选面交叠区间取最早起点，并在该时刻重新核实真实边界交点。

`pair.mesh_mesh.first_ratio` 是归一化首触时刻的整数分子、分母。`entry` 和其中每条材料支撑对应这个精确有理数；`pair.sweep.first_t` 是包围它的二进区间。`sweep.witness` 另外在公开的 FP64 `witness_time` 求值，所以当首触为 4/5 时，精确入口与舍入采样如实分开。`end` 和 `pair.witness` 始终属于末时刻。首次进入后可以先分离、再接触；该查询不发布连续受力时长。单步非零旋转、最小推出位移和 GPU 窄相另行验收。[过壁、双端分离、共面运动与单刻度近失](../../tests/collision/test_mesh_mesh_sweep.py)
