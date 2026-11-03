# 有限高度场上的带半径材料节点

`rounded_particle` 用一个材料顶点和一个正半径登记节点的球形接触表面。该节点属于可变形物体，运动输入为节点中心的世界坐标，可给静止姿态或仿射首末位置。固定有限高度场参与双面及真实外缘查询，配对族为 RD。既有 `particle` 继续表达零半径点。[形状登记](../../src/phymeo/constraints/collision_shapes.py) · [地形查询](../../src/phymeo/geometry/_heightfield.py)

```python
node = ph.CollisionShape(node_key, "rounded_particle", (material_id,), radius)
land = ph.CollisionShape(land_key, "heightfield", terrain_vertex_ids, heightfield=grid)
model = ph.CollisionModel((land, node))
paths = {
    node_key: ph.CollisionMotion([center_start], [center_end]),
    land_key: ph.CollisionMotion(grid.world_vertices()),
}
with ph.CollisionWorkspace(model) as workspace:
    result = ph.detect(model, paths, ph.CollisionPolicy(), workspace)
```

节点侧的材料地址为 (material_id, 1)。将节点中心加上 CollisionWitness 对应一侧的世界 offset，得到球面上的实际见证点。地形侧的三个材料权重重建三角面上的见证点。根据 pair.first 和 pair.second 选择 first/second 字段，法向从第一方指向第二方，两个见证点之差沿该法向的投影等于发布的间隙。共享同一材料顶点的不同注册须提供一致的首末路径。

## 原输入的独立节点读数

`RD-HEIGHTFIELD-NODES` 读取原 `RD-HEIGHTFIELD` 的三个节点、半径 1/16 m、3×3 高度样本和平面内半宽。原样本标高为 1/4 m，附加间隙与形状 margin 均为零。HOST 与实际 CUDA 下三行的材料编号和几何结果一致。[冻结输入](../../tests/reference_newton/rd/cases.json) · [独立读回和归档比对](../../tests/reference_newton/rd/heightfield_nodes.py)

| 节点 | 中心 m | 原查询 | 几何核对 |
| --- | --- | --- | --- |
| 0 | (−0.125, 0.125, 0.28125) | overlap，gap = −0.03125 m | 地形点 (−0.125, 0.125, 0.25)，节点侧 offset (0,0,−0.0625)，球面点 (−0.125,0.125,0.21875)。地形→节点法向 (0,0,1)。 |
| 1 | (1.09375, 0.5, 0.21875) | separated，AABB 排除 | 最近点为有限边缘 (1,0.5,0.25)，几何间隙为 (√10−2)/32 m。 |
| 2 | (−0.5, 0.5, 0.5) | separated，AABB 排除 | 正上方投影在域内，间隙为 3/16 m。 |

节点 0 位于单元 (行 1, 列 0) 的三角面 0，地形材料 ID (3,4,7) 的权重依次为 (1/8,3/4,1/8)，节点材料 ID 0 的权重为 1。原查询中两个 AABB 排除项没有最近点读数。专项另用 0.25 m 候选 margin 请求这两个位置的最近点，核对了表中所列的正间隙。

历史 Newton 1.7 CPU、CUDA 的 particle_default 各有一条节点 0 记录，soft_point_m 是节点中心。逐字段核对了材料编号、中心、地形点、法向及间隙。两份原始结果的 SHA256 与归档中的 original_report_sha256 一致。本轮未重新运行 Newton。历史 full-surface 请求对高度场发出降级告警，边、面候选均为零。该请求、告警及完整布面能力仍属于原综合案例的独立条件。逐节点读数不改变原案例库数量。[原始节点与 full-surface 记录](newton-1.7-reference-observations.json)

## 留出与执行范围

- 背面节点得到 −Z 法向，有限边缘的侧向相切得到 (3/5,0,−4/5)。调换形状顺序后，两侧点、材料权重和世界 offset 均随所属方交换。
- 非正方形网格在不同 XY 间距、刚体旋转和平移下保留了独立三角面的材料 ID (20,23,32) 与权重 (1/4,1/2,1/4)。另一路径边跨斜面边切向漂移，首末地形见证的世界高度分别为 0.6640625 和 0.7109375 m。
- 从 z=0.5 m 向 z=0 穿过地形的节点，首末间隙均为 +3/16 m，入触与离触分别位于归一化时间 3/8、5/8。反向穿越时两侧接触次序反转。真实 TYX CUDA 对四个单元完成一次逐面候选核，返回面 ID 4 和 5，掩码读回和同步各一次。事件隔离、双侧见证与材料重建在 HOST 完成。
- 原静态三节点集合在 CUDA 完成两个包围盒/配对内核，一次动态上传和两次下载。静态查询的运动逐面掩码调用数为零。
- 同一材料节点作为球形节点及三角面顶点注册时，起点不一致或终点不一致都被拒绝，workspace 保留上次成功结果。独立负控还覆盖移动地形、对材料节点传入刚体转弧，以及当前未支持的整三角面–地形和刚球–圆角节点组合。[专项及物理预期](../../tests/collision/test_rounded_particle_heightfield.py)

项目 FP64 HOST 专项 17 项、实际 TYX CUDA 1 项均通过。两端的独立节点读回各自通过旧 Newton 粒子子集的核对。旧源码 `6fc17b2` 的首红是形状构造阶段拒绝 `rounded_particle`。验收范围为节点接触几何和材料地址，分布载荷及整面接触需各自的物理判据。

```powershell
python -m pytest -q tests/collision/test_rounded_particle_heightfield.py -m "not tyx"
python -m pytest -s -q tests/collision/test_rounded_particle_heightfield.py -m tyx --require-tyx --tyx-mode=cuda --basetemp artifacts/node-terrain-cuda-local
python tests/reference_newton/rd/heightfield_nodes.py --device cpu --out artifacts/node-terrain-cpu.json
python tests/reference_newton/rd/heightfield_nodes.py --device cuda:0 --out artifacts/node-terrain-cuda.json
```

从仓库根目录运行，PYTHONPATH 指向当前 src。读回报告与 CUDA 临时目录使用新路径。
