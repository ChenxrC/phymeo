# RD04 / DD04 · 四面体外表面的碰撞检测

同一组四面体单元可以供刚球和布面查询。`CollisionKey(world, body, shape)` 的 `shape` 是原单元的稳定 ID，四个材料节点按正向绕序登记。构建工作区时，同一世界、同一物体内的单元按三节点身份配对，反向的双方共有面消去，单侧面按原单元向外定向。节点或单元重排保留稳定 ID 即可。不同世界或不同物体可以复用这些 ID。单元的碰撞掩码与排除表只筛选查询对象，不改变体网格的外表面。此处采用“仅一个三维单元使用的面属于边界”的拓扑判据，并保留每个面的原单元归属。[VTK 外表面与原单元 ID](https://vtk.org/doc/nightly/html/classvtkDataSetSurfaceFilter.html) · [实现](../../src/phymeo/geometry/_tetrahedral_surface.py)

```python
import phymeo as ph

cell, ball = ph.CollisionKey(0, 0, 7), ph.CollisionKey(0, 1, 0)
model = ph.CollisionModel((
    ph.CollisionShape(cell, "tetrahedron", (0, 1, 2, 3)),
    ph.CollisionShape(ball, "sphere", (0,), 0.125),
))
paths = {
    cell: ph.CollisionMotion([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    ball: ph.CollisionMotion([[-0.25, 0.25, 0.25]], [[0, 0.25, 0.25]]),
}
with ph.CollisionWorkspace(model, "cuda") as work:
    outward = work.exterior_faces[cell]
    pair, = ph.detect(model, paths, ph.CollisionPolicy(), work).pairs
    assert pair.sweep.first_t == (0.5, 0.5)
    face, = [f for f in pair.facets if f.sweep.first_t is not None]
    weights = dict(face.sweep.witness.first_weights)
    assert weights == {0: 0.5, 1: 0.0, 2: 0.25, 3: 0.25}
```

## 范围与结果

- **球–软体。** 支持静止或平移的球，以及节点各自仿射运动的正向四面体。查询覆盖外面的内部、边和顶点。球在体内而未碰到边界时，窄相返回 `surface_clear`。若确实由包围盒排除则返回 `separated/aabb`。消费者需另行查询实体占据才能判断完全包含。
- **布–软体。** 异体零厚、非退化布面允许任意仿射顶点运动。正厚布面使用完整三角棱柱。当前支持整个步长内派生法线方向恒定、且单位法线可由有理数精确表示的情形，包括法线 (3/5, 4/5, 0) 的倾斜面。查询保留两张帽面和三张侧壁，并裁剪四面体外面与整个厚度层的交集。布的材料地址由三角形权重和有符号高度 `first_height_m` 或 `second_height_m` 构成。相交区域严格位于层内时，高度也可以严格介于上下表面之间。
- **首次接触与末态。** 每个原形状对返回 `facets`，每张外面保留定向节点、末态见证和扫掠结果。两侧权重随配对顺序互换；四面体侧始终包含四个原材料节点，对面节点权重为零。最早事件、后续面命中和末态可独立读取。可证明精确时刻的共面事件另外提供 `event_time_ratio` 和 `event_witness`。普通 `sweep.witness` 对应时间区间中的实际采样时刻。[公开见证类型](../../src/phymeo/geometry/collision_witness.py)
- **失效处理。** 重复单元、同向共面或三重共面在创建工作区时拒绝。候选体任一单元在步内失去严格正向体积时返回 `unresolved`，包括两端正向而中途塌陷的路径。改变方向或超出当前数域的厚布法线返回 `unsupported`。本批按一致的四面体网格查询外面，不提供任意网格的全局嵌入证明；四面体–四面体的占据、同体布–体和一般旋转厚布仍有独立开发项。
- **后端分工。** tyx 在选定的 interpret 或 CUDA 后端执行宽相及球–面、点–面、边–边的首次事件候选。HOST 从原始坐标复核时间和材料权重，裁剪并汇总各张外面的结果。静态汇总用窄相的原始有符号量分类，发布的浮点距离可以舍入到零。[分派入口](../../src/phymeo/geometry/collision_pipeline.py) · [tyx 批处理](../../src/phymeo/backends/tyx_collision.py)

## 2026-11-02 本地验收

| 独立构型 | 公开入口的观测 |
| --- | --- |
| RD04 基准及存储重排 | 两个单元共六张外面。首触区间包含 0.75，点为 (0, 0.2, 0.2) m，单元 0 的权重为 (0.6, 0, 0.2, 0.2)。 |
| 球从另一半接近、外部掠过与内部共享面陷阱 | 下半边在约 0.8 首触单元 1，点为 (0, 0.25, −0.375) m。外部负控由宽相排除。完全位于体内的小球到两单元外面的最小间隙均为 0.1875 m，均返回 `surface_clear`。 |
| DD04 零厚、半厚 0.02 m、反绕序及侧移 | 零厚布首触约为 0.75，增厚后约为 0.70。反绕序保持首触与 0.02 m² 区域，接触侧的有符号高度从 +0.02 m 变为 −0.02 m。侧移负控无接触。 |
| 变形四面体与厚布留出 | 第一面在精确时刻 9/14 接触，区域为 0.046875 m²。面 (1, 2, 3) 在 [0.9920061345801514, 0.9920061345810609] 首触；末态保留含 (3/14, 3/16, 9/16) m 的截面。半步拆分和随动坐标对照保留首触。 |
| 仅侧壁首触与初始层内 | 两张帽面始终避开体面，侧壁在 1/2 接触，三点的材料高度均在半厚度之内。整个体面在入步已严格处于布层内时，首触为 0。 |
| 舍入回归 | 球心距最近顶点的距离平方超出半径平方 2⁻¹⁰⁸⁰ m²。显示间隙为零，查询仍返回 `surface_clear`。旧汇总曾误报 `touching`，对应首次失败与修正保留在原日志。 |

十个原构型都在 HOST 对换形状顺序，保存二十份观测。设备混合查询覆盖十个世界、25 个形状、300 对，其中 14 对进入窄相。两端设备保存的目标结果一致，十五条目标单元记录均与独立 HOST 查询的完整记录一致，比较时只对应查询的世界编号。CUDA 首次执行及同输入复查合计处理 126 条球–面条件、1200 条顶点–面条件和 808 条边–边条件。容量不足时报告需要 14 对并保留上次结果；注入后端故障时也保持上次发布。[HOST 多案例回归](../../tests/collision/test_tetrahedral_surface_detection.py) · [tyx 和事务回归](../../tests/collision/test_tyx_tetrahedral_surface_detection.py)

最终 HOST **118 passed / 4.88 s**，tyx interpret **20 passed / 9.21 s**，实际 CUDA **20 passed / 9.99 s**，均 exit 0。三组各核对 31 项源文件、输入和配置的运行前后与交付绑定。8 个改动的 Python 文件通过 Ruff 和格式检查，执行槽已释放。原始记录在 `artifacts/p65-collision-m2-tet-exterior-local-20261102-01` 的 `host-final-pytest-182456232`、`interpret-final-pytest-182604255` 和 `cuda-final-pytest-182619889`。沿用固定 tyx `99f6df7c0fd8fdeef77f97df83914da27b64b7ee`。本机复现时将当前工作树的 `src` 加入 `PYTHONPATH`，逐条执行并分别使用新的绝对 `--basetemp`。

```text
python -m pytest tests/collision/test_tetrahedral_surface_detection.py tests/collision/test_collision_pipeline.py tests/collision/test_round_collision.py tests/collision/test_sphere_face_sweep.py tests/collision/test_triangle_pair_sweep.py tests/collision/test_adjacent_triangle_sweep.py -q -p no:cacheprovider
python -m pytest tests/collision/test_tyx_tetrahedral_surface_detection.py tests/collision/test_tyx_collision_pipeline.py tests/collision/test_tyx_round_collision.py tests/collision/test_tyx_sphere_face_sweep.py tests/collision/test_tyx_triangle_pair_sweep.py tests/collision/test_tyx_adjacent_triangle_sweep.py -q -p no:cacheprovider --tyx-mode=interpret --require-tyx
python -m pytest tests/collision/test_tyx_tetrahedral_surface_detection.py tests/collision/test_tyx_collision_pipeline.py tests/collision/test_tyx_round_collision.py tests/collision/test_tyx_sphere_face_sweep.py tests/collision/test_tyx_triangle_pair_sweep.py tests/collision/test_tyx_adjacent_triangle_sweep.py -q -p no:cacheprovider --tyx-mode=cuda --require-tyx
```

DD05 已通过完全包含、交换内外、分离、塌陷和变形外部首触等构型，将实体占据与边界扫掠分别发布。[闭体实测](p65-collision-m2-closed-volumes.md) · [行动计划](../superpowers/plans/2026-11-02-p65-tyx-collision-pipeline.md)
