# DD03 · 布面自碰撞中的材料邻接

`phymeo.detect` 现在可以区分同一布体上的合法邻接与异处自接触。两张非退化的零厚三角面可以共用顶点或边，每个顶点按自己的起点和终点作仿射运动。同体内同一材料节点的起点、终点必须一致，试算在发布前检查这一条件。[注册与分派](../../src/phymeo/geometry/collision_pipeline.py) · [材料接触实现](../../src/phymeo/geometry/_triangle_pair_sweep.py)

```python
import phymeo as ph

a, b = ph.CollisionKey(0, 1, 0), ph.CollisionKey(0, 1, 2)
model = ph.CollisionModel((
    ph.CollisionShape(a, "triangle", (0, 1, 2)),
    ph.CollisionShape(b, "triangle", (0, 3, 4)),
))
paths = {
    a: ph.CollisionMotion([[0, 0, 0], [2, 0, 0], [0, 2, 0]]),
    b: ph.CollisionMotion([[0, 0, 0], [1, 0.5, 0.5], [1, 0.5, -0.5]]),
}
with ph.CollisionWorkspace(model, "cuda") as work:
    pair, = ph.detect(model, paths, ph.CollisionPolicy(), work).pairs
    assert pair.status == "touching"
    assert pair.witness.first_point == (0.5, 0.25, 0.0)
    assert pair.witness.shared_material_points[0].point == (0.0, 0.0, 0.0)
```

## 如何读结果

| 情形 | 返回值 |
| --- | --- |
| 只有同一材料点或同一材料边重合 | 保留 `candidate=True`，返回 `adjacent/shared_material_only`。几何间隙为零，`shared_material_points` 给出合法交集的生成点，`intersection_points` 为空。全程只邻接时 `sweep.first_t` 为 `None`。 |
| 采样时刻存在不同材料地址相交 | 静态返回 `touching`。合法共享点与不同材料的交点分开列出，两组点的凸包给出实际几何交集。二者同时存在时，以全部生成点的均值给出一个远离共享特征的代表点及两侧权重。 |
| 步内先在别处碰到 | 返回 `hit` 和原归一化时间区间，宽度至多 2⁻⁴⁰。当入步已有异处交集时首触为零。入步只有合法共享点时，零几何间隙不会触发首触。 |
| 瞬时命中落在两个可表示时刻之间 | `sweep.witness` 记录 `sweep.witness_time` 的实际几何。若取样时只有合法邻接，`sweep.feature_witness` 另存该时刻两侧独立材料特征的最近点、权重和实测距离。它用于核对事件附近的两套材料地址。 |

材料身份由世界、物体、顶点 ID 和权重共同决定。比较发生在精确存储坐标导出的重心权重上，去除零权重后逐 ID 比对。不同物体复用同一个顶点号、或同一物体的两个不同顶点恰好处于同一世界位置，都保留为真实交点。显式排除、跨世界和掩码仍先执行。有厚度的面保留 `unsupported`，步内面退化的候选返回 `unresolved`。[公开见证类型](../../src/phymeo/geometry/collision_witness.py)

## 首触如何避开永久共享特征

宽相保留整对三角面。窄相继续检查非共享顶点对整面的运动，以及没有共同材料端点的两条边。边的一端可以是共享顶点，只要对边端点与它不同，该边–边组合就会保留。入步是否已在异处相交，单独用完整静态交集检查，包括边穿过面内部。

两面只共享一个顶点时，可从该点观察两面覆盖的方向。若从合法邻接变为真实相交，初次相交的方向至少在一面的边界上。沿该方向的远端要么是落在另一面上的非共享顶点，要么是从共享点发出的边与另一面对边的交点。共享整条边时，非共面的两面只沿此边相遇。同侧共面重叠的额外交点由非共享顶点或两条不共端点的边给出，平行重叠时有端点可供检查。三个 ID 都相同则是同一个材料三角形。两个面均非退化是这些判据的前提。

tyx 复用已有点–面和边–边时间内核。HOST 复核原坐标上的闭条件，确定与后端无关的首触区间。此处验收的是检测和材料见证，接触力由后续动力学消费者产生。[tyx 内核](../../src/phymeo/backends/_tyx_collision_kernels.py) · [IPC 候选特征](https://ipctk.xyz/cpp-api/candidates.html)

## 2026-11-02 本地验收

| 构型 | 结果 |
| --- | --- |
| base、renumbered | 原点只有一个材料地址，另外一端 (1, 0.5, 0) 有两个地址。代表点 (0.5, 0.25, 0) 的两组权重分别为 (0.625, 0.25, 0.125) 和 (0.5, 0.25, 0.25)。节点重排保持该结果。 |
| flat、hinge | 分别只保留一个共享顶点、两个共享边端点。均为 `adjacent`，没有不同材料的接触记录。 |
| holdout | 代表点为 (0.625, 0.25, 0)，两侧权重为 (0.5625, 0.3125, 0.125) 和 (0.5, 0.375, 0.125)，与独立目录一致。 |
| 共边折弯的两条路径 | 向公共边另一侧经过共面姿态时，全程只有合法邻接。向同一侧共面时，两个端点姿态均只有合法邻接，首触区间 [0.33333333333303017, 0.33333333333393966] 包含 1/3。取样时两个不同材料特征的距离为 2.2737367544323206×10⁻¹³ m，两点都位于 y=0.5 m。 |
| 顶点均未到达对面的首触 | 一条连着共享顶点的边在 t=0.5 碰到另一面的对边。实际交线从合法原点延伸至 (1, 0, 0)。代表点 (0.5, 0, 0) 的两套权重为 (0.875, 0.125, 0) 和 (0.5, 0.25, 0.25)。在 1/4 处分步后，后段局部首触区间包含 1/3。 |
| 身份、面积和事务负控 | 只拆材料 ID 或改成不同物体后，相同坐标会产生真实接触。高度 ±2⁻⁴⁶ m 的同侧与异侧薄三角形分别判为重叠接触和合法共边。三个 ID 全相同时没有远端事件。退化路径和显式过滤有各自状态。 |

设备把原五个构型及三个运动对照放在八个世界，一次提交 24 个形状、276 对，28 对进入窄相。所有形状对的结果与 HOST 相同，八个指定目标的完整返回也从保存的收据逐对核对。首次查询和恢复原输入的复查合计在 CUDA 运行了 16 条顶点–面条件及 14 条边–边投影条件，共 2 次三角面窄相启动，其中 4 个候选时刻来自边–边内核。另一个坐标变换、共同运动与顶点重排对照仍在 1/2 命中，代表点变为 (2.25, −2.375, 0.75)。零容量报告所需的 28 对，窄相启动数不增加，注入故障后上次成功结果保持可读。[HOST 用例](../../tests/collision/test_adjacent_triangle_sweep.py) · [tyx 用例](../../tests/collision/test_tyx_adjacent_triangle_sweep.py)

最终 **HOST 87 passed / 1.41 s**、**tyx interpret 18 passed / 3.62 s**、**实际 CUDA 18 passed / 6.02 s**，均 exit 0。三组包括受共享分派影响的 M1、RR、RD 与异体 DD 回归，分别核对了 21、22、22 项源文件、输入与配置的运行前、后及交付时绑定。9 个改动的 Python 文件通过 Ruff 和格式检查。初次三个功能反例保留。第一次完整 HOST 运行还纠正了一项旧构型的预期：重合边的第一端共用 ID，另一端在两面各用不同 ID。保留原几何后，断言分别核对了合法端点和真实接触端点。

原始收据位于 `artifacts/p65-collision-m2-dd03-local-20261102-01`。最终三组为 `host-final-pytest-175857694`、`interpret-pytest-175910431`、`cuda-pytest-175928020`。新增测试保存 8 份 HOST 观测及各 2 份设备观测。执行环境是 Python 3.12.14、pytest 9.1.1 和固定 tyx `99f6df7c0fd8fdeef77f97df83914da27b64b7ee`。复现时将工作树的 `src` 加至 `PYTHONPATH`，逐条执行并各自使用新的绝对 `--basetemp`。

```text
python -m pytest tests/collision/test_adjacent_triangle_sweep.py tests/collision/test_triangle_pair_sweep.py tests/collision/test_collision_pipeline.py tests/collision/test_round_collision.py tests/collision/test_sphere_face_sweep.py -q -p no:cacheprovider
python -m pytest tests/collision/test_tyx_adjacent_triangle_sweep.py tests/collision/test_tyx_triangle_pair_sweep.py tests/collision/test_tyx_collision_pipeline.py tests/collision/test_tyx_round_collision.py tests/collision/test_tyx_sphere_face_sweep.py -q -p no:cacheprovider --tyx-mode=interpret --require-tyx
python -m pytest tests/collision/test_tyx_adjacent_triangle_sweep.py tests/collision/test_tyx_triangle_pair_sweep.py tests/collision/test_tyx_collision_pipeline.py tests/collision/test_tyx_round_collision.py tests/collision/test_tyx_sphere_face_sweep.py -q -p no:cacheprovider --tyx-mode=cuda --require-tyx
```

后续 RD04/DD04 已将定向四面体外表面接到球、布的独立扫掠案例。DD05 的闭体包含单独验收。[外表面实测](p65-collision-m2-tetrahedral-surfaces.md) · [多案例计划](../superpowers/plans/2026-11-02-p65-tyx-collision-pipeline.md)
