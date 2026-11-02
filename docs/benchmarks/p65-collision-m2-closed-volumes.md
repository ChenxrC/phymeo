# DD05 · 两个变形体的占据与外边界首触

`phymeo.detect` 可以查询不同物体的四面体材料域。每个候选单元对分别报告实体交集和外表面交集，`result.solid_pairs` 再按世界和物体汇总。一个小物体完全位于较大物体内时，实体交叠自入步就存在，外表面仍可以有正间隙。四面体的内部、外部、边界和有向体积是独立的几何判定。[CGAL 四面体定义](https://doc.cgal.org/latest/Kernel_23/classCGAL_1_1Tetrahedron__3.html) · [Phymeo 实现](../../src/phymeo/geometry/_tetrahedral_solid.py)

```python
import phymeo as ph

outer = ph.CollisionKey(0, 2, 7)
inner = ph.CollisionKey(0, 5, 9)
model = ph.CollisionModel(tuple(
    ph.CollisionShape(key, "tetrahedron", (0, 1, 2, 3)) for key in (outer, inner)
))
paths = {
    outer: ph.CollisionMotion([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]]),
    inner: ph.CollisionMotion([
        [0.25, 0.25, 0.25], [0.5, 0.25, 0.25],
        [0.25, 0.5, 0.25], [0.25, 0.25, 0.5],
    ]),
}
with ph.CollisionWorkspace(model, "cuda") as work:
    result = ph.detect(model, paths, ph.CollisionPolicy(), work)
    body, = result.solid_pairs
    assert body.start.relation == "first_contains_second"
    assert body.start.volume_ratio == (1, 384)
    assert body.first_t == (0.0, 0.0) and body.boundary_first_t is None
    point = body.start.witness
    assert min(w for _, w in point.first_weights) > 0
    assert min(w for _, w in point.second_weights) > 0
```

## 输出和适用范围

| 输出 | 语义 |
| --- | --- |
| `body.start`、`body.end` | 在各自姿态区分分离、零体积相触、部分体积重叠、包含方向与两体重合。包含允许两边界相触。`volume_ratio` 是基于存储坐标的精确体积比值，`volume_m3` 是其浮点读数。正体积交集给出两侧的单元 ID、形状内四个稳定节点的权重以及共同内部点。 |
| `body.first_t` | 全部有效单元的填充材料域第一次相遇的时刻区间。初始重叠或相触返回零。 |
| `body.boundary_first_t` | 两个物体实际暴露的三角面第一次相交的时刻区间。在同一物体的相邻单元之间转移时，这个字段可以始终为空。 |
| `pair.solid.boundary` | 候选单元具有相对的外面时，保存起点和终点最近的外面、见证与首触取样。并列的最近面选择一个代表。可独立证明精确首触时刻时，另给出 `event_time_ratio` 与该时刻的 `event_witness`。 |
| `body.status` | 完整放行是 `ok`。屏蔽了部分单元对时返回 `partial`。塌陷、翻转、或已证实自身单元内部重叠返回 `invalid_geometry`。自身非共享边界在步内相遇而超出当前体网格合同，返回 `unresolved`。这些状态均不发布整物体包含结论。 |

每个单元的四个材料点各自沿仿射路径运动。单元间按四个半空间裁剪，取交多面体顶点的平均位置作共同材料见证，并由其各个边界多边形计算交体积。共面的重复边界在计体积前合并。整物体比较使用原单元体积和精确交体积；先检查同一体各单元在整步保持正向，且只在共同材料处邻接。反向共享面的两个正向单元由其面平面分开，其他可能相遇的同体单元还检查非共享特征的扫掠。这个验证不提供同体自接触响应。当前适用于无额外偏移的线性四面体与不同物体的查询。[状态分派](../../src/phymeo/geometry/collision_pipeline.py) · [结果类型](../../src/phymeo/geometry/collision_witness.py)

tyx 执行宽相和仿射点–面、边–边的时间候选。HOST 复核原坐标上的闭条件，计算交体积、材料权重和整物体关系。首触区间的宽度最多为归一化步长的 2⁻⁴⁰。没有可证明精确时刻的事件仍按区间返回，取样点保留当时的实际间隙。[点–面与边–边检测](../../src/phymeo/geometry/_triangle_pair_sweep.py) · [设备执行](../../src/phymeo/backends/tyx_collision.py)

## 2026-11-02 本机结果

| 构型 | 观测 |
| --- | --- |
| 原基准和交换两体 | 外表面相距 0.2 m；实体交叠约 0.00133333 m³，内部见证为 (0.25, 0.25, 0.25) m。交换后包含方向反转。 |
| 原分离和塌陷 | 分离给出零交体积与空首触；塌陷给出无效几何。另一负控的包围盒相隔很远，塌陷仍返回无效。 |
| 两个原变形首触 | 分别在约 0.75 与精确 0.5 首触，点为 (0, 0.4, 0.4) m 与 (0.4, 0, 0.4) m。两条路径的末态均有严格内部见证。 |
| 小体跨两个单元 | 两个单元分别只与小体部分重叠。整物体报告完全包含，交体积精确为 1/384 m³。另一条运动路径在 0.375 进入上单元，全窗没有外边界首触。 |
| 无顶点在另一个体内 | 两个四面体相交成八面体，交体积精确为 4/3 m³。另外两条异向边在 0.5 首触于原点，两侧材料权重均为 (0.5, 0.5, 0, 0)。 |
| 端点、拓扑和材料身份 | 两端分离的穿越在 7/16 首触。从包含状态出发的另一条路径，外边界在 1/2 首触。共面、共边、共点相触均给出零交体积；完全重合的单元体积只计一次。材料重排、交换物体、旋转及共同运动的对照也保留了首触与权重。 |
| 网格有效性 | 不同节点 ID 代表同一物理单元会使体积重复累加，现返回无效。两个同体单元在步内互穿而两端分离，返回未决。仅共享合法材料边或顶点的正例通过。 |

最终 HOST **144 passed / 7.37 s**，tyx interpret **21 passed / 13.46 s**，实际 CUDA **21 passed / 14.07 s**，均 exit 0。设备混合查询含 12 个世界、28 个形状和 378 对，其中 19 对进入窄相。原始体关系和 16 条跨物体单元记录在两个设备后端一致，测试还逐对核对同一混合场景的 HOST 返回。六个原案例及跨接缝包含与独立保存的 HOST 记录等价，只对应了世界编号。CUDA 初次查询和同输入复查合计执行 280 条点–面条件、164 条边–边条件，返回 4 个边–边时间候选。容量不足和注入后端故障均保留上次成功发布。[HOST 回归](../../tests/collision/test_tetrahedral_solid_detection.py) · [tyx 回归](../../tests/collision/test_tyx_tetrahedral_solid_detection.py)

初次四个功能反例均返回了旧的 `unsupported`。实现后的首轮 HOST 另暴露了时间刻度缺少归一化，原七项失败修正后全部通过。无效网格的两个反例也保留原始失败和修正。三个最终运行各核对 34 项源文件、输入与配置的运行前后和交付绑定；7 个改动的 Python 文件通过 Ruff 和格式检查。原始证据位于 `artifacts/p65-collision-m2-dd05-local-20261102-01`，三组分别为 `host-final-pytest-184317746`、`interpret-pytest-184359065` 和 `cuda-pytest-184420491`。唯一执行槽已释放，tyx 仍为 `99f6df7c0fd8fdeef77f97df83914da27b64b7ee`。

复现时将本工作树 `src` 放入 `PYTHONPATH`，在同一执行槽逐条运行，各用新的绝对 `--basetemp`。下面是本模块的直接入口，完整回归文件清单在各运行的 `final.json`。

```text
python -m pytest tests/collision/test_tetrahedral_solid_detection.py -q -p no:cacheprovider
python -m pytest tests/collision/test_tyx_tetrahedral_solid_detection.py -q -p no:cacheprovider --tyx-mode=interpret --require-tyx
python -m pytest tests/collision/test_tyx_tetrahedral_solid_detection.py -q -p no:cacheprovider --tyx-mode=cuda --require-tyx
```

随后 RD02 已接入胶囊整轴对有限布面的查询，轴部材料参数与步内时间分别发布。[胶囊–布面实测](p65-collision-m2-capsule-face.md) · [三族案例计划](../superpowers/plans/2026-11-02-p65-tyx-collision-pipeline.md)
