# RD02 · 胶囊整轴与有限布面

`phymeo.detect` 可以查询静止或平移的刚胶囊与双侧、零厚三角形。三角形的三个材料点可以各自作仿射运动，整步须保持非退化。胶囊使用两个有序的轴端点和一个正半径。线段到有限三角形的最近点带有空间参数和三角形重心坐标；扫掠另外求步内首次接触的时间。[线段–三角形距离参照](https://raw.githubusercontent.com/davideberly/GeometricTools/master/GTE/Mathematics/DistSegment3Triangle3.h) · [带最小距离的连续检测参照](https://ipctk.xyz/cpp-api/ccd.html) · [Phymeo 实现](../../src/phymeo/geometry/_capsule_triangle_sweep.py)

```python
import phymeo as ph

face = ph.CollisionKey(0, 0, 4)
cap = ph.CollisionKey(0, 1, 8)
model = ph.CollisionModel((
    ph.CollisionShape(face, "triangle", (7, 13, 31)),
    ph.CollisionShape(cap, "capsule", (11, 29), radius=0.25),
))
paths = {
    face: ph.CollisionMotion([[0.5, -1, 0], [0.5, 1, 0], [0.5, 0, -0.25]]),
    cap: ph.CollisionMotion([[-1, 0, 0.75], [1, 0, 0.75]],
                            [[-1, 0, -0.75], [1, 0, -0.75]]),
}
with ph.CollisionWorkspace(model, "cuda") as work:
    pair, = ph.detect(model, paths, ph.CollisionPolicy(), work).pairs
    lo, hi = pair.sweep.first_t
    assert lo <= 1/3 <= hi
    first, last = pair.sweep.witness.capsule_axis_span
    assert first.axis_parameter == last.axis_parameter == 0.75
    assert first.face_point == (0.5, 0.0, 0.0)
```

## 结果与条件

- `pair.sweep.first_t` 给出归一化步内时间区间，宽度至多 2⁻⁴⁰。`pair.witness` 记录末态，`pair.sweep.witness` 记录区间内实际采样姿态。静态查询只返回该姿态的见证。
- `witness.capsule_axis_span` 给出轴上所有最近位置的两个极端，按输入轴端方向排列。每端保留轴参数、轴上世界点、三角形上世界点及三角形材料权重。唯一最近点重复两端；线接触保留整个区间。代表见证使用区间中点，胶囊两节点的权重为 (1−s, s)。交换形状顺序仍按同一胶囊方向解释区间；反转胶囊轴后参数随之反转。[公开见证类型](../../src/phymeo/geometry/collision_witness.py)
- 有符号间隙是最近轴–面距离减去胶囊半径，以原始坐标上的平方距离判定正负。当轴穿过面内时，见证保留轴上位置及布面材料点；此时接触法线和胶囊边界上的唯一点为空。零轴长有球的几何，两端材料 ID 仍可追踪，代表权重各为 1/2。
- 对于旋转轴或正厚度布面，进入窄相的候选返回 `unsupported`。布面在步内退化，候选返回 `unresolved`。本批输出几何检测结果；力与运动响应由动力学模块承担。

## 为什么能检出轴身首触

静态查询覆盖端帽对面、轴段对三条有限边，以及轴实际穿入面内的交点。对连续运动，初始间隙为正且胶囊有正半径时，首触处的轴–面距离必等于半径。若最近轴点和最近面点都在各自内部，轴必平行于面；沿最近点区间走到一端，会碰到胶囊端帽或三角形边。由此，端帽–三角形、布顶点–有限轴与轴身–有限边三类条件覆盖首触。

对轴方向 u、布边方向 v 和两起点差 p，取 n=u×v。非平行的边–轴内部首触满足 r²(n·n)−(p·n)²=0，两个线段参数还必须落在闭区间内。实现从等式中去掉与 n·n 公有的根，避免两支撑线瞬间平行时把 0=0 当作接触。平行或持续相切时，最早有限接触由端帽或布顶点覆盖。tyx 在选定后端执行宽相、端帽与轴身条件的候选计算；HOST 依据原坐标复核闭条件，包含两个设备采样点之间的重根。[设备批处理](../../src/phymeo/backends/tyx_collision.py) · [时间区间求解](../../src/phymeo/geometry/_sweep_polynomial.py)

## 2026-11-02 本机验收

| 构型 | 公开入口的观测 |
| --- | --- |
| 原五个静态构型 | 两端帽各留 0.4 m 净隙，轴在 s=0.5 穿过三角形内部。端帽相切给出 s=0 和零间隙，侧移负控分离。斜轴在 (−0.25, 0, 0) m 穿面，颠倒端点后 s 从 0.25 变成 0.75，布面位置与权重保持。 |
| 两端分离的轴身–边穿越 | 首触区间 [0.33333333333303017, 0.33333333333393966] 包住 1/3，轴参数是 3/4。布边内部的材料权重为 (0.5, 0.5, 0)。 |
| 变形边与平移胶囊 | 在 t=1/2 首触，s=5/8，布面权重为 (3/4, 1/4, 0)，接触点 (0.25, 0, 0) m。刚体坐标变换、共同平移、材料重排、交换双方以及在 1/4 处分步均保留物理事件。 |
| 平行面及平行边 | 平行面在 t=1/2 首触，整段 s∈[3/8, 5/8] 同时相切。瞬间平行但相隔的边保持分离；另一构型在平行姿态之后到 t=3/4 才真正接触。 |
| 孤立的轴身–顶点、轴身–边擦碰 | 两条轨迹均只在 t=1/3 接触，s=11/16。顶点对照向外移动一个浮点刻度，边对照将半径缩小一个浮点刻度，各自返回分离。在专用 interpret 和 CUDA 查询中，采样内核均给出零命中候选，原坐标复核仍找到两个真实擦碰。 |
| 事务与失效 | 零容量报告实际需要的 14 对且不启动额外窄相。同输入复查得到相同结果；随后注入设备错误保留故障前最近一次成功发布。退化布面、旋转轴和厚布各有独立负控。 |

轴身–边孤立擦碰使用可手算的留出。令 v=3t−1，投影到垂直轴线的 yz 平面后，布边的支撑线为 z−vy/4=1/4+v²/16。其到原点的平方距离比半径平方 (1/4)² 多出一个在 v≠0 时严格为正的量，v=0 时最近点在两条线段的内部。这个对照同时检验重根和近失，不依赖设备是否恰在根上采样。[HOST 用例](../../tests/collision/test_capsule_face_sweep.py) · [实际后端用例](../../tests/collision/test_tyx_capsule_face_sweep.py)

HOST 主回归 **171 passed / 7.63 s**，随后新增的轴身–边留出 **2 passed / 0.40 s**，两次均 exit 0。最终 tyx interpret **24 passed / 15.40 s**，实际 CUDA **24 passed / 14.89 s**，均 exit 0。14 个世界混合查询 28 个形状、378 对，14 对进入窄相。两端设备的目标记录相同，也与 14 份独立 HOST 记录逐字段一致，只对应了世界编号。CUDA 成功查询与同输入复查共处理 350 条表面条件，其中轴身–顶点 54 条、轴身–边 44 条；这两类分别产生 4 和 16 个设备候选。

四个原始功能反例保留。第一次 interpret 运行 23 项通过、1 项事务断言失败：它把故障后的结果与首次发布对象比较，实际在此之前已成功复查并发布了新对象。修正后直接断言故障前后对象身份相同。所有产品源码在四组功能运行间相同。HOST 主回归与新增两项的源绑定分别为 36 和 29 项；为追加留出而改写的两个 HOST 测试文件均核回原段字节，设备专用测试的改动在最终两组独立核对。9 个改动的 Python 文件通过 Ruff 和格式检查，串行执行槽已释放。

原始日志位于 `artifacts/p65-collision-m2-rd02-local-20261102-01`。四组分别为 `host-pytest-185844161`、`grazing-pytest-190118735`、`interpret-final-pytest-190333931`、`cuda-pytest-190418406`，tyx 固定为 `99f6df7c0fd8fdeef77f97df83914da27b64b7ee`。复现时将本工作树 `src` 加入 `PYTHONPATH`，逐条执行并使用不同的新绝对 `--basetemp`。下列为直接模块入口，完整回归清单见各组 `final.json`。

```text
python -m pytest tests/collision/test_capsule_face_sweep.py -q -p no:cacheprovider
python -m pytest tests/collision/test_tyx_capsule_face_sweep.py -q -p no:cacheprovider --tyx-mode=interpret --require-tyx
python -m pytest tests/collision/test_tyx_capsule_face_sweep.py -q -p no:cacheprovider --tyx-mode=cuda --require-tyx
```

随后 RR03 已接入凸刚体的支撑特征、接触区域及精确进出时刻，沿滑动箱与异向边的独立基准验收。[凸体接触实测](p65-collision-m2-convex-hulls.md) · [三族计划](../superpowers/plans/2026-11-02-p65-tyx-collision-pipeline.md)
