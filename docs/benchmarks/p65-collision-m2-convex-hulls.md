# RR03 · 凸刚体的接触区域与连续区间

`phymeo.detect` 可以从两个无序顶点集构造实心凸包，查询静止或平移中的刚体。输入给出顶点的稳定材料 ID 和位置，检测器自行提取外面与真实棱边、选择支撑方向、裁剪接触区域。一个步长内可以同时取得开始接触和最后接触的时刻，即使两个端点都分离。[凸包构造](../../src/phymeo/geometry/_convex_hull.py) · [碰撞与区域](../../src/phymeo/geometry/_convex_contact.py)

```python
from itertools import product
import numpy as np
import phymeo as ph

a, b = ph.CollisionKey(0, 1, 0), ph.CollisionKey(0, 2, 0)
box = np.array(list(product((-0.5, 0.5), repeat=3)))
upper = box + [0, 0, 1]
model = ph.CollisionModel(tuple(ph.CollisionShape(k, "hull", tuple(range(8))) for k in (a, b)))
paths = {
    a: ph.CollisionMotion(box),
    b: ph.CollisionMotion(upper, upper + [1.25, 0, 0]),
}
with ph.CollisionWorkspace(model, "cuda") as work:
    pair, = ph.detect(model, paths, ph.CollisionPolicy(), work).pairs
    trace = pair.convex
    assert trace.start.patches[0].area_m2 == 1.0
    assert trace.last_time_ratio == (4, 5)
    final_face = next(p for p in trace.last.patches if p.normal == (0, 0, 1))
    assert final_face.dimension == 1 and len(final_face.points) == 2
    assert trace.end.relation == "separated"
```

## 读取结果

- `pair.convex.start` 和 `end` 是入步及末态。`first_time_ratio`、`last_time_ratio` 是闭接触区间与 [0, 1] 的交集端点，以原始存储坐标导出的有理数表示；对应的 `first`、`last` 保存这两个精确时刻的区域。整步分离时四个字段均为空。当接触延续到步末，最后时刻为 1。已有 `pair.sweep.first_t` 仍提供宽度至多 2⁻⁴⁰ 的区间，`sweep.witness` 保留实际可表示的取样姿态。[结果类型](../../src/phymeo/geometry/collision_witness.py)
- 每个 `patch` 给出法线、两侧支撑特征的极端材料 ID、区域维数、面积及全部角点。点、线、面的维数分别为 0、1、2。每个角点含两侧世界坐标和权重，可用原节点重建。面内点采用由稳定材料 ID 确定的扇形剖分；点云重排、附加内部点和共面非极端点均不改变这些控制的结果。
- 分离时区域为空，间隙是两个有限凸体的欧氏最近距离，包括沿对角方向分开时。接触余量只影响宽相候选。相触时区域是两个支撑特征的真实交集；有多个并列最优检测轴时分别保留区域，代表见证的法线为空。发生体积重叠时，负间隙给出最短平移脱离深度，区域记录移动前双方的支撑点，二者的法向距离等于该深度。输出不表示交叠体积或接触力。
- 本批支持三维满维凸包、零形状偏移和固定姿态平移。共面或重复位置输入返回 `invalid_geometry`。每凸包最多 64 个输入点，每对最多 4096 条不同无向检测轴，超限返回 `unresolved`。旋转凸包和圆角偏移候选保留 `unsupported`。这些范围以外的刚体查询另行验收。

## 检测过程

凸包从正体积单纯形开始，逐个加入外部顶点。共面三角片合并为凸多边形，只留下外面轮廓上的棱边。对两个满维凸体，分离轴测试使用双方的面法线和两侧棱方向的非零叉积。姿态固定时，这些方向在步内恒定；每条轴上的两个投影区间只作线性平移，各自的重叠时间区间可以直接求得。所有轴的时间区间相交后得到整对的闭接触窗口。[凸包背景](https://doc.cgal.org/latest/Convex_hull_3/index.html) · [分离轴与运动凸体](https://www.geometrictools.com/Documentation/MethodOfSeparatingAxes.pdf)

tyx 在所选后端把两个完整的极端顶点集投影到每条检测轴上，批量计算 FP32 候选时间区间，包括静态查询。原坐标复核优先检查设备报空或时间窗口最窄的轴，最终时刻、符号及材料位置由原数据确定。检测器用全部并列最大轴裁剪支撑多边形；对分离且支撑投影不相交的情形，另查真实顶点–面与棱–棱距离。[tyx 投影内核](../../src/phymeo/backends/_tyx_collision_kernels.py) · [后端缓冲与分页](../../src/phymeo/backends/tyx_collision.py)

## 2026-11-02 本机结果

| 构型 | 公开入口的观测 |
| --- | --- |
| 原基准与顶点重排 | 初态接触面积 1 m²，t=1/5 时为 0.75 m²，t=4/5 时只剩一条长 1 m 的线，步末欧氏间隙为 0.25 m。顶点重排后接触区间和区域保持。 |
| 正间隙与尺寸留出 | 提高上箱 0.02 m，0.03 m 余量使其仍进入候选，整步接触窗口为空。留出箱的初态区域为 0.375 m²，最后相触时刻改为 7/10，留下长 0.5 m 的线。 |
| 异向边内部首触 | 两斜盒需要 17 条轴，首次接触约为 1/2，唯一接触点为 (0, 0, 0.1) m。双方材料地址都由各自支撑边的两个端点等权组成。末态重叠深度约 0.05 m。 |
| 端点都分离 | 穿过一个箱体的轨迹从 1/4 接触到 3/4。另一条轨迹只在精确 1/3 发生角–角擦碰，三个并列轴都返回同一物理点。取样姿态仍有微小正间隙，精确事件单独保留。 |
| 不限于箱形 | 一个四面体顶点在 1/2 命中另一四面体的面内部。另两个各含 16 个顶点的凸体需要 1811 条检测轴，设备以 8 页完整处理。对角分离的两箱返回真实距离 √14/4 m；同心包含的两箱给出 1.25 m 的脱离深度及全部六个并列方向。 |
| 数值、容量和事务 | 间隙为 2⁻³⁰ m 的静止构型在设备的 3 条轴上都得到重叠候选，原坐标复核返回分离。零宽相容量报告实际所需的 8 对，窄相启动数不增加。设备故障关闭工作区并保留故障前最后一次成功发布。 |

补充轨迹用同一工作区连续提交四个子步。接触面积依次为 **1、0.75、0.25、0 m²**，维数为 **2、2、2、1**。正 z 支撑始终对应第一体的 [1, 3, 5, 7] 和第二体的 [0, 2, 4, 6]。每一步末态完整记录等于下一步入态，最后一个子步从接触边走向分离。两个 40 点凸体超出不同轴上限时，三端都返回 `convex_axis_capacity`，设备没有启动不完整的投影计算。[HOST 回归](../../tests/collision/test_convex_hull_contact.py) · [tyx 与事务回归](../../tests/collision/test_tyx_convex_hull_contact.py)

主回归 **HOST 194 passed / 8.16 s**、**tyx interpret 27 passed / 18.97 s**、**实际 CUDA 27 passed / 16.73 s**，均 exit 0。八个世界含 16 个形状、120 对，8 对进入窄相；两个设备后端和八份独立 HOST 目标记录的完整返回一致，只对应了世界编号。CUDA 主场景的首次查询与复查合计投影 108 条轴，16 次窄相启动，取得 106 条非空候选区间。分页与精度控制分别核对原始结果。

追加历史与超限控制分别通过 HOST **2 / 0.62 s**、interpret **1 / 0.82 s** 和 CUDA **1 / 1.62 s**；四段记录与超限结果逐字段一致。三个最初的功能反例保留。产品源码在六次运行间相同，主回归各绑定 42 项文件，追加组各绑定 34 项，前三份测试文件保留了已验证的原段。11 个改动的 Python 文件通过 Ruff 与格式检查，最终串行执行槽已释放。

原始收据位于 `artifacts/p65-collision-m2-rr03-local-20261102-01`。主组是 `host-pytest-192409938`、`interpret-pytest-192459635`、`cuda-pytest-192528054`；追加组是 `host-history-pytest-193425112`、`interpret-history-pytest-193438145`、`cuda-history-pytest-193449942`。沿用 tyx 固定提交 `99f6df7c0fd8fdeef77f97df83914da27b64b7ee`。复现时将工作树 `src` 加入 `PYTHONPATH`，逐条执行并各用新的绝对 `--basetemp`。以下是模块直接入口，完整回归清单在各组 `final.json`。

```text
python -m pytest tests/collision/test_convex_hull_contact.py -q -p no:cacheprovider
python -m pytest tests/collision/test_tyx_convex_hull_contact.py -q -p no:cacheprovider --tyx-mode=interpret --require-tyx
python -m pytest tests/collision/test_tyx_convex_hull_contact.py -q -p no:cacheprovider --tyx-mode=cuda --require-tyx
```

下一批 RR04 查询绕轴转动的胶囊与球。先用两端分离、中途碰撞的原转弧和整段分离负控验证首次接触，再用变更球位置的留出构型检查首触两侧的符号。[三族实施计划](../superpowers/plans/2026-11-02-p65-tyx-collision-pipeline.md)
