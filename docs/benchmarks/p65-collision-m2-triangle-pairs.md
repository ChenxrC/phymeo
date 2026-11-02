# M2-DD 首批 · 两张独立布面的首次接触

`phymeo.detect` 支持不同物体上的两张双侧、零厚三角面。每个顶点可以独立作仿射运动。静态查询返回最近点、实际交集和两侧材料权重。扫掠查询还返回首次接触时间区间，与步末间隙分开保存。同体邻接留给 DD03，有限厚度留给后续壳表面。两者在宽相入选时继续明确返回 `unsupported`。[实现](../../src/phymeo/geometry/_triangle_pair_sweep.py) · [公开几何记录](../../src/phymeo/geometry/collision_witness.py)

```python
import phymeo as ph

a, b = ph.CollisionKey(0, 1, 0), ph.CollisionKey(0, 2, 0)
model = ph.CollisionModel((
    ph.CollisionShape(a, "triangle", (11, 7, 3)),
    ph.CollisionShape(b, "triangle", (31, 41, 53)),
))
paths = {
    a: ph.CollisionMotion([[-1, 0, 0], [1, 0, 0], [0, 0, -0.25]]),
    b: ph.CollisionMotion(
        [[0, -1, 0.5], [0, 1, 0.5], [0, 0, 1.5]],
        [[0, -1, -1.5], [0, 1, -1.5], [0, 0, -0.5]],
    ),
}
with ph.CollisionWorkspace(model, "cuda") as workspace:
    row, = ph.detect(model, paths, ph.CollisionPolicy(), workspace).pairs
    assert row.sweep.first_t == (0.25, 0.25)
    point, = row.sweep.witness.intersection_points
    assert point.point == (0.0, 0.0, 0.0)
    assert dict(point.first_weights) == {11: 0.5, 7: 0.5, 3: 0.0}
    assert dict(point.second_weights) == {31: 0.5, 41: 0.5, 53: 0.0}
```

- **首触和取样。** 首触区间宽度至多 2⁻⁴⁰ 个归一化步。事件见证对应 `sweep.witness_time`，`row.witness` 对应步末。瞬时擦碰落在两个可表示时刻之间时，区间可以证实命中，而取样时刻的 `intersection_points` 可以为空。
- **零厚交集。** 两张面相交时间隙为零，法线不唯一。每个 `MaterialContactPoint` 独立携带世界点和两侧的材料权重，物体与形状身份来自 `CollisionPair`。记录按材料地址排序，这些点的凸包是取样时刻的三角交集，可为点、线段或共面区域。若原输入的某张面在闭合步内退化，命中的候选返回 `unresolved`。
- **入步已经相交。** 独立检查边穿过对方面内部的交点。初态横穿即使没有顶点落在对面、也没有边与边相碰，仍返回首触 `(0, 0)`。共面交集保留落入另一面的顶点和边界之间的交点。

## 连续检测路径

每个形状对检查六个有方向的顶点–面组合和九个边–边组合。顶点必须落在对面平面上，三个重心分子均须非负。边–边事件首先要求两轴共面，然后在有效投影内检验两条闭线段的参数。只有在当前投影非奇异的根上才使用该投影。平行共线段相触时至少一个端点属于另一段，这类情况已由包含边界的顶点–面条件覆盖。持续共面的内点边–边交叉可在邻近时刻延续，从分离进入时由端点首次接触对边。入步已有交集单独判断。

tyx 在设备上计算三次共面方程，查找采样时刻之间的符号变化，细化候选根并检验最多三个材料条件。HOST 使用原存储坐标的多项式检查候选两侧、重根和共根，发布与后端无关的时间区间。FP32 没有给出候选的组合也会检查，根隔离达到既有上限时明确返回未决。这是几何检测，没有产生接触力或摩擦历史。接口与移动端点的定义参考 IPC Toolkit 的点–三角和边–边 CCD。[tyx 内核](../../src/phymeo/backends/_tyx_collision_kernels.py) · [根与投影处理](../../src/phymeo/geometry/_sweep_polynomial.py) · [IPC Toolkit CCD](https://ipctk.xyz/cpp-api/ccd.html)

## 2026-11-02 本地验收

| 构型 | 公开结果 |
| --- | --- |
| DD01 基准、复用 ID、交换两侧 | 首触 [0.5, 0.5]，世界点为原点。面权重各 1/3，点侧权重为 1。两块布复用顶点 11 后仍保留两个物体的地址。 |
| DD01 留出和分离负控 | 留出区间 [0.6666666666660603, 0.6666666666669698] 包含 2/3，点为 (0.125, 0.375, 0)。侧移负控默认由 AABB 排除，扩大检测余量后仍由窄相判全段分离。 |
| DD02 基准、换序和留出 | 两条边首次在原点相触，轴参数各 0.5，点–面基线的最短距离仍为 1 m。基准与交换构型的首触均为 [0.5, 0.5]。留出区间 [0.5999999999994543, 0.6000000000003638] 包含原输入首触。分离负控末态为 +0.05 m。 |
| 双端分离的穿越 | 步初 +0.5 m、末态 +0.25 m，步内首触 [0.25, 0.25]。在 1/8 处切步后，后段局部首触区间包含 1/7。 |
| 孤立的边–边擦碰 | 两端分离，只有 t=1/3 相触，返回 [0.33333333333303017, 0.33333333333393966]。边抬高 2⁻³⁰ m 后全段分离。降低边的位置后另检验了两个无理事件中的较早者。 |
| 共面及初态交集 | 共面重叠得到六角形，完全包含得到三个内面顶点，滑动首触得到两个端点组成的线段。初始横穿得到长 0.5 m 的交线，首触为 0。边在步内变平行但几何分离的反例也保持分离。 |
| 坐标变换与共面负控 | 整体旋转、共同平移、顶点重排后，首触仍为 1/4，世界点为 (3.125, −4.0625, 1.1875)，两侧材料权重均为 (1/2, 1/2, 0)。另一对共面三角形的 AABB 重叠，但末态正间隙为 0.08838834764831845 m。 |

三端按九个原目录构型及两个新增控制逐对核对了状态、见证和时间区间。22 个形状在一次查询中产生 231 对，220 对跨世界过滤，11 对进入窄相。在首次查询与恢复原输入后的复查中，CUDA 累计处理 132 个顶点–面条件和 136 个边–边投影条件，完成 4 次三角面窄相启动。单独运行 DD02 留出构型时，实际 CUDA 搜索给出了非采样网格上的边–边候选。零容量返回所需的 11 对，窄相没有新启动。注入故障后仍可读上次成功结果，出错工作区关闭。[HOST 回归](../../tests/collision/test_triangle_pair_sweep.py) · [设备回归](../../tests/collision/test_tyx_triangle_pair_sweep.py)

完整受影响组为 **HOST 74 passed / 1.26 s**、**tyx interpret 14 passed / 3.01 s**、**实际 CUDA 14 passed / 4.93 s**，均 exit 0。每组包括 M1、RR 和 RD01 回归，21 项源、输入及配置在运行前后一致。随后只追加坐标变换与共面分离两个回归，分别在新建工作区与 HOST 对照运行，interpret **2 passed / 0.71 s**，CUDA **2 passed / 1.49 s**，各自 18 项运行绑定一致。13 个 Python 文件通过 Ruff 和格式检查，追加后的设备测试文件也通过相应检查。Python 3.12.14、pytest 9.1.1，tyx 固定在 `99f6df7c0fd8fdeef77f97df83914da27b64b7ee`。

本机原始收据在 `artifacts/p65-collision-m2-dd01-dd02-local-20261102-01`。完整三组分别为 `host-pytest-161102208`、`interpret-pytest-161112930` 和 `cuda-pytest-161127612`，另有两个 frame 组和最终释放记录。新增用例保存 12 份 HOST 物理几何观测，完整设备组各 2 份，增量设备组各 2 份。原版三个功能失败均已保留。复现时使用固定 tyx 环境，将本工作树的 `src` 加入 `PYTHONPATH`，逐条执行并各自指定新的绝对 `--basetemp`。

```text
python -m pytest tests/collision/test_triangle_pair_sweep.py tests/collision/test_collision_pipeline.py tests/collision/test_round_collision.py tests/collision/test_sphere_face_sweep.py -q -p no:cacheprovider
python -m pytest tests/collision/test_tyx_triangle_pair_sweep.py tests/collision/test_tyx_collision_pipeline.py tests/collision/test_tyx_round_collision.py tests/collision/test_tyx_sphere_face_sweep.py -q -p no:cacheprovider --tyx-mode=interpret --require-tyx
python -m pytest tests/collision/test_tyx_triangle_pair_sweep.py tests/collision/test_tyx_collision_pipeline.py tests/collision/test_tyx_round_collision.py tests/collision/test_tyx_sphere_face_sweep.py -q -p no:cacheprovider --tyx-mode=cuda --require-tyx
```

下一批为 DD03，同一张布上的共享顶点或共享边需要区分合法邻接与远离该材料特征的真实相交。[多案例实施计划](../superpowers/plans/2026-11-02-p65-tyx-collision-pipeline.md)
