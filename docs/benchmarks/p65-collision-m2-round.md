# M2 首批 · 球、胶囊与移动半空间的首触

公共查询 `phymeo.detect` 现可计算球–球、球–胶囊、胶囊–胶囊，以及球或胶囊–无限半空间的静态间隙和直线平移首触。两侧都可以运动。平面由一个材料锚点和非零法向定义，正法向指向允许的一侧，平移时法向固定。有限板的边缘、背面以及转动胶囊的扫掠分别留在 RD03 和 RR04。[运动输入](../../src/phymeo/constraints/collision_shapes.py) · [公开结果](../../src/phymeo/geometry/collision_witness.py)

```python
import phymeo as ph

floor = ph.CollisionKey(0, 1, 0)
ball = ph.CollisionKey(0, 2, 0)
model = ph.CollisionModel((
    ph.CollisionShape(floor, "plane", (19,), normal=(0, 0, 1)),
    ph.CollisionShape(ball, "sphere", (7,), radius=0.1),
))
paths = {
    floor: ph.CollisionMotion([[0, 0, 0]]),
    ball: ph.CollisionMotion([[0, 0, 0.3]], [[0, 0, -0.1]]),
}
with ph.CollisionWorkspace(model, "cuda") as work:
    pair, = ph.detect(model, paths, ph.CollisionPolicy(), work).pairs
    assert pair.status == "hit"
    first_contact = pair.sweep.first_t  # 步内归一化时间，约 0.5
    end_gap = pair.witness.gap_m        # 末态约 −0.2 m
```

- **运动查询。** `sweep.first_t` 是闭合的首次接触时间区间，最大宽度为 2⁻⁴⁰ 个归一化步。初态已经接触时返回 `(0, 0)`。进入窄相后全程分离时返回 `None`。事件附近的几何在 `sweep.witness`，对应的实际取样时刻在 `sweep.witness_time`。步末另有 `pair.witness`，因此两端都分离的穿越仍能报告首触。
- **材料见证。** 球和胶囊保留原形状键与材料权重。两个非零胶囊轴平行时，`axis_parameter_span` 保存所有最近轴参数对所在线段的两个端点。反向轴保留参数配对。轴重合时可报告负间隙且法线不唯一。平面侧的世界点由锚点与切向位移 `first_offset` 或 `second_offset` 重建。
- **筛选与后端。** 无限半空间在结果中没有有限 AABB。世界、同体、掩码和显式排除仍先执行。半空间通过过滤后保守入选，即使远离锚点也能检测。检测余量影响候选，物理首触按原半径计算。超容量查询在窄相启动前返回，并保留上次成功结果。

## 首触怎样得到

每条有限轴有端点和轴内两类特征。两条轴相对平移时，固定特征上的最近参数随时间仿射变化。将参数限制在两条闭线段上，得到每个特征的有效时间区间和一条凸二次式。所有边界最小值及可行的内部驻点都参与比较，平行轴的并列最近点由可行边界端点覆盖。半空间单独保留负半空间和半径触壁两种进入条件，支持任意非零法向的尺度。[特征构造与首触求解](../../src/phymeo/geometry/_round_sweep.py)

interpret 和 CUDA 在每个有效特征上实际运行 FP32 首触内核。HOST 使用原存储坐标构造系数，并用有理数谓词验证设备给出的时间、处理舍入分歧，发布可复现的时间区间。完整比较会覆盖设备未报命中的特征，因此擦边仍可检出。形状对分块处理，设备每页最多 512 条特征，查询间复用缓冲。HOST 也保留不依赖设备的独立执行入口。[tyx 内核](../../src/phymeo/backends/_tyx_collision_kernels.py) · [设备工作区](../../src/phymeo/backends/tyx_collision.py)

## 2026-11-02 本地验收

原 M0 目录里的 RR01 五项、RR02 六项均通过公开入口。[固定构型](p65-general-collision-cases.json) · [HOST 预期与回归](../../tests/collision/test_round_collision.py) · [设备回归](../../tests/collision/test_tyx_round_collision.py)

| 构型 | 检测结果 |
| --- | --- |
| RR01 球穿过平面 | 首触区间 [0.4999999999990905, 0.5]，末态间隙 −0.2 m。分离负控末态为 +0.1 m，显式排除返回 `filtered`。 |
| RR01 球–球与换轴留出 | 球–球首触区间 [0.7499999999990905, 0.75]。换法向与平面偏置后，区间 [0.6666666666660603, 0.6666666666669698] 包含 2/3。交换两侧后时间不变，法线反向。 |
| RR02 轴部、端帽、平行与重合 | 轴部间隙 +0.05、0、−0.05 m。端帽间隙 +0.1 m，轴参数 (1, 0)。平行留出的参数对从 (0.25, 0) 到 (0.75, 1)，间隙 −0.0125 m。重合轴返回 −0.2 m，法线不唯一。 |
| 双端分离的胶囊穿越 | 初末间隙均为 +0.75 m，首次接触为 [0.375, 0.375]，两轴参数均为 0.5。前 1/4 步分离，随后 3/4 步的局部首触区间包含 1/6。 |
| 端帽擦边与近失 | 轨迹在 t=0.5 擦过端帽，横向坐标从 1.25 m 改为下一个 FP64 可表示值则全段分离。两者在宽相均保留。 |
| 无限平面锚点 | 沿平面平移锚点 10¹⁰⁰ m 后，首触仍为 3/8，平面见证点仍在原点。如果实际法向间隙超过 FP64 可发布范围，则明确返回 `unresolved`。 |

三端对 11 个构型的配对状态、见证和时间区间一致。设备侧把 22 个形状放在 11 个世界，一次查询返回 231 对，220 对跨世界过滤，一对显式排除，10 对进入窄相。另用 19 根平移胶囊验证跨页处理。171 对均在步初相交，CUDA 对 1,368 条有效特征完成了 3 次窄相启动，共上传 30,720 B、下载 6,144 B 的窄相缓冲。容量不足时未启动窄相，原结果可重试。窄相注入故障也保留旧结果并关闭故障工作区。

HOST **38 passed / 0.54 s**，tyx interpret **10 passed / 0.88 s**，实际 CUDA **10 passed / 3.15 s**，三组原始退出码均为 0。每组包括受影响的 M1 回归，各自 15 项源码、输入及配置的运行前、运行后和交付字节一致。10 个 Python 文件通过 Ruff 与格式检查。环境为 Python 3.12.14、pytest 9.1.1、固定 tyx `99f6df7c0fd8fdeef77f97df83914da27b64b7ee`。

本机收据位于 `artifacts/p65-collision-m2-rr-local-20261102-01`。最终组为 `host-final-pytest-151114426`、`interpret-checked-pytest-151141434`、`cuda-checked-pytest-151151914`。新增配对测试分别保存 16、8、8 份观测，另保留 M1 混合场景的原始输出。初次缺首触与轴并列字段的两项失败、tyx 内核说明字符串的编译失败，以及远锚点被当作有限包围盒的失败均保留。最终实现使用内核注释，过滤后的半空间跳过有限盒判别。复现时使用已安装固定 tyx 的环境，从仓库根目录把 `src` 加入 `PYTHONPATH`，逐条执行并各自指定新的绝对 `--basetemp`。

```text
python -m pytest tests/collision/test_round_collision.py tests/collision/test_collision_pipeline.py -q -p no:cacheprovider
python -m pytest tests/collision/test_tyx_round_collision.py tests/collision/test_tyx_collision_pipeline.py -q -p no:cacheprovider --tyx-mode=interpret --require-tyx
python -m pytest tests/collision/test_tyx_round_collision.py tests/collision/test_tyx_collision_pipeline.py -q -p no:cacheprovider --tyx-mode=cuda --require-tyx
```

随后交付的 [RD01](p65-collision-m2-sphere-face.md) 已支持整张变形三角面的首触。DD01/DD02 的点–面、边–边及 RR03–RR05 的其余刚体配对仍待接入。[多案例实施计划](../superpowers/plans/2026-11-02-p65-tyx-collision-pipeline.md)
