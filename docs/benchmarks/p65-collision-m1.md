# M1 · 通用碰撞检测入口

同一个模型可以同时登记刚体、布面和四面体，用 `phymeo.detect(model, motion, policy, workspace)` 查询全部形状对。一次调用返回每一对的稳定身份、筛选原因和几何结果。求解器可以独立消费这份结果。[公开模型](../../src/phymeo/constraints/collision_shapes.py) · [配对和见证](../../src/phymeo/geometry/collision_pipeline.py)

## 最小使用示例

```python
import phymeo as ph

ball = ph.CollisionKey(world=0, body=1, shape=0)
cloth = ph.CollisionKey(world=0, body=2, shape=0)
model = ph.CollisionModel((
    ph.CollisionShape(ball, "sphere", (7,), radius=0.25),
    ph.CollisionShape(cloth, "triangle", (12, 18, 31)),
))
motion = {
    ball: ph.CollisionMotion([[0.0, 0.0, 0.20]]),
    cloth: ph.CollisionMotion([[-2, -1, 0], [2, -1, 0], [0, 2, 0]]),
}
with ph.CollisionWorkspace(model, backend="cuda") as workspace:
    result = ph.detect(model, motion, ph.CollisionPolicy(), workspace)
    pair, = result.pairs
    assert pair.family == "RD" and pair.status == "overlap"
    gap = pair.witness.gap_m  # 约 -0.05 m
```

模型按世界、物体、形状 ID 排序。材料顶点 ID 与形状的几何版本独立保留。柔体顶点可以各自作仿射运动。同一世界和柔体内，重复出现的材料顶点 ID 必须提交相同起点和终点。查询会在上传前拒绝冲突。刚体可提供平移端点，或用 `RigidTurn` 指定移动转心、世界轴和有符号转角。转动时用原顶点到转心的 L1 距离包住整段圆弧，允许候选偏多。同体刚体形状互相过滤。同体柔性面即使共享顶点仍保留候选，后续窄相负责判断合法邻接。

| 输入或结果 | 本批行为 |
| --- | --- |
| 球、胶囊、有限凸包、三角面、四面体 | 计算全段 AABB。形状半径或表面偏移按几何长度计入，配对检测余量在两侧各加一半。 |
| 跨世界、同一刚体、双向掩码、显式排除 | 以固定优先级返回 `filtered` 和单独原因。包围盒严格分离返回 `separated/aabb`。 |
| 静态球/胶囊两两配对、球–零厚三角面 | 可返回间隙、两侧代表点与材料权重。唯一法线由第一侧指向第二侧。同轴线点重合时记录重叠与法线不唯一。 |
| 包围盒相交但尚无窄相 | 保留 `candidate=True, status="unsupported"`。运动候选在本批不返回首次接触时间。退化三角面在需要计算见证时返回 `unresolved`。 |
| 容量不足 | 返回 `capacity_exceeded` 和完整所需候选数，不发布截断的行。上次成功结果保留，可用更大容量重试。 |

`host` 使用 NumPy 穷举。tyx 的 `interpret` 和 `cuda` 都实际运行形状包围盒归约、形状对过滤与重叠筛选两个内核。原始存储坐标在 HOST 生成向外取整的 FP32 顶点区间。已接的静态窄相也在 HOST 计算，并在见证里标识算法来源。CUDA 工作区固定上传拓扑、掩码和配对表，查询间复用设备缓冲。首版上限是 256 个形状、8192 个登记顶点，按固定顺序穷举所有形状对。[内核](../../src/phymeo/backends/_tyx_collision_kernels.py) · [工作区](../../src/phymeo/backends/tyx_collision.py)

## 本地串行验收 · 2026-11-02

混合场景保留 M0 的 RR04 旋转胶囊、RD01 穿越大三角面的球和 DD02 两条运动边。附加的远物体、跨世界形状、双向掩码、显式排除和共享边覆盖了另外几种筛选结果。11 个形状产生 55 对，分属 RR 21、RD 28、DD 6。三组保存的原输入、55 行结果和 11 个包围盒分别一致。[混合场景与独立穷举](../../tests/collision/m1_cases.py)

| 结果 | 形状对 | 说明 |
| --- | ---: | --- |
| 筛选规则排除 | 21 | 跨世界 10、同刚体 1、掩码 9、显式排除 1 |
| AABB 严格分离 | 11 | 全段包围盒可排除这对 |
| 候选且静态重叠 | 4 | 已有静态窄相确认 |
| 候选且缺窄相 | 19 | 18 对需连续查询，1 对缺静态配对。RR04、RD01、DD02 的已知命中均在候选内。 |

验证使用 Python 3.12.14、NumPy 2.5.4、pytest 9.1.1 和固定 tyx `99f6df7c0fd8fdeef77f97df83914da27b64b7ee`。HOST 完整文件 14 passed / 0.42 s，tyx interpret 2 passed / 0.62 s，实际 CUDA 2 passed / 1.31 s。三个进程均 exit 0，各自 10 个源和配置文件的运行前、运行后及交付字节一致。八个 Python 文件通过 Ruff 和格式检查。[HOST 回归](../../tests/collision/test_collision_pipeline.py) · [设备回归](../../tests/collision/test_tyx_collision_pipeline.py)

设备测试依次查询原场景、零容量、移动一个物体后的场景和恢复后的原场景。两个后端各完成 8 次内核启动，复用 6 个缓冲。CUDA 四次查询共 3 次静态上传、4 次动态上传、8 次下载和 4 次同步，总上传 2848 B、下载 1936 B。容量请求 0 时返回所需的 23 对，旧结果保持可读。另有独立的注入故障，确认错误能传播且未覆盖旧结果。恢复原场景的单次整查询耗时为 interpret 6.0572 ms、CUDA 2.4810 ms，含 HOST 预处理、静态窄相及结果发布，仅记录这次运行。

两条回归保留了原失败。近乎平行的两根胶囊轴在 (1/4, 0, 0) 相交，原双精度最近点给出分离。现在对条件不足的轴查询，在 HOST 对闭线段参数的二次式比较四条边与内部驻点。该例返回间隙 −2×10⁻¹⁴ m，两侧轴参数分别为 5/8 和 1/2。第二条回归让两张同体布面对共享顶点提交矛盾的起点或终点，两者均在改动前意外发布，现已被拒绝且能用原输入重试。

本机原始收据位于 `artifacts/p65-collision-m1-local-20261102-01`。最终三组分别为 `host-checked-pytest-140946137`、`interpret-checked-pytest-140954160`、`cuda-checked-pytest-141004135`。对应 basetemp 下的 JSON 保留完整输入和每一对的返回。复现时从仓库根目录使用已安装固定 tyx 的环境，将本工作树的 `src` 加入 `PYTHONPATH`，逐条执行并分别指定新的绝对 `--basetemp`。

```text
python -m pytest tests/collision/test_collision_pipeline.py -q -p no:cacheprovider
python -m pytest tests/collision/test_tyx_collision_pipeline.py -q -p no:cacheprovider --tyx-mode=interpret --require-tyx
python -m pytest tests/collision/test_tyx_collision_pipeline.py -q -p no:cacheprovider --tyx-mode=cuda --require-tyx
```

M2 首批已补球、胶囊和固定法向移动半空间的平移首触，扩展后的支持范围见[M2 记录](p65-collision-m2-round.md)。SAP、BVH 与动力学消费者按后续节点验收。[分族计划](../superpowers/plans/2026-11-02-p65-tyx-collision-pipeline.md)
