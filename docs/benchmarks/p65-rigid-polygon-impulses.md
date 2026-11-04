# 有限刚体接触面的多点冲量

普通 HOST `ph.detect` 发布的静态凸包接触若恰有一个正面积多边形，可将全部裁剪角点交给 `ph.solve_witness_impulses` 或 `ph.step_floating_contacts`。顶点次序取自 `pair.convex.end.patches[0].points`。以下构造假定两端已用各自局部材料坐标在同一时刻登记点端口。

```python
patch = pair.convex.end.patches[0]
rows = [
    (pair, {pair.first: ("jaw", i), pair.second: ("box", i)}, i)
    for i in range(len(patch.points))
]
result = ph.solve_witness_impulses(
    ports, rows, [0.5] * len(rows), [0.35] * len(rows), local_vertices=registered_body_vertices
)
```

每个提交角必须在两端实际姿态上共点。每对凸包在一个共同原点坐标系内重新检测一次。新的完整多边形必须具有相同的两组支撑材料 ID 和相同顶点数；各实际端口点要和新多边形形成一一映射，法向与两侧原始材料权重重建的位置也要匹配。可乱序提交。遗漏、重复、拓扑或材料变化、同一形状切换到不同实体都会在发布前被拒绝。已脱离时按当前结果提交空接触行。

| 公开入口验收 | 实际物理量 |
| --- | --- |
| 1 kg 方块垂直以 1 m/s 落在有限支撑面上，四角方形与八角裁剪两种面 | 合冲量向上 1 N·s，合力矩冲量为零，动能减少 0.5 J。测试同时核对更新后的姿态和各刚体速度 |
| 带滑移的方块，初速度 (3,0,−1) m/s，摩擦系数 0.25 | 合冲量 (−0.25,0,1) N·s，末速度 (2.75,0,0) m/s，各行满足摩擦锥；同面改为开离速度后冲量为零 |
| 三个自由刚体，两侧有限指面同时撞击中间方块 | 八条材料行，每侧 1 N·s，三体各方向末线速绝对值低于 1.6×10⁻¹¹ m/s、末角速低于 4.6×10⁻¹¹ rad/s，动能变化 −1 J |

单角分力可有多解，验收使用总动量、角动量、能量及接触点末速。适用域是已定位的静态硬接触多边形。软压层、转动中首次到达的面、穿透恢复和圆盘面均有各自的几何及受力条件。

2026-11-04 以 11 项新增测试和 25 项相邻回归验收，36 项在 1.37 秒通过。最终格式和 lint 检查均正常退出。父目录 `artifacts/p65-20261104-hull-polygon/polygon-full-host/` 保存完整输出与双指八行原始冲量，旧行为的单项红测保存在 `artifacts/p65-20261104-panda-batch/rr-polygon-baseline/`。[独立公开入口测试](../../tests/test_floating_hull_polygon.py)
