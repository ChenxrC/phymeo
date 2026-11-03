# 有限三角高度图与刚球

`Heightfield` 将每个样本放在局部坐标 `(col*dx, row*dy, heights_yx[row][col])`，再用 `world = rotation @ local + origin` 变换。两个轴的间距可不同，行数和列数也可不同。每个单元按左下到右上的对角线划成两个实三角形，0 号三角形沿左下、右下、右上排列，1 号沿左下、右上、左上排列。行列从 0 开始。有限边缘和四个外角也是地形几何的一部分。

```python
import phymeo as ph

grid = ph.Heightfield([[0, 1, 0]] * 3, (1, 1), origin=(-1, -1, 0))
land = ph.CollisionKey(0, 1, 0)
ball = ph.CollisionKey(0, 2, 0)
model = ph.CollisionModel([
    ph.CollisionShape(land, "heightfield", tuple(range(grid.vertex_count)), heightfield=grid),
    ph.CollisionShape(ball, "sphere", (0,), radius=5/16),
])
paths = {
    land: ph.CollisionMotion(grid.world_vertices()),
    ball: ph.CollisionMotion([[0, 1/8, 5/4]]),
}
with ph.CollisionWorkspace(model, "host") as workspace:
    (pair,) = ph.detect(model, paths, ph.CollisionPolicy(), workspace).pairs
```

球心到整张有限曲面的最短距离减去球半径得到带符号间隙。上例的屋脊最近点为 `(0, 1/8, 1)` m，间隙为 −1/16 m，球面点为 `(0, 1/8, 15/16)` m。曲面从两侧起作用，球在面下远离时分离；球在外缘以外时按真实边或顶点测距。平面下方没有无限延伸的实体或竖直裙边。球心恰在曲面上时，读取归属三角形的正向法向。换序只改变见证的第一、第二侧及法向方向。

`pair.heightfield.cell`、`triangle`、`vertex_ids` 与 `barycentric` 对应同一原始材料面。按行优先的材料 ID 取三个世界样本并加权，可重建地形支持点。球拥有自己的中心材料 ID，结合半径及法向即可重建另一侧支持点。多个最近三角面同距时，按行、列、面号取最小者。宽相分离的配对可直接返回 `separated` 和 `aabb`，需要读正间隙时可用 `CollisionPolicy.margin` 扩大候选范围。

每轴至少有两个样本，样本总数受公共模型的 `MAX_VERTICES` 限制。构造器拒绝无效姿态或在世界坐标中塌陷的三角形。查询的初始顶点必须等于登记后的世界网格。窄相在 HOST 上用实际单元的世界包围盒筛选，再复用球–三角形几何核核对候选。首次相触和内核运行设备分别验收；本阶段覆盖离散姿态，进入宽相的扫描路径明确发布 `unsupported / heightfield_swept_pair`。

[冻结的 RR 输入](../../tests/reference_newton/rr/cases.json) · [两种次序、非方形旋转留出及有限边界](../../tests/collision/test_heightfield_sphere.py)
