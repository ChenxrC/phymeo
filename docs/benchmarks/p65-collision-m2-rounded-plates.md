# RD03 · 有限圆边板与柔性表面

`CollisionShape(key, "plate", vertex_ids, radius=r)` 表示按边界次序给出的凸平面多边形与半径为 r 的球的 Minkowski 和，正反两面、圆边、圆角属于同一实体。多边形须有 3–32 个不同的极点和非零半径。布侧可登记零厚三角面或零半径的材料 `particle`。板可以平移，布的各顶点可以独立做仿射运动。板的非零转角、带额外偏移的布面分别返回明确的未支持结果。需要测量分离距离时可用 `CollisionPolicy(margin=...)` 保留宽相候选，半径及窄相判定照原几何计算。[注册](../../src/phymeo/constraints/collision_shapes.py) · [查询与材料区域](../../src/phymeo/geometry/_plate_contact.py)

`pair.plate` 保存初态、末态和首触区间取样时刻的姿态。法向唯一时，通过投影并裁剪两侧完整的支撑特征，返回最近点、线段或带面积的多边形。区域中的每一点保留双方表面点和材料权重。若布穿过板的中面，间隙为负半径，该处法向不唯一。连续查询复用板顶点对布面、板边对布边和布点对板面等多项式特征，tyx 在批内计算候选，HOST 在原始运动上隔离首触。

| 构型 | 实测 |
| --- | --- |
| 点分别靠近板正面、背面、侧边和圆角 | 均相距 0.05 m，方向分别为 +z、−z、+x 和平面对角线 |
| 大三角面覆盖板，布顶点都在板外 | 首触区间包住 2/3，区域有四个材料点，面积 1 m² |
| 绕侧边到背面，同高程路径穿过板中心 | 前者全程分离，后者首触区间包住 1/6 |
| 角部留出点 | 间隙 −0.01339745962155614 m |
| 两条边的内部首先接触 | 首触为 3/4，布边权重各 1/2；另一个平行构型返回完整 1 m 接触线段 |

HOST 集成回归 242 项通过，tyx interpret 与实际 CUDA 各 31 项通过。两种 tyx 后端的同一批十对几何结果均与 HOST 一致，各有 4 次表面窄相核启动，覆盖 146 组表面特征，其中 11 组为胶囊轴身–边特征。完整记录见 `artifacts/p65-collision-m2-rd03-local-20261102-01`，三组分别为 `08-host-pytest-201012435`、`09-interpret-pytest-201044499` 和 `10-cuda-pytest-201121971`。后两组的 basetemp 内有原始 `rd03.json`。[HOST 回归](../../tests/collision/test_rounded_plate_contact.py) · [设备回归](../../tests/collision/test_tyx_rounded_plate_contact.py)
