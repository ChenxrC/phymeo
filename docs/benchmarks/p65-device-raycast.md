# 静态碰撞世界的 TYX 射线观测

`TyxRaycaster.query` 为每束射线在它指定的世界中返回最近表面的米制距离、原碰撞模型中的形状编号和世界法向。相机可以共享原点，扫描仪的每一束也可以拥有各自原点。每次显式传入当时的形状姿态。

```python
import numpy as np
import phymeo as ph
from phymeo.backends.tyx_raycast import TyxRaycaster

key = ph.CollisionKey(5, 7, 2)
rod = ph.CollisionShape(key, "capsule", (89, 7), radius=0.25)
model = ph.CollisionModel([rod])
poses = {key: ph.CollisionMotion([[-1, 0, 0], [1, 0, 0]])}
origins = [[0, 2, 0], [3, 0, 0], [1.4, 2, 0]]
directions = [[0, -4, 0], [-1, 0, 0], [0, -1, 0]]
with TyxRaycaster(mode="cuda") as rays:
    hit = rays.query(model, poses, origins, directions, world=5, far_m=4)
    np.testing.assert_equal(hit.distance_m, [1.75, 1.75, np.inf])
    assert tuple(hit.shape_index) == (0, 0, -1)
```

HOST 复用同一套已注册资产、世界筛选和当前姿态。无限平面和有限三角面各占一个面片，球占一个，非退化胶囊占两个外半球和中间的有限杆身。每个椭球、有限圆柱或有限圆锥各占一个解析形状条目。网格、凸包、四面体和规则高度场沿 HOST 的真实面拆分。设备上的每个工作项负责一束射线，测试所有面片和可见根，对闭合的 `[near_m, far_m]` 选最小距离，并一起写回原始形状索引和法向。距离相等时保留原稳定顺序。圆面的法向朝外，双侧平面的法向朝向来射线。未命中返回 `(+inf, -1, [0,0,0])`，索引对应的是全模型 `keys`，因此选中世界的局部编号可以重复。

计算为 FP32，以首束原点平移并应用二进制统一长度尺度，读回时恢复米。面片几何、归一化和几何输入校验在 HOST。有限特征尺寸小于本次计算尺度的四倍 FP32 机器精度会被明确拒绝，无法表示的输入、全批有歧义的运算和超过容量的查询返回错误。这里的深度和边界比较服从 FP32 的精度，特别贴近轮廓或两种表面深度接近时可能与 HOST 的 FP64 读数有差别。支持闭区间端点，若入口在近裁面之前则仍测试出口。默认容量最多 65,536 个面片、合计 67,108,864 次射线–面片测试，可通过 `max_ray_patch_tests` 减少每次上限。空世界或零束射线保留空观测且不发起内核。选中带圆角的有限板等尚未支持的形状会显式拒绝。此接口观测一个已给定的当前姿态。

## 2026-11-05 球、胶囊与平面本机实测

固定 HOST 接口基线 `1817a506`，设备实现及测试提交 `5d6c5e8`。冻结旧版本从单独的 `git archive` 导入，公共 `TyxRaycaster` 缺失产生预期的 `ModuleNotFoundError`。第一次新 CUDA 编译暴露了 TYX 对整数充当条件的拒绝，改为显式布尔判断。最终同一源码在 RTX 4060 Laptop GPU 与 TYX 解释器上各通过 8 项，原错误和两轮输出分开保存。

| 269 束斜向、不同起点的球面读数 | TYX CUDA | TYX interpret |
| --- | ---: | ---: |
| 命中 / 未命中 | 243 / 26 | 243 / 26 |
| 相对独立球面解析值的最大距离绝对误差（m） | 4.685 × 10⁻⁷ | 4.879 × 10⁻⁷ |
| 最大法向分量绝对误差 | 3.003 × 10⁻⁷ | 5.332 × 10⁻⁷ |
| 原生 CUDA 内核提交数 | 1 | 0 |
| 原生 H2D 次数 / 字节 | 5 / 11,900 | 0 / 0 |
| 原生 D2H 次数 / 字节 | 2 / 5,380 | 0 / 0 |

原生量由 `tyx.runtime_stats()` 在单次查询前后直接相减，并与算子统计逐值核对。共享调用适配器同时传入三份输入与两份初始化输出，再读回标签及距离/法向。其余专项覆盖了裁后出口、闭区间端点、共重球体的稳定形状编号、胶囊杆身与两端、两侧三角面、有限斜坡与闭网格面、远离世界原点的凸包、极大非单位平面法向、域外曲面、容量拒绝及设备失败后的重新查询。相机对接和 Newton 观测由独立集成验收记录。

复现命令在项目根目录运行。

```text
python -m pytest -q -s --require-tyx --tyx-mode cuda tests/collision/test_tyx_scene_raycast.py
python -m pytest -q -s --require-tyx --tyx-mode interpret tests/collision/test_tyx_scene_raycast.py
```

本机完整原始记录在父集成工作区 `artifacts/p65-device-raycast-20261105-01/`。固定旧源保存在 `01-red-frozen-source`，新源首个编译失败保存在 `02-cuda-primary`，最终物理读数及原生计数保存在 `07-final-cuda-native`、`08-final-interpret`。本次验收没有计时排名。

## 2026-11-05 有限解析曲面

同一设备查询现支持任意刚体姿态的三不等轴椭球、有限圆柱与有限圆锥。HOST 从登记资产读取实际半轴、半高度和材料到世界的转动。TYX 核逐束转入当前局部坐标，求完整二次或线性根，并在有限高度、真实端盖和距离裁剪区间内选最近表面。椭球用局部隐式梯度计算法向，有限曲面的法向朝外，圆锥尖返回零向量；侧壁与端盖同距时沿用侧壁优先次序。射线恰沿母线时仍需在裁剪后保留真实有限区间。几何尺度、精度和世界过滤约定与上文一致。

使用冻结的 Newton 1.7.0 CUDA SensorCamera 原始数据，复现 97×43 个不同起点的射线及三种独立转动。在 RTX 4060 Laptop GPU 上，全部 4,171 个形状编号与 Newton、Phy HOST 一致，其中 3,103 束未命中。未命中的 Phy 距离均为正无穷，法向均为零。下表的误差只在双方命中同一表面的束上计算。

| 形状 | 命中束数 | 对 Newton CUDA 最大距离误差（m） | 对 Newton CUDA 最大法向分量误差 | 对 Phy HOST 最大距离误差（m） | 对 Phy HOST 最大法向分量误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 三轴椭球 | 179 | 4.054 × 10⁻⁶ | 6.378 × 10⁻⁶ | 5.770 × 10⁻⁷ | 1.676 × 10⁻⁶ |
| 有限圆柱 | 491 | 1.050 × 10⁻⁵ | 2.064 × 10⁻⁵ | 8.694 × 10⁻⁷ | 9.699 × 10⁻⁷ |
| 有限圆锥 | 398 | 4.340 × 10⁻⁵ | 6.936 × 10⁻⁴ | 7.515 × 10⁻⁷ | 5.635 × 10⁻⁶ |

同一帧在 TYX 解释器也得到相同编号和命中数。解释器对冻结 Newton CUDA 最大距离误差为 4.340 × 10⁻⁵ m，最大法向分量误差为 6.911 × 10⁻⁴。两种模式对 Newton 和 HOST 的所有命中束均在原 10⁻⁴ m 距离、10⁻³ 法向分量门限内。真实 CUDA 帧由原生计数器读到 1 次内核提交、5 次上传共 183,728 字节、2 次读回共 83,420 字节。解释器的 CUDA 计数均为零。

独立的 6 项解析专项覆盖侧面、真实端盖、圆锥尖、内部出射、母线裁剪、混合形状遮挡、两次非交换转动及过小尺寸拒绝。每种执行模式还通过原有 8 项，原来的域外形状选为带圆角的有限板。源码为 `7e227ba`。全帧参考来自 `artifacts/p65-newton-night-20261105-01/curved-rays-gpu.json`，SHA-256 为 `bdc40eb0dffeb3b196421c540fb3d3a3e58ef0def59651fba587bf331af079c4`。独立重放只需该数据文件，保留原生输入设备与结果。该帧没有对应的 Newton CPU 原始记录。

```text
python -m pytest -q --require-tyx --tyx-mode cuda tests/collision/test_tyx_curved_raycast.py
python scripts/check_curved_device_rays.py --mode cuda --native curved-rays-gpu.json --output device-frame.json --cache cache-rays
```

完整形状编号、距离、法向和超限计数位于父集成工作区 `artifacts/p65-curved-device-rays-20261105-01` 的 `09-final-frame-cuda/all-rays.json` 与 `10-final-frame-interpret/all-rays.json`。旧源明确拒绝圆柱的原始结果在 `01-initial-red`。

## 每束射线选择世界

`phymeo.raycast_scene(..., world=...)` 和 `TyxRaycaster.query(..., world=...)` 接受同一标量或逐束整型世界编号。方向形状为 `(N,3)` 或 `(H,W,3)`，对应的逐束编号必须严格为 `(N,)` 或 `(H,W)`。距离与形状编号保持这个布局，法向在末尾再加 3 个分量。可以共享单个原点，也可传与全部方向逐项同形的原点。观察某个缺少形状的世界会直接得到未命中。

```python
import numpy as np
import phymeo as ph
from phymeo.backends.tyx_raycast import TyxRaycaster

shape = [ph.CollisionShape(ph.CollisionKey(w, 4, 0), "sphere", (3,), radius=r)
         for w, r in [(2, 0.5), (9, 1.0)]]
model = ph.CollisionModel(shape)
poses = {s.key: ph.CollisionMotion([[0, 0, -2]]) for s in shape}
worlds = np.array([[9, 2], [2, 90]])
directions = np.zeros((2, 2, 3))
directions[..., 2] = -1
with TyxRaycaster("cuda") as op:
    hit = op.query(model, poses, [0, 0, 0], directions, world=worlds)
np.testing.assert_equal(hit.distance_m, [[1.0, 1.5], [1.5, np.inf]])
np.testing.assert_equal(hit.shape_index, [[1, 0], [0, -1]])
```

HOST 按已注册世界对束做稳定分组，然后将结果写回原顺序。TYX 将选中世界的面片连接成一份表，每束只遍历自己世界的整数起止区间，在同一个内核完成求交与最近表面选择。尺度、原点基准和米制裁剪按各世界独立处理。相同本地物体及形状编号可出现在多个世界，返回的索引始终指向全模型的 `keys`。单次查询的容量按各世界“束数×本世界面片数”之和计算，空世界不消耗面片测试数。所有选中形状在发起设备调用前校验，未选择世界无需提供姿态。

### 同输出的 32 世界实测

RTX 4060 Laptop GPU 在 32 个几何重叠的世界分别放置球和平面，各世界使用相同本地物体编号。每个世界 25 束，合计交错排列 800 束。逐世界标量查询也采用同一个 `TyxRaycaster`。两条路径的距离、全模型形状编号和世界法向逐值相同，每个世界都实际命中了自己的球和平面。

| TYX 原生计数 | 一次多世界查询 | 逐世界调用 32 次 |
| --- | ---: | ---: |
| CUDA 内核提交 | 1 | 32 |
| H2D 次数 / 字节 | 5 / 52,096 | 160 / 39,296 |
| D2H 次数 / 字节 | 2 / 16,000 | 64 / 16,000 |

多世界路径额外上传 12,800 字节，用于每束的面片区间和按本世界尺度换算的裁剪界限。本次比较提交、传输与实际输出，未测墙钟用时。另一组 `2×3` 交错矩阵读得全模型编号 `[[2,1,-1],[3,1,2]]`，同一位置分别由不同大小、姿态的资产或空世界产生结果。设备路径与独立公式、HOST 和原标量路线一致。相距 2³⁵ m 的大坐标组与半径 2⁻³⁵ m 的小球也能在一次查询中按各自尺度恢复米制距离。

冻结产品基线 `5b6c854` 对原矩阵输入返回参数错误。最终源码 `c47e306` 在 CUDA 设置下通过本次 7 项、前两批设备专项及 HOST 射线近邻共 29 项。解释器另通过 7 项新增专项并确认没有 CUDA 提交。选中世界含圆角板、非法世界编号、错位输入形状和容量拒绝都在发起设备调用前被发现；在已执行内核后注入读回异常，重新排序后的下一次查询仍与独立结果一致。证据保存在 `artifacts/p65-batched-world-rays-20261105-01` 的 `05-final-cuda` 与 `06-final-interpret`。

```text
python -m pytest -q -s --require-tyx --tyx-mode cuda tests/collision/test_world_routed_raycast.py
```
