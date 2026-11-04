# 静态碰撞世界的 TYX 射线观测

`TyxRaycaster.query` 为一个指定世界的每束射线返回最近表面的米制距离、原碰撞模型中的形状编号和世界法向。相机可以共享原点，扫描仪的每一束也可以拥有各自原点。每次显式传入当时的形状姿态。

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

HOST 复用同一套已注册资产、世界筛选和当前姿态。无限平面和有限三角面各占一个面片，球占一个，非退化胶囊占两个外半球和中间的有限杆身。网格、凸包、四面体和规则高度场沿 HOST 的真实面拆分。设备上的每个工作项负责一束射线，测试所有面片和可见根，对闭合的 `[near_m, far_m]` 选最小距离，并一起写回原始形状索引和法向。距离相等时保留原稳定顺序。圆面的法向朝外，双侧平面的法向朝向来射线。未命中返回 `(+inf, -1, [0,0,0])`，索引对应的是全模型 `keys`，因此选中世界的局部编号可以重复。

计算为 FP32，以首束原点平移并应用二进制统一长度尺度，读回时恢复米。面片几何、归一化和几何输入校验在 HOST。有限特征尺寸小于本次计算尺度的四倍 FP32 机器精度会被明确拒绝，无法表示的输入、全批有歧义的运算和超过容量的查询返回错误。这里的深度和边界比较服从 FP32 的精度，特别贴近轮廓或两种表面深度接近时可能与 HOST 的 FP64 读数有差别。支持闭区间端点，若入口在近裁面之前则仍测试出口。默认容量最多 65,536 个面片、合计 67,108,864 次射线–面片测试，可通过 `max_ray_patch_tests` 减少每次上限。空世界或零束射线保留空观测且不发起内核。选中椭球、圆柱、圆锥等尚未实现的曲面会显式拒绝。此接口观测一个已给定的当前姿态。

## 2026-11-05 本机实测

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
