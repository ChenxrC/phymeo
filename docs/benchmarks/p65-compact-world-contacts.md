# 单页世界的紧凑形状对接触缓冲

提交 79bd045 增加可选的 CompactWorldContactWorkspace。它给求解器返回每世界的阳性接触、诊断与所需容量，记录沿用原 CollisionPair。一个形状对可以携带多个材料点、曲面支撑或首次时间区间。运动接触的终点见证和首次接触见证分别保存在原字段。统计单位是形状对。每个四面体对的 solid 轨迹随阳性记录保留，整身四面体占据汇总和完整包围盒在原诊断入口中。[实现](../../src/phymeo/geometry/compact_contacts.py) · [专项测试](../../tests/scale/test_compact_world_contacts.py)

```python
from phymeo.geometry.compact_contacts import CompactWorldContactWorkspace, ContactBufferLimits

with CompactWorldContactWorkspace(world_models, "cuda") as workspace:
    by_world = workspace.detect(
        motion_by_world, policy_by_world,
        limits_by_world={37: ContactBufferLimits(contacts=512, diagnostics=32)},
    )
    result = by_world[37]
```

## 接触、诊断与发布

hit、touching、overlap 进入 contacts，保留原顺序、两侧形状键、法向、材料地址及全部原迹线。候选中的正间隙分离和仅共享材料点的相邻形状不进入阳性接触。invalid_geometry、unsupported、unresolved 与未来未知状态进入 diagnostics。已确认重叠但明确缺少可用支撑深度或唯一法向时，阳性记录和对应诊断可以同时出现。原未触发碰撞的 1/16 m 正间隙留出有一个候选、零接触。

原 CollisionPolicy 的候选预算先执行。候选超限时有实际 required_candidates，两个尚未求出的输出需求为 None。窄相完成后再分别校核接触和诊断容量，任一超过时返回两项真实需求和具体 overflow 类型，两个输出缓冲均为空。该世界的上次接受记录保留，邻居仍能推进查询。整体设备故障当次全部撤回，重试会重建设备工作区。

CUDA 继续执行原包络归约和宽相分类。HOST 读取完整边界和分类标记，在建立形状对对象前选择原全局槽，再运行既有窄相。掩码、排除和普通球等已证明分离的槽可直接略过。闭网格、凸包、板、四面体及新形状的分离标记仍送原验证，以保留可发生在 AABB 分离处的几何错误。没有改变设备配对表的大小。

真实试验中，自交网格即使宽相候选数为零，仍给一条 mesh_self_intersection 诊断。退化凸体同样给 convex_not_full_dimensional，两者各自启用掩码或显式排除后才按原规则过滤。真实未支持的转动圆柱扫掠保留为 curved_sweep。另核对刚–柔球穿三角面内部的 −1/16 m 间隙、三节点权重，以及两个异体三角面的边中段在 t=1/4 首触时的双方材料点。专项通过 10 项 HOST、7 项真实 CUDA，旧分区与混合查询相邻组通过 13 项 HOST、6 项 CUDA。

## 同输出成本

对照在完整 detect 返回后、同一计时窗口内调用 compact_contacts 投影。新入口直接返回相同字段。两条路径逐配置比较完整输出 SHA-256、候选/接触/诊断数，再从实际球心、半径和两侧见证独立核验。每个方法各自新建 Python 进程，先预热 3 窗、再测 7 窗，每窗连续查询 4 次。CUDA 的三个配对轮次为完整/紧凑、紧凑/完整、完整/紧凑。HOST 一对为完整/紧凑。下表为各窗口中位数折合的 ms/次，所有场景均为静态查询。[几何留出](../../tests/scale/compact_contact_cases.py) · [测量入口](../../scripts/benchmark_compact_contacts.py)

| 场景 | 进入原窄相 / 可省槽 | HOST 完整 / 紧凑 | CUDA 第 1 对 | CUDA 第 2 对 | CUDA 第 3 对 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 稀疏 65 球 | 0 / 2,080 | 20.210 / 15.760 | 9.360 / 2.485 | 10.513 / 2.351 | 10.354 / 2.125 |
| 稀疏留出，只有球 0 与 64 命中 | 1 / 2,079 | 19.814 / 12.604 | 11.072 / 2.478 | 10.562 / 2.537 | 11.089 / 2.772 |
| 密集 34 球 | 561 / 0 | 85.958 / 83.506 | 83.249 / 90.326 | 87.902 / 91.751 | 85.748 / 88.010 |
| 密集留出，按局部 ID 奇偶形成双簇 | 272 / 289 | 55.444 / 43.039 | 44.260 / 41.212 | 43.529 / 56.318 | 45.821 / 44.228 |

65 个分离球实测没有构造 CollisionPair。单命中留出只选择真实的全局槽。第 2 轮稀疏配置在中位窗内，完整路径的结果阶段折合 7.793 ms/次，紧凑路径为 0.028 ms/次。该窗的两种路径都执行 8 次真实 CUDA 核，H2D 为 6,240 B，D2H 为 39,520 B。数据改善发生在 HOST 结果构造。密集全接触配置没有可省槽，三轮 CUDA 紧凑耗时均较长。密集双簇三次的改善方向不一致。

本机证据在 `artifacts/p65-night-compact-world-contacts-20261104-01/`。八个轮次进程各自的 report.json 保留逐窗原始时长、窗长 p50/p95、阶段时钟和设备传输，comparison.json 核对跨进程和跨后端的完整输出。远距坏几何与容量收据分别见 `compact-cuda/pytest-temp/` 下的 compact-invalid.json 和 compact-overflow.json。本轮不含 Newton 求解器物理步进或引擎间吞吐对照。
