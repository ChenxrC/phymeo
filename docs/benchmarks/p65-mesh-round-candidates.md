# 闭网格与球/有限胶囊的逐面候选

本片在已有 `CollisionWorkspace(..., mesh_midphase="bvh")` 开关下，为**静态或双方刚性平移**的闭网格–球/有限胶囊接入真实逐面 BVH。默认 `all_facets`、旧精确特征算法、公开材料坐标和闭体语义不变。双转轴及其他非零转动仍走已验的完整枚举。本片没有增加响应求解，也不声称 GPU 精确窄相或性能收益。

作者只进行了文本、文件和 Git 操作，未执行项目。父唯一串行队列的已测与待测范围分列如下。

## 接口与保守依据

新增内部模块 `src/phymeo/geometry/_mesh_round_candidates.py`：

```python
selector = MeshRoundFaceSelector(shape, prepared_mesh, query_boxes)
indices = selector.sample(start, end, radius, vertex_distance2)
indices = selector.translation(start, end, drift, radius)
```

`start/end` 是当前有限轴段的两端点，球为同一点；坐标在当前 prepared mesh 的初始世界坐标系内，已扣除网格平移。`drift` 是圆体相对网格的整个单位时间平移。`sample` 的距离表为完整 `{原材料ID: 精确平方距离}`，父 `MeshRoundQuery.sample` 原本就计算该表。

静态取完整顶点表的最小值 U²，另用对应真实顶点与有限轴段重算该一项，避免内部调用误传一个过小上界。该顶点属于有限面，所以完整边界最近距不大于 U。使用既有 `sqrt_bounds(U²,80)` 的有理上端，令 padding=max(radius,U_upper)。轴段 AABB 各方向扩 padding，因欧氏近距蕴含每坐标近距，包含全部最近面并列及半径内边界面。这里不把轴的无限延长线当有限胶囊。

平移取两轴端在初末时刻的四点 AABB，再扩真实半径。这是整条仿射路径的保守候选，不以有限时间采样判定净空。父仍执行原有精确 AABB 下界、有限面/边/点特征和时间根检查；候选不是实际接触。

callback 合同固定为：

```python
query_boxes(binding, leaves_fp32, queries_fp32) -> list[list[int]]
# leaves: (F, 6), queries: (1, 6)
```

`MeshRoundFaceBinding(key, revision, vertex_ids, faces)` 为冻结值，faces 保留原有向面顺序。构造时核对原登记、prepared points、各面坐标和边界；查询盒与叶盒从当前真实坐标向外舍入为 FP32，并保守处理设备 flush-to-zero。超出有限 FP32 存储明确拒绝。传入 callback 的数组拥有不可写 bytes，不与原始几何或后续查询缓冲共享写别名。

回读只能为一行合法原面索引，排序去重后返回 tuple。重复索引最多接受 2F 个，超量、非法类型/索引不截断、不补全。面数保持原每网格 128 的上限。父树 key 为 `("mesh", binding)`；workspace 本身绑定不可变 model。当前坐标不进入长期 layout key，每次 refit 新盒，同 key 姿态更新不会积累逐姿态树或复用旧占用。

## 完整闭体与失败边界

`ClosedMesh` 完整拓扑验证、inside parity、全顶点最远距，以及圆体包住网格时的全部最远顶点支持记录始终使用完整网格。selector 只替换原 sample 最近特征队列和 `_first` 的逐面入口，不构造“候选闭体”。

- 正常 sample 空集不可能满足 U 的证明，返回具名 `SolverFailure`，公开为 `unresolved`；不能当 separated 或退回全脸掩盖遗漏。
- translation 空集只排除边界事件。只有原完整初态已判 separated 的查询才消费该结论；初始包含不能据此变成全程净空。
- 真实 callback/BVH 回读或设备失败关闭工作区并向外抛错，旧发布和 layout 不变；恢复方式是新建 workspace 后重试原输入。正常几何未决与设备损坏不混同。
- pair 容量不足保持既有整次不发布，原 workspace 可在容量恢复后重试；外层 transaction 取消也不发布暂存几何。

## 冻结用例与独立依据

`tests/collision/mesh_round_candidate_cases.py` 复用旧测试专用的精确有限段–三角特征 oracle，独立逐脸枚举，不调用生产最近特征或 BVH。主盒为 [-1,1]³；面球中心 (35/32,0,0)、r=1/8，间隙 -1/32。边外偏移 (3/32,4/32) 构成精确 3–4–5 三角形；另有有限杆身、端帽、轴外近失、轴穿过闭体、内含、球包住全网格和凹 U 槽/壁。

平移球和有限轴均从 x=-2 相对前进4m，r=1/8，独立首次时间为 `(2-1-1/8)/4=7/32`；两体另加相同非零平移，初末分离。净空留出 y=73/64，球最下方全程高于盒顶 1/64m。108 面细分盒保留原边界，逐脸 oracle 验证设备候选包含所有最近并列/实际半径内面，并要求真实剪枝。

还覆盖反序、原材料 ID 重排、刚体协变和镜像、FP32 边界内外、同 key 实际坐标更新、版本/模型绑定、过滤码和无候选时仍作完整拓扑验证。全部支持与边界见证以同一采样时刻的原材料点、权重、圆体 offset 重建；完整 `(status, reason, witness, sweep, mesh_round)` 保持原值。

设备专项把6个共址世界放入一个真实公开查询，各世界重复局部 ID，包含108面胶囊、内含、轴穿面、途中首触、凹槽净空及空扫掠候选。比较同设备 all_facets/bvh 的完整 `asdict`，另对 HOST 全 pairs/bounds；禁止 `BoundsTree.host_hits`，记录实际 BVH 调用、原 face row、completed launches、传输与同步。每次面 BVH 为真实 refit/traverse 两个 kernel，精确几何和时间根继续 HOST。新增3种设备异常分别为运行错误、回读形状错误和越界 facet bit，要求整次失败保前态并可由全新 workspace 恢复。

## 父串行收据

证据根：`artifacts/p65-top-three-batch7-20261106-01/`。

| 收据 | 实际结果 | 范围 |
|---|---|---|
| `03-b-valid-red` | RAW1 | 独立 gap、材料和原完整输出先通过，旧公共 bvh 未调用圆体逐面候选 |
| `04-b-host-first` | 44 passed，21.86s，RAW0 | helper/协议、静态/平移/完整输出、108面、包含/并列、交换、资源与事务 |
| `05-b-readback-red` | RAW1，DID NOT RAISE | 新延迟 callback 的 SolverFailure 被原 `_pair` 转成 unresolved，缺整次原子失败 |
| `06-b-readback-green` | 1 passed，0.69s，RAW0 | 父 closure 关闭工作区后转换 RuntimeError；旧 result/layout 保持 |

首个有效 RED 提交 `0060333`；helper/HOST 检查点 `26e71db`；坏回读独立回归 `bf01e16`。首错误全部保留，没有改变输入或放宽几何容差。已绿 HOST 源码保持冻结，父未重跑同源完整组。

本报告提交时，新增设备4节点尚待父执行，不能据代码宣称 interpret/CUDA 已通过。父建议只选新设备标记，以免重复已绿 HOST：

```text
pytest tests/collision/test_tyx_mesh_round_candidates.py -m tyx --require-tyx --tyx-mode=interpret
pytest tests/collision/test_tyx_mesh_round_candidates.py -m tyx --require-tyx --tyx-mode=cuda
```

父还负责窄范围共享接缝兼容、格式与静态收据。没有新增性能计时或旧矩阵重跑；本批交付目标是真设备逐面候选可用且完整结果不变。
