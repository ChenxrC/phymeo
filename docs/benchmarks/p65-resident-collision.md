# 接受设备状态直接供球宽相（C1）

状态：AUTHORING / UNEXECUTED。作者只编写文本、源码和测试；本批项目执行由父统一串行。基线为 `2684ba9c6664ffb84125a902ac1be93ac5320595`。尚无性能或设备通过声明。

C1 每个已注册 world 最多绑定一个真实 FreeSphere，查它与同 world 显式注册的有限静态几何。输入是当前已接受姿态，不是 CCD，也不做接触响应。C2 的滑轨映射、转动刚体、机器人、变形面和候选积分状态不在首片。

公开入口为 `batch.prepare_collision(model, bindings, *, candidate_capacity=4096)`，以及 `plan.query()` / `batch.collision(plan)`。绑定为 `ResidentSphereBinding(key, world)` 和 `ResidentStaticBinding(key, motion)`；后者只接受静止的有限 CollisionMotion。所有模型 key 必须恰好覆盖，球半径必须与 resident 的真实属性一致。候选仅覆盖显式返回的 `registered_pairs`，不将静态–静态配对隐藏成已检查。

结果保存 owner、topology_generation、单调 state_epoch、world_keys、ticks、shape_keys、shape_revisions、bounds、registered_pairs、candidates。公开数组有独立且不能重新开启写入的存储。`plan.require_current(result)` 检查所属 owner、拓扑和接受版本；tick 可由 restore/reset 回退，不能代替 epoch。失败不得替换 last_result；资源和运行时生命周期仍由原 batch 控制。

父维护的准确旧接缝：

- `TyxWorldBatch._borrow_collision_state(*, owner=None, topology_generation=None)` 在同一 RLock 和原 `_device()` 内 yield NEW `AcceptedCollisionState`。字段依次为 `owner, topology_generation, state_epoch, active, world_keys, bindings, slots, states, ticks, props, runtime, mode, tx`。states/ticks 仅为当前接受的一个缓冲；空 batch 的设备字段为 None。借用不得逃逸。
- `TyxWorldBatch` 挂 `prepare_collision` / `collision`，初始化并关闭 `_collision_plans`；真实 step 发布或 restore/reset 成功后推进 epoch，设备失败不推进。对单世界数值拒步允许保守地失效整批旧引用，但该世界状态/tick仍原样。
- `TyxCollisionBatch(..., runtime=None)` 在显式传入 runtime 时只借用，不关闭它；`query_device_bounds(*, trees, pending)` 消费已写好的 `_bounds`，不上传 envelope、不调用 reduce_bounds。
- `TyxCollisionBatch.prepare_shape_bvh(tree)` 使用原 `('shape',)` 缓存键，预分配 `TreeBuffers(self, tree, self._count, True)`，只准备静态布局。NEW plan用已知初始球位置和静态包围盒建立 BoundsTree；真实查询仍从当前设备 bounds refit，不借 HOST 初始位置代替运动。

不承诺零读回。静态注册和索引上传、原 BVH 参数上传、状态戳/完整公共 bounds/候选读回均须分别计数。暖态不得 snapshot/kinematics 下载完整动态状态、不得重新上传动态 envelope、不得新增设备分配。interpret 是真实 TYX 解释路径，CUDA 必须实际使用同源设备核。

固定测试库依计划涵盖真实推进后的分离→恰触、W1/W4/W32 与同 ID 异 world、FP32 外向恰触与近失、mask/exclusion、restore同tick不同位置与reset、数值拒步/设备失败、两个active buffer、空/close/容量和输入绑定错误。每项保持独立几何 oracle；保守候选允许 false positive，但不能遗漏独立HOST oracle的可能碰撞对。完整宽相输出不得冒作原窄相材料点或接触力结果。

## 冻结实现与待验证项

首有效 RED 为父作业 `05-c-valid-red`：绑定 tests-only `fc19dfe`，真实 interpret 球推进 0.25 s 后，在 `batch.prepare_collision` 缺失处失败，1 failed / 0.93 s / RAW 1。此前作业04使用旧虚拟环境，缺 tyx 的 fixture 错误保留，不能当有效功能 RED。作者没有执行这些作业。

完整用例检查点 `d25ae60` 先于两个产品模块，保留原 C01 输入。当前源码和其余用例均 UNEXECUTED，待父统一真实 interpret/CUDA 与静态检查。不得把下述实现说明作为通过证明。

设备核取当前 state 的 xyz 和真实 props 第7列半径，对每个 RN 的 c-r/c+r 端点分别增加外扩量 `max(FP32 normal_min, 2*FP32_epsilon*abs(endpoint))`。显式 add_rn/sub_rn/mul_rn 保留舍入屏障；正常范围内保守包围原 FP32 球，零/FTZ附近也外扩。它可能比既有 Fraction→最小外向 FP32 盒更宽，测试比较包含关系及已分辨的完整候选，而不谎称盒逐bit相同。任何溢出/非有限结果在 BVH 查询及结果发布前拒绝。极接近 FP32 最大值的合法中心也可能因无法表达有限保守盒而拒绝。

查询先读 compact shape status/tick 表，再用原 shape BVH 从同一设备 bounds refit 和筛选；不读 xyz/v 的完整状态。每个计划、每个 accepted-buffer 方向各有一个核绑定句柄。静态表和布局准备时分配，暖态继续上传原 BVH 参数元数据并诚实计数。`plan.stats()` 记录本计划碰撞资源/传输和所有 HOST copy，原 batch.stats 仍记录原世界状态与提交/同步；它们不是同一全局聚合计数。宽相只描述当前接受姿态，对运动段的穿越和实际施力没有新保证。

容量为显式0至32640的整数，超过容量抛 ValueError，不发布部分候选。完整输入校验在碰撞设备分配前完成；静态几何复用既有有限几何包络，shape revision随结果返回。模型固定后不支持原地重新绑定；需要关闭旧计划并创建新计划。restore/reset改接受epoch；它们不是拓扑变更。


## 父集成与有限验收（2026-11-07）

状态更新：FOCUSED_INTERPRET_CUDA_VERIFIED。作者提交295e448的四个新文件由父正常合并，父独占挂接 tyx_world_batch.py / tyx_collision.py，保留原事务与运行时所有权。首轮完整解释35项和CUDA35项分别在23/24通过，均RAW_EXIT=0。

原作者交叉审阅发现同batch/epoch/key的不同静态注册可混用结果；27-c-registration-red以近墙/远墙同身份实际复现，1 failed/RAW1。父给每个计划独立registration对象，结果保存它，require_current先检查同计划身份，再检查batch/topology/epoch。C19负控纳入最终完整集。当前registration也属于公开版本绑定，不能只用tick或shape revision判断结果有效。

最终完整 tests/scale/test_resident_collision.py：33-c-reviewed-interpret为36 passed/1.57s；34-c-reviewed-cuda为36 passed/8.32s，均RAW_EXIT=0，没有skip或HOST代替CUDA。包含W1/W4/W32、暖态两buffer、当前真实位置、FP32恰触/近失、过滤、同tick恢复/重置、状态/查询失败、容量、关闭异常、静态mesh和旧envelope路线同输入比较。完整静态检查31/32通过（9文件包括A，原首lint/format记录保留）。

该节点验收不涵盖CCD、窄相、受力、C2滑轨或通用机器人设备积分。每计划真实碰撞资源与原世界资源分别计数；零动态envelope上传不等于零传输。没有主张性能快于Newton。阶段组合回归、独立审阅和源码外wheel仍待A/K3与其它节点合并后执行。原记录均位于 artifacts/p65-generic-collision-20261107-01。
