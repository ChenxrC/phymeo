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
