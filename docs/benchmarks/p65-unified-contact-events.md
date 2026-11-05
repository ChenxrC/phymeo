# B2：普通与机器人场景的统一接触事件

本批提供同一种 `scene.contact_events` 只读快照。它记录已接受物理步的**正法向载荷 episode**，不增加静摩擦状态、恢复系数或新的接触求解器，也不把所有几何相交都叫作受载事件。B1 的 `ContactHistory`、普通场景完整结果以及两套原材料律保留。

## 源码和验证归属

作者树基线 `69bd12d7abd322a894c45b8a15a5d9d3bb6187c8`；父负责两场景薄接缝，作者树接缝提交 `76225f9`。本批只新增：

- `src/phymeo/runtime/contact_events.py`
- `tests/test_contact_events.py`
- `tests/test_robot_contact_events.py`
- 本报告

父已运行两个公开入口 RED：真实普通 4 N 和机器人两条 2 N 链的物理前置通过，随后均因缺少 `.contact_events` 失败；`2 failed / 1.17 s / RAW1`。随后两个公开 GREEN 为 `2 passed / 1.38 s / RAW0`，普通 4 N 与机器人双 2 N 链、事件公开视图均实际通过。其余专项由父独占串行执行，作者不运行项目命令；结果待实测后补入。

## 统一结果与保真的范围

`SceneContactEvents` 记录 `step_index / sample_time_s / commit_time_s / active / events / generations`。时间含义是入步载荷采样时刻和接受末态时刻；不把载荷标成末态力。观察、几何读回不推进 episode。

每条 `SceneContactRecord` 使用 B1 的 `(first CollisionKey, second CollisionKey, generation)` 身份；其中两侧 key 保留 world 身份。`first_owner / second_owner` 另记实际物理归属。机器人 collision body ID 经编译重映射后仍不能替代登记的 robot body ID。移动连杆材料坐标属于真实计算模型 link frame；折叠的固定子链不冒称其原 URDF authored frame。刚体坐标属于入步 COM frame；普通可动三角片保留原粒子材料权重。

`components` 按原采样顺序保留所有 `PairLoad`，每个 component 内保留全部 `LoadSite`、双侧真实材料点和权重、两侧施力点、法向、间隙、法向力、牵引、功率和势能。一个 pair 上的非共线采样不会合成平均法向。没有跨步稳定 component/site ID；只比较实际采样材料值及离散非零支撑集是否改变。

每个 component 保留原不可变 `law` 和明确 `material_kind`：

| material_kind | 原始材料 | 间隙与力含义 |
| --- | --- | --- |
| `normal_layer` | `NormalLayer` | 有限开距层内压缩；保留宽度、N/m 刚度、s/m 速率因子与原滑移正则化 |
| `robot_contact_material` | `RobotContactMaterial` | 原负穿深、N/m 刚度、Ns/m 阻尼与原滑移正则化 |

两类不做单位或数值转换，不声称同输入必然同轨迹。只要一条 component 的 `force_n > 0`，该 pair 受载；该 pair 当前所有 component（包括同时存在的零力 component）完整保留。全部为零时结束原因是 `zero_force`；采样器未返回该 pair 时是 `absent`。结束不夹带旧载荷。generation tombstone 每个曾受载 pair 只有一项，不积累完整历史。

## 父维护的最小共享接缝

普通场景在锁内直接适配已提交 B1 快照，不在 `step()` 增加一轮采样、推进或结果发布：

```python
return contact_events_from_history(
    self._contact_history,
    layers=self._system.layers,
    bindings=self._system.bindings,
)
```

机器人构造器使用当前真实变量名：

```python
self._event_adapter = RobotContactEventAdapter(
    robot_addr, self._tree, self._items, self.rigid_rows, pairs, self._robot_keys
)
self._contact_events = SceneContactEvents()
```

在 `public = _owned(reading)` 成功后，最终 state/observation/geometry 单次赋值之前纯 staging：

```python
staged_events = self._event_adapter.stage(
    self._contact_events,
    measured.loads,
    geo=geo,
    rigid=body,
    sample_time_s=self._time_s,
    commit_time_s=end_time,
)
```

`geo / body` 是同次入步真实几何与刚体状态；stage 不改变 previous、不推进求解器、不持有可变历史。所有分配和材料验证成功后才与物理末态一起发布。异常由场景保留旧状态、旧观察、旧时间、旧序号和旧事件，恢复后可用相同输入重试。新模块不 import `robot_scene`，避免两入口循环依赖。

## 验收域与限制

冻结测试包括普通/机器人真实多体载荷、有限箱台抬离后回返、同 pair 多站点的非共线法向与纯扭转摩擦、材料/物理 owner、world 隔离、不可变旧快照、失败后同输入重试。普通设备测试沿已有 TYX 路径；RobotScene 仍沿原 HOST 物理路线。本批不会把机器人 HOST 接口标成 CUDA，也不增加连续碰撞、机器人自碰或布–机器人共同求解能力。

## 父实际验收

完整HOST范围分组完成：首两public 2 passed/1.38s；其余首次22 passed/1 failed/1.44s，失败仅RobotScene的external关键字参数在测试中误作位置参数，c3b092e只修调用后该节点1 passed/.88s。因此25个HOST实例均已通过，不重复已绿节点。普通场景interpret 1 passed/1.14s、实际CUDA 1 passed/2.06s，均保真实有限台退出/重入和原law，RobotScene仍HOST。

包括非共线pad法向、两种旋转与刚变换、原摩擦合力和非零力矩、12站点普通接触、folded/root owner及两种实际stage异常后同输入重试。原作者之外的既有agent只读检查了组件/owner/原子发布，无阻断发现。

首Ruff仅一个测试import区排版；首format-check四文件需排版。父保存原5个Python文件后修style，5份raw AST逐一严格相同；最终Ruff/format-check全部RAW0，不重跑语义未改的已绿物理。证据在父 artifacts/p65-postnight-batch2-20261105-01/b2-*。整组没有增改物理力律或积分器。