# B4：从真实场景载荷登记材料站点

本批为普通 ContactScene 与 RobotScene 的真实法向采样提供显式材料站点适配。输入是同一入步时刻的 `SceneContactEvents` 和实际 COM/link/root 坐标系，输出可直接交给 B3 `TangentHistory`。适配器不求摩擦、不推进场景、不创建第二套 episode，也不按最近点或输出行序自动继承弹簧。

## 来源与验证状态

基线 `9e1693110880f23245d8d2e780d37316f1d7d21b`。tests-only 提交 `65ba4b0b0a58876f66e3d28de09cdeb045142cab`；父已用集成基线执行 `test_real_finite_face_load_is_bound_to_explicit_material_station`，真实有限箱面–球的 4 N 入步前置通过，随后缺少 `runtime.material_stations`，RAW1。父保存收据名为 `03-material-first-red`。首次完整功能运行在 `3 passed / 1 failed` 停止：同一有限面移动的 `[17]` 用例错误期望仅凭 chart ID 就能跨过稀疏支撑变化。第一次诊断的完整 dataclass 被 pytest 截断，保留 RAW1；随后短诊断实际确认 first.feature `(1,7)→(1,5,7)`、second.feature `(0,)→(0,)`、ended=`feature_changed`，箱面局部点由 `(0,0,0.125)` 移到 `(0.001953125,0,0.125)`，球侧 `(0,0,-0.125)` 不变。这是 B4 初版源/测试对 feature 定义不完整，B3 的保守重置正确。修复新增显式完整 feature 声明，以预先登记的 +z 面 `(1,3,5,7)` 验证同面连续性，同时保留未声明时的原 reset 和越出声明子集的拒绝负控。修复与其余专项待父串行 GREEN；作者没有运行项目 import、测试、数值、lint 或格式化。

仅新增：

- `src/phymeo/runtime/material_stations.py`
- `tests/test_material_stations.py`
- 本报告

本批复用 B2 的真实 `PairLoad`/`LoadSite`、双方 material local point、物理 owner 与原法则标记；B3 的身份、运输、丢弃储能和纯事务保持原实现。`NormalLayer` 与 `RobotContactMaterial` 仍是不同原力律，没有把开距改成穿深或互相换算。

## 公开适配合同

`OwnerPose(owner, position_m, rotation_world)` 保存实际 owner 原点和正交右手旋转的独立不可变值。支持普通 rigid COM、URDF computational link、fixed root；粒子/柔体 owner 暂时明确拒绝。机器人物理 body ID 与碰撞 remapped body ID 分开。

`MaterialStationRegistration(key, first, second, anchor_side, anchor_local_m, tangent_axis_local, first_chart_id, second_chart_id, topology_revision=0, match_tolerance_m=1e-8, first_feature=None, second_feature=None)` 声明一个稳定 token、canonical 同世界 shape pair 及一侧实际局部材料锚点。切向轴在该侧 owner 材料坐标系中声明，现场投影到真实法向的切平面；退化轴拒绝。匹配容限是调用者明确的局部坐标容限，不用来挑最近的一行。可选 `first_feature/second_feature` 是各侧完整 chart 的预声明原始材料顶点 IDs，必须非空、非负整数且互异；bind 逐次核实真实 weights 的所有 IDs 都在声明集合内。默认 `None` 仍将当前稀疏支持作为 feature，不凭 chart ID 自动扩展。

`MaterialStationRegistry(registrations)` 冻结最多 128 个登记，允许空集合；重复 key 拒绝。只读 `pairs` 是选中的 canonical pair tuple，供父层仅对这些 pair 使用 normal-only 原采样；未配置 pair 不进入适配，保留旧摩擦路径。

```python
selected = registry.bind(entry_events, entry_frames, sample_time_s=t)
trial = history.prepare(base_history, selected.samples, sample_time_s=t, dt_s=h)
# A 使用真实法向、完整跨站点响应和自由运动，得到物理候选。
material_delta, application_delta = selected.measure_motion(candidate_frames)
# 只有实际 motion/work 审核和全部场景检查通过，才共同发布物理状态与 history.commit(...)。
```

选中 pair 的每个正法向 site 必须恰好对应一个登记，登记也不能消费两条源样本；缺漏、歧义、重复样本拒绝。没有活跃站点的帧合法。`samples` 按显式 StationKey 排序，直接使用各 `LoadSite.force_n`，不再乘一次 fraction，不用合力平均掉不同法向或非共线作用点。

每个 `bindings` 项公开 `key`、完整拥有的 `record/component/site`、`component_index/site_index`、双方 `first_owner/second_owner`、`first_material_local_m/second_material_local_m`、`application_point_first_m/application_point_second_m`。两个 index 只定位**当前** record 中的真实载荷，用于回填；不作跨步身份。父通过完整 `binding.component.load` 与当次输入 PairLoad 做唯一值定位，不能把重复同值 PairLoad 任意分配。

bind 会校验每方实际 pose 能在登记容限内将局部材料坐标重建成该 LoadSite 的原表面点。输入列表、数组与 pose 数据被复制；后续修改调用者数组不会改变旧 selection。选择本身不消费 B3 generation，也没有可变时钟。

## 真实材料位移和最近点漂移

`measure_motion(end_frames)` 返回两个独立且不可写的 FP64 `[n,3]` 数组，顺序与 samples 相同：

- `actual_material_displacement_m`：将**同一入步**双方局部材料点随实际末态 pose 推进，取 second 减 first 的位移。
- `actual_application_displacement_m`：先在入步把原两端力作用点逆变换到各自 owner，再随末态推进，同样取 second 减 first。

RR 的旧摩擦可在中层公共点施力，robot pad 可在 pad 点施力；这些点不一定等于表面材料点。平移时两套位移可相同，偏心转动时一般不同。此模块如实返回差别，不能把 `h J v` 或重新采样的末态最近点当作真实材料运动。A 的实际 motion/work 门仍决定候选能否发布。

`chart_id` 是用户对该侧有效同一材料图的显式声明；适配器只保存当次真实权重/材料点，既不证明连续跨面，也不把 chart 内的最近点漂移计成塑性滑移。完整 feature 已明确声明时，当前稀疏 weights 可以在该集合内变化；对真正离开声明集合的采样直接拒绝。B3 对未声明坐标变化、owner/feature/拓扑变化仍保守释放重建，丢弃储能可见。夹持面裁剪后若原局部 anchor 不再对应唯一真实采样，本版拒绝，不自动将附近顶点升级为同一材料身份。

公共 `.contact_events` 表示上一成功步的**入步**载荷。父场景接缝必须从当前 `sample()` 的真实 loads 和当前 entry state/FK 纯 stage B2 view，再调用 bind；传旧 view 与新 sample_time 会拒绝。源码入口默认物理/摩擦行为本批作者没有改动。

## 新专项与范围

新专项包含真实有限台 4 N 及三步退出重入、有限四角 patch 每站点 0.25 N（合计 1 N）、真实 URDF 同滑轨双 pad 的转序与整体静态变换、真实有限面滑动时相同 chart 的见证漂移和固定入步材料位移分离、真实刚体转动时 material/application 两套运动，以及缺漏/歧义/重复、错时/错 owner、输入拥有、空集、不同世界同 token 和非法姿态负控。

这些适配测试没有凭手填法向替代真实采样。B3 中手填的弹性更新只验证存储/退出释放，不能当成新的摩擦物理结果。普通/机器人 scene 的新共同物理步、全 W、反力和总功验收由父与 A 另行交付；本文件本身不宣称任意转动接触已通过，也不宣称旧 mimic 抓持失败已解决。
