# 同一 world 的多个自由刚体与变形表面：M-B1 用法

这条接口让同一逻辑 world 的多个独立运动刚体共同作用于一个变形节点模型。每个刚体保留独立状态、控制、质量、COM 力矩和载荷；节点与全部刚体在一次 step 中共同接受或共同拒绝。零接触刚体也推进，不会从场景里消失。

以下是按当前公共源码核对的**文本示例，尚未单独执行**。它展示怎样修改已有合格单体声明，不提供一组保证稳定的材料/初态参数。完整固定案例和保存协议见 [安装消费者](../../tests/installed/check_deformable_multibody.py)；源码和安装资格以 [实施合同](p65-multibody-deformable-implementation-20261113.md)及父验收记录为准。

## 1. 从一体声明增加第二个 participant

保留原 `ParticipantDeformable`：其中的 `worlds` 是 `DeformableNodeWorld`，`surfaces` 是原面身份的安装记录。增加的是 `ParticipantArticulation`，不复制节点世界，也不手工拼接设备 bank。

下面复制原单 body 的资产作为另一个独立刚体。`q_b`/`v_b` 是调用者给定的第二刚体真实初态；其 root q 是 xyz + **xyzw** 四元数，v 是 root COM 的世界线速度 + 世界角速度，长度分别 7、6。若使用不同资产，须按该资产原 collider keys 重新建立完整安装表。

```python
from dataclasses import replace
import phymeo as ph

# base: 已有单 world / 单自由体的 ParticipantSceneDeclaration
# base.contact_response == "deformable_normal"
# base.deformable 是既有 ParticipantDeformable；保留原材料、节点和预算。
a = base.articulations[0]
world = a.key.world
b_key = ph.ResidentParticipantKey(world, 37)  # 与 a.key 不同
b_body = 303  # 与该 world 的其它 body/surface body ID 均不同
new_keys = {
    source: ph.CollisionKey(world, b_body, installed.shape)
    for source, installed in a.collider_keys.items()
}
b = ph.ParticipantArticulation(
    b_key, a.asset, ph.FloatingRobotState(q_b, v_b),
    kp=a.kp, kd=a.kd, effort_limit=a.effort_limit,
    body_ids=(b_body,), collider_keys=new_keys,
    initial_control=a.initial_control,
)
laws = dict(base.materials_by_pair)
for surface in base.deformable.surfaces:
    if surface.key.world != world:
        continue
    for source, old_key in a.collider_keys.items():
        old_pair = tuple(sorted((surface.key, old_key)))
        if old_pair in laws:  # 复用原过滤后完整 pair 的材料或显式 None
            laws[tuple(sorted((surface.key, new_keys[source])))] = laws[old_pair]

declaration = replace(base, articulations=(a, b), materials_by_pair=laws)
```

`initial_control` 也按 participant 独立持有；可在编译后用 `scene.set_controls({b_key: command}, expected_revision=scene.revision)` 修改其中一个。此处 command 沿用公共 `FloatingRobotCommand` / `FloatingRobotInputs`，不是给物体固定轨迹。材料表必须完整覆盖允许的表面—圆形 collider pair；不施加载荷的 pair 用显式 `None`，不能漏项。增加 collider 后，原 `site_capacity` 不会自动变大，需在原0..128域内按预定场景声明足够容量。

## 2. 推进并读取各刚体的实际载荷

```python
import numpy as np

inputs = {
    w.key: ph.StepInputs(np.zeros_like(w.initial_state.position, dtype=np.float32))
    for w in declaration.deformable.worlds
}
# h 复用原场景已选择的步长；inputs 是节点外力，不替代刚体控制。
with ph.compile_participant_scene(declaration, mode="cuda") as scene:
    receipt = scene.step(inputs, h, expected_revision=scene.revision)
    row = receipt.worlds[world]
    for key, body in row["rigid_participants"].items():
        wrench = body["uploaded_rigid_wrench"]  # (1, 6)：世界力 + COM 力矩
        raw_and_loss = body["rigid_wrench_delivery"]
        state = scene.snapshot()[world].rigid.participants[key]
        # state.position / velocity / tick / control 保留该 participant 的状态。

    for site in row["sites"]:
        key = site["participant"]
        force, torque = site["rigid_force"], site["rigid_torque"]
        # site 同时保留 round/surface/facet_id/point_ids/barycentric/body_slot。
```

`mode="interpret"` 使用同一公共接口。`uploaded_rigid_wrench` 是实际交付的合力/COM 合矩；delivery 另保存原 FP32 合成值、subnormal 规范化的实际值和损失，不把数值损失隐藏为零。每 site 的反力与节点散射、所有 participant 的 M/J/投影力及候选状态在同次 receipt 中；`material_impulse` 是原弹性行冲量，启用 D6 时另有 `damping_impulse`，两者不混称接触力。

## 3. 查询、保存与整个 world 恢复

独立 `query` 读取当前已接受几何，不推进物理，也不产生接触力。可以从声明建立 round→participant 映射，按原形状身份查看查询结果。以下代码应放在上一段 `with` 块内部，确保 `scene` 尚未关闭：

```python
owner_of_round = {
    installed: row.key
    for row in declaration.articulations
    for installed in row.collider_keys.values()
}
# 下列操作位于同一个仍打开的 scene 生命周期中。
query = scene.query(expected_revision=scene.revision)
for group in query.groups:
    for distance_row in group.packet.rows:
        key = owner_of_round[group.round_keys[distance_row.segment_id]]
        status, gap = distance_row.status, distance_row.gap

saved = scene.checkpoint()
scene.step(inputs, h, expected_revision=scene.revision)
scene.restore(saved, expected_revision=scene.revision)
scene.step(inputs, h, expected_revision=scene.revision)  # 真实恢复后的后继
scene.reset((world,), expected_revision=scene.revision)  # 此 world 的节点与所有刚体
```

checkpoint绑定原 scene owner、模型和布局，不能拿给另一个 scene 使用。`restore(..., worlds=(world,))` 可恢复选定**完整 world**；不能只恢复其中一个 participant。partial reset保留未选world的状态，所选world回到声明初态及初始控制。step、控制、reset或restore后旧query都应重新取得，不能因恢复到相同tick就继续使用旧结果。调用失败时保留 `last_attempt` 与原异常；完整故障保存/关闭异常聚合可参考安装消费者。

## 4. 当前边界

- 总 participant 数1..8，logical worlds数1..8；每个world至少一个participant。每个participant必须是 ordinary floating **单 body**，每world一个原D1节点模型，原节点/面/内存限制保持。
- 当前移动 collider 为原资产中的 sphere/capsule，作用于离散变形三角面；NormalLayer为mu=0的入步采样响应。一个body可有多个collider，全部反力累加到该body；同坐标的不同world互不作用。
- 本功能**不新增刚体—刚体碰撞响应**，也不支持关节树、affine/mimic树、摩擦、CCD或任意机器人—布料。它不承诺原D4/D6持续支撑稳态已通过。
- 多体 receipt 使用 `tyx-deformable-multibody-contact-step-v1`，world记录提供完整 `rigid_participants`，不再提供含义不清的单个 world wrench。原单体仍使用 `tyx-deformable-contact-step-v1` 及原字段，原核/计费路径保持。
- 超容量、未解析且必须承载的接触、错误身份或最后participant失败会拒绝整个共同step；不会截断site或发布一半world。这是功能组合资格，不是全GPU无HOST读回或Newton性能比较。
