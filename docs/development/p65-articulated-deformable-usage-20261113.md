# 非根连杆与变形表面的共同响应：M-B2

现在同一公共声明可以把一个 ordinary `FloatingRobotTree` 的 R/P 子连杆与一个真实变形节点模型连接起来。碰撞点的力和 COM 力矩按其实际连杆交付，再经原完整 J/M 推进整棵树；没有碰撞体或没有接触的连杆仍保留质量、惯量、状态和零接触载荷行。

这扩展的是原 `deformable_normal` 路径。已有的多个独立单体 world 仍可用，也可与关节树 world 放在同一个 batch。一个 world 含多 body participant 时，该 world 目前只允许一个 participant。它不是 MuJoCo-TYX 的 FREE＋BALL 路径，也没有新增两机器人之间的响应。

## 声明和使用

复用已有 `FloatingRobotAsset`、`FloatingRobotState` 和 `ParticipantArticulation`。关键是完整安装身份：

1. `body_ids` 顺序为根 body、原树各 child body，长度等于完整 body 数。没有 collider 的 body 也必须列出。
2. `collider_keys` 从资产的原 shape key 映射到实际 world/body/shape；非根 collider 必须安装到其原 link 对应的 body ID，不能统统写到根 body。
3. `ParticipantDeformable` 保存原节点、距离材料、初态及稳定 point/facet IDs。材料表完整覆盖允许的 surface–round pair，使用 `NormalLayer` 或显式 `None`。
4. `initial_control` / `set_controls` 仍用原 `FloatingRobotCommand`。有界 PD/前馈通过 `RobotCommand`，完整每 body 世界力/COM 力矩通过 `FloatingRobotInputs`；不以手改位置代替控制。

下面是消费已有合法声明的接口片段，**不是另外执行过的新场景**；完整已执行 C/D 数据见[固定案例](p65-articulated-deformable-cases-20261113.md)和[独立安装消费者](../../tests/installed/check_deformable_articulated.py)。

```python
import numpy as np
import phymeo as ph

# declaration: 已声明关节树、完整 body/collider 安装和节点材料。
# h: 该场景选定的真实物理步长。
inputs = {
    w.key: ph.StepInputs(np.zeros_like(w.initial_state.position, dtype=np.float32))
    for w in declaration.deformable.worlds
}
with ph.compile_participant_scene(declaration, mode="cuda") as scene:
    receipt = scene.step(inputs, h, expected_revision=scene.revision)
    for world, row in receipt.worlds.items():
        for key, participant in row["rigid_participants"].items():
            body_ids = participant["body_ids"]
            wrench = participant["uploaded_rigid_wrench"]  # (B, 6)，世界力和 COM 力矩
            delivery = participant["rigid_wrench_delivery"]  # raw / delivered / loss
            state = scene.snapshot()[world].rigid.participants[key]
        for site in row["sites"]:
            owner = (site["participant"], site["body_group"])
            force, moment = site["rigid_force"], site["rigid_torque"]
```

`mode="interpret"` 使用同一公共入口。新域的 step schema 是 `tyx-deformable-articulated-contact-step-v1`；原单体和 M-B1 schema 保持。`body_slot` 仍表示该 participant 的首 body，新增 `body_slots` 覆盖完整树；应使用这些身份映射，不能假设“一个 participant 等于一个 body”。

`receipt.rigid_publication` 同时保存完整 entry/candidate geometry、J/M/bias、控制、projected generalized load 和实际后继。每 site 的速度、COM lever 与有限步功都使用真实 body group。节点接触力、材料冲量及固定反力单独保存；它们不能混称为接触冲量。

## 生命周期与资源

`query` 不推进物理；step、控制、reset、restore 后应重新查询。`checkpoint` 绑定原 owner；恢复与 partial reset 按完整 world 同时处理节点和整树。不能只恢复其中一根连杆，也不能把 checkpoint 移植到独立 scene。

完整候选和 receipt 验证后只共同发布一次。实际晚查询、完整 publication 和跨 world body 地址故障已验证保留旧状态；随后通过真实重试与独立 owner 的后继比较。释放状态 UNKNOWN 时不重复关闭或假报已释放。

容量仍为总 participants≤8、worlds≤8、每 world physical bodies≤33，以及原节点/面/site/query 限额。声明 cap 不等于极限组合已测：本批关节正控是两个 world、每树 3 或 4 bodies；旧 M-B1 的 4 world/8 participant 资格按原结果保留。新预算按完整 body 数计算，包括零接触行；这只是明确 owned arrays 的预算，不是整个进程或显存峰值。

## 已验与下一边界

已验 R–P 串联末端胶囊、非对称 R/P 分支的两连杆同时受力、非零根/关节入速、控制变化、每族八步连续参考、输入倒序、原 M-B1 world 混合、零响应和资源/事务负控。源码解释 15 项、CUDA 10 项；安装两后端各 40 worldsteps。详见[报告](p65-articulated-deformable-report-20261113.md)。

法则仍是按 facet 声明的入步采样 NormalLayer，零摩擦，有 HOST 载荷/诊断往返。短窗资格不证明持续稳态支撑、摩擦换面历史、CCD、任意关节树组合或 dense cloth；D4/D6-B 原持续支撑失败继续保留。可选 D6 透传复用原实现，本批新关节树消费者未另验带 D6 的组合矩阵。
