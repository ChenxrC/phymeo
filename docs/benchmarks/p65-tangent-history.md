# B3：显式材料站点与切向状态事务

本批新增一个可供普通刚体、粒子和机器人材料适配器使用的状态工具。调用方明确登记站点，求解器产生切向候选，本工具验证并保存；它不搜索最近接触、不计算摩擦力、不把 B2 事件观察变成求解状态。默认 ContactScene、RobotScene 和旧动摩擦/静持入口均未改变。

## 源与当前验证状态

基线 `ff6efeb07143cb2b3eb671a48d26d4ad6bd2e06c`；tests-only 提交 `6dc3c05f7b518ab6a632126095098845dd728013`。父在 `artifacts/p65-postnight-batch3-20261105-01/b3-history-red` 实际运行首节点：普通场景的真实 4 N 接受步和净线动量先验通过，随后缺少 `phymeo.solvers.tangent_history`，`1 failed / 1.05 s / RAW1`。

首份产品 `19d2392` 已由父实际执行原专项 `15 passed / 1.11 s`；父样式提交 `52975a8` 对两个 Python 文件核得 raw AST 不变，Ruff/format 通过。独立审查随后发现完整 trial 内容未绑定及 owner 字段可能持有外部可变引用。回归提交 `6793619` 的 9 个新增实例在 `artifacts/p65-postnight-batch3-20261105-01/b3-review-red` 实际得到 `9 failed / 1.28 s / RAW1`：释放记录/能量篡改 1 项、四个求解数组替换 4 项、非机器人或固定根 owner 字段 4 项。本次修复候选待父完整 24 实例验收，不用先前 15 绿代替修复验证。作者未运行 import、测试、数值、lint 或格式化。新增文件仅：

- `src/phymeo/solvers/tangent_history.py`
- `tests/test_tangent_history.py`
- 本报告

## 已有机制与 pinned Newton 证据

- 现有 `MaterialStickSlipStepper`、`MovingMaterialStickSlipStepper`、共面/变形/小折角材料模块已共用 `solvers/_material_stick_law.py::return_traction`。前者有独立动态节点、固定有限面等限制；可动面会运输面内极旋转，未知倾斜/越边拒绝。专门 seam 程序具有自己的已验过边判据，不能凭通用事件的同一 body/shape 标签复用其结论。本批不复制返回映射。
- B1/B2 的 identity 是已受载 **shape pair** 的 generation。B2 的 component 是当次真实采样，不是稳定站点编号；跨三角片可能改变 shape pair，纯事件身份不能自动继承弹簧。
- 本机 `.venv-newton-1.7.0/Lib/site-packages/newton-1.7.0.dist-info/METADATA:2–3` 确认 `newton / 1.7.0`。以下均查本机源码，没有新增 Newton 物理运行：
  - `_src/geometry/contact_match.py:5–55`：刚体接触先按 shape pair 找范围，以世界接触中点距离和法向点积阈值匹配，并做单射冲突决胜；这是一种有阈值的几何对应，不等于任意拓扑材料连续性。
  - 同文件 `:65–89`：匹配器持有独立旧缓冲；sticky 额外保存双方 body-local 点/offset，并在保存下一帧前重放匹配几何。
  - `_src/sim/contacts.py:347–365`：match index 明确引用上一排序接触缓冲，`rigid_contact_match_generation` 说明其对应哪次 contact set。
  - `_src/solvers/vbd/solver_vbd.py:544–566` 与 `rigid_vbd_kernels.py:243–268`：VBD 分开数值 warm-start 和 sticky 材料锚点；只从确实解过的同 buffer/generation 恢复。latest 的切向乘子是投影且限于摩擦锥的数值初值，不能称其为已验证的通用软接触材料身份。
  - `_src/sim/contacts.py:390–440`：soft contacts 有真实节点 indices/barycentric 和刚体侧表面点。该数据本身没有为本批提供软特征跨步匹配合同。

## API 与边界

`TangentHistory(capacity, max_frame_turn_rad=...)` 创建一个独立命名空间。`initial_state()` 得到零时刻不可变状态。

```python
trial = history.prepare(state, samples, sample_time_s=t, dt_s=h)
candidate = history.commit(state, trial, updates)
# 只有全部物理验证也通过，调用方才将 candidate 与真实物理末态一起发布。
```

`commit` 也是纯函数式候选生成：同一 base 上同输入重试合法；将旧 trial 用于后来的 state、另一 owner 或改过的时间会拒绝。准备快照绑定所有公开 trial 字段：站点、代际、结束原因/释放能量、键、时间，以及四个求解数组的拥有关系、dtype/shape/stride 和实际字节。替换或篡改这些内容会在接受更新前拒绝；数组的数据只读不被误当作其 NumPy 元数据也不可修改。没有全局 consumed 标记，也没有内部钟；失败不推进 generation 或历史。当前样本数及曾受载 token 的 tombstone 总数均受固定 capacity 约束，不隐式淘汰、增容或回收旧 generation。

- `StationKey(world, token)` 的 token 必须由调用方对真实材料站点显式登记；不能把当前 component 下标、世界坐标 hash、最近邻顺序当作长期身份。
- `MaterialAddress` 保留实际 `ContactOwner`、shape、topology revision、原 feature 节点 IDs、weights 与 rigid/link-local surface point。机器人物理登记 ID 与 remapped collision body ID 分开。粒子侧使用材料节点权重。 接受前按 owner kind 验证全部字段并重建规范值对象：普通刚体/粒子不得携带 link 字段，robot link 为非负整数索引和非空字符串名，fixed root 为整数 `-1` 和非空字符串名；不保留调用者的可变字段。
- `StationSample` 另给真实法向、法向载荷和 `frame_world=[t1,t2,n]`。frame 必须正交、右手且与法向一致。运输使用完整 `F_new F_old^T`，包括绕不变法向的材料共转，不能只投影旧世界向量而缩减储能。转角超预设范围直接拒绝。
- 默认材料地址变化即释放并重建。可选非空 `chart_id` 仅是生产适配器对有效同一材料图的**明确声明**；在 owner、feature、topology 都不变时，才允许该图内坐标漂移。此声明和 weights 不证明实际连续运动，也不生成位移/滑移/功。适配器仍必须独立提供真实材料位移，不能以最近点漂移冒充滑动。
- owner、拓扑版本、shape/feature、chart 改变均保守重置并递增 generation；未验证的相邻面、同空间重合面、非连通面都不会自动继承弹性。零载或缺席明确结束。`ended` 给原因和旧弹簧能量，`dropped_spring_energy_j` 给本次总额；不静默抹去。

trial 返回按显式 key 排序的 `stations / keys` 与拥有且不可写的 `frames_world / normal_force_n / elastic_displacement_m / elastic_tangent_m`。候选更新使用：

```python
TangentUpdate(
    elastic_displacement_m=world_vector,
    stored_energy_j=energy,
    plastic_slip_m=cumulative_plastic_distance,
    tangent_work_j=cumulative_work,
    regime="stick",  # 或 slide、正法向载荷下的 frictionless
    stiffness_n_per_m=kt,  # 可选；提供则核对 E=kt*|elastic|^2/2
)
```

B 验有限数值、切向性、非负能量/塑性距离、同 episode 塑性累计不倒退，以及可选的弹簧储能一致性。FP64 frame/能量一致性容限为 `1e-10` 量级的表示检查，不是物理准确度门。`open` 行由 prepare 的零载结束表示；A 核的卸载释放与 B 的 identity reset 释放在集成时必须分清来源，不能重复计能量。

## 冻结案例与待验范围

专项取实际普通 4 N 球对、实际 2 N 球–有限可动三角片、实际机器人双 pad 的 owner/local surface 地址作为输入；覆盖显式 token 转序与不同 world 同 token、共同刚转/法向不变的面内 twist、零载结束/新代重入、face/topology/owner/chart 重置、容量、坏 frame、stale base/time、坏 update 及同输入重试。材料运输的刚转样本来自两个真实几何快照，是坐标协变测试，不冒称已运行完整转动动力学轨迹。

测试中的手填弹性/能量只用于状态工具验证，不是新摩擦求解结果。A3 的 HOST 全耦合切向核独立输入完整 W 和真实自由材料增量；父另有两正质量、两共享站点的组合推进验收。B 不 import A，也不自动 hook 生产 scene。本批通过后仍不能宣布一般非线性旋转接触、任意跨面继承或原 mimic 抓取失败已解决。
