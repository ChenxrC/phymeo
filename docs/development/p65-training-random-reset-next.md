# N5 首片：可复现 q/v 随机 reset

状态：SOURCE_RESEARCH_ONLY / PROPOSED_NOT_IMPLEMENTED。2026-11-12。
只读当前 runtime、VecEnv、N3/RSL 和固定 Newton 1.7 源码；未运行项目、
采样、训练、测试或数值探针。此能力可独立于 J/R 的学习收益推进；不以
随机初态重训来改写旧学习结果。N4e 的实际结果由父另行记录。

## 1. 已有机制和准确缺口

| 层 | 源码事实 | 首片需要补的部分 |
|---|---|---|
| TYX runtime | `_mujoco_tyx_runtime.py::reset(keys)` 从 `_initial[key]` 生成候选，重建设备，清所选 world 的 controls/tick/receipt；原构造初态不可由公开 reset 替换 | 公开、带 CAS 的一次性 reset-state 输入；不能临时修改 `_initial` 或把每步状态写入伪装为 reset |
| VecEnv | 手动 reset 和 done 自动 reset 均调用上述固定初态；之后 export、更新 episode/returns/previous_action/frame；已接受 physics 后失败会 poison | 两条 reset 路径复用同一随机计划、每 world 计数与提交收据；成功的下一 observation 必须来自该次实际 reset |
| N3 | `training_binding()` 和完整 `capture_training_state()` 已进入版本 bundle；恢复前校验/候选 learner，随后 physics CAS，再刷新 task/RNG | 在 opt-in env binding/state 中增加计划身份与 reset 流计数；无需复制 PPO 或再造 checkpoint 格式 |
| RSL | `reset(indices)` 将行号转为稳定 world key；自动 done 由 env 完成，后续 upstream 清相应 RNN hidden；禁止仅随机 episode length | 保持行号/稳定 key 的区分，继续清正确 done mask，不开启 `init_at_random_ep_len` |

当前 runtime reset 重建整个设备。未选 world 不能只“看起来 q/v 相同”：
完整 act、held 三通道、tick、receipt、CPU integration mirror 与原始双字 clock
均须保留。新共享 seam 应复用 A1 的 accepted-bank 捕获/核验和退休 owner
处理，不能由 HOST time 重新拼出未选 world 的原始 clock 字节。

## 2. 固定 Newton 参照及边界

本机 `.venv-newton-1.7.0/Lib/site-packages/newton-1.7.0.dist-info/METADATA`
明确版本1.7.0；以下原源码只读，未 import Newton：

- `examples/selection/example_selection_cartpole.py:30-35,99-112`：
  `wp.rand_init(seed, tid)` 产生分 world q，并通过 ArticulationView 写入。
  SHA256 `57e19a5a34cec98f592663451201c690264efe2f9471624dbe2dda8a9bdb254b`。
- `example_selection_articulations.py:43-64,266-303`：mask 控制各 world 的
  root velocity/transform/DOF reset，随机输入使用 seed/world。
  SHA256 `25b54bc57b2264321dc454b7405188db5cc50aa94497707a629b7a763f73f51d`。
- `example_selection_materials.py:45-53,202-254`：修改模型 friction 后明确
  `notify_model_changed(ModelFlags.SHAPE_PROPERTIES)`，这是另一条模型缓存更新链。
  SHA256 `93c0c04d0a7db5a457058b6095177ef4e964814b89c97499dc80df66f0df1195`。

这些固定 selection 示例是可复用状态/材料操作参照，不是带完整 episode、
PPO/RNG checkpoint 的训练验收。Newton1.7 的[官方训练生态入口](https://newton-physics.github.io/newton/1.7.0/lab/isaac-lab.html)
将 Isaac Lab 支持交给外部集成。上述示例本身不承诺 Phymeo 所需的 subset
顺序无关、CAS 或恢复后随机流一致，不能照搬 `seed=step_count` 当作此合同。
远程固定 tag 源页本次未取回；具体判断来自上述本机固定版本文件，而非猜测 main。

## 3. 最小公共 API 和采样协议

建议三个组成部分，不加新的物理求解核：

1. `MuJoCoTyxRuntime.reset_states(states, *, expected_revision)`：输入是非空或
   空的稳定 world-key 子映射，值为完整 `MuJoCoInitialState`。仅 reset 边界；
   time=0，所选 world 的 controls/tick/receipt 按原 reset 清零。复用原 FP64
   初态合法性和 FP32 packing，不归一化坏输入。原 `_initial` 默认模板不变。
   返回不可变 receipt，含 source/destination revision、所选 keys、请求/实际
   stored q/v/act、新 clock/tick 与源/布局 binding。旧 `reset(keys)` 默认语义不变。
2. NEW `MuJoCoTyxResetPlan`：不可变完整 anchor states、q/v 原生索引及 uniform
   bounds、seed、stream namespace、algorithm/version、模型/布局/key 身份。
   首片只随机已支持 hinge/slide 的 q/v；自由根和 quaternion、ball、act 都保持
   合法 anchor，不随机化。没有质量、惯量、阻尼、摩擦、contact law 或模型修改。
   不重采坏点、不裁剪结果；采样后原初态准入失败则整次请求失败。
3. `MuJoCoTyxVectorEnv(..., reset_plan=None)`：默认路径不变；显式计划使手动
   reset 和 done reset 使用同一 sampler。构造不隐式增加随机 reset，使用者要
   首个随机 episode 时明确调用 reset。允许读取最后 reset receipt，但不增加
   “任意训练时刻 restore”或完整 Env clone 的支持声明。

建议首片采用 HOST 无可变 RNG 的计数映射，避免新的 CUDA RNG kernel 和
Torch/NumPy全局状态副作用：

```
key = versioned_encode(namespace, seed_u64, stable_world_key,
                       reset_index_u64, channel_q_or_v, native_index)
z = little_endian_u64(SHA256(key)[0:8]) >> 11
u = z * 2**-53                     # [0,1)
sample = RN32(lower + u*(upper-lower))
```

编码、字节序、浮点计算顺序在实施前固定并保存；这是工程可复现采样，不宣称
跨所有软件/硬件精度普适。plan hash 绑定完整参数，但**不把整个 world 集合/
调用顺序的 hash 放进随机 key**，否则 subset/clone 会改变其余 world 样本。
同一 world 的隔离编号流由稳定 key 和计数决定；顺序、页、batch row 不参与。
这里承诺复现与流隔离，不另宣称随机独立性的数学证明。
全局 Python/NumPy/Torch RNG 保持不变。HOST采样与上传成本明确计入，不称全GPU。

每 world `next_reset_index` 从0开始，selected 每次实际 reset 提交后加1，
未选不变；空 subset 不增，重复/未知 key、计数溢出在提交前拒绝。episode_id
是用户 episode 身份，reset_index 是随机流账本：正常成功二者同样推进，但
不能在部分发布失败时用旧 episode_id 倒推出已经消耗的随机样本。

## 4. 成功发布、晚失败与恢复

在 Env 排他 owner 锁内：先取得当前 frame/revision，纯采样且预分配下一份
episode/returns/action/计数，再调用公开 `reset_states` CAS。所有 world 候选
通过且设备构建/验证完成后才发布 physics；不逐 world 提前写入。

| 失败时刻 | 必须呈现的实际状态 |
|---|---|
| 参数、采样、初态、容量、stale CAS、末 world 构建失败 | physics、episode、current frame、reset indices 全不变；同请求重试得到同样样本 |
| physics 已发布，旧设备 cleanup 失败 | 如现有 A1：退休 owner 留存重试，不假装 reset 被拒；receipt 说明实际提交 |
| physics 已发布，task export/同步/episode 包装失败 | 所选 reset indices 已消费一次，记录 actual accepted reset receipt，env/adapter poison；禁止暴露成功 transition 或假称 RNG 回滚 |
| 恢复到此前一致 checkpoint | physics、episode、plan/counters 共同恢复；之后下一次相同 key reset 得到相同样本；运行资源计数和目的 CAS revision 不回退 |

正常返回时实际 q/v、episode、previous_action=0、returns=0、新 current frame
共同组成一个可用发布。现有 API 没有跨 runtime/task 的通用两阶段提交，因此
这里明确保留“CAS后失败须恢复”的既有 N3 语义，不宣传全部晚失败可原子回滚。
恢复必须显式：一致 N3 bundle 或完整新 episode reset；不能偷偷继续旧frame。

`training_binding()` 在 opt-in 时加入 plan hash/algorithm/seed/namespace；
`capture_training_state()` 加入固定 schema 的 key→next_reset_index 和最近提交
身份（语义计数可恢复；目的 operational revision 另记且不倒退）。`restore_training_state()` 先完整校验/复制，再随现有物理CAS发布；若
CAS后刷新失败，复原状态中的 reset 流已是该次实际 accepted physics 的账本，
owner 继续 poison。plan/source/seed/布局不匹配不得在 physical CAS 后才发现。
N3外层已有 env state 和 binding 容器，预计不需改 `mujoco_tyx_checkpoint.py`
或 RSL 算法；若实现发现严格 schema 阻挡，只给父最小具体差异。

runtime.clone 只复制物理，不自动复制 episode/随机流。首片同 world/order 的
训练克隆通过原 N3 bundle 恢复组合状态；不新增泛化 Env.clone。子集计划的
纯采样 key/counter 保持可限制性，但不冒称支持跨 world-order N3 恢复。

## 5. 文件所有权与固定多案例

| 文件 | 变化及 owner |
|---|---|
| NEW `src/phymeo/training/mujoco_tyx_reset.py` | 不可变计划、版本化纯采样及采样收据；训练作者 |
| MOD `src/phymeo/backends/_mujoco_tyx_runtime.py` | 父唯一 owner：公开 reset_states、backend reset receipt、候选/双clock保留/CAS/退休资源；backend不导入training模块 |
| MOD `src/phymeo/training/mujoco_tyx_env.py` | 手动/自动共同 reset、计数、poison、N3 env binding/state；训练作者与父先冻接口 |
| NEW `tests/test_mujoco_tyx_reset_states.py` | runtime reset事务，interpret/CUDA |
| NEW `tests/training/test_mujoco_tyx_random_reset.py` | HOST sampler + CUDA Env/N3组合 |
| NEW `examples/mujoco_tyx_random_reset.py` | 两任务、稀疏 worlds、选取reset、保存/恢复后后继一致；不重写训练循环 |

不改现有任务/XML、物理核、N4学习预算、RSL算法、CPU runtime或 package exports。
默认旧Env/RSL接口继续可用；只跑实际受影响 reset/checkpoint 子集，不重跑旧学习。

预登记基础库建议如下；所有数值在执行前冻结，不能按采样结果换分布：

- 两物理模型仍取原 J 和 S，原h/质量/接触/动作范围不变；W4稳定稀疏key。
  J：每轴 q uniform[-.10,.10]、v[-.04,.04]；S：q uniform[.00065,.00085]、
  v[-.002,.002]。seed固定6511；这是新随机reset资格，不是原N4训练变体。
  每次reset后至少两次真实受控step，S读实际contact/wrench，不能仅看采样数组。
- HOST：精确重复、反序subset、分两次独立subset与一次并集相同、空集、未选
  world计数不变；改namespace/seed的身份拒绝；重复key/非法q索引/NaN/越界/
  非法anchor/溢出，不消耗流。硬编码采样字节golden向量跨单独重建比对。
- 实际两后端：晚一个 world 失败、stale CAS、末资源构建失败均无部分发布；
  unselected q/v/act/controls/receipt/tick和双clock逐字节保留；旧frame/动作包失效。
- CUDA Env：不同world在不同步done，final_obs取reset前，下一obs取随机后；
  手动乱序subset也同样正确。export晚故障证明physical已reset、计数已消费、
  owner不可继续，完整恢复后行为明确；cleanup重试不重复消耗样本。
- N3：两模型短原上游更新后保存，A继续跨随机done，恢复后B相同后继；核完整
  action/obs/reward/done/q/v/held inputs、episode、reset counters、RNG、Adam。
  一例GRU非零hidden+部分done；同world/order runtime.clone再组合恢复，验证
  下一次reset和延续一致。不是新128-update学习或更大规模实验。

## 6. 可实施性与止损

这是中等规模、可独立开发的首片，最小顺序是父先冻/实现公开 reset-state
事务，再并行纯采样/Env测试author，最后父唯一串行验收。它可在一批内交付，
但前提是双clock与晚故障语义实际闭环；不承诺研究文档即今晚验收。
若共享seam必须绕过原状态准入或修改 `_initial` 才能工作，停止该捷径并报告
具体反例；若两次定向排查仍未缩小同一事务问题，留最小失败，不扩随机分布。
模型参数随机化另设后继，需要重编译/缓存刷新与模型绑定变化，不能搭本次reset
偷偷修改质量、摩擦或动力学。G1训练、ball/root姿态随机化、更多奖励和全GPU
采样均不在首片承诺内。
