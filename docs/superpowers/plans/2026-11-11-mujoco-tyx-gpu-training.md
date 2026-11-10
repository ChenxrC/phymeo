# GPU 训练连接：后续实施计划

日期：2026-11-11。状态：PLANNED / SOURCE_RESEARCH_REVIEWED / NOT_IMPLEMENTED / NOT_EXECUTED。
父已完成源码调研与计划审阅。本计划列为下一批入口；本夜先完成当前已冻结增量的验收，不插入 GPU 环境安装或训练实现。当前 CPU 训练环境与原物理预算保持。父唯一串行项目执行，隔离工作树只可在另行授权后并行纯编码。

依据：[已审阅的源码调研](../../development/mujoco-tyx-gpu-training-research.md)。本稿只读取该报告及相关本地源码，未新增网络调研或项目执行。以下源码路径均相对父仓库 C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo；调研基线 d9e5bf33，本次只读接口核对父 HEAD 19feafc2，具体验收仍绑定执行时冻结源码。

## 目标与已有边界

目标是先连通两个通用任务的真实 GPU 策略、真实物理 rollout、真实 PPO 更新及更新边界恢复。G1、W256 训练吞吐、随机化、多 GPU、任意时刻恢复和更多奖励表达式留在后续，不用一个舞蹈代表整个训练功能。

| 已有，可复用 | 尚未实现，不能提前宣称 |
| --- | --- |
| MuJoCo-TYX CUDA 物理、公开 action/task、物理 checkpoint/CAS；已声明有限模型域 | 独立 MuJoCo-TYX device VecEnv、真实 CUDA Torch→物理→RSL PPO 整体资格 |
| TaskSpec 的 FP32 仿射 observation、负二次 reward、区间 terminated；每次 export 是 Torch-owned D2D copy | 任意接触/传感/history reward DSL、物理 scratch 的零复制训练 view |
| action 接受 TYX CudaArray 与 producer stream；复制成 owned packet，HOST 全字节审计后 CAS 提交 | 完整流程无 D2H/无同步/全 GPU 控制流 |
| CPU Torch 2.15.0+cpu、RSL 5.5.2、TensorDict 0.15.5 的既有训练环境 | 合格 CUDA Torch 环境；现有 RslRlVecEnv 的 NumPy/HOST 路径不等于新 device 路径 |

复用数据流：task refresh → to_torch(copy=True) → CUDA policy → FP32 contiguous action → 固定 TYX from_torch/stream_from_torch → set_controls_from_device → 一次真实物理 step → 新 task frame → episode/rollout。保留 HOST 审计和同步并分别计数；不为速度删诊断，不承诺超越 Newton。

## N1：独立 CUDA Torch profile 与真实策略消费（中等；前置门）

**输入**：原 TYX pin 99f6df7、Python ABI、现有公开 task/action 与两类已验 fixture。**输出**：隔离依赖 profile/lock、公开最短闭环 example、focused 互操作测试和带来源的运行收据。优先零产品改动。

1. 父后续授唯一执行槽后核 driver/device、磁盘和官方可用 CUDA Torch wheel；固定 wheel/hash、CUDA build、Python、TYX、RSL 5.5.2、TensorDict 与模块来源。新建独立 GPU profile，禁止覆盖 .venv-training、原 G1/物理环境或更换 TYX pin。没有兼容 wheel/容量时止于环境资格，不能 CPU fallback 或编译 Torch 来掩盖。
2. 执行前把引用 fixture 的 XML/模型 hash、h、质量、初态、控制/外载、native channel 映射与原数值门写进新测试常量和案例表。复用原物理门，不用实际输出调参数。
3. 固定两模型：hinge+slide 的 W=1/8/32；有限平面支撑 slide 的 W=8。每例 8 个连续真实步，非零入步速度、控制与外载；使用真实 torch.nn MLP CUDA forward，记录 weights/input/output 的 device。默认及一个非默认 producer stream 均覆盖。此节点是 inference 闭环，不叫 PPO。
4. HOST 对照只取同一次 CUDA forward 的 FP32 action 副本，各自连续演进；逐步核完整 ctrl/qfrc/xfrc、q/v/clock、真实接触力与收据，沿用已验同输入门。不能把每步状态复制到对照使其追随。

**固定负控**：CPU tensor、dtype/shape/noncontiguous stride、非有限/禁止的 subnormal、通道 alias、stale revision/frame、已关闭 view/stream；验证 submit 返回前 tensor/stream 存活，返回后可释放，workspace.close 后 Torch-owned observation 仍可用。只有一张卡时跨卡项标未测，不伪造通过。

**通过**：全部模型完成实际 CUDA inference→真实物理；公开 CAS/所有权负控正确；输入 bits 与结果绑定、资源关闭完成；复制/同步/HOST 诊断分开记录。**停止**：资格失败，或同一可归因修复后同阻断仍在；交最小错误和下一决策，不改动作/物理范围。不通过 N1，不运行 N2 PPO。

## N2：独立 device VecEnv 与固定 RSL 实际优化（较难；依赖 N1）

**输入**：N1 profile、两模型、冻结 world/order/action/episode contract。**输出**：独立 MuJoCo-TYX device wrapper、薄 RSL adapter、两任务短程训练 example 与协议/优化测试。复用上游 PPO/rollout/optimizer，不复制算法，不将旧 HOST wrapper 改名为 CUDA。

- wrapper 自有执行锁并独占所包装 runtime 的写入，串行 step/reset/restore/checkpoint；只调用公开方法，不假设公开 runtime lock 或读取私有 storage。绕过 wrapper 的外部并发写入不在首版合同；revision/CAS 只在现有公开 action/restore 提交点拒绝 stale，不能宣称防住任意外部竞态。
- 固定 world_keys 与原生坐标映射；外部入口接显式 world-order metadata，RSL 内部绑定冻结顺序。同形状且无标签的 tensor 行被偷偷交换无法自动识别，测试只能验证 metadata 不匹配被拒；不得宣称识别所有语义行交换。
- 默认 decimation=1；先用已有 TaskSpec。分别保存 final_observation/final_mask、terminated、time_outs，再 subset reset 返回新初态 observation；未结束世界 state/history 不变。终止与超时同时出现以 terminated 优先，超时 bootstrap mask 只取未 terminated 的 timeout。
- episode_id/length 可先 HOST 管理并诚实计读回。关闭 init_at_random_ep_len；不能只随机改计数而不推进物理。done 只清相应 actor/critic RNN 状态。
- 固定 RSL 5.5.2 原语义：process_env_step 的 timeout bonus 使用入步 act 的 values，不读取 final_observation。独立验 mask/target；本节点不偷偷修改成 final-state critic bootstrap。

**两事务必须显式输出**：

| 失败位置 | 真实状态与 wrapper 行为 |
| --- | --- |
| action preflight/CAS 前 | 原 accepted state 和 held control 保留；不写成功 transition |
| action 已提交、physics 失败 | q/v 不前进，新 held action 已接受；记录 action_accepted=true/state_accepted=false，停止该 rollout |
| physics 已接受、task/export/包装失败 | 物理已经前进；标需恢复并停止 learner 消费，不声称回滚，不自动 reset 伪续跑 |
| 完整 step/terminal bookkeeping 成功 | 才向 rollout 提交完整 transition；重复调用不得重复插入 |

**固定案例**：上述两个任务分别 W=8、固定不同 seed，MLP/FP32、固定学习率，4 次 update×16 steps/world（每任务 512 transitions），关闭 AMP、torch.compile、RND、distributed。测试常量在首次执行前登记 seed/网络/学习率/episode horizons，不根据 reward 调参。每任务至少两个世界在不同时间结束；另短 GRU rollout 验部分 done mask、未 done hidden 保留。负控覆盖 bad action、metadata 行序、旧 frame、terminal observation 被 reset 覆盖和上表两个后提交失败点。

**通过**：两个任务各 4 次实际 CUDA loss.backward/optimizer.step；有限 loss/grad、optimizer 步数和参数前后 hash 证明实际优化，全部 transition/episode/reset/timeout/RNN mask 可独立核对；原物理门仍通过。无需承诺 4 次更新学会任务或 reward 单调上升。**停止**：同失败两轮未缩小到接口/环境/物理层；保最小输入与阶段，不扩大训练预算、换任务或重写 PPO。只有网络 CUDA、动作仍经旧 HOST adapter，判本节点未完成。

## N3：PPO 更新边界的组合 checkpoint（较难；依赖 N2）

**输入**：N2 已实际运行的 learner、env、物理 capsule 与独占 wrapper。**输出**：版本化组合 manifest、边界 save/restore helper、两任务与短 RNN 恢复测试。物理 capsule schema 保持；首次只支持 update 完成、rollout storage 已 clear、无 pending transition/异步计算的边界。

**完整保存集合**：物理 capsule/hash、model/backend/device/world/task/action/h/decimation binding 和 held controls；actor/critic/optimizer/normalizer buffers、固定配置、next_iteration；CPU/CUDA/Python/NumPy及实际自有 Generator RNG、determinism/TF32设置；episode_id/length/horizon、累计 reward、当前 obs/previous action/action-hold；真实 GRU hidden 或 LSTM(h,c)及 world 轴；rollout empty/step0、transition cleared 证据。RNN hidden 是普通属性，不能只靠 state_dict。沿用上游 save 的 current_learning_iteration 时须显式转换/核对 next_iteration，不能重复最后一次更新。

**恢复与文件发布**：临时完整目录写完且全部 hash/schema 验证后发布 manifest，旧 bundle 不动。wrapper 自有执行锁内先完成候选 learner/env 的全部可失败加载/分配和绑定校验，再以公开 restore(checkpoint, expected_revision=...) 执行物理 CAS；成功后切换训练所有者、丢弃旧 task frame 并 refresh。CAS 前失败应保旧组合；CAS 后失败须报告“物理已恢复、需恢复/停止”，不得声称跨 learner/runtime 的魔法原子回滚。不得访问私有 runtime lock，也不得初态重跑冒充 restore。

**固定案例**：两任务各在第 2 次 update 后保存，A继续一个 rollout+update；B在同一运行实例后继状态实际 restore 后执行相同后继。比较物理收据、obs/action/reward/done、optimizer步数、参数及 RNG 消费；短 GRU 分支在非零 hidden 保存，覆盖部分世界 reset前后。负控：坏hash/截断、错model/world/task/action/policy尺寸、缺RNG/RNN、stale CAS、非空rollout/mid-update保存、恢复后旧frame。

**通过**：绑定同设备/后端/依赖/确定性配置，明确后继轨迹与更新的逐项一致；不能把权重重载或seed重启叫精确续训，也不声明跨平台 bitwise。**停止**：实际非确定性给首个分歧字段/原始差值；两轮无缩小则停止精确续训资格，分别报告“可重新开始训练”和“精确恢复未通过”，不静默降门或重写 RSL。

## 文件责任与并行顺序（建议，父冻结后才成为授权）

| Owner/阶段 | 独占新增文件建议 | 不跨界修改 |
| --- | --- | --- |
| N1 环境与互操作作者 | requirements/training-cuda.txt；examples/mujoco_tyx_cuda_policy.py；tests/training/test_mujoco_tyx_cuda_policy.py | 旧 CPU profile、TYX依赖与物理 backend；环境安装仅父串行 |
| N2 wrapper作者 | src/phymeo/training/mujoco_tyx_env.py；tests/training/test_mujoco_tyx_env.py | 旧 ResidentVectorEnv/RslRlVecEnv、N1 fixtures |
| N2 RSL作者（可由同一作者顺序承担） | src/phymeo/training/rsl_mujoco_tyx.py；examples/train_mujoco_tyx_ppo.py；tests/training/test_rsl_mujoco_tyx.py | 不改上游 PPO；必须等 wrapper 输出/失败语义冻结 |
| N3 checkpoint作者 | src/phymeo/training/mujoco_tyx_checkpoint.py；tests/training/test_mujoco_tyx_training_checkpoint.py | 物理 capsule schema 与 N2 owner字段由公开 composition contract引用 |
| 父集成 | pyproject.toml 的可选 profile入口、training/__init__.py导出、正式计划/验收 | 不在子树并行改上述共享接缝；如需 backend窄修另行定owner |

建议顺序：N1资格与测试先冻结 → 父唯一串行 N1 → N2 wrapper/RSL在接口冻结后可隔离并行 author → 父按协议/两任务顺序执行 → N3边界实现与恢复。N3可提前读源码和设计 manifest，不抢写尚未稳定的N2对象。每节点小提交、完整失败保留、通过同源组不重跑；最终仅跑受影响相邻 HOST 兼容、fresh build及源码外 wheel 公共闭环，不借机扩大为旧全套重验。

每节点的完成证据必须是上述可运行案例的结果与明确边界。只增加文档、继续讨论更强保证、重复同源绿组或不断扩大案例而没有消除同一阻断，都不算推进；到停止条件就交具体缺口/选择，不进入无限精确证明。N1–N3通过后才根据真实复制/同步/审计计数选择性能工作，W256吞吐和对Newton性能比较另立计划。

## 审查入口

- 公开 task/action：src/phymeo/backends/mujoco_task_workspace.py、mujoco_device_controls.py、_mujoco_tyx_runtime.py。
- 物理恢复：src/phymeo/backends/mujoco_checkpoint.py、docs/development/mujoco-tyx-checkpoint.md。
- 已有同输入与负控：tests/test_mujoco_tyx_device_controls.py、tests/test_mujoco_tyx_task_workspace.py；训练兼容参考 src/phymeo/training/resident_env.py、rsl_rl.py。
- RSL/TYX版本、具体上游文件及官方来源链接沿用首段调研报告，不把本草稿当新增实测或安装证明。
