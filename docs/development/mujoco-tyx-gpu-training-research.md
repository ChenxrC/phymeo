# 下一批：真实 CUDA 策略到公共 rollout 与训练恢复

日期：2026-11-11。状态：SOURCE_RESEARCH_ONLY / NOT_IMPLEMENTED / NOT_EXECUTED。
作者：既有 /root/mtyx_t2_audit；父独占全部项目执行。本轮仅读源码、已结束的小型环境元数据和官方资料，未 import、执行策略、数值 probe、测试、安装、构建或读取 G1 轨迹。
父源码观察点：d9e5bf3344519d57b39ade32acfb71f30ae2a411；本报告不改变正在运行的 G1 v2 run93，也不改变其资产、策略、观测或门。
本文件由父审阅后归入正式研究入口；原作者交接保存在本轮 artifacts。本文件是源码研究，不是新增执行 grant。

## 结论与最小路线

已有真正可用的 CUDA tensor/stream 互操作实现；下一步应连接它们，而非再写一套指针桥或训练框架。最小缺口是：一份合格的 CUDA Torch 依赖环境、直接消费 MuJoCo-TYX task/action 的 RSL adapter、把物理 capsule 与学习器状态组成的版本化 checkpoint。三项按此顺序完成。

现有 Torch CPU PPO、TYX 策略 kernel benchmark、物理 checkpoint 各自已有价值，但任何一项都不能单独证明 GPU PPO rollout 或完整训练续跑。GPU 数据入口也仍有明确的同步和 HOST 审计，不能称整个引擎零读回。

拟用已有受力机器人构建两个独立任务：hinge+slide 跟踪、有限平面支撑上的 slide 控制；不以 G1 一个舞蹈作为全部训练接入标准。第一批不改 PPO 算法、不训练 G1、不做多 GPU、不追求 Newton 性能胜出、不消除物理审计、不扩任意奖励表达式。

## 已核实的能力与缺口

| 层 | 当前源码事实 | 下一步判断 |
|---|---|---|
| 物理设备态 | MuJoCo-TYX 持有接受态 q/v、补偿 clock、真实 controls/forces 与 step receipts；公共 reset/restore 有 revision/CAS | 复用，禁止读取私有 storage 代替公共契约 |
| C2 task | MuJoCoTyxTaskSpec 从显式 q/v 索引产生 FP32 仿射 observation、负二次 reward、区间 terminated；1..256 features，<=256 worlds | 可覆盖最小机器人训练任务；还不是任意 contact/sensor/history reward 系统 |
| CUDA observation | TaskWorkspace.to_torch(frame) 已调用固定 TYX to_torch(copy=True)，生产者等待、Torch 当前 stream 同步；返回独立 Torch-owned CUDA 张量 | 可以直接接真实 Torch policy；每次是 D2D copy，不是共享物理 buffer 的零复制 view |
| CUDA action | set_controls_from_device(ctrl, qfrc_applied, xfrc_applied, expected_revision, producer_stream) 已支持相同 device 的 TYX CudaArray | 通过 tyx.torch.from_torch / stream_from_torch 包装实际 Torch CUDA action；无需 HOST action 中转来完成接口传递 |
| 动作验收 | own packet D2D copy 后完整 native channels 下载至 HOST 审计，pack/status 同步读回，再 CAS 发布 | 保留；“action 输入在 GPU”不等于“动作没有 D2H” |
| GPU互操作 | 固定 TYX 99f6df7 提供 tensor borrowed view、external stream、copy-only Torch 输出 | 源码可复用，尚需 CUDA Torch 的真实跨 stream/生命周期验收 |
| 已安装训练依赖 | 父 .venv-training 的 torch/version.py 为 2.15.0+cpu、cuda=None；RSL 5.5.2、TensorDict 0.15.5，Python 3.11.15 | 当前训练环境不能完成 CUDA Torch consumer。未执行新 availability 查询，不能从源码保证另一环境的 GPU runtime |
| 现有 RSL wrapper | RslRlVecEnv 仅接受 ResidentVectorEnv；actions.detach().to(cpu,float64).numpy()，输出从 NumPy 再构造 Torch Tensor | device='cuda' 也只是 GPU 网络加 HOST env 桥。不能原样宣称 C2/C2b 训练路径 |
| PPO算法 | 已安装 RSL 有 PPO、rollout storage、MLP/RNN、optimizer、TensorDict；既有 HOST 三任务接入可作 API 参考 | 复用固定 RSL，不复制 PPO、不新造 trainer |
| 物理恢复 | MuJoCoTyxCheckpoint 字节 capsule 保存接受态与 held controls、clock、tick、last_step 等，严格模型/世界/后端绑定 | 可作组合 checkpoint 的物理分量，不能替代学习器/episode/RNG/RNN 保存 |
| 训练保存 | RSL save 为 actor/critic/optimizer、可选 RND，再加 iter/infos；普通 RNN hidden_state 是 Python 属性 | env episode、RNG、当前 RNN 状态、rollout、框架运行配置仍须显式保存。禁止把 .pt 权重重载叫精确续训 |

直接读取的环境位置：
- 父 .venv-training/Lib/site-packages/torch/version.py、rsl_rl_lib-5.5.2.dist-info/METADATA、tensordict-0.15.5.dist-info/METADATA、pyvenv.cfg。
- C:/Users/38347/Documents/ChatGPT/phymeo/.venv/Lib/site-packages/tyx/torch.py、cuda_array.py、cuda_stream.py；同环境 tyx_compute-0.1.0a0.dist-info/direct_url.json 将 TYX 绑定到 99f6df7c0fd8fdeef77f97df83914da27b64b7ee。
- 项目 pyproject.toml training extras 当前为 torch>=2.6,<3、tensordict>=0.7,<1、rsl-rl-lib==5.5.2。宽范围不是已验证 CUDA 组合的 pin。
- 既有环境 run17/run75 元数据与上述 CPU Torch 文件一致；run52 是真实 TYX consumer 的两条路径比较，不能替代 Torch/PPO 证据。

## 实际数据流、复制及所有权

建议第一实现走安全的已有 copy-only 公共路径：

物理接受态 CUDA q/v
→ C2 refresh：设备 task kernel；HOST status gate
→ to_torch：三个 D2D copy 到 Torch-owned observation/reward/terminated；同步
→ Torch CUDA actor / critic / distribution / optimizer
→ FP32 contiguous native action tensors
→ from_torch 的 borrowed CudaArray + borrowed producer stream
→ set_controls_from_device：D2D owned packet + HOST 全字节 action 审计 + status gate + CAS
→ env.step：真实物理与完整 HOST diagnostics
→ task refresh / episode bookkeeping / 可选 subset reset
→ 下一条 observation。

具体生命周期要求：

1. TYX from_torch 的 CUDA view 持有 tensor owner，owns_memory=False；free() 只释放 view 的 owner 引用，不 cudaFree Torch allocation。stream_from_torch 也借用 stream handle，不销毁 Torch stream。调用方保留原张量与 stream 至 submit 返回。
2. submit 先同步 producer stream，再复制/审计/发布，因此返回后调用方才可释放或修改原 action。用另一 stream 写同一 tensor 时必须由调用方显式同步；一个对象持有 owner 不等于所有异步写操作自动有序。
3. to_torch(copy=True) 的结果可活过 workspace.close，不依赖旧 task bank；frame 自身则有 owner/epoch/revision 约束。控制提交、step、reset、restore 后应 refresh 取得新 frame，拒绝复用 stale frame。
4. 第一版不导出私有物理 buffer 的 zero-copy Torch view，不让 rollout storage 持有会被下一帧覆盖的 scratch。以后可预分配 Torch-owned outputs 再由 copy_to 写入；仍须保证 rollout 已复制其 transition，不把重复使用的同一 storage 当历史。
5. Task refresh 本身 source sync + status D2H；action submission 有完整 ctrl/qfrc/xfrc D2H；physics 现有 receipt/observer/失败门仍有 HOST 工作。subset reset 的 indices 也可能由少量 done mask 读回生成。各项都计数，不能只列观测 D2D。
6. action 与 physics 是两阶段：动作已接受而下一 physics step 失败时，q/v 不前进但 held action 保留。env 应记录 action_accepted/state_accepted；不得偷偷 reset、忽略 failure 或声称整个 action+step 自动回滚。若 physics 已接受而 task/包装失败，env 标为需恢复，不虚构物理拒绝。

官方 CUDA 文档要求不同 stream 间明确建立顺序，并保证 allocator 不会在异步使用结束前复用内存；本批先保持已有同步路径，不以省同步为验收条件。[PyTorch 2.15 CUDA semantics](https://docs.pytorch.org/docs/2.15/notes/cuda.html#cuda-streams)

## 三个实施节点

### N1 — 真实 CUDA Torch consumer 资格与最短闭环（中等，先做）

产物建议：隔离 GPU training dependency profile / lock、一个公开接口 example、一个 focused interop test 文件。优先零产品改动；只有发现公开接口本身缺陷才提出窄修。禁止在当前已验 CPU 训练或父 G1 环境中替换 Torch。

依赖动作：父另行授唯一串行槽后，先核现有 GPU driver、Python ABI、TYX 原 pin、空闲磁盘；选择官方 Windows CUDA Torch wheel，记录 wheel hash、CUDA build、模块来源、device ordinal、RSL/TensorDict pins。复用现有 TYX 源码版本，不混用另一个 Python 的 native extension。没有兼容已发布 wheel或空间不足时停止资格阶段，不构建 Torch源码、不用 CPU fallback 报通过。官方安装页将 CUDA build 与 CPU build 明确区分；具体组合留到该时刻环境资格检查确定，研究不虚构已验证版本。[PyTorch 安装入口](https://pytorch.org/get-started/locally/)

正例固定为两种独立公开模型：
- hinge+slide，W=1/8/32，相同模型各自独立初态、非零入步速度、非零控制与外载，8 个真实连续步。
- 已有有限支撑 slide 模型，W=8，同样 8 步，确认 task→Torch动作→真实接触力与运动共发布。
每例用真正 torch.nn 模块在 CUDA 上 forward，且权重、输入、输出都记录 CUDA device；先 deterministic MLP inference，不把它叫 PPO。固定动作缩放/限幅作为 task action spec，FP32 contiguous native channels；含一个非默认 Torch producer stream。HOST入口对照只接同一次计算出的 FP32 action副本，物理各自连续演进，不中途复制状态追随。记录完整输入/输出/controls/receipts 与各阶段计数；由已冻结现有门比较力与状态，原误差预算不放宽。

独立负例：CPU tensor、错误 dtype/shape/stride/device、非有限/不允许的 subnormal、跨通道 alias、stale revision/frame、提前关闭 view/stream；有限单GPU环境无法真实制造第二卡时明确该项未测，不能伪称跨设备通过。关闭 workspace 后已导出的 Torch-owned observation 仍可读；释放 view 后原 action tensor 仍有效；失败前后公共 capsule/state/held controls 是否变化按对应阶段准确核对。

成功：真实 CUDA Torch运算、原模型输入和 action bits 绑定、连续真实 physics过门、原公共所有权/CAS负控拒绝、close后owned资源归零；分开报告 D2D、必要 D2H、HOST诊断和同步耗时。无吞吐倍数或 Newton 优势门。
停止：环境资格失败先交精确缺依赖/ABI/空间原因；一次可归因修复后仍相同阻断则结束本节点，不改物理/动作范围来让 consumer过门。未过N1不启动PPO接入。

### N2 — 独立 MuJoCo-TYX device VecEnv + 固定 RSL PPO（中高）

产物建议：新 training/mujoco_tyx_env.py、新 training/rsl_mujoco_tyx.py 或同等隔离模块、短程训练 example 和 focused tests。现有 ResidentVectorEnv/RslRlVecEnv HOST路径保留，其兼容性只做受影响相邻测试。新 adapter 不通过旧 NumPy controls/observe callbacks偷偷转HOST。

公共 contract：
- 固定 world_keys顺序、observation/action shape和native坐标映射；torch.device必须与TYX一致。
- step(actions) 按 N1路径提交动作、推进一次env步；默认 decimation=1。先只启用已有TaskSpec的仿射obs/二次reward/区间termination，不增加任意reward DSL。
- 返回 TensorDict policy obs、reward、done；保持 final_observation/final_mask、terminated与time_outs分开。先保存入步与物理完成后的 final数据，再reset完成世界并返回其新初态obs；其它世界不reset或改计数。
- episode_id/length可先小规模HOST管理，明确读回；CUDA网络/rollout/优化器是真实GPU，不把这叫全设备控制流。
- 上游 init_at_random_ep_len=False。不能随机改episode计数而不推进/重置物理；随机化物理初态与domain randomization不在本节点。
- RSL actor/critic.reset(dones)只清done的RNN状态；env不自行清所有world隐状态。普通CPU包装异常与已接受物理失败须保留阶段标记。

固定上游细节不能隐去：已安装 RSL5.5.2 PPO.process_env_step 第154–157行按 time_outs 给 reward 加 gamma*transition.values，values来自入步act；它不读取 final_observation。第一版明确复用这一原生约定并写独立mask/target测试，不能宣称已经使用最终截断状态的 critic值。若以后改终端bootstrap，必须单列算法语义变更，不在GPU桥中静默“修正”。RSL现有PPO/runner本身直接复用。[RSL-RL 官方项目](https://github.com/leggedrobotics/rsl_rl)

正例：上述两个任务各 W=8、预声明seed、4次update×16steps/world；小MLP、固定学习率、FP32、关闭mixed precision/torch.compile/RND/distributed。真实 action→physics→reward进rollout，真实PPO loss.backward与optimizer.step；保参数前后SHA、有限loss/grad、optimizer step计数、实际device与非零物理响应。这是训练链路成功门，不是4次update足以学会任务或长期reward提升的承诺。每例至少包含两个世界不同结束时刻；另以小GRU做一次短rollout验证done mask只清对应world，尚不宣称完成长程recurrent PPO学习。

负例：交换同形状world行（依靠显式world-order绑定负控，不能期待设备猜出错误）、badaction全批拒绝、staleframe、term与timeout并发时既定优先级、reset后的finalobs被覆盖、action已提交后physics拒绝、physics成功后task读回失败。失败路径不把数据写进成功rollout，不允许自动freshreset伪续跑。
成功：两个任务都完成固定4次真实GPU update，transition数量/episode/reset/timeout/RNN mask独立匹配，现有力运动门保持；完整分类CPU协调、GPU计算和读回。只有网络在CUDA而action仍经旧HOSTwrapper时判本节点未完成。
停止：同一失败两轮仍无法归因到接口/环境/物理层，保存最小输入和失败阶段并停，不换任务参数、扩大训练预算或追求reward曲线；缺少任意任务表达式不成为重写框架的理由。

### N3 — 组合训练 checkpoint / restore（中高，依赖N2）

产物建议：新 training checkpoint composition/helper、版本化manifest和恢复测试；复用 MuJoCoTyxCheckpoint 与固定RSL save/load，避免修改物理capsule schema。第一版仅在**完成一次PPO update、rollout storage已clear、没有pending transition/异步计算**的边界保存；任意mid-rollout、mid-backward、多GPU checkpoint显式不支持。

保存必须组成同一事务manifest：
- 物理capsule bytes/hash、model/source/backend/world-order binding、task/action spec、h/decimation、实际heldcontrols；
- actor/critic/optimizer及注册normalizer buffers、固定训练config、learning rate/迭代位置；记录next_iteration，而非把lastcompleted重复执行。上游runner.save存current_learning_iteration，learn从该值开始，这个边界需要wrapper明确验证；
- CPU与CUDA RNG、所有实际使用的NumPy/Python/custom Generator state、determinism/TF32等执行配置；
- env episode_id/length/horizon、累计reward/历史、当前obs与previousaction/action-hold计数；
- actor和critic的真实GRU hidden或LSTM(h,c)，含world轴shape、None状态及done/reset语义。rsl_rl/modules/rnn.py的普通hidden_state不是register_buffer，不能指望state_dict自动保存；
- rollout empty/step0、transition已清的证据；RND、mixed precision、编译图、分布式仍关闭，不假称遗漏字段也兼容。

恢复先验证所有文件hash/schema/设备/模型/world/task/policy尺寸和RNG种类，在候选对象中加载policy/optimizer/env状态；持新 wrapper 自有执行锁用公开当前revision CAS恢复物理，随后一次切换训练所有者。任何可失败分配/反序列化尽量在CAS前；CAS后的异常要准确标为已经恢复或需恢复状态，不能冒称旧状态完全未动。恢复后旧taskframe/actionpacket必须失效，refresh新frame；不能通过重建初态+追跑替代restore。文件写入以临时完整bundle加manifest发布，旧bundle原样保留。

正例：每个任务在第2次update后保存；A继续一个完整rollout+update，B从同一个运行实例的后继状态执行真实restore后继续，比较obs/action/reward/done/物理receipt、optimizer步数、参数及RNG消费。固定同设备/后端/依赖/执行配置，先追求精确对应；不要跨CPU/CUDA/版本宣称bitwise。另一个短GRU分支在非零hidden状态保存恢复，含部分world reset前后，确认未reset世界history保留。
负例：坏hash/截断文件、错模型或world顺序、缺RNG/RNN、stale CAS、错误task/action spec、非空rollout保存、未声明mid-update保存、恢复后使用旧frame。每项检查拒绝时作用在哪一阶段，禁止部分权重加载配旧物理继续跑。
成功：真实恢复不是seed重启；指定update边界的整个后继轨迹与训练更新可复现；RNN分支保持真实hidden/cell；其他时点明确拒绝。PyTorch本身不保证跨版本、平台或CPU/GPU重复，因此验收域绑定具体运行环境。[PyTorch 2.15 reproducibility](https://docs.pytorch.org/docs/2.15/notes/randomness.html)
停止：若同环境存在不可消除的非确定性，不静默放宽为“保存成功”；报告首个分歧字段/原始差值，区分“可重新开始训练”与“精确续训未验”。两轮没有缩小分歧就停止，不把问题扩大成重写RSL。

## 源码入口与分工建议

以下路径均相对父checkout；本报告只读，未给这些文件授权写入：

- docs/development/phymeo-newton-panorama-2026-11-11.md：优先级与功能边界。
- src/phymeo/backends/mujoco_task_workspace.py：spec、refresh、frame验证、copy_to、to_torch（约434行）。
- src/phymeo/backends/mujoco_device_controls.py：preflight、borrowed/owned资源、prepare/_audit、submit/CAS；src/phymeo/backends/_mujoco_tyx_runtime.py：set_controls_from_device（约380行）、reset、checkpoint/restore、prepare_task_workspace。
- src/phymeo/backends/mujoco_checkpoint.py 与 docs/development/mujoco-tyx-checkpoint.md：物理恢复范围。
- src/phymeo/training/resident_env.py、rsl_rl.py：可复用episode/failure经验，不是设备MuJoCo adapter；examples/train_scene_ppo.py、train_contact_ppo.py：固定算法与已有HOST案例。
- .venv-training/Lib/site-packages/rsl_rl/algorithms/ppo.py：act/process_env_step 125–164、compute_returns、save/load 374–410；runners/on_policy_runner.py：learn 56–134、save/load 144–169。
- 同目录 modules/rnn.py：普通hidden_state/部分reset；modules/normalization.py：注册normalizer buffers；storage/rollout_storage.py：step/clear及隐藏态轨迹存储。
- 固定TYX源码 torch.py：from_torch约394、to_torch约441、stream包装约498、_wait_for_producer约593；cuda_array.py的external owner与free；cuda_stream.py的borrowed handle。

可并行的工作只限纯编码且无重叠：一位负责N1资格脚本/桥接案例，一位在冻结TensorDict/episode contract后author N2 adapter和负控，父负责唯一串行环境与验证。N3设计可提前读源，但实现依赖N2真实保存边界，不能先猜定API。任何执行都先父明确唯一owner；本报告没有启动安装、测试或新agent。

本报告的三个交付点是功能连接，不是“大规模训练已完成”。W256训练吞吐、随机化、复杂sensor/contact任务、多GPU、全异步、任意时点恢复与G1训练保留为后续；只有前三项实际通过后再按计数选择热点。若某节点只产出长文而没有进入其预声明的可运行正负例，应视为尚未完成，不能继续增加理论要求掩盖阻塞。


## 父审阅澄清

无标签且同形状的 tensor 行交换不能自动识别；负控仅对显式 world-order binding/metadata 作验证。新 wrapper 自己拥有执行锁并独占包装 runtime 的写入；原公开 action/restore CAS 只保护相应提交点，不存在供此方案直接借用的 public runtime lock。当前 G1 run93 已结束并保留失败；以上运行中表述仅描述研究时点，不是最新执行状态。具体实施以[后续计划](../superpowers/plans/2026-11-11-mujoco-tyx-gpu-training.md)为准。
