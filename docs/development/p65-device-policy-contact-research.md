# 设备策略与受载接触连接调研（2026-11-10）

**RESEARCH_NOT_IMPLEMENTED。** 这是下一批连接顺序和验收建议，不是冻结实施合同。本轮只读源码/官方文档并新增此文；没有项目 import、探针、数值运行、lint、build、安装、pin 或容量修改。父当前仍为唯一执行 owner。观察基线为父 7e378f7 与固定 TYX 99f6df7c0fd8fdeef77f97df83914da27b64b7ee。

今晚已经完成的能力应复用：G1 实际全 131.48 s 未跌倒，但峰值足力 14.9124 倍体重、参考 XY 偏差 1.2749 m，原两工程门仍失败；公共 HOST SceneBackend→Env→Policy 的 20 s 全门通过；CPU PPO 学习/重载通过；W256 混合运动各端 2048 world-steps 通过。受载 W256 的 512 shapes 被现 256 全局登记上限挡住。它们不是完整设备摩擦、GPU PPO 或受载 W256 的通过证据。

## 1. 已有接缝，先补什么

| 已核源码 | 事实与含义 |
|---|---|
| [ResidentContactPlan](../../src/phymeo/backends/_resident_contact_plan.py) 初始化材料检查；[bridge._law](../../src/phymeo/backends/_resident_contact_bridge.py) | 两处都明确拒绝非零 friction。只删其中一处不能接通，更不能把两个 guard 都删掉当成摩擦实现。 |
| 同一 bridge 的 ContactLoads.__post_init__ 与 sample_tree_contacts | group_count 仍硬限 1..17，而 group_index 已允许 0..32；plan 对浮根传 N+1，完整 G1 是 30 bodies，当前还会在这一包边界拒绝。32 关节无载通过不等于完整受载入口已通。 |
| [sample_loads](../../src/phymeo/solvers/sampled_box_contact.py)、[sample_patch_loads](../../src/phymeo/solvers/contact_patch_loads.py) | HOST 已有通用速度正则化库仑力，patch 权重只乘一次。G1 当前普通足地接触用这类速度律，不依赖固定材料锚点历史。 |
| [floating_contact_project](../../src/phymeo/backends/_tyx_resident_floating_contact.py) | 设备已把每个 body 的三维 COM force/torque 经完整 linear/angular J 投影到全部 6+N 自由度，再进入原全 M 求解，不是只支持一个法向标量。 |
| [RslRlVecEnv](../../src/phymeo/training/rsl_rl.py) | 现适配明确复制 action 到 CPU/NumPy，再复制 observation/reward 到 Torch。构造 TensorDict 或选择 CUDA device 不会消除这些传输。 |

**建议最高优先：先核对 17-group 旧包门与完整 30-body 来源，再保持 G1 已用物理语义完成非零 μ 的通用受载桥；不把“精确静摩擦/滚动历史”作为设备 G1 首个窗口的前置条件。** 速度摩擦没有有限静态保持区，零切速时切力为零；需要小滑动才能平衡侧载。固定材料站点迁移、滚动压力和完整静摩擦是另一个节点，不能假装速度律已经包含。

## 2. 力律、时刻和双方力学必须同源

取法向 n 指向第一参与者，入步相对速度 v=v1−v2、v_n=n·v、v_t=v−v_n n，压入 δ=max(0,−gap)，正则化速度 v_s>0。HOST 现有定义为：

- δ>0 时 F_n=max(0,kδ−c v_n)，否则 F_n=0；
- F_t=−μF_n v_t / max(||v_t||,v_s)；
- 每站 F1=w(F_n n+F_t)，F2=−F1，w 仅出现一次。

它保证瞬时切向功率不正，不保证任意有限 h 的离散总能量都下降。端点实际位移功、材料点功和入步功率×h 要分别保存，不能互相替代或引入新“能量补贴”。k、c 属于原 patch/样点的定义，不能因为点数改变再次乘面积或归一化。

每个站点必须保留真实两侧材料、normal、gap、weight、同一 world/application point p、两侧 COM 和所有者。v_i 用该入步的 v_COM+ω×(p−COM)，或等价完整 J(p)u；不能用策略帧速度、上一控制周期速度或把材料见证差当速度。双方 COM 力矩均为 (p−COM_i)×F_i；共用 p 使全世界合力和总矩相消。静态地面的反作用也要记录。

设备入口可以复用现 COM wrench→全 J 投影和原 full M/有界 drive，而不是重新定义叶节点质量响应。保留根–关节、跨分支耦合；执行器物理 M 与隐式步进 M+hD 不混为动能矩阵。同树动态双方与静态环境先按已支持 owner 域验收；跨独立 participant/batch 的联合求解另需共同发布接缝，不能从现同树成功外推。

时间协议不变：动作/目标可按 50 Hz 保持，PD 和接触速度律按每个真实物理 h 重新计算。load_time=t、accepted state=t+h；不能将步末速度反填为入步载荷来源。新入层、分离和再接触都从真实连续轨迹产生，拒步不能推进 held action、策略时钟或接触历史。

## 3. 建议连接顺序与依赖

| 次序 / 难度 | 先交什么 | 边界与依赖 |
|---|---|---|
| A：完整 body 包与非零 μ 混合桥 / 中 | 先为现 17-group 残留门作完整容量/来源审查和多关节正负控，再 HOST 几何和原 HOST 力律，全部三维双方 wrench 走已存在设备 full-J/full-M 响应；保原来源、功与拒步记录 | 最快验证通用受载能力。需同时核材料 admission、站点计费、全部诊断和失败路径。仍是 HOST/设备混合，不称 GPU 摩擦内核。 |
| B：站点力与投影设备化 / 中高 | HOST 只输出有界站点几何/材料系数；设备用原入步速度计算 F_n/F_t、双方 COM wrench、投影并推进一次 | 用 A 作同物理对照。先不用浮点原子累加的不确定顺序改变结果；空站点、分离、动态双方、根/child 和所有 guard 都保留。几何仍在 HOST。 |
| C：公开 Tensor 边界 / 高 | Torch-owned 连续 action/obs/reward/done 工作区，TYX 公共互操作；明确运行时、stream、生命周期和 epoch | 先独立无接触/单受载任务；逐项撤掉不必要 HOST readback，不读写 private bank 绕过事务。不依赖完整设备窄相先完成策略张量数据面。 |
| D：几何与规模 / 高 | 设备有限 patch、变量站点集合、完整接受检查；按 world 分片或有界容量化 | 512>256 是独立真实容量边界。先量模型/候选/站点/工作区成本，再比较可验证分片与扩容；不直接提高常数，不把 W256 自由运动当 W256 受载。 |

A/B 可按不同新文件并行作者，公共接缝仍父 owner、执行严格串行。C 可以先做公开缓冲合同/独立算例，不必等全设备几何；最终“端到端无 HOST 张量往返”要等数据面和相应接受检查都已迁移。此顺序是研究建议，文件范围和执行授权另行冻结。

## 4. HOST 几何暂留时，哪些量仍传输

现 [contact plan](../../src/phymeo/backends/_resident_contact_plan.py) 与 [floating packet](../../src/phymeo/backends/_resident_floating_contact.py) 在实际读函数计量 D2H/H2D，并同步原 runtime：

| 阶段 | 仍需传输/同步的量 |
|---|---|
| 入步几何 | 候选/边界、选中或整组 root/joint/FK/COM twist/tick；HOST 重建有限世界材料点、patch 和容量结论 |
| 现 A 力律 | HOST 三维站点力汇总的 FP32 body wrench、拒绝标记 H2D；normal/gap/系数本身在 HOST |
| B 力核后 | 站点 p/n/gap/weight/双方 owner/material 系数 H2D；设备可避免速度与 wrench 为求力而往返，但 HOST 几何仍需位姿读回 |
| 当前验收 | entry/candidate 状态、能量、实际作用点位移与功、完整 M/J/vector/残差等仍用于 HOST 诊断；不是只传一个 acceptance bit |
| 当前 RSL 连接 | actions.detach().to(cpu).numpy().copy()；观察/奖励/终止标记经 NumPy 再 torch.tensor；策略若在 GPU，还产生额外 H2D |

应分别报告大数组、状态标量、几何包、接受标记和 D2D 复制的字节/次数，以及同步与端到端 wall time。即使策略 action/obs 已共享显存，仍不能把 A/B 叫零拷贝仿真。维持工程容差与失败检查，再将可等价的检查改为设备 status kernel；不能仅为降低 D2H 删除验收。

## 5. 固定 TYX pin 的公开 Torch 能力

本机主环境 direct_url.json 的 commit_id 与 requirements/tyx.txt 同为 99f6df7…；以下签名来自该已安装 tyx/torch.py 的公开 __all__ 和函数体，未执行。固定上游 [互操作文档入口](https://github.com/ChenxrC/tyx/blob/99f6df7c0fd8fdeef77f97df83914da27b64b7ee/docs/pytorch-interoperability.md)本轮无法远程取回，本地同 SHA 源码为主要依据。

- tyx.torch.from_torch：借用连续 Tensor；rank 1–4、各维正数；支持 f16/f32/i32/bool，CUDA 另支持 BF16。FP64、int64、非连续和零维度不能默认接受。CPU 路返回 detached NumPy 视图；本身不构成 GPU 训练。
- tyx.torch.to_torch(array,copy=False,stream=...)：共享或显式复制；TYX-owned 普通 CUDA 数组通过公开 GPU array interface 导出并核指针，BF16 共享导出受限。已核 CudaArray 接口不是可随意假设的 __dlpack__ 路线。
- tyx.torch.launch、stream_from_torch、stream_to_torch、execution_context、dedicated_execution_context 都是公开入口。producer stream 必须明确传递；不能默认其工作总在 Torch default stream，也不能擅自关闭全局共享 execution contexts。
- 现 TyxWorldBatch 仍自有 runtime/双 bank，未公开训练用 mutable-state lease。K0 内部已有 owner/registration/state_epoch/生命周期校验的借用模式，可借鉴约束；不能把下划线函数或内部 device_ptr 当应用 API 直接调用。

当前已用训练环境为 Torch 2.15.0+cpu、TensorDict 0.15.5；CPU 学习/重载通过不证明 CUDA Torch+TYX 组合已验证。后续需要独立匹配的 GPU 依赖 profile，但本研究不改 pin 或安装。

PyTorch 官方说明共享存储意味着原地写可改变原对象，跨非默认 stream 需要显式同步和生命周期安排；所以仅相同 data_ptr 不能证明安全。[共享存储](https://docs.pytorch.org/docs/2.15/dlpack.html) · [CUDA stream/lifetime](https://docs.pytorch.org/docs/2.15/notes/cuda.html)。[TensorDict](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html)提供批维/设备容器，不能替引擎决定物理所有权或把 Tensor 变成不可写物理状态。

## 6. 真正 DeviceVectorEnv 还需明确的合同

1. **所有权**：引擎独占模型与接受/候选物理状态；policy 只能交 action。先公开专用 observation/reward 输出，不把 Torch 可写别名直接指向 q/v。允许显式 D2D 复制保留 rollout，报告其成本；“无 HOST 张量往返”与“绝不复制”分开。
2. **stream 与释放**：记录 producer/consumer stream、设备和完成事件；从策略完成到物理读取、从物理完成到策略读取都有 happens-before。借用 Tensor/owner 至消费完成，close/resize/pool-reuse 不得使悬空视图仍可用；同流、跨流、显式等待和延迟释放分别验收。
3. **epoch**：每份 action/observation 绑定 owner、model/topology generation、world keys、episode/reset generation、accepted step。双 bank 会复用，旧 rollout 必须是独立副本或受租约保护；仅 Python 对象引用或 TensorDict lock 不足以保护内容不被覆写。
4. **拒步与共同发布**：设备先写 candidate/status；按现 acceptance_known/accepted_worlds 规则发布。known all-False 可重试而不推进策略时钟；部分/未知接受不伪造 reward/done，保原 mask 并明确完整恢复/复位要求。GPU 标记迟到/读回失败同样不能先发成功 transition。
5. **reset 与终态**：先保存同 epoch 的 final_observation、terminated/truncated、final_mask，再 reset 选定世界；其它世界状态/动作节拍不变。RSL 对接的 float32 obs/reward 与 bool flags 明确形状；若计数需 Torch long，从 pin 可互操作的 i32 做显式设备转换并检查溢出，不偷换物理 tick。
6. **训练边界**：PPO 不要求对物理 rollout 反传，明确 detach 行为。当前无状态 PolicyRunner 的成功/失败节拍要保留；RNN hidden state、normalizer/RNG、异步 rollout 和中途恢复需独立事务。TYX 有 VJP 入口不等于接触分支及整个引擎已经可微。

## 7. 可复用验收，而非只看 G1

| 多案例族 | 必须测到的物理/连接量与失败 |
|---|---|
| 平台滑块/胶囊、μ=0/非零、切速 0、v_s/2、2v_s | FN、FT/v_s 分段、切力方向/瞬时非正功率；开离 FN=FT=0；权重只一次；HOST、interpret、CUDA 同律误差 |
| 偏心受载盒、转动胶囊、倾面与边缘 | 双方总力/总矩、各站真实共同作用点、根/关节响应和全 J/M；法向切换/歧义不选任意旧轴 |
| 完整树/浮根、root/child、同树动态双方与静态反力 | N=16/17/29/32 对应完整 B，错 B/越界 group 拒绝，不能只保留 17 bodies；μ 切力不能只改叶节点；cross-branch、根 COM/世界角速度、armature/D、原 drive cap；独立所有者联合域另测 |
| 持载→离地→再接触与接触出生 | 负载采样 t、发布 t+h、真实速度连续；每 h 求力、每 K h 更新策略；不注入参考 q/v，不假称精确静摩擦 |
| 无接触 + 受载混合 W1/W32/W256 | 非连续 world keys、仅一世界扰动/reset、容量与晚失败、empty sites；各世界完整 acceptance/reward/done 时刻；先解 512-shape 登记边界再叫受载 W256 |
| Torch/TensorDict 端口 | 同流/跨流、非连续/错 device/FP64/i64/空维拒绝、owner 提前释放、陈旧 epoch、双 bank 覆写、终态快照、部分接受；指针/字节/同步记录，不只断言 tensor.is_cuda |

先用小模型解析量和多案例短窗锁语义，再跑 G1 新设备受载短窗及选定持续窗口；原 HOST 131.48 s 同源证据复用，不重复长前缀。新后端若失败须保留首个输入、原始状态/双方载荷与接受 mask。峰值 8 倍体重和参考 1 m 门不因已有超限而悄悄放宽。

**完成判断**：先分别交付“非零 μ 设备响应已接通”“站点力核已设备化”“策略张量不绕 CPU”“受载规模通过”四项可调用能力与真实结果；未满足的连接仍画虚线。连续两轮只减少日志或包装 API、却不增加受载能力/减少实测传输/闭环一个失败，应停止该支线重新裁定瓶颈。本文件不授予这些实施或执行权限。
