# Newton 功能差异补齐：2026-11-11 至 08:00

用户已确认本计划。北京时间约 00:54 开始，08:00 收口；06:30 后不扩大功能范围。固定基线 e4d00bfc，Newton 对照 1.7.0，MuJoCo 3.3.7 / mujoco-warp 0.0.2，既有 TYX pin 不变。目标是新增可调用、可验证的通用能力，不以单个 G1 动作或测试数代表整体对齐。

## 本轮实际交付（功能范围已收口）

本节为当前结果，后文是原计划。产品/测试冻结e3d15707；本批源码外新wheel、公共调用及保存证据已验。原G1全窗失败保留。06:30后仅文档、交付核对和dev同步，08:00结束本轮，不追加GPU依赖或第三次舞蹈。

| 节点 | 当前结果 | 尚未包含 |
|---|---|---|
| A1 公共物理恢复 | 已验并通过新wheel恢复后继续 | 完整learner/RNG/RNN组合恢复 |
| A2 G1完整动作 | 未通过：v2在46.29s触发原参考XY门；保存诊断结束 | 131.48s完整跟踪成功、原训练run配置绑定 |
| B1 凸体patch资格 | 已验十类配对及四类树来源 | 通用设备mesh/变形窄相、广泛CCD |
| B2/B3 设备patch真实受力 | 单页与world-complete分页已验，完整力/力矩/共同发布 | HOST转换与装配尚存；不是全GPU接触流水线 |
| C1 受载多世界 | W127/128/129/256限定库与失败隔离已验 | W256不是256台G1或大规模训练吞吐 |
| C2/C2b 训练数据面 | task/action、29执行器宽度与生命周期已验 | 真实CUDA Torch/PPO尚未实现 |

[正式验收](../../benchmarks/p65-parity-20261111-connections.md)列实际测试与新wheel范围；[G1诊断](../../development/g1-drift-diagnosis-20261111.md)保留同输入接触集合边界与部署来源未知项；[八方向全景](../../development/phymeo-newton-panorama-2026-11-11.md)列剩余功能。

下一开发直接从[GPU训练N1–N3](2026-11-11-mujoco-tyx-gpu-training.md)开始：独立CUDA Torch资格/真实策略消费→两个通用任务PPO→更新边界组合恢复。原三个agent可以在各节点公开接口和独占文件冻结后继续隔离author；父保唯一串行执行。接触边界与G1来源问题用小案例并行研究，不占用训练主线，不能继续重跑已绿组或把局部精度无限升级。

## 优先级、所有权和当前事实

沿用场景与机器人、碰撞、GPU与多场景三线；按用户后续指示，把训练与规模提升为最高主线。可微与训练、传感与交互、跨域编排、布与体软体、杆缆/MPM/流体继续列在全景，当前先解决前三线连接。

MuJoCo-TYX T4 与 G1 原策略 20 s 已验收，四窗口共 14000 接受步；完整 131.48 s、mesh部署、GPU端到端训练仍未验。旧 resident 与 MuJoCo-TYX 是不同能力域，前者的 K1/K2 和分页不由后者通过自动覆盖。既有 210 节点、新 wheel 及保存审计复用，不重复运行。

父为共享接口和唯一项目执行 owner；三个原 agent 只在各自旧 worktree 的新隔离分支编写代码/测试。所有 import、collection、数值、pytest、lint、format、compile、build、install、profile 均交父单槽执行，执行前 fresh entrypoint-aware guard。禁止辅助 agent、远程 CI、main merge、改 pin、覆盖 WIP。三个原 K1 untracked 文件保留，只可显式复制到隔离树作为参考，不静默纳入提交。

## 并行节点

| 节点/难度 | 作者 | 最小交付 | 验收与后继 |
|---|---|---|---|
| A1 公共中途恢复 / 中高 | 原 mtyx_t2_source | 接受实际已发布 FP32 状态的原样检查点恢复，绑定 compiled model、后端、布局和控制/时钟；不经初态 FP64 单位四元数门重新解释或归一化 | 多机器人、free/fixed、非零速度、继续一步一致；非法/错源/晚失败原子拒绝。已有 reset/clone 不重做。策略历史若由更高层持有，明确组合边界并补对应测试 |
| A2 完整动作 / 高、组合留出 | 父串行执行 | 原 G1 原初态/原策略/h=.002 连续 131.48 s；使用既有同源 harness，保持全部工程门 | 保全部实际失败；无 reset/姿态特判/参数调参。它不替代多案例库，也不以 raw exit0 冒所有功能完成 |
| B1 K1 资格与数据适配 / 高 | 原 mtyx_t2_tests | 复用已存在设备 patch kernels，核清旧 WIP，补 floating/affine accepted source 适配和相应身份/布局；普通树 K1 剩余矩阵先补具体缺口 | 点/线/面、S/C/OBB/H、旋转/反序/重入/边角/部分交面、权重与一阶矩、资源与陈旧来源；独立参考、interpret/CUDA。不得把 query witness 当 manifold 或载荷 |
| B2 设备 patch 力消费 / 高、条件后继 | 父裁定后沿原作者 | 仅 B1 具备设备几何资格后冻结最小 K2 消费，原双侧力/力矩/一次权重/共同发布不变 | 无 HOST 搜索偷换、受载多案例及晚失败；若前置未完成，本轮停在 B1 明确交付，不抢写共享力学 |
| C1 受载多世界分页 / 高、优先 | 原 mtyx_t2_audit | world-complete bounded pages，单 batch/单物理发布，保原每页 shape cap，每 storage 只装配和 solve 一次 | 单页等价，W127/128/129/256，四类树，世界/页顺序隔离、partial reset、site cap和最后页故障；真实受载而非自由推进 |
| C2 公共设备训练数据面 / 高、条件后继 | 父与原作者 | C1交付后冻结 action/observation/reward/done 受控工作区、stream/epoch/lifetime；先做不依赖完整GPU策略的可调用接口 | 不暴露可写 accepted bank；同/跨流、过期租约、reset、部分接受；CUDA Torch环境能力先核，未验证时不称GPU PPO。按余时完成最小片，不强装依赖 |

父在作者工作期间执行 A2，随后安排各交付的 focused 新测试；不让 coding 变成并发项目执行。旧共享文件的重叠改动先报父，父选择一个 owner 或集中接线。新报告必须区分 AUTHORING/TESTS_PENDING 与实际通过。

## 分批时间与停止条件

### 01:35 后 C2 首片裁定

A1/B1/C1 的首批作者交付已完成，等待父串行验证。原 A 作者在冻结 A1 提交后继续 C2 的独立切片：MuJoCo-TYX 自有设备 task workspace，按显式 native q/v 索引与 scale/bias 生成观测，按目标和非负权重计算二次跟踪奖励，按显式 bounds 生成终止标记。它直接读取接受 bank，输出不别名物理状态，绑定 owner/model/world/device/revision；reset/restore 后旧帧失效。工作区、kernel、专项测试和报告各一新文件，runtime 仅加最小准备/受锁读取接缝，不改物理核、原 Env 或 RSL。

首片保留 HOST action 与完整物理诊断读回，不称端到端 GPU PPO；CUDA Torch 尚未安装，不临时更换依赖。输出可由真实 TYX consumer 验证，Torch 导出仅允许正确同步的独立设备副本，环境缺能力明确拒绝。作者仍无项目执行权限。原 C 作者另行编写公开分页 benchmark/example，只有 C1 功能通过且时间允许才执行；不预先声称加速。

1. 00:54–01:20：落盘、明确 file ownership、现有实现与固定来源对照；长窗预检和串行启动。每作者先提交精简接口/案例和预期文件表，随后在允许范围编码，不另等逐阶段用户许可。
2. 01:20–06:30：A/B/C 隔离实现；父验证、诊断首错、集成。依赖未满足的后继不提前实施。性能测量只在相同物理输入和完整输出下，记录 wall time/传输/显存，不承诺超越 Newton。
3. 06:30–08:00：停止新增范围，完成已具备条件的验证、安装包公共接口检查、小提交、dev同步和差异全景。08:00 若命令尚在执行，准确记录运行状态与安全收尾，不把未结束命令写通过。

每个节点的成功必须是新增公共能力及正负控/原始证据；不是文档长度或测试数。连续两轮没有缩小具体失败或形成可验证增量，就停止该支线的扩张，保最小反例，转向独立项。精度在已有工程门内就推进；不回到无限精确相触证明，不改物理/质量/初态/阈值追舞蹈。

## 最终交付

更新八方向全景与已实现连接图，标实际完成、部分完成、阻碍及下一优先级；提供代码提交、测试范围、G1完整结果、真实受载规模和性能边界。完成的代码分小提交；源码外新 wheel 只验本批公共新能力，不重跑所有旧绿集。用户已授权 dev 分支同步，不合 main，不发包。

研究与入口：[当前续作](../../development/p65-continuation.md)、[全景](../../development/phymeo-newton-panorama-2026-11-07.md)、[训练优先级](2026-11-09-training-humanoid-priority.md)、[分页研究](../../development/p65-resident-contact-scale-research.md)、[设备训练研究](../../development/p65-device-policy-contact-research.md)、[MuJoCo-TYX扩展](../../development/mujoco-tyx-expansion-plan.md)。
