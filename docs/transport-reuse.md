# Transport 现有能力复用验证（2026-11-13）

原交付 SOURCE.json 声明基于 39cd6a6e 并包含未提交改动；当前集成检查点为
08c7a48c。原交付的 MuJoCoCudaSolver、device_state、device_observations 和
articulation 事务 API 并不在该检查点，不能只凭相同包名或版本字符串视为已接入。

## 已完成的实际验证

独立消费者 consumer/transport/reuse-probe 保留原 PHYMEO 算法/状态事务/观测、
原 Transport controller、checkpoint 和成功判据。仅绑定当前 physolve_tyx 和
逐实例 frozen Host，启用条件 Graph，并准备已有 frontend workspace。
没有导入 Newton、Warp 或 mujoco_warp；没有改变六个 CUBIN。

原模型 19 项数组和初态 11 项数组散列均与交付报告一致。N=16、700 ticks、
25×2 ms 的 17500 次批量推进全部接受，零拒绝；16/16 finite、16/16 OSC 满秩，
原 success 判据9/16、payload9/16、trash16/16。正常关闭且进程退出0。
热循环原 GPU guard 通过，TYX 计数 H2D/D2H=0、graph_builds=0。
这不意味着无设备复制，原事务仍以 kernel 复制状态/工作区。
冷准备337.095s，控制周期109.327ms，3658.75 env-steps/s；不是 P2PE 性能。
证据在隔离实验根 evidence/transport-reuse-n16-v2，原始失败和完整记录保留。

## 接入限制归属与最小复用边界

- native_assembly 的 child.option 非默认拒绝是资产合并政策；完整场景应复用
  MuJoCoCompilation 的完整模型快照，不经 scalar_pd 重建原 actuator。
- template/runtime 的全部 nsensor 拒绝属于当前资格范围，原生 Host 已能保留原
  四个 F/T sensor 运行本案例；这不是对所有 sensor 输出的独立测量资格。
- NativeInitialState 的1e-10检查比原交付FP32合同1e-5更严；不能归一化原输入
  或放宽任务判定。正式入口应显式承接原状态精度合同，保留其他入口默认检查。
- 原 device_observations 在当前 live state 运行 forward，提供 owned qM/bias/
  axis/anchor/site/contact 输出。复用其采样语义和已有 native forward，不能用
  solve-time 残留字段代替。接入当前 generation/canonical stream/error gate。
- 原 source 采用 live/safe/scratch；当前 P2PE 使用双 bank ticket。正式接入需
  选择唯一权威状态，不叠加两套事务或让消费者读取 PHYMEO 私有属性。
- P2PE 公开原 actuator 命令和动力学字段后，消费者保持原每2ms OSC以及20Hz策略。
  不修改为4×10ms或25子步保持一次力矩，不改原控制器算法。

本记录只证明原生能力复用可行。当前安装 P2PE 的完整模型/动作/动力学桥接仍未
完成，真实 Phyzoic GUI 未执行；不能据此声明整条消费链完成。下一步应按完整模型
与原生索引映射、控制/观测公开绑定、C ABI无头、GUI的顺序接入。
