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

## 公共接入增量（尚未完成 Transport）

`compile_native_scene` 保留一个完整 MJCF 的原模型、actuator、world geometry，
公开名称独立绑定，不能按 scalar_pd 资产重建。初态 NPZ 包含 11 个 integration
字段及根 source/model SHA256；include 资源变化也拒绝旧初态。该模式显式使用
原交付 FP32 quaternion 合同（1e-5），普通 NativeInitialState 的 strict_v1
仍为 1e-10，不归一化输入。球关节消费绑定、额外 Stage 物理覆盖仍拒绝。

`PhySolveRuntime.prepare_current_observations(fields)` 只创建一份派生 scratch
Data 和输出，不增加权威状态。两份捕获分别借用原 accepted/candidate bank，
warmup 使用当前 active bank；integration D2D copy 字节与刷新计数可查询。
scratch 原生总显存尚未测量，不宣称零复制或零额外显存。

idle `refresh()` 后必须 `complete()`，否则禁止 begin_execution；候选刷新使用
runtime.complete_execution，completed 后禁止追加。失败不发布新输出；close
确认完成后释放 Graph/Host，失败保留 closing owner 供 runtime.close 重试。
v1 只允许一个 live frame 且固定 stream；已有活跃 frame 时换流明确拒绝。
公开借用数组只在 owner 存活期内有效，输出是 provisional frame，不是独立物理
state。qM 保留原生 tiled/padded 格式，不等于已完成 dense mass matrix ABI。

初态、模型与默认合同主机回归通过；当前 forward 的 N1/8/64、F/T sensor
配置 admission、NaN错误恢复、固定stream拒绝和关闭重试逐项实测。F/T 此处
仅验证载荷保留/有限，不作为单位与物理量验收。P2PE原actuator C ABI在N1/8/64
已验证写/读/推进和局部reset。完整Transport的C ABI commit仍因原材质含纹理，
被native_visuals现有门禁拒绝；未改模型或去掉纹理规避。动力学P2PE绑定、原
25×2ms控制节奏、任务轨迹、独立安装与GUI均未完成。证据在实验根
`evidence/transport-public-*`，原失败日志保留，不能由历史原生9/16成功代替。
