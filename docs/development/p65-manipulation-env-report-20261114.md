# B3 异构操作 Env 源码资格

2026-11-14，第六批。产品快照 `135752e4d9050b89fbdb5fc08e1784e3e44ffe13`。
本批把既有公共 GPU 状态、动作、入步接触和成功帧连接成多任务 episode；不改求解器、碰撞、物理门或原八场景。
用法见[Env 公共用法](p65-manipulation-env-usage-20261114.md)，后续见[主计划](../superpowers/plans/2026-11-14-language-conditioned-gpu-manipulation.md)第14节。

## 已连接

- `HeterogeneousManipulationEnv` 借用 CUDA heterogeneous batch；只用公共 workspace、layout、action、step、reset 和 checkpoint API。
- 每 world 原 native ctrl/qfrc/xfrc 地址独立映射。策略动作先 tanh，再 scale/bias；padding 不参与动作，保存值为零。每步新建外力通道，避免继承上一请求。
- 观察为 Torch 独立 CUDA 副本；`action_mask` 随观察返回。目标上下文提供中英自然语言、物体标签及所有可见目标，不用任务整数代表语义。
- Scene/Goal、episode、history、长度、稀疏回报和 horizon 各自绑定 world。每次 step 只消费一份真实 accepted-entry；actor 端点与评价入步相位分列。
- 先发布结束 episode 的 final observation，再按世界独立 reset。terminated 优先于 truncated；关闭 auto_reset 时必须显式重置已结束世界。
- 一个不可变 capsule 保存原类型 physics bytes 与目标/history/reader cursor/episode。先验证来源和元数据，物理 restore 后再验证公共 entry 配对。后者失败会保留物理已恢复事实并要求恢复处理。
- 物理已接受后的评价或导出失败不发布 learner transition；旧 transition 不改。此时 step/reset/checkpoint/observation 阻止继续，仅显式配对 restore 可恢复。

## 验证结果

| 范围 | 实际结果 |
| --- | --- |
| checkpoint 传输、完整性、资源上界 | run06：22 passed / 2.53 s；使用一个真实 interpret 初态 capsule，无物理 step |
| binding JSON 类型区分 | run08：2 passed / 2.63 s；拒绝 bool/int/float 等值冒充同一 binding |
| CUDA Env 前六节点 | run15：6 passed，随后第七节点 fixture 失败；整次 95.88 s，原 exit1 保留 |
| 第七节点修复及后三节点 | run16：4 passed / 38.62 s，exit0 |
| 静态检查 | run17 四个新文件 Ruff 通过；run18 三文件格式通过、测试修复行需换行；run19仅格式化该文件，run20格式检查通过 |

共 **24 个 checkpoint 节点、10 个 CUDA 节点**取得通过证据。不是一次完整套件全部绿；前六个同源 CUDA 节点未重复执行。

CUDA 矩阵包含原八场景正常顺序32步、逆序4步，各自与独立CUDA W1逐字节对照；原八物体在32步中均出现真实支撑，入步与端点确实不同。
另覆盖多通道动作/异步producer流、NaN/Inf padding、Torch所有权、异步horizon、两种reset模式、换目标、同字节tick0 reset、输入拒绝、外部突变、新Env恢复与下一步精确重放、错误metadata，以及两种物理接受后的晚失败。
唯一模拟成功标志的测试仅验证 terminated/truncated 调度优先级，明确保存真实评价结果仍为失败，不能计为抓取或学习成功。

run15 的首失败：测试调用旧 `batch.set_controls` 只提供一个 world，该接口要求完整 world 集合，因此尚未造成预期外部修改。
仅把该测试输入改为所有世界的零控制后继续四个剩余节点；未改产品实现、物理模型、成功判据或精度门。
首 lint/format 检查及其修复记录均保留。Env作者实现只做格式整理；codec另补严格JSON类型比较。

## 保存与版本

父归档：`artifacts/mujoco-tyx-manipulation-env-20261114-01`。
20次命令的原始 command/stdout/stderr/exit、选中源码before/after摘要与diff保留。
12个完整压缩测试payload/物理任务capsule归档，共5,901,259 bytes，源/目的SHA一致；没有删字段或归一化重写。
`final-receipt.json` SHA256：`f04b8c417b7a8d11c43e1cd5d2b711d390389df1478ce0c4bb8d91d6d1a764af`。
最终 fresh matching execution=0，root串行槽释放；两作者只编码/文本审阅，未执行项目命令。

作者提交：Env `046350df` →父 `3af8c278`；测试 `3aafb274` →父 `e98d97bc`；codec `4b98f4a0`；最终资格修正 `135752e4`。
现有 Python3.13.3、MuJoCo3.3.7、TYX0.2.0a1验证wheel、CUDA Torch环境复用，无安装/构建/push/远程CI。
仅新增两个产品模块和两个测试模块，旧已验组件按既有证据复用，没有全仓库回归或新wheel资格。

## 当前边界

这是训练环境协议的源码资格，尚未完成真实抓取/放置/PushT诊断控制器、语言编码器、共享策略或优化训练。
成功评价仍是显式 HOST 消费者；CUDA物理/观察可用不等于全链路无HOST同步。
一动作对应一个物理步，没有动作decimation；padding还须在未来策略概率/损失中正确遮罩。
checkpoint摘要提供完整性与绑定，不是对恶意改写后重新计算摘要的任务历史作真实性认证。
Goal必须取自该Scene的原目录；Env拥有其借用batch的唯一写入口，直接外部修改后须配对restore。
原三项无关未跟踪WIP保持原样。P6.5、Newton功能对齐及八任务共享训练仍未完成。