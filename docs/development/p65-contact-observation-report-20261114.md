# M2-B2c/d：公共 GPU 接触观察与同入步成功帧

日期：2026-11-14。产品提交：`3ceb77cbc87396ba7c704ad433769ec2e8a46674`。
[实施计划](../superpowers/plans/2026-11-14-language-conditioned-gpu-manipulation.md)；
[公共用法](p65-heterogeneous-manipulation-io-usage-20261114.md)。

## 交付与范围

`prepare_device_workspace(include_contacts=True)` 已提供已接受物理步的入步 q/v/act、时钟、contact 身份/几何/完整载荷、
同入步 body origin 世界位姿/线角速度、接触点世界力/力偶，以及双方各自 body origin 的世界力/力矩。
全部设备输出从原 accepted-entry bank 和工作区 FK 构造，不从 HOST receipt 重建并上传。
原端点 `body_*` 与新入步 `entry_body_*` 独立开关、独立相位，所有桶成功后共同发布。
默认 contact 输出仍关闭；批次 capability 现在声明可显式启用。

contact-frame 三行转世界使用转置；geom2 为正，geom1 为反作用。
双侧 body-origin 力矩保留真实 torsion/rolling couple，再加各自力臂项，不误用 COM 原点。
无历史记录、成功空接触、inactive检测和实际受载分别表示。旧胶囊恢复无记录时不伪造过去载荷。
controls-only revision不产生新物理样本，partial reset/restore自动重新解析当前设备owner。
Torch 导出是独立 D2D 副本，修改与关闭均不改变物理状态。

`ManipulationSuccessReader` 从公共 readback 明确进行 HOST 成功判据适配：
绑定原 Scene XML、native FREE 物体及 named finger/support 地址，位姿/速度/载荷来自同一 input tick。
每 tick 只消费一次，跳步、改历史数据、伪 episode 变化和错配恢复均拒绝，晚 world 失败不部分推进游标。
它不是 actor 特征生产器，不实现 Env、语言编码或训练。

## 重置与恢复的通用边界

文本审查找到了“同字节 tick0”反例：初始首步和实际 reset 后首步可以完全相同，仅凭内容去重会丢掉新 episode，
而仅允许改 episode 又会把同一旧步消费两次。本批以公开 `frame.world_boundaries` 解决：
每 world 为 batch UUID、单调 boundary serial 和 operation。普通 step/controls不变，
成功 reset/reset-states 只更新选中world，restore更新所有world；失败不发布新token。
这是事务元数据，不是物理模型状态、任务ID或语言输入，不修改旧物理capsule格式。

真实 reset 需要新 episode；可以先读取无记录帧，保存 tick=-1 的 empty cursor，也可以直接推进首步后读取。
cursor restore 必须提供实际恢复后的公共 frame，检查完整 entry 指纹/tick/time后绑定当前token；
旧 cursor、最早未读取的空 cursor 都不能绕过新的 reset。物理restore若不同时恢复配对cursor会被拒绝。
下一批 B3 再将物理、cursor、目标/history与episode计数组成环境恢复协议。

## 实际验证

/root 唯一串行执行，原两个agent只在隔离树 author/review。
Python3.13.3、MuJoCo3.3.7、NumPy2.4.6、pytest9.1.1、TYX0.2.0a1；沿用已验环境与TYX site313，
每命令先 fresh entrypoint-aware guard。没有安装、改pin、远程CI、全仓库重跑或新wheel。

| 组 | 实际结果 |
| --- | --- |
| 新公共接触观察与SuccessReader，两完整文件 | run09：19 passed / 727.88s；interpret9、CUDA10 |
| 受影响原body观察与设备IO，两完整文件 | run10：52 passed / 148.69s |
| 最终8文件静态 | run12 lint、run13 format-check均0 |
| 最终释放 | fresh matching project processes=0 |

合计 **71个不同测试节点通过**。新验证覆盖：
- W1/W8、稀疏IDs、model/bucket/world独立排列、非首FREE、多树/固定体与原路由。
- 同入步 native FK/速度/接触 oracle；端点与输入故意不同；body origin与偏心COM故意不同。
- contact-map非slot身份、condim4/6非零力偶、inactive/empty、activation/capacity0 sentinel。
- 全 physical manifest 查询前后字节不变，封锁 HOST state/receipt/native FK 作为GPU producer。
- 最后桶失败、有效接触载荷NaN拒绝、旧frame保持与重试、controls-only、partial reset、legacy restore、下一步重放。
- CUDA Torch 独立storage、修改隔离、跨关闭持有。
- 原8个操作场景每后端各32步，同一entry SuccessFrame逐步检查并送入原目标/history判据；所有场景均出现真实支撑。
- 重复消费、跳步、错误episode、同字节首步reset、空cursor旁路、错配恢复和多world消费原子性。

原 pose/twist、native force rtol/atol=3e-3门保持；世界wrench另有预登记代数变换门。
32步真实支撑和接口正确不等于实际抓取、放置或PushT成功。

## 原始失败与交接证据

run02首次lint发现1个import排序，run03首次format-check发现6文件待排版；原退出码与日志均保留。
run08首次完整组在保存输入XML原bytes到JSON时失败（首节点，观察refresh之前）。
测试证据保存改为原XML bytes的无损hex，未修改任何物理输入、产品算法或阈值；run09完整19节点通过。
最后run11只将测试payload字典排版展开一处；7文件字节保持，产品源码全部与run09/10相同，不重复已绿数值组。

两个功能组各8份source-before/after与 `tested-source` 精确相同。
最终8文件SHA/blob、13命令原始退出码/log hash和36份保存文件归档在：
`artifacts/mujoco-tyx-contact-observation-20261114-01`。
保存文件合计12,339,571 bytes，包括run08原69-byte不完整失败gzip；不把它宣称为有效完整payload。
原Temp内容保留，新canonical压缩/胶囊副本逐文件size/SHA相同，没有全解压、删除或重复运行验证器。
`final-receipt.json` SHA256：
`2b8ccf0b914301233e587d20c35eb8201ef09d98ace3b9e554668d0553eb6987`。

没有性能胜出、零HOST往返、总显存预算、新安装包或完整操作训练资格。
下一批按计划第13节完成B3异构Env生命周期，随后T1真实可控性及L1/L2/T2语义与共享策略。
