# G2b-A：共享数组预算与 32 构型 CUDA 交付

2026-11-13。产品快照 `e1401fbf22396e802af18c3b8d7b6d1187d6ffbe`。
状态：**SOURCE_AND_INSTALLED_QUALIFIED / STRICT_PASS**，限下列已执行范围。
对应[实施合同](p65-g2ba-resource-gallery-20261113.md)、[固定案例](p65-g2ba-cases-20261113.md)与[公共用法](p65-g2ba-resource-gallery-usage-20261113.md)。

## 交付的公共能力

- 六类 solver-bank 的分配清单驱动实际数组 shape/dtype/bytes，覆盖双银行、scratch、padding、activation/CSR；alias 不重复计费。
- `MuJoCoTyxResourceBudget` 可由多个 runtime/facade 共享。构造前预留，真实分配记 current/peak，确认释放才减账；UNKNOWN_RELEASE 保留占用与强 owner。
- clone/reset/restore 检查 accepted 与候选重叠的额外峰值；预算不足前置拒绝。共享 owner 操作采用同一可重入锁，避免公开 admit 与 owner 锁的顺序反转。
- 异构 facade 扩至 32 models / 32 worlds。超过旧 8 models / 16 worlds 必须显式给预算。各模型原准入、DOF/接触/约束和 checkpoint 128 MiB 门保持。
- 32 个不同来源常驻同一 GPU，逐 source bucket 推进、整批共同发布。晚桶状态/下载/聚合失败保留全部旧状态，恢复与实际后继已测试。

本片计量的是 **已登记 CUDA 数组 payload**。HOST/native arena、context、JIT、allocator overhead、JSON、外部 device-control/task-workspace 的总峰值仍 UNKNOWN；不能把清单字节称为整个进程显存。完整 G2b-B 资源观察仍未交付。

## 已完成的验证

| 证据 | 实际结果 |
| --- | --- |
| 16 HOST/interpret | 40 passed，65.61 s；新预算/32构型/事务及受影响清理与33容量边界 |
| 17 CUDA | 24 passed，201.59 s；真实数组、六分配路径、32短窗、晚失败/retry与资源释放 |
| 最终静态 | 18 Python 文件 Ruff/format 通过；renderer 后续展示改进另有27/28静态通过 |
| 18–20 新包 | 源码外 build/install；536 产品成员 source/wheel/site 同字节，issues=0 |
| 21 安装 interpret | 32模型×12步=384 worldsteps，13帧；STRICT_PASS，正常关闭 |
| 22 安装 CUDA | 32模型×256步=8,192 worldsteps，257帧，2 s物理时间；STRICT_PASS，正常关闭 |

源码两组共64个唯一通过节点，其中新文件38项，其余为受影响兼容/容量节点；没有重跑旧 G2a/BALL-S完整矩阵。两个包外消费者全部 Phymeo 模块来自新 site。初次 lint/format 失败与修正前快照保留，功能验证没有失败或按输出改案例/阈值。

## 正式 CUDA 轨迹

RTX 4060 Laptop GPU，device 0。32 owner 的实际 CudaArray 指针、device ordinal 与 runtime 身份一致；累计24,576 kernel完成计数。四组各八种固定根R/P、固定根BALL、FREE R/P、FREE BALL，包含串联、分支、偏心惯量、不同关节轴、真实初速/电机与body wrench。32物理结构指纹不同，不靠名称/颜色制造差异。

固定根有重力；FREE组为零重力太空机构。geom mask过滤全部接触，因此本视频不授予地面支撑、场景间碰撞或任意机器人接触组合资格。CPU MuJoCo用于编译和独立参考，TYX真实step期间禁止原生step/forward等替代；视频只使用实际TYX返回状态。

原严格连续门全部通过，strict_failure_count=0。全轨迹最大连续q分量差 `5.800953241319462e-6`、v分量差 `1.2729747225215249e-5`；不同广义坐标单位不可混称统一米误差。同入步质量矩阵/J/力/加速度与一步推进原门也全部通过。未启用预登记宽功能门来隐藏严格失配。

solver-bank manifest/current/peak为144,124 bytes；正式轨迹未clone，故该峰值不代表恢复峰值。关闭后 current=reserved=unknown_retained=0。总显存仍UNKNOWN，不能称仅用144 KB。所有重叠峰值与UNKNOWN情形由独立短测试覆盖。

保存1028条完整typed事件（原逻辑4,681,912,134 bytes、gzip145,153,082 bytes）；另有257条轻量实际状态（gzip1,147,920 bytes）。完整输出、参考、预算与原输入保留。运行中保存占419.36 s，batch.step累计24.29 s；这是带同步/核查的观察值，不是纯kernel性能或Newton公平比较。

## 视频与证据

canonical证据目录：`artifacts/p65-g2ba-resource-gallery-20261113-01`。
`installed-closed-archive`保留80个同字节文件，包括完整两次消费、原模型、脚本和新wheel；源外原件仍保留。没有重解压大轨迹或重复运行producer。

- 原固定镜头：`video/phymeo-32-models-cuda.mp4`。
- 展示版本：`video-tracking/phymeo-32-models-cuda.mp4`；主镜头跟随当前真实几何，小窗保持原全程固定视角，画面明确标注。只改变摄影机，原q/v/act/time/tick、碰撞几何与物理输入不变。
- 两版均257帧、1920×1140、30fps、约8.57秒，显示2秒物理过程，播放速度0.234375×。renderer读取原XML和状态SHA，不插值/改q，不调用mj_step。
- 状态gzip SHA256：`e7dc845de90791170252455f95f3f78f9ff6dcad7f155ca89fbffc385cea954e`；两版render-receipt绑定同一个值。

初版全程固定视角使大行程slider不易看清，因此追加带固定小窗的摄影机版本；首版保留。没有为画面重新跑仿真或改动控制。原材质/几何尺寸保持。

下一步优先G2b-B：补完整HOST/native/context等观察范围与外部owner计量，再单独冻结更大W与同模型、同质量、同输出性能格。32构型通过不是融合异构kernel、跨world碰撞、全部训练组合或整个P6.5完成。
