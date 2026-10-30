# 有限厚度棱柱与有限面摩擦验收

修复产品 `33ec9ea`（I2 `e8c821d`）、最终报告 `a59182a` 已提交并释放数值/index。唯一 fresh review 的 0 Critical / 2 Important / 0 Minor 已由原实现者定向修复，父核对关闭。修复版实际 tyx interpret/CUDA 各 10 项通过；父 C++ 短例、构建和源码外 wheel 均已通过，产品5150c3e已推送，准确CI因CPU任务超时取消，交付尚未通过。

依据：[冻结合同](../superpowers/specs/2026-10-29-p65-thick-friction-design.md)、[选测计划](../superpowers/plans/2026-10-29-p65-thick-friction.md)、[验收政策](../development/p65-validation-policy.md)、[工程证据](p65-thick-friction-engineering.json)、[唯一审阅](p65-thick-friction-review.md)。

## 已实现范围

真实厚度体为三角节点上下偏移构成的完整凸棱柱，机器人为有限双面零厚度凸 patch。自动建立/分离、完整有限流形极点、固定材料 anchor、初态 held 与当前 terminal 两阶段摩擦、真实原生 J.T 反作用及最终实际路径 CCD 已集成。全局力矩、功率、材料、能量、实际 FP32 提升和失败事务门槛保留。

新程序支持坐标 local 轴的全 prismatic 树、任意合法固定 frame/origin、共享祖先与旁支、动态 FP64 三角节点及 distance 材料或 None。一般树和厚度转子仍不支持；host 几何/认证与 direct CUDA 原生执行分开，不宣称整步 GPU 驻留或 Graph。

## 已核对选测

| 证据 | 结果 | pytest 时间 | 版本范围 |
| --- | --- | --- | --- |
| host 新功能及直接旧消费者 | 60 passed | 1140.40 s | 42 新 host + 18 旧消费者，原快照 |
| 收紧原 tol 的 CPU 完整窗口 | 1 passed | 514.87 s | 124 实际逐步记录，原窗口快照 |
| 新候选覆盖及相关回归 | 4 passed | 9.80 s | 定向修复 |
| 斜 frame 正常控制及 anchor 预算 | 3 passed；另 1 passed | 11.64 / 0.89 s | 最终新分支 |
| 实际 interpret | 13 passed | 328.57 s | 修复前 `1fcd0e7`，7 新 + 6 直接旧消费者 |
| 实际 CUDA | 13 passed | 764.24 s | 修复前，与 interpret 完全同源 |

退出码均为 0。选测不是整分支全量。修复前源码 153 文件摘要为 `14c69ce1b64661639d382e8dbe08576ebc507909707f429a0640e291599c0bdd`；各版本逐文件摘要与命令在工程 JSON 和 SDD 报告中保留。CPU 原版本窗口不改写为最终源码全集；未受影响 canonical 窗口复用，变化路径已有定向回归和最终实际后端验收。

两侧夹压分别验证高摩擦静止、低摩擦滑落及步长减半；斜撞验证三个真实角点的独立分数解和材料变体；有限小 patch 完全位于大三角内部的场景验证部分重叠、退出和真实外力反向再进入。修复前两 backend 各 50 条实际窗口记录，实际状态与完整 history 在两次新建运行间一致；原 device tol 2e-5、能量/角动量预算 1e-5 保持。CPU 124 条只比较已保存的 9 个字段，未宣称未保存的完整 history 已作逐步 bitwise 比较。

斜 frame 暴露的真实候选缺失极点已在拒绝候选前补入，全部 owned anchor 从零原状态重放，原总迭代/重建预算不重置。θ=.3 测例是 FP64 近似变换输入的稳健性控制，不是严格代数同构证明。完整窗口实际误差减半由 raw 状态独立计算，没有用解析期望标量冒称实测。

## 审阅修复后的定向证据

Host 7 passed/13.10s、后段及事务 17 passed/433.27s、actual interpret 10 passed/134.13s、CUDA 10 passed/327.19s，原始 exit 均 0。最终 source/tests/scripts 摘要 cd6d77917dee8af09f13a430f4fdd4a5b744461874364a142aff0d8e6722d84b，153 source 摘要 df8c2ad9698f3f4c958b75ad9eee4135162fac1be490e56dd29cfce3df1f8d94。三模式同次 C actual/fullhistory 39 条，host 最大解析误差 8.595465450511597e-10，设备 1.8835067748801393e-6，均在原 tol 内；合力、原物理预算、FP32、CCD 和有限退出再入检查通过。没有新增步长/重建窗口或整分支全量声明。

原 diagonal square-corner 输入的候选可达性限制仍开放并记录，未把其失败改写为成功；新精确单顶点支撑正常控制不意味着任意点接触都收敛。R4 只明确离散候选成员判断的输入，完整理想体认证与原门槛保持，见唯一审阅记录。

## 尚待交付

唯一 review 两项发现已关闭；实际 C++ 短例、build、源码外新 native-pip 环境按未变 runtime.lock 安装及完整 installed smoke 均已通过，三个 raw exit 为 0。C++ 短例实际完成 52 launch、FP32 断言通过；153 个 wheel 源码文件由父独立逐字节复核相同，wheel SHA256 `7298fd1e2ff7429c8ff8a1cd0edf1f56b93840654bbb0497e12cfb011a359d68`。runner 墙钟分别 34.9892111 / 2.7996965 / 14.7534786 s，是工程耗时而非 pytest 或正式性能。工程前后冻结摘要相同。5150c3e已推送且远程SHA相同；其准确CI40950637127的8个CPU任务30分钟超时取消，required失败，Documentation40950637280成功。合同 `6e818f5` 的两项 CI 已成功，但不代表新产品 CI。

本切片不完成 H1。一般机器人、弯曲/自碰撞、真实有限驱动单爪承重/低μ滑落、IK/双臂交接放台和同源 Three.js 仍须推进。NUM005、TYX003、TYX004 仍开放；此阶段没有正式性能测量。

## 远程 CI 时限阻塞

产品 `5150c3e` 已推送并核对远程 SHA。准确 CI `40950637127` 已结束为 cancelled：8 个 CPU 矩阵任务均由 GitHub 确认超过现有 30 分钟时限，required 因 CPU 未成功而 failure；Documentation `40950637280` success。不能计为产品交付通过。2026-10-30 用户明确选择“保持 30 分钟，另行优化耗时”。90 分钟建议未应用并已撤回；完整矩阵、测试、required 和 fail-fast 保持。当前按剖析结果优化厚度精确算术的重复规范化，先证明输出等价并验证受影响路径，再推送取得新 head 的准确 CI；不得盲重试旧 head 或提前推进下一生产合同。

原始最小元数据及仅分类诊断分别留在 `artifacts/ci-5150c3e-checkpoint.json`、`ci-5150c3e-cancellation-metadata.json`、`ci-5150c3e-cancellation-classification.json`。未公开完整断言、参数或日志。该诊断只证明时限取消，既不证明数值断言失败，也不证明未完成的全量测试通过。建议补丁 `artifacts/ci-cpu-timeout-90m-proposal.patch` 仅变更一个运行时限和解释注释，未应用。
