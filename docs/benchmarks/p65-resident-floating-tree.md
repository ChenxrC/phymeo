# C9 普通浮根设备树：正式限域接受

状态：**LOCAL_VERIFIED_ACCEPTED**，2026-11-08。产品与完整 CUDA / wheel 源码基线
`4927ef9414754cb13bf3a2edab97f334025e49e8`；最终保存读取器已提交
`cd454cf9475de80cc41eb296011302675f8077c9`。本批项目执行已结束。
这是[冻结 C9 计划](../superpowers/plans/2026-11-08-resident-floating-tree.md)的局部接受，
持续总目标仍 ACTIVE，无 08:00 截止。

## 可以交付的能力

公开 `FloatingRobotTree` 使真实浮动根和普通 R/P 子树在 TYX interpret/CUDA
持续推进，0–16 个物理关节、B=n+1、D=n+6、Q=n+7，各维度分开处理。
完整根–关节质量、惯性项、COM Jacobian、外载与有界 PD 一起求解，
根没有伪电机，实际内部反冲与偏心运动保留。根姿态使用一次 exp-map 漂移。

它与既有固定普通、固定 affine、自由球共用一个 batch 的接受/拒步/restore。
完整检查点保留真实 local_origin_m，避免大世界坐标相减丢失局部运动。
root、child 和真实折叠工具的 S/C/H 材料直接由接受态设备 FK 供 AABB/BVH；
暖路径禁止 HOST snapshot/FK、动态包络上传和新设备分配，必要状态同步如实计数。

## 实际验证

| 层 | 已结束原任务 | 结果与边界 |
|---|---|---|
| HOST 合同 | 453 | 15 passed；固定独立首步锚与输入/资源拒域 |
| interpret 分组 | 459/460/469/470/476/479 | 86 个独立 device 身份的联合覆盖；原首失败与必要源码变更复验全部保留 |
| 最终源码 CUDA | 455 | 86 passed、15 HOST deselected，234.86 s，RAW_EXIT=0 |
| 受影响旧接缝 | 466/467 | interpret 8、CUDA 8 passed；混合恢复、设备宽相与 affine 晚故障事务 |
| 独立 Newton 1.7.0 | 463/464 | CPU/CUDA 各七例×40步，全部原检查通过；真实 FREE 根、实际 COM/solve-origin 读回 |
| 新 wheel | 465/481/482/483 | 构建、源码外独立安装、双端 W4 消费均通过；每端40步+21重放、65 batch attempts、258接受/2拒绝 world-steps |
| 同次保存审计 | 495 | 20 个原任务、7284 文件绑定、172 gzip、192752739 解压字节一致；新增物理步0 |

interpret 的 460 通过前缀使用主轴陀螺修正之前的源码；469 的两 pi 节点、
470 的四个异惯量/偏心/16关节原回归、476 的六个节点和479的剩余25项
按各自真实源码记录。明确的变更后重复身份只计覆盖一次。
**不把这组历史并集写成最终同一源码的整套 interpret 重跑结果。**
最终同源完整 CUDA，以及最终产品的双端源码外消费者另有实际验证。

原工程门保持：same-entry atol 5e-6 / rtol 5e-5，40步 atol 2e-4，
质量/完整方程 2e-5，原有限 PD 门 5e-5。结构、材料身份与恢复重放完整字段精确核对。
Newton 采用自身归一 Euler 姿态更新，与本项目 exp-map 的算法差别保留，
不声称任意轨迹逐位相同或性能领先。

## 首错、修复与证据

浮根专用有限中间量下溢归零，NaN/Inf 仍拒绝；旧严格算术没有被全局替换。
原 exact-pi 第二步的各向同性伪陀螺力用通用主轴公式修复，
没有按物体名、测试参数或坐标特判。详[算术裁定](../development/p65-resident-floating-arithmetic-ruling.md)。
476 的固定根统计键误用、以及485/488/492保存读取器的格式假设错误保留原RAW1。
读取器只按实际 producer schema、工作目录和公开计数纠正；
未补造缺失首失败文件，未改保存字节，也未重跑物理窗口。

- [固定身份 manifest](../../artifacts/p65-generic-collision-20261107-01/c9-saved-manifest.json)：SHA256 `095b3e963ad2d39d03bfa427d208cfbc6730713bf82a6aeee7c5155549e7f646`。
- [最终完整保存审计](../../artifacts/p65-generic-collision-20261107-01/495-c9-saved-audit-reader-v4/audit)：SHA256 `7b0a831eaac34e2036a5c1a0770c0dda650e05d5e185dab4c74e7e3a1c353fe2`。
- [新 wheel](../../artifacts/p65-generic-collision-20261107-01/465-c9-wheel-build/wheels/phymeo-0.1.0a0-py3-none-any.whl)：SHA256 `4022b455f2219aab6f61a66eee000c2cf3deab7ac3efba5916dcb48c56a08e57`，429 个安装产品文件、287 个加载模块来源已核。
- [读取器及原保存边界](p65-resident-floating-saved-audit-plan.md)。
- [原实现交接](p65-resident-floating-tree-implementation.md)、[原测试交接](p65-resident-floating-tree-tests.md)、[独立外部合同](p65-resident-floating-external-validation.md)保持原历史状态。

## 下一步及限制

C9 是普通浮根设备运动和当前姿态宽相；不是浮根接触、浮根 affine/mimic、
持续摩擦、CCD、全设备窄相、跨独立 participant 共同求解或完整 Newton 对齐。
C8 已有固定根同树双侧接触继续复用，不重跑旧验收。
现在解除 [C10 浮根接触](../superpowers/plans/2026-11-08-resident-floating-contacts.md)
的 C9 前置：父导入已审 NEW 交付、最小公开有效 RED、接旧共享接缝，
再验多案例、事务、真实原生接触及新安装包。其执行仍由父独占串行槽；
原 agents 可在隔离树继续无依赖纯作者工作。
