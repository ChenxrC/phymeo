# C11 浮根 affine/mimic 常驻树与接触：正式限域接受

**LOCAL_VERIFIED / EXECUTION_SLOT_RELEASED，2026-11-08。** 产品与父共享接缝为 `17769c5`；后续 `593a81d`、`5856404`、`45868ba` 只修保存表示、审计排版和源码外观察器。完整执行依据为[冻结计划](../superpowers/plans/2026-11-08-resident-floating-affine.md)与[案例清单](../development/p65-resident-floating-affine-case-manifest.md)。不是整个 P6.5 或 Newton 功能对齐完成。

## 已交付能力

公开 FloatingAffineRobotTree 及 resident binding 复用同一浮根求解器。完整真实部件的 6+n 维 M/b/J 和所有接触 COM wrench 先保留，再由 S=diag(I6,P) 投影至 6+K；原图逐行约束反力、功和根六维余额进入实际诊断。根仅漂移一次，外层 storage 仅登记一次；full/reduced/root/control history、接触与几何在同一接受发布中保持一致。

已实际覆盖普通与 affine 混合世界、非零入步运动/即时外载、非对角惯量与真实 COM、不同 full n 与 independent K、原关系图、root/child 载荷与同树双动态反馈、当前姿态宽相、容量/晚读回故障/恢复/两 bank 拒步和合法重试。最大 full n=16 的不同 K 短例、W1/W4 代表窗以及后置 W32 均有独立记录。

范围仍是当前支持的关节/affine 图、HOST 有限 mu=0 离散窄相及 TYX 完整响应。没有全设备窄相、持续摩擦、CCD、同 world 独立 participant 联合求解或全部闭环机器人结论；普通浮根 n=0 已有的能力不被冒称为本次 affine n=0 新验收。

## 实际验证

以下均为父唯一串行执行，原始记录在 [C11 所在证据目录](../../artifacts/p65-generic-collision-20261107-01/)；分组联合覆盖按固定身份和实际日志确认，同源已绿节点没有为了计数重复运行。

| 层 | 实际结果 |
|---|---|
| 核心 interpret / CUDA | 每端 52 唯一节点：step 15、contact 5、transactions 32；每端 46 完整保存身份，14 个 preflight 无独立 gzip。551–564 分组完成。 |
| 代表物理窗 | A04-p1 和 contact W1/W4 保原 40 步及保存19后重放至21；lifecycle 实际正载→零载→再正载。563/564 均 2 passed。 |
| 相邻兼容 | 565/566 每端 6 passed，覆盖 ordinary 浮根、延迟/即时外载、固定 affine 以及宽相事务；不混入新核心52计数。 |
| 后置 W32 | 569/570 每端 1 passed，4 batch steps / 128 world-steps；同端原 W4 保存输出比较，无 W4 重跑。 |
| pinned Newton 1.7 原生 | 567 CPU、568 CUDA 各 A01-motor 1、A03-fan 4、A05-native-contact 4 步，共18步。保原生接触律，不喂 Phymeo 力；原行反力是完整余额观察，不冒原生反力传感器。 |
| 新 wheel 源码外 | 584 interpret、585 CUDA 每端 266 接受 / 2 预期拒绝 world-steps，其中 W4 262 接受、A04 4 接受。完整安装成员、loaded origins、独立材料/力矩/功与资源关闭核完。 |
| 静态 | 548/549 exact18 最终通过；550 限定 AST/import 证明明列局部改名和死赋值移除。保存审计与消费者各自最终静态通过，审计排版 raw AST 完全相同。 |

原始故障保持：

- 542 在 HOST 物理锚之后被测试自己的 FP64 等式断言阻断，不能当有效 RED；改用精确 Fraction 前提后，543 在缺失公开 API 处取得有效 RED。
- 553 前11个完成节点保持通过；n16-k1 在物理步骤和检查之后保存 opaque owner/registration 时失败，最后断言未执行，不能计该节点通过；n16-k15 未到达。593a81d 只为这两种进程内 token 保存明确不可重建标记，557 复验两受影响节点通过，其他物理字段完整保留。
- 580 wheel 观察器只累计活跃静态 site，错误拒绝产品保留的已注册零反力项。45868ba 从独立静态声明预置零向量，仍严格核全部 key、参考点、力和力矩。584/585 使用同一未改变产品 wheel 完整通过；580 原16接受 world-step 的失败前缀保留。
- 首 lint/format 与所有 RAW1 原输出、源快照和异常均保留。没有修改质量、运动、物理门或原数据来通过。

## 同次保存与发布证据

[587 保存审计](../../artifacts/p65-generic-collision-20261107-01/587-c11-saved-audit/audit/audit.json) RAW0，status=complete-saved-evidence、errors=0。固定20原任务、7751绑定文件、177194314计费解码/输入字节；这是来源、结构、协议与已保存独立比较的一致性审计，不是重新求解。

共105个 gzip 物理文件：102份完整输出（92核心、2 W32、6原生、2成功wheel），另有543有效RED前缀、580失败wheel前缀，以及553不能解码的原始不完整gzip。**不能把105全叫完整通过。** 无保存 preflight、只读标记及回调时序依原实际 pytest 断言，不声称能从 JSON 重建。

新 wheel SHA256 `0b851953ba6f04ad34105384c4cdfb2c040b84f3ef5746e79a943e2c68f77919`，1291063 bytes。[源码/构建/包内逐字节核对](../../artifacts/p65-generic-collision-20261107-01/c11-wheel-source-parity-01.json) 覆盖435产品成员，0 mismatch。原 C11 四个 URDF 已包含在当次源快照，C10 独立的 retrospective qualification 历史不改写。

保存审计 SHA256 `3bb37f7f94290511a026bc703c73ae0ccbcbfb3965371cf24b3940695b745fac`；正式原文件为权威。来源核对收据 SHA256 `4e65bc125be22162d0306854b2296b7fcf1d9e14bf3e06a72de6bdaa14ed411`。[执行释放](../../artifacts/p65-generic-collision-20261107-01/c11-execution-release.json) 为 10:42:31+08，matching execution=0。全部原 session 结束，禁止重启旧任务、重算已通过证据或重打同源包。

## 下一节点

继续 [K0 整族设备查询](../development/p65-device-convex-query-case-manifest.md)：先冻结同源姿态借用、十种 S/C/OBB/H 组合的距离和双侧材料见证、明确 UNKNOWN/预算失败，再由独立作者并行开发产品和案例；父持有旧接缝与唯一执行槽。完整 patch 和受载消费者分为 K1/K2，不把单个最近点当全部接触集合。
