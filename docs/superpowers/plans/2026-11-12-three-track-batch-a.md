# 三主线 Batch A：设备来源、持久节点、球电机动态

状态：AUTHORING / TESTS_PENDING。实施基线79e3e089，2026-11-12，用户已授权按计划继续；无新截止。上层范围见[三主线路线](2026-11-12-scene-deformable-scale-roadmap.md)。本批只关闭实际通过的子节点。

## 交付与所有权

| 节点 | 作者 / 隔离分支 | 本批交付与文件边界 |
| --- | --- | --- |
| S1/S2 | mtyx_t2_tests / codex/p65-scene-boundary-20261112 | NEW场景来源adapter、独立测试、安装消费者和报告。父负责既有source/query、ResidentSceneBatch、transaction/public compiler接缝及必要设备anchor转换 |
| D1 / M2-P | mtyx_t2_source / codex/p65-deformable-bank-20261112 | NEW持久节点owner/buffers/helpers、三类真实形变与事务测试、安装消费者。复用原ordered节点核；不改旧布/R7/StVK语义。完整资源表先由父确认 |
| R1/首R2 | mtyx_t2_audit / codex/p65-ball-activation-20261112 | direct BALL FILTEREXACT与原normal/pyramidal接触；必要tree/actuator有限扩展和新多案例。父独占model/runtime/shared contact ABI及导出，按S0确定最小变更 |
| 集成 | root / codex/p65-parallel-modules | 共享协议、审阅、正常整合、小提交；全部环境/静态/测试/build/install与源码外验证串行 |

三个作者先用现有研究核对实际源码，给出简短S0接口/资源/测试登记；父快速裁定后继续纯编码，不重新做泛调研。未分配路径先报父，避免双写。

## 开发与验证顺序

1. 并行tests-first和隔离模块编码，全部标UNEXECUTED。父同时实现共同来源协议；公共结果保逻辑身份，物理row只作存储地址。
2. 逐片源码审阅和接口对齐。新scene查询复用真实transaction的底层bank/lock/runtime，不HOST重建当前几何；多anchor需设备侧转换到同一逻辑world参考框架。
3. 父唯一串行执行：fresh entrypoint-aware guard、源快照、精确选中静态检查、先解释后CUDA的新增及确实受影响测试。首失败保留，诊断后只重跑受影响节点；同源完整绿组不重复。
4. 新产品整合后一次构建，源码/wheel/installed成员字节一致；各新增公共消费者在源码外运行解释/CUDA，确认import来源和真实后继状态。该资格不称全包历史回归。
5. 更新结果、续作入口、全景与上层节点；记录实际能力、原失败、资源和未覆盖组合。执行/作者正式释放。

## 本批通过条件与边界

- S：四family、工具/自由物体、跨participant复用local IDs、稀疏多world、不同anchor、反注册/存储序、过滤、连续真实步、partial reset/restore、晚查询失败；核原shape/facet身份、同world全participant tick、旧epoch拒绝。独立query失败不倒滚先前已成功physics。
- D：原D1解析三节点、D2共享边弹性片和D3多world固定节点均真实变形；完整ordered约束/力输入，accepted双bank、单调revision、lease、末world/读回/收据/清理故障及恢复后真实后继。资源cap/cap-1与实际buffer账一致。仅节点生产者，不含D2查询或D3接触响应。
- R：不同轴/gear、共享关节、混合关节、非零入步速度和卸载，实际activation、力/运动、原接触以及同入步和连续独立MuJoCo参考；失败不部分发布。仅声明的直接球电机FILTEREXACT组合，不扩FREE+BALL/limit/dry/training。
- 两后端及源码外调用均按实际结果记。工程门沿已声明模型/单位，不追求精确相触，不看到输出后改fixture/质量/阈值。
- 同一假设两次诊断没有缩小问题就保存最小反例和可证伪下一问题，转独立节点；不以文档数量或单个演示关闭功能。

## 操作约束

沿用三位原agent；无新助手。所有项目import/collection/数值/probe/lint/formatter/compile/build/install/profile由root唯一串行槽执行。没有owner时不得启动项目命令。只用现有环境/pins；旧K1三未跟踪文件、旧作者稿、全部证据/Junction保留。无CI、push/mainmerge、发包、清理环境、G1长窗或无假设训练。功能完成按feature-commit-checkpoint分小提交，不覆盖WIP。
