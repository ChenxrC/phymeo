# P6.5 仿射特征整段净距认证验收

状态2026-10-29：第16项有限能力已完成本地验收，准确远程交付待闭环。产品`700fd325c3a61b20202a4994c25336e3bfe9524a`，最终测试/计量快照`61ea634`，126源码hash `67ae299638326bf2018ee781a3bceadb5f54fbd84a7238f0740d9c27be3a6b60`。依据[冻结设计](../superpowers/specs/2026-10-29-p65-linear-sweep-design.md)与[实施计划](../superpowers/plans/2026-10-29-p65-linear-sweep.md)，工程数据在[JSON](p65-linear-sweep-engineering.json)。

## 已验收合同

`linear_feature_sweep`对FP64点/三角、反向、边/边端点定义的精确仿射轨迹，在归一化t∈[0,1]上认证严格正clearance_m。同步顶点投影在区间端点的有理下界证明整段distance>=clearance；浮点closest只提供权重候选，违例见证在真正有理轨迹上重建。不能以中点舍入或有限采样代替证明。

返回certified、violation或unresolved；后者不能作为安全。有界深度/区间预算、精确可表示时间分区，早停仍完整保留已认证和未决区间。violation的时间是严格违例见证，非首次TOI；允许接触等号，零厚度拒绝。退化/候选不足显式保守处理，结果是owned不可变标量/tuple，不保存外部可变别名。

两个非递布小场景独立验收：探头穿板（含根中点之外t=.25穿越）、平行等号滑动；两根独立杆件穿越和安全转向。安全转向距离²=(1-t)²+t²>=.5，而clearance=.5，整段没有一个固定分离方向；预算1未决、128/20细分认证，避免“一律未决”蒙混通过。另验证次正规阈值、ULP中点、最大深度52、变换/同步大位移/顺序、退化、完整分区及异常传播。

薄SceneVerifier检查真实before/after候选，最后子步violation/unresolved和资源close失败均全局恢复state/history/输入，retry=fresh。对仍不安全的同一输入，retry与fresh都拒绝；close故障移除后正常接受。实际interpret/CUDA短测各2条确认原生执行与独立端点值，几何认证仍在host。

## 验证与版本

- 缺API2RED、细分2RED、source-smoke1RED后实现。边界58条和最初4条事务立即GREEN，不虚称都曾RED；阈值浮点平方1RED、舍入有理见证2RED、取消验证拒绝2RED负控制均在恢复源后通过。
- [唯一fresh review](p65-linear-sweep-review.md)为0Critical/0Important/0Minor；另做48条三维有理路径、晚段违例与右半区深度52探针及4条事务，均通过，所有探针退出。没有第二次review。
- 最终main/core311/nativepip相关回归分别241通过、4未选；Ubuntu新host67通过、2未选，使用已有依赖，非干净安装。actualinterpret/CUDA各2通过、4未选。没有把这些选测称为新全分支全量；准确远程CI另核对。
- 126文件源码外wheel逐文件一致，新native-pip安装及solver/sweep smoke通过；ruff/docs/导入边界/compile、uvpip/锁导出通过。无新tyx kernel，不重跑已关闭的旧1950 CUDA轨迹或未修TYX-003压力。
- 所有数值和审阅退出后，三种host路径各3×100，共900计时/909含warmup成功，状态/区间数与源/脚本hash一致；[PERF-018](../performance-bottlenecks/PERF-018-linear-sweep.md)只记录，不优化。

## 仍待实现

该函数认证的是调用方声明的仿射几何路径。它不枚举碰撞对，不选择厚度侧/零距离法向，不维护接触身份/摩擦history，不产生接触反作用，也不认证机器人FK弧或返回TOI。不能把自由预测证书用于不同的最终接受候选，不能用机器人端点连线冒称其旋转轨迹安全。

下一阶段应交付能产生真实反作用的自动建立/分离闭环，并把必要侧/法向/身份/history及FK路径规则一起冻结；不要继续无限堆静态工具。弯曲/自碰撞、单爪承重与低mu失败、双臂交接和Three.js仍依序推进。框架有限能力不等于H1或完整P6.5完成；NUM-005、TYX-003/004保持开放。

准确交付5c352006442641c21faf1684fe6427b10960a930的CI40756643301及Documentation40756643278均已核对completed/success；本阶段交付闭环，不代表完整H1或P6.5完成。
