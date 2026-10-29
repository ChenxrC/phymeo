# 有限厚度摩擦唯一最终审阅

唯一 fresh review `thick_friction_final_review` 已结束：**0 Critical / 2 Important / 0 Minor，request changes**。BASE `6e818f574d097ce415940289a58847d4c8ba49fb`，HEAD `f2d499914e109fdc9194a8f64c5e56544475ca8c`。两个发现经父读取源码、冻结合同和独立探针后确认，已由原实现者 e8c821d / 33ec9ea 定向 TDD 修复，父核对源码、原反例及原始退出码后关闭；未安排第二次 fresh review。

审阅覆盖全部 11 个新模块、5 个测试文件、导出和 smoke。依据为[生产合同](../superpowers/specs/2026-10-29-p65-thick-friction-design.md)、[验收](p65-thick-friction-acceptance.md)及[工程记录](p65-thick-friction-engineering.json)。两探针独立核对 307 文件摘要 `e3ee006f6ae79baa808b679ce4958464b5af28aef928f815a7dc685b603e8637`；观察到的父 HEAD `14097418` 只修改文档。审阅未改源码、测试、index 或 HEAD，全部资源已关闭，最终探针进程数为 0。

## I1 — 侧接触使用整个 patch 最近点，丢失选中支撑面的 signed gap

位置：`src/phymeo/solvers/_thick_contact_workspace.py` 的 `_current()`。侧法向可以位于 patch 平面内；anchor 越过有限边后，其整个 patch 最近点成为自身，当前实现因而给出零 held gap，尽管完整 SAT 已判定穿透。独立精确侧碰撞中，两个观察 sweep 均回到同一零冲量状态；独立候选经过实际 native 重放、全部位置/末速度/能量/角动量/映射门槛及完整 CCD，得到合法接受结果。

父裁定：属于支持域内 false rejection，没有证明 unsafe acceptance。按已冻结 g_gen 使用选中支撑特征的有限投影成员关系和有向支撑面，外部仍真实有限距离/inactive；不能无条件采用无限平面。新增自动响应回归，不能只验证静态几何。证据为 `artifacts/p65-thick-friction-review/side-feature-probe-02.py`、同名 `.log/.exit`，退出 0。首次探针调用参数错误保留为观察器错误，不记产品失败。

## I2 — 独立映射账本未检查实际合力及施加冲量后的映射误差

位置：`src/phymeo/validation/thick_contact_mapping.py` 的 `ThickContactMappingLedger.require()`。原始 Jacobian 系数误差很小，不代表乘以实际冲量后的误差很小；请求又与同一 supplied blocks 的聚合量相比，力矩/功率为零时不能保证合力平衡。独立合法几何中的小 block 扰动被大实际冲量放大，账本仍接受超过原阈值的合力误差。

父裁定：这是独立审核门禁的错误接受，探针未证明端到端 native 错误步接受。补独立实际节点合力与 world 请求的差，并用独立精确 expected J.T 施加真实冲量后逐节点检查映射；原尺度、容差、力矩、功率和 FP32 门槛保留，不能除以大冲量掩盖误差。证据为 `artifacts/p65-thick-friction-review/mapping-force-probe.py`、同名 `.log/.exit`，退出 0。

## 已评判但不作为发现的项目

父接受 reviewer 的逐项判断：坐标 local 轴限制属于冻结支持域；SAT 精确并列及 clipping、Fraction/二次域运算、CCD 面积/全区间/曲率/全部轴违例及未知拒绝、材料 anchor 转换/理想极点独立审核、terminal-only 的零 held 容量与完整 pre 身份、两相位实际法向容量、共享响应/聚合末能量、有限 bad trial 与执行异常的区分、缺失极点从零重放及总预算、native live/close 生命周期、版本化证据复用、附加导出/smoke 均未产生另一项发现。空 workspace close 不持有未关闭的原生资源。

一般机器人、厚度转子、完整 H1/P6.5 仍为后续目标。两项已关闭且实现者释放数值/index；父 C++ 短例、构建与源码外 wheel 已全部退出 0，准确产品 CI 尚待。已有 CPU/interpret/CUDA 结果保留修复前版本，修复后只按直接影响选测，不复制旧长套件或 900 计量。

## 父核对的修复闭环

I1 修复按初始选中支撑轴保留当前有限 support feature，R4 明确 g_gen 使用实际公开 port 的 FP64 点精确转 Fraction/Q 作成员判定，向外 signed gap 同时包围 actual/owned 值。独立表示误差仍受原 b 限制，完整理想棱柱 SAT/manifold/CCD/极点速度门槛不变。原边接触自动求解与独立合法候选均通过；新增运算精确的单顶点支撑控制通过。有限外侧仍 inactive，无 union、epsilon 或吸附。

R4 的限制明确保留：原新增 diagonal square-corner 输入曾迭代预算拒绝，实际点落有限点外而理想点在内；该输入和失败记录未删除，最终以有界成员/inactive 控制记录，不称自动收敛。撤回的是新增且未经全局收敛合同保证的断言；原 A/B/C、原审阅边接触和物理门槛未放宽。

I2 保留系数检查，补独立 expected J.T 施加实际冲量后的节点聚合误差及独立实际合力；原单位化尺度保留。原大冲量缺陷现被拒绝，合法控制四残差零；合力相消但节点映射错误及共享节点控制通过。

修复版 307 文件摘要 cd6d77917dee8af09f13a430f4fdd4a5b744461874364a142aff0d8e6722d84b，153 源摘要 df8c2ad9698f3f4c958b75ad9eee4135162fac1be490e56dd29cfce3df1f8d94。父独立核对 307 文件及 73 证据 hash 均相符，host 7/13.10s、后段及事务 17/433.27s、actual interpret 10/134.13s、CUDA 10/327.19s 原始 exit 全为 0。26 个未受 R4 影响的 host03 通过项保留原版本；旧长窗口不改写成新源码全量。三模式同次 C 轨迹共 39 条 actual/fullhistory 记录经独立只读审计，原 tol/budget、FP32、force、CCD 和退出再入均通过。
