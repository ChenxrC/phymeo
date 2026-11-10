# B2：设备接触面进入受载事务

状态：2026-11-11 03:36，B1 新增资格矩阵 interpret/CUDA 各 96 项已通过，解锁 B2 纯编码。原 B 作者独占下列六文件，父独占公共参数转发；全部项目执行仍只由父串行安排。06:30 后不扩展本片。它是[本轮计划](2026-11-11-newton-parity-until-0800.md)的条件后继。

## 要补的连接

设备宽相和凸体 patch 已有独立查询接口。当前受载 resident 路径仍调用 HOST `contact_patch_sample`，查询结果不能直接冒充实际载荷。B2 要让合格的设备 patch 进入现有完整 Jacobian、质量矩阵、双侧力矩和一次状态发布。先支持单个 contact plan；分页接入在共享接口稳定后扩展。

当前代码依据：`backends/_device_convex_query.py` 的 `query/_patches`，`backends/_resident_contact_plan.py` 的共享力学事务，`backends/_resident_contact_bridge.py` 的采样与材料点输运，以及 `geometry/contact_patch_adapter.py` 的严格数据构造门。以下保留工程精度，不要求理想数精确相触，也不放宽既有几何门。

## 必须先冻结的三件事

1. **同一个事务发布。** 把凸体查询拆为暂存结果和无失败操作的最终发布，消费 `_begin_attempt` 已暂存的同一宽相候选/source loan，不再做第二次宽相。设备宽相、凸体结果、原始/effective patch、力收据与 accepted bank 在同一提交点生效；query 仍绑定原入步 epoch/bank，不改标为步末。后续材料转换、装配、求解或收据构造发生异常，不能留下看似成功的新查询。独立 `query()` 保持原公共语义。失败的资源计费保留。
2. **显式表示转换。** 旧 `ContactSite` 要求 FP64 单位法向、重心和积分权重严格归一，不能把近似 FP32 字直接塞入，或降低旧构造门。仅允许确定性表示转换：保全部原始字；对非负、有界且已合格的重心/积分权重按各自原和归一，保留零权重项和全部 ID；归一后的距离行法向是唯一共同法向，站点法向作一致性检查；按原材料 ID、同入步 `ContactShapeFrame` 重建材料核点，圆形 offset 使用原半径的 `±radius*n`，hull offset 为零。公共施力点取双方中点，gap 为法向投影。不能做新的 HOST 接触搜索、按力选点、截掉负权重、增删接触点或修复未知几何。
3. **转换有证据和限界。** 收据同时保存 raw 与 effective 数据及每项差值。重建点与设备原点、gap、offset、法向和一阶矩必须满足 B1 原有相应工程门。所有比较使用原局部几何尺度，巨大世界坐标不能放宽容差。shape/world/owner/topology/epoch/active bank、原局部点与实际上传点的来源分别绑定。验收还必须独立检查 raw 和 effective 对原几何参考均满足原门，不能把两段误差预算重复使用；这项测试不引入 runtime HOST 几何搜索。已有门不能覆盖的情况明确拒绝，先保最小反例，不用未声明的额外容差解释。

## 载荷和失败语义

- 所有 candidate 都有原显式材料律；只有合格 `RESOLVED` patch 可消费。`UNRESOLVED/ABSENT/RESOURCE_LIMIT` 不能落回 HOST 搜索或算作零力。设备资源状态没有精确 required-site count 时，失败整个尝试，不能编造逐世界溢出计数。
- 真实几何站点先全部计数，含零力站点。既有逐世界 site-cap 保留，失败世界保留原状态及上一接受收据。
- 原正常 `REJECTED` 状态允许其他世界前进；不能把它误改成整批回滚。上文的整批不发布指异常、未知几何和全局资源错误，完整保留 C1 已有区别。
- 用真实入步 COM 线速度与角速度计算共同施力点速度。原 `sample_patch_loads` 只施加一次积分权重；双方力相反，分别对真实 COM 计算力矩。各 body 的所有贡献完整汇总后，每 storage 一次完整求解。
- 候选末态只输运原材料点，不能重采样改变入步力。保留 application/material work、量化误差、原控制、真实输入、完整动态与静态反作用。
- 普通、affine、floating、floating-affine 使用各自完整物理 FK/J/M；不能把约化自由度当 link 索引。第一片仍是设备几何加 HOST 载荷装配，不声称全 GPU 接触或训练。

## 开发顺序与测试库

| 子项 | 交付 | 必须通过的独立检查 |
|---|---|---|
| B2a 表示适配 | 一个有界、无搜索的适配器及 immutable 转换记录 | 全十类无序 S/C/OBB/H 组合中 B1 已解析的点/线/面；反序、旋转、部分交面、稀疏材料 ID；原始字段不变、完整双射和一阶矩；伪造来源/权重/法向/不完整 patch 拒绝 |
| B2b 共同发布 | query 暂存接缝、可选设备 narrow 路径、原力学复用 | 后置 patch 失败、最后载荷失败、求解后收据失败均无部分发布；计费保留；重试和 stale result；原 HOST 默认路径受影响专项回归 |
| B2c 真受载矩阵 | 多初态连续步，含真实入步速度、执行器和外载 | 四类树、动态–静态及双方动态，双侧力/COM 力矩、权重一次、完整 J/M、逐世界隔离；同一输入 HOST 与设备几何路径的工程误差对照；不可仅验证 query 状态 |

首片候选七文件：新 `backends/_resident_device_patch_bridge.py`、`tests/collision/test_resident_device_patch_bridge.py`、`tests/scale/test_resident_device_contacts.py`；修改 `backends/_device_convex_query.py`、`backends/_resident_contact_bridge.py`、`backends/_resident_contact_plan.py`；父持有 `backends/tyx_world_batch.py` 的公共参数转发。其余共享物理核、bundle 和严格 `ContactSite` 不改。共享 mechanics 的默认 geometry hooks 不改变旧 plan/bundle；设备 query 只在单页 plan 注册完成后创建，不能由 bundle 共用的准备函数隐式创建。所有转换记录和收据在 swap 前构造。

拟定公共参数为 `narrow_phase="host_patch" | "device_patch"` 和 `device_patch_site_capacity=None`；设备模式的 `None` 使用原 4096 上限，原 `site_capacity` 仍管每世界真实载荷点数，原 `candidate_capacity` 仍管宽相。首片不开放外部 query-plan 注入。编码前根据已集成 C1 版本冻结精确范围，不并行编辑共享接缝；仍不得修改旧三个 K1 untracked 文件。

首轮专项验证保留原始失败。两个修复循环仍不能缩小具体缺口时，停止扩展并保留最小反例；先完成已独立可验收的检查点、训练数据面与分页。判定完成依据是公共受载能力及共同发布证据，不能用计划行数、组合数或单个舞蹈作为验收。

## 本轮编码授权与交接

沿用原 `mtyx_t2_tests` 作者的已清洁 worktree，新建隔离分支指向父冻结提交，保留原 B1 分支，不 reset、不覆盖原产物。作者先编写两个新测试文件，再实现上述三个修改文件和一个新 bridge；父随后补 `tyx_world_batch.py` 的两个参数及转发。原 C1 bundle、物理核、`ContactSite`、依赖、七处排序和 CUDA 清零/保留字修复均不改。

需显式覆盖独立 query 的原发布语义、共享 `_ResidentContactMechanics` 默认 HOST 路径及 C1 bundle 兼容。新 staged query 的尝试/费用可见，未共同提交前 `last_result` 与成功计数不变；后续装配/求解/收据异常不得留下新发布。构造所有 immutable 记录后才交换接受 bank，提交后不可再做可能失败的表示转换。B1 的积分权重本已严格为正；材料重心中的合法零权重和全部 ID 保留，不能通过删点绕过旧构造门。

固定测试库先保多形状点/线/面与四种树布局，至少两个有非零入步速度、驱动及外载的连续步，真实双侧力/COM 力矩与逐世界拒步。原始设备和有效材料点分别对独立参考满足原门；禁止只比较两个共用转换器的输出。拒绝 HOST `contact_patch_sample` 的 monkeypatch 作为设备消费路径证据；HOST 默认专项仍真实执行原搜索。晚失败、最后世界/站点容量、未知 patch、stale/reset、重试与关闭必须覆盖。

作者交清洁小提交、精确六文件清单、手工枚举的测试域和未执行标记。无需另行逐阶段用户确认。任何范围外数学/实现冲突报父；作者不运行 import、collection、pytest、数值、lint、formatter、compile 或 build，不另派助手。父全文 review、补公共转发后，才开始同一串行槽的首轮检查与 interpret/CUDA 测试。
