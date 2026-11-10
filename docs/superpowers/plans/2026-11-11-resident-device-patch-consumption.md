# B2：设备接触面进入受载事务

状态：接口准备，尚未授权编码或执行。B1 的实际 interpret/CUDA 资格验证通过后，父再冻结本片文件范围；06:30 后不扩展本片。它是[本轮计划](2026-11-11-newton-parity-until-0800.md)的条件后继。

## 要补的连接

设备宽相和凸体 patch 已有独立查询接口。当前受载 resident 路径仍调用 HOST `contact_patch_sample`，查询结果不能直接冒充实际载荷。B2 要让合格的设备 patch 进入现有完整 Jacobian、质量矩阵、双侧力矩和一次状态发布。先支持单个 contact plan；分页接入在共享接口稳定后扩展。

当前代码依据：`backends/_device_convex_query.py` 的 `query/_patches`，`backends/_resident_contact_plan.py` 的共享力学事务，`backends/_resident_contact_bridge.py` 的采样与材料点输运，以及 `geometry/contact_patch_adapter.py` 的严格数据构造门。以下保留工程精度，不要求理想数精确相触，也不放宽既有几何门。

## 必须先冻结的三件事

1. **同一个事务发布。** 把凸体查询拆为暂存结果和无失败操作的最终发布。设备宽相、凸体结果、原始/effective patch、力收据与 accepted bank 在同一提交点生效。后续材料转换、装配、求解或收据构造失败，不能留下看似成功的新查询。独立 `query()` 保持原公共语义。失败的资源计费保留。
2. **显式表示转换。** 旧 `ContactSite` 要求 FP64 单位法向、重心和积分权重严格归一，不能把近似 FP32 字直接塞入，或降低旧构造门。仅允许确定性表示转换：保全部原始字；对非负、有界且已合格的重心/积分权重按和归一；使用一条明确的共同单位法向；按原材料 ID、同入步接受帧和原半径重建双方材料点，公共施力点取双方中点，gap 为法向投影。不能做新的 HOST 接触搜索、按力选点、截掉负权重、增删接触点或修复未知几何。
3. **转换有证据和限界。** 收据同时保存 raw 与 effective 数据及每项差值。重建点与设备原点、gap、offset、法向和一阶矩必须满足 B1 原有相应工程门。shape/world/owner/topology/epoch/active bank、原局部点与实际上传点的来源全部绑定。已有门不能覆盖的情况明确拒绝，先保最小反例，不用未声明的额外容差解释。

## 载荷和失败语义

- 所有 candidate 都有原显式材料律；只有合格 `RESOLVED` patch 可消费。`UNRESOLVED/ABSENT/RESOURCE_LIMIT` 不能落回 HOST 搜索或算作零力。设备资源状态没有精确 required-site count 时，失败整个尝试，不能编造逐世界溢出计数。
- 真实几何站点先全部计数，含零力站点。既有逐世界 site-cap 保留，失败世界保留原状态及上一接受收据。
- 用真实入步 COM 线速度与角速度计算共同施力点速度。原 `sample_patch_loads` 只施加一次积分权重；双方力相反，分别对真实 COM 计算力矩。各 body 的所有贡献完整汇总后，每 storage 一次完整求解。
- 候选末态只输运原材料点，不能重采样改变入步力。保留 application/material work、量化误差、原控制、真实输入、完整动态与静态反作用。
- 普通、affine、floating、floating-affine 使用各自完整物理 FK/J/M；不能把约化自由度当 link 索引。第一片仍是设备几何加 HOST 载荷装配，不声称全 GPU 接触或训练。

## 开发顺序与测试库

| 子项 | 交付 | 必须通过的独立检查 |
|---|---|---|
| B2a 表示适配 | 一个有界、无搜索的适配器及 immutable 转换记录 | 全十类无序 S/C/OBB/H 组合中 B1 已解析的点/线/面；反序、旋转、部分交面、稀疏材料 ID；原始字段不变、完整双射和一阶矩；伪造来源/权重/法向/不完整 patch 拒绝 |
| B2b 共同发布 | query 暂存接缝、可选设备 narrow 路径、原力学复用 | 后置 patch 失败、最后载荷失败、求解后收据失败均无部分发布；计费保留；重试和 stale result；原 HOST 默认路径受影响专项回归 |
| B2c 真受载矩阵 | 多初态连续步，含真实入步速度、执行器和外载 | 四类树、动态–静态及双方动态，双侧力/COM 力矩、权重一次、完整 J/M、逐世界隔离；同一输入 HOST 与设备几何路径的工程误差对照；不可仅验证 query 状态 |

候选文件：新 `_resident_device_patch_bridge.py`、新专项 tests；最小修改 `_device_convex_query.py`、`_resident_contact_bridge.py`、共享 `_resident_contact_plan.py`，公共准备接口由父持有。编码前根据已集成 C1 版本明确精确路径，不并行编辑共享接缝。不得修改旧三个 K1 untracked 文件。

首轮专项验证保留原始失败。两个修复循环仍不能缩小具体缺口时，停止扩展并保留最小反例；先完成已独立可验收的检查点、训练数据面与分页。判定完成依据是公共受载能力及共同发布证据，不能用计划行数、组合数或单个舞蹈作为验收。
