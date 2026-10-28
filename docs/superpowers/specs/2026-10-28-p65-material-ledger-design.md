# P6.5 距离材料原式账本设计

用户已授权持续P6.5与按小切片先测试后实现。本阶段延续已验收真实材料路径，实施者选择先交付可复用生产认证器，再将其用于接触outer；不扩大旧FreeParticleLedger许可，不替换native状态，不改原积分/容差。旧69b866a准确CI需先闭环；产品路径dc245f7/120文件已验收。

## 选择与边界

现有18列路径只是证据。仅检查终点、仅用最终lambda乘最终方向、或重新host算一条候选替换设备状态均不能满足独立认证。选择用存储输入的有理数与平方根包络核对实际记录和candidate，沿用项目已有Fraction/sqrt_bounds；算术只用于验证。逐项容差仍取调用者原StepSettings，不按世界坐标/边数/迭代次数扩大。

本切片认证距离XPBD的离散方程和载荷记账，**不认证物理角动量守恒、能量单调或完整接触**。这些需要下一outer明确物理误差/步长门槛。把原NUM-005角动量损失精确报告为材料离散贡献，不将其默认为接触摩擦。不引入机器人/碰撞求解器或新的材料积分器。

## 公共接口和所有权

新增`MaterialPathLedger(model, constraints, initial, original_inputs, h, settings, context=BackendContext())`，位于`validation/material_path.py`并公开导出。仅FP64存储；context.backend限numpy、tyx:interpret、tyx:cpu_cpp、tyx:cuda。模型/约束/初态/原输入独立快照，h正且settings.tolerance>0。constraints允许None/零边；固定点允许但初速度必须零，固定点不可请求非零新增冲量。未知backend/precision拒绝。

`require(candidate, effective_inputs, expected_impulses, diagnostics, *, sweeps)`返回owned只读mapping。expected_impulses是(N,3)有限实数或Fraction的原始请求，不拿ReplayTrial.integrated_loads的已舍入差冒充原请求；零请求也必须显式给出。diagnostics必须含完整material_path和distance_multipliers；sweeps显式正整数、<=原position_iterations、等于frames长度。失败抛SolverFailure，配置错误在构造时抛TypeError/ValueError。认证器不调用求解器、不修改输入、没有持久history，不关闭调用者资源。

输出scope固定`distance-material-equations-v1`、residuals/max_residual、每节点材料路径冲量、固定点支持反冲量、转换位置差/投影舍入位置差，以及K/U/重力势能/端点位移功/线角动量与材料贡献的区间或向量。报告不是新的SceneStepReport.accepted；后续outer仍须自己的物理门禁。

## 原式与路径

所有下面的浮点数据先用精确Fraction表示，norm使用sqrt_bounds的保守上下界。非有限/非法shape/错误schema和列名先拒绝；原记录方向零哨兵仅在wi+wj+alpha=0时允许。NumPy存储为FP64；tyx记录每个值必须恰为FP32升FP64，execution mode/compute_dtype/正completed_launches必须与context相符（这是可信程序合同检查，不是设备真实性密码学证明）。

1. 输入一次记账：逐节点检查`h*(effective.force-original.force)=expected_impulse`，动态节点误差乘w/reference_speed。固定节点要求新增请求及表示差均为零。
2. 预测：独立核对`v_pred=v0+h*(g+w*effective.force)`和`x_pred=x0+h*v_pred`，使用原FP64输入；固定点要求x_pred=x0,v_pred=0。位置误差除h/reference_speed，速度误差除reference_speed，不因大坐标放宽。初/末/预测数组shape/dtype严格核对。
3. alpha：diag的alpha与`compliance/h²`按`max(alpha_exact,wi+wj)`作相对核对；分母0要求alpha=0。每条实际增量原式仍使用原compliance/h²、原w/rest，alpha近似不能替代物理输入。
4. 路径连续性：从真实predicted_position、lambda0开始。每sweep非空tyx投影先独立形成输入FP32表示；逐行before_i/j和lambda_before必须等于当前实际序列值（严格相等，不用tol容纳未记录载荷）。行after作为下一条边的实际状态；末lambda_after链必须等于最终multipliers。每sweep结束仅固定点按原native规则恢复host位置。None/空边不转换整域位置。最终candidate.position必须恰等于重建的真实操作序列，拒绝隐藏position_load。
5. 逐边原式：用before差向量独立norm包络得到方向（coincident X），检查实际方向和增量方程`length-rest+alpha_exact*lambda_before+(wi+wj+alpha_exact)*dl=0`。方向误差受原tol且其等效位移贡献计入累计预算；before/after变化与原w*dl*n逐分量核对，lambda_after-lambda_before与dl核对。wi+wj+alpha_exact=0时要求方向/增量零、位置与lambda未变。正分母的零dl不豁免方向。
6. 累计而非摊薄：逐节点累计每次投影更新、方向/增量原式误差和FP32位置转换的绝对等效速度误差；按h/reference_speed归一，最终与原tol比较，不除以边数/sweep数。转换差与投影舍入差分项报告，固定位置转换也受同一绝对范围门禁。fixed host恢复是预先声明的路径操作，不算物理材料位移。
7. 重建：逐节点检查`v1=v_pred+(x1-x_pred)/h`，固定v1=0；原式residual不得由模拟同一舍入路径而被掩盖。未记录velocity_impulse会被拒绝。单独检查末本构残差（原rest、compliance/h²、实际末lambda，按max(rest,reference_length)归一）。

字段级上界与累计项均输出residuals，经upper_float保守转FP64再返回；所有门禁使用未舍入Fraction与原tol比较。任何报告值不可表示/非有限拒绝，不返回误导性成功。

## 物理记账（不把损失宣称守恒）

材料配置载荷G_i=sum(dl*n)，材料等效冲量G_i/h。每边相反分配，包括固定点；固定支持反冲量为固定节点材料冲量的相反数。线动量只对动态质量m=1/w求和，报告总变化、外部h*(m*g+effective.force)、材料路径冲量与实际差值。

角动量用真实初末x×mv独立计算。报告外载荷关于x0的力矩冲量、材料路径冲量关于x0的离散贡献，以及实际角动量变化与二者之差。纯硬边旋转例应显示非零材料贡献，不能将其抵消后宣布世界角动量守恒。含固定点时只报告动态部分与明确支持反作用，不冒称封闭系统。

能量：动态K=Σ|v|²/(2w)，重力势能-Σg·x/w；软边U=(|xi-xj|-rest)²/(2compliance)，以独立sqrt包络给保守区间；硬边无弹性势能项但末约束必须合格。端点位移功W=Σeffective.force·(x1-x0)，同时分原输入与新增请求两项，差异属于输入表示误差而非接触力；明确是离散位移功，不是连续路径精确功/接触摩擦热。报告E1-E0-W区间，暂不把其正负作为通用接受门槛；下一接触程序必须预定义能量与离散角动量允许预算和细化验收，不能直接以require通过替代完整物理验收。

## 验收与交付

TDD：单硬边解析、三节点两边路径、两固定/混合固定/None/零边/coincident、多sweep，所有权与配置验证；损坏before/after/direction/dlambda/alpha/lambda/预测/候选及隐藏load/重复输入必须拒绝，累计小误差不能按迭代数稀释。大坐标舍入/FP32转换失真拒绝；未使用零分母方向哨兵不误拒绝。

实际NumPy/interpret/CUDA匹配原候选并通过账本；有限C++短例；三步长旋转硬边和软材料K+U证据。SceneLoadReplay同初态多请求、两个节点/小三角与真实单关节旁域映射的相反输入账本检验，不能称已求出接触。集成故障在后子步污染路径/关闭失败时全局state/history恢复，重试=fresh。公开导出/源码外wheel、两CPU/完整CUDA、Linux新核心、唯一阶段fresh审阅及修复、串行性能只观测。TYX-003长期门禁保持开放。

Builder-owned：精确辅助函数可拆到`validation/_material_path_math.py`，不迁移旧robot helper；测试数据与故障包装细节由实施者选择。用户未另选实现细节，此为实施者决策，连续授权内不再询问常规实现。

Builder check裁定：唯一缺项为逐边增量与乘子更新的无量纲尺度，尤其两固定点的compliant边不能靠乘w的速度预算认证。现冻结：L_e=max(rest_e,settings.reference_length)，D_e=wi+wj+compliance_e/h²。逐边增量残差上界为maxabs(length-rest+alpha_exact*lambda_before+D_e*dl)/L_e；乘子更新为D_e*abs(lambda_after-lambda_before-dl)/L_e。各项沿同一边的全部sweep累加并分别与原tol比较，同时保留动态节点等效速度累计预算；不因wi=wj=0跳过标量方程。D_e=0时要求lambda与坐标严格未变、dl和方向零。方向本身逐分量无量纲误差<=tol，另计其位移影响。fixed compliant正反例必须进入TDD。无需更改用户目标。
