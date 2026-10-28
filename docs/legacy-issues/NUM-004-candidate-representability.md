# NUM-004：接触候选的可表示范围与独立验收

状态：错误成功路径已修复并有针对性回归；扩大可表示范围尚未实施。此条不允许在候选审计失败后继续成功提交。

## 问题与原因

端口 `response(p)` 的小速度增量在FP32中可表示，不代表 `v+response(p)` 在大基准速度上仍可表示。候选存储为FP64也无法补回先前FP32算术丢失的位。原H2b验收只检查响应预测，漏掉实际候选，这属于phymeo集成错误。

实际tyx节点逆质量1、初始速度 `(1e4,-.03,0)`、μ=.3、y法向、速度/响应预设容差2e-5；预测x约9999.991，实际x为9999.9912109375。基准1e8时切向冲量更新完全消失。较小速度1e2也可能先触发更严格的做功/动能一致性拒绝。不能只用相对误差把大基准速度下的物理冲量丢失隐藏掉。

## 位置与复现

- `src/phymeo/backends/tyx_participants.py`：`_NodePort.apply_impulses`、`_RobotPort.apply_impulses`是真实候选路径；新增`candidate_audit`只验证。
- `src/phymeo/coupling/candidate_audit.py`：实际速度和全域动能差；机器人使用独立FP64质量/运动学验收，未替换设备答案。
- `src/phymeo/coupling/domain_graph.py::audit_candidates`：逐域一致性，防止相对量抵消错误。
- `src/phymeo/constraints/contact_impulse.py`：全图再次验收后才返回候选。
- 最小集成复现：`tests/test_tyx_contact_impulse.py::test_actual_candidate_rounding_is_rejected_atomically`，`--tyx-mode interpret|cpu_cpp|cuda`；固定tyx `99f6df7c0fd8fdeef77f97df83914da27b64b7ee`、Python3.12.14、NumPy2.5.4。当前正确行为为明确拒绝且原状态不变。

## 当前合同

每域初速/候选作用点速度与实际响应的欧氏误差不得超调用方原有速度容差；全域动能差与做功误差不得超 `velocity_tolerance * sum(norm(point_impulse)) + 1e-12J`。全图能量仍不得增加超过同一预算；局部域可接受能量转移。所有自由度计能量，同一域只计一次。既有容差未放宽。

几何另有同类尺度边界：共同法向距离很大时完整距离不能区分面内候选。现用面内目标排序，精确有理数只认证原始浮点输入/实际世界候选的KKT；预算与特征边平方有关，不随远处点间距放宽。认证失败拒绝查询，后续碰撞程序必须传播失败，不得忽略碰撞对。

## 给后续 agent

1. 先保留原有失败例及未污染状态测试；收集目标场景实际速度、质量条件数、坐标尺度与冲量分布，判断是否进入该边界。
2. 选择正式FP64计算路线、混合精度误差控制或有物理依据的尺度策略；任何方案先独立证明实际候选通过原有物理门禁，不仅比较response。
3. 依赖侧FP64能力请求已在[TYX-001](../../tyx_flaw/TYX-001-fp64-compute.md)，不要新建重复的“FP32加法错误”条目；没有Warp实测。
4. 不允许用状态覆盖、事后增大容差、删掉审计、将失败统计成成功来扩大范围。若完整P6.5场景触发，此问题立即成为综合验收阻塞。

## H3a主机控制预测补充

独立控制切片也出现同类问题：q=1e8、target=q+1e-8、kp=1e8时，展开后的隐式PD方程可收敛，但返回的q+h*u舍入到target，使原始位置反馈控制律不再成立。该问题发生在主机FP64，不属于tyx缺陷。

现于`control/bounded_pd.py`返回前检查实际预测q/v上的原式，正负大坐标反例先失败后修复。超effort_tolerance明确拒绝，不调整控制增益/命令或覆盖真实状态。后续选择坐标中心化或其他数值表示策略时，必须保持原始物理方程与输出坐标的可审计关系，不能仅验证一个代数展开式。

## 面厚度端口补充（2026-10-28）

`coupling/offset_port.py::TriangleOffsetPointPort`在固定面材料点上使用完整块Jacobian。非退化/可表示域内，法向偏置的速度和切向载荷反力矩已由独立解析及旋转/虚功例覆盖。实现对缩放后面积<=1e-10的面明确拒绝；它没有给退化面自动选法向。

实际世界点仍存在FP64表示边界：三角面整体z=1e8m、偏置1e-8m时，存储点实际增量不等于1e-8m。构造阶段检查关于实际点的力矩恒等式，不满足冻结门槛则SolverFailure。最小复现位于`tests/test_offset_port.py::test_lost_world_offset_is_refused_by_actual_point_moment_check`。这是phymeo主机坐标/数值域限制，不是tyx缺陷。

后续agent先读取[冻结合同](../superpowers/specs/2026-10-28-p65-offset-port-design.md)。边/顶点厚度包络、接触侧和动态材料点重建仍属于P6.5必要开发，不能因记录在此而从完整验收中删除。极端尺度扩域可后续研究局部坐标，但对外返回点、速度、载荷与矩必须保持同一物理语义。

审阅另发现更小的±1e-12m偏置在z=1e8m时可落在同一个存储点，却对相反1e12冲量产生2N·m·s力偶；它小于原面尺度门槛。现补偏置自身相对表示门禁，保持原门槛不变。对应回归`test_tiny_lost_offsets_cannot_create_a_couple_at_identical_actual_points`。另以显式受浮点陷阱保护的乘积替代einsum，拒绝1e-200乘1e-200静默下溢；这是NumPy调用语义和phymeo检查边界问题，不新增tyx_flaw条目。

## 固定区间输入重放补充（2026-10-28）

`runtime/load_replay.py::SceneLoadReplay._compose`把积分载荷变为原输入上的恒定平均率。背景输入很大时，较小增量会被FP64加法吞掉；注册输入构造器也可能量化、限幅或耦合其他字段。账本现在从构造后的实际字段回算，逐分量误差不超过`1e-10*abs(requested)`；零请求必须零差，未声明字段必须不变。输入/状态快照必须保值，否则在创建workspace前拒绝。

一次独立审阅用量化构造器复现：原tau=2、dt=.2、积分请求=.1，临时2.5构造成2；旧账本记.1但实际零增量。四个RED→GREEN回归见`tests/test_load_replay.py`末四项（量化、未声明字段、输入复制、额外状态复制），产品修复`3f23378`。这是Phymeo插件合同检查问题，不是tyx缺陷。扩大数值可表示范围仍未实施；不能改大背景误差地板掩盖载荷丢失。

后续agent注册新求解器时，构造器/validator须保持快照值；真实持久物理记忆进入注册state/history。输入是率的字段由channels显式声明。该账本记录host FP64输入表示，原tyx program仍在FP32执行，其实际状态/物理残差门禁不可省略。完整接触外层、几何与CCD仍是P6.5必需能力。
