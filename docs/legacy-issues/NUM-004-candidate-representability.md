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
