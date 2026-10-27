# NUM-003：P6 FP32 材料数值尺度边界

- 状态：已确认范围；普通场景中由极小扰动引发的错误拒绝已修复待阶段验收，扩大极端尺度范围延期。
- 类别：精度/可表达性；不改变 P5 FP64 或 P6 冻结验收预算。
- 位置：`backends/_tyx_node_kernels.py::_bounded_material_product`、`stvk_material`；
  `backends/tyx_node_math.py::material`；`backends/tyx_articulated.py::TyxNodeRobotConstraints._dot`。

## 现象和原因

24 四面体的小软块压缩后，接触会留下约1e-24量级横向分量；参考逆矩阵中约1e-15的舍入余项与其相乘，
或归一化法向的小分量相乘，可能产生 FP32 subnormal。
此前对每个乘法一律拒绝 subnormal，导致主响应完全可表达的场景也无法提交。
解释器在外层 `np.errstate(under='raise')` 抛错，CUDA 的显式算术检查产生非有限结果并拒绝，均不是成功。
独立小复现表明这属于 phymeo 的数值策略，未发现 tyx 执行错误。

## 当前处理及限制

法向长度已按最大分量缩放；和式至少有一项1，因此低于 FP32 最小正规数的平方项不影响长度，原法向分量仍保留。
接触内积在设备端忽略不可表达的小乘积后，主机独立核对相对反向误差≤2e-5；若整个答案被丢失会拒绝，不用主机答案替换。

材料矩阵/能量归约采用同类有界乘积；静态尺度门禁使用保守上界
`1000*tiny*(mu+abs(lambda)+bulk)*sum(volume)*(1+9*max(abs(rest_inverse))) <= 1e-6`。
系数覆盖既有形变域内的乘积、应力、参考梯度及所有单元装配传播；此预算小于冻结材料绝对误差2e-4。
独立单元、24单元、扰动及能量梯度测试仍使用原阈值。
边界转换、自由积分和一般矩阵原语的非零下溢拒绝不变；没有打开全局 flush-to-zero，也没有修改模型参数。

该上界保守，极高刚度/特殊尺度的模型可能被拒绝，即使特定构型没有发生下溢。
32项材料/节点和26项接触相关用例已在解释器与实际CUDA通过；最终C++、全轨迹和独立审阅仍需完成。

## 后续 agent 建议

先读取 P6 最终验收记录，按 `test_material_keeps_resolved_response_with_tiny_transverse_perturbation`
和 `test_material_rejects_scale_with_unsafe_omitted_term_error_bound` 重现。
独立推导更紧的逐单元/逐阶段误差界，或评估有明确类型合同的 FP64/混合精度路线；不得只删除门禁或把阈值按失败结果扩大。
扩域时增加高刚度、小单元、近抵消及长时间积分测试，检查小内力经大时间尺度放大的影响。
这不属于本轮性能优化，也不等于 tyx 或 Warp 性能差距。
