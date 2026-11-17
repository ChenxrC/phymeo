# I3j：自动面积接触接入异构任务环境

2026-11-18；codex/exp，源码提交9b928c2d。4个新CUDA节点与6个受影响native兼容节点通过；dev未变。本批补齐任务Env生命周期，尚未把面积字段送入策略编码器。

## 实际链路

`HeterogeneousManipulationEnv(..., area_scene=..., area_roles=...)` 在原动作被接受后，按新的batch revision调用借用的area Scene。Scene从当前状态/FK生成真实有限接触面，整批共同求解后，Env导出设备观察、读取同入步载荷、评价任务并管理独立episode。调用者不再逐步提供接触面。

面积模型必须显式提供完整角色，Scene必须借用同一个batch；静态资产摘要与reader身份加入Env binding。原native-only binding不变。面积力已经是N，沿I3i reader消费，不再次乘面积。当前面积法向律无摩擦，原生手指接触仍走原生路径。

自动和显式选择reset调用Scene；checkpoint沿原codec保存typed batch快照和任务元数据，恢复时重新封装相同资产身份，不重算历史载荷。Scene资产摘要每次核对实际owned资产，防止构造后改压强而静默换模型。Env借用Scene/batch，仅关闭自己的workspace。

## 固定测试与结果

三个world顺序41/7/305，分别为area-soft/native/area-stiff。真实3P机器人、独立FREE立方体(.08kg，边长.1m，中心z=.049m)、有限小支撑顶面(.02×.02m²)，h=.0005s；完整8角点+中心/12tet压力体，中心压力100000/125000Pa、外边界0。最长4次真实步调用，没有改质量、几何、力或成功门槛追绿。

- 独立对照：Env和另一个真实batch+Scene+reader各2步，完整物理状态、几何、载荷、任务历史和设备数组相同。操作性revision/configuration身份允许不同；科学输出按原规则逐字节对照。
- 自动几何：每个面积world的有效总面积.0004m²，来源仅bottom两tet，法向+ez；真实支撑载荷为正。native world面积payload为空，保留真实原生支撑载荷。
- 生命周期：不同horizon的自动与手动选择reset、终止前final_observation、episode/tick和原观察一致；fresh Env成对恢复后继续真实步一致。
- 失败：缺角色、材料错配、错误Scene owner及构造后合法数值资产变更在提交前拒绝。动作接受后的末world几何退化不提交物理；物理接受后的导出故障明确标记state_accepted。两者都不发布新transition，后续操作要求成对恢复，恢复后可继续。

这些短步没有产生任务成功，也不声称稳态承重、抓取或共享策略训练完成。

## 验证记录

证据根：`artifacts/area-pressure-manipulation-env-20261118-01`。

- 04-native-affected-compat：6passed，4个未变化长组deselected，82.34s，RAW_EXIT0。
- 08-new-cuda：4passed/48.22s，01:55:57–01:56:47，RAW_EXIT0；四份完整gzip保存独立对照、reset、restore及失败记录。
- 09-final-ruff、10-final-format：精确3文件全部通过；执行前后3份source/test SHA相同。
- 首源码/测试Ruff仅import排序，修复与排版在新测试执行前结束，没有数值失败或物理参数修复。

下一节点由实际策略入口决定：若原编码器不消费area载荷，先补设备特征和身份/phase/mask，再做有界共享网络更新。不会把Env可运行等同网络已学会操作。执行顺序见[07:10计划](../superpowers/plans/2026-11-18-area-pressure-through-0710.md)。
