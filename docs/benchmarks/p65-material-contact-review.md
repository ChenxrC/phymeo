# P6.5 材料接触阶段唯一独立审阅

范围：d32199e96836cf6f53348d4d1720d5d287cd5e0c..7e004ac3882e4ce21aebb315831452713e02c4e5。fresh reviewer为material_contact_final_review，未修改checkout、未派子代理。结果0Critical/1Important/0Minor；本阶段不派第二次审阅。

## Important与修复

MaterialPathLedger使用了外层position_iterations上限，合法inner_settings更大时误拒真实材料路径。独立三角原生内层需3sweep、接触p为0，外层1/2会失败而外层3通过。两个NumPy外层配置与一个actualinterpret配置共3RED确认；工作区保存真实inner设置供材料账本使用，外层GS预算、原tol与尺度保持不变。修复后相关与旧接口86通过/18未选，22.28s。源码最终完整工程测试另列[工程记录](p65-material-contact-engineering.json)，不把定向测试当全量。

## 独立检查

审阅者7个精确能量舍入、域间角动量、原始外功、owned、后段物理拒绝、trace回滚与同节点双平面测试通过，2.02s。压缩边实际第24sweep恢复完整最佳trial，全部25次replay仍计数，history仅推进一次。共享摩擦与支撑探针分别37/24次replay，created=closed=replays。

## 未裁定项目的实施者处理

- 实际设备完整结果、C++短测、wheel、CI、串行性能：必须单独取得证据；不因审阅未重复便豁免。
- 一般M(q1)世界能量/CCD：仍属明确后续阶段，当前诊断只称冻结模型能量。
- 后续细化trial的native/trace/close失败仍拒绝全帧：接受这一严格事务规则，不以已有可行candidate为由吞掉故障。

本报告不代表P6.5整体完成或允许合并/发布。
