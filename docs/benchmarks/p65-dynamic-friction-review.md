# P6.5 动态摩擦：唯一独立审阅

审阅已完成：**0 Critical / 0 Important / 1 Minor**。reviewer `dynamic_friction_final_review`，BASE `b7dd5b55451295434191eecc0bbedf3a1c46f712` → HEAD `cb209b8c5f5738d2527ef6e03dbe685b6cecf763`。生产6df5b14和父基准在完整147930字节差异包中。父f697f53只改文档，源码134 hashd18e557c…、脚本c9426b7a…保持一致。审阅所有probe已退出，没有改tracked文件/index；不是最终工程通过。

## Minor M1 与技术裁定

原brief的“physical failure even after best”混淆了试算候选超预算和执行失败。新workspace `_refine`可以在后续完整、有限候选物理残差超门槛时恢复原完整合法best，再执行全部接受检查。一次只对新候选注入`frozen_model_energy=2*tol`的探针仍接受旧best；这不是接受越限的状态。

主agent裁定明确：有限、已定义的不合格试算候选可丢弃；最终选定候选必须重新通过全部原式/物理/几何门槛。若最终best也越限则整帧回滚；native/trace/terminal/close/ledger执行异常、非有限或未知结果不能被best掩盖。成本是有限越限候选可能终止进一步细化，未改变law/tol/物理预算。冻结spec和brief已澄清，回归提交dc59c3968b305833e941e68ac1170c695a15bbcb：三种区别加正常对照共4项直接GREEN（11.60s），transaction文件27通过（23.92s），没有伪造RED。父已核对差异、注入触发、完整best重验/解析状态与回滚重试断言、日志及生产hash不变；M1关闭，生产未修改。

## 本次独立证据

- 只读差异包与指定git diff一致；实际读取三个新生产模块、四测试、source smoke/benchmark和直接复用ledger边界。源码/version校验通过。
- CPU3.11.15/NumPy2.4.6，stdin串行斜轴解析探针exit0：轴a=(3/5,4/5)、M2/node m1，静支撑p1/4；低μp3/14,nodev=(11/56,-1/28),robotv=-5/112；新接触p25/66,eta5/11,tauX=-5/12,nodev=(7/12,-1/6),robotv=-5/24，位置/速度均独立核对。
- Actual interpret3.12.14/NumPy2.5.4同斜轴冲击exit0；23workspaces、798真实launch逐项与read-only engine.stats一致，selected pre/post FP32提升、上下文全部关闭。
- CPU两次故障探针exit0：一次有限越限回选、最终best越限回滚、after-best ledger执行异常回滚，均确认实际触发及预期。

未执行full suites/正式900/CUDA重跑/CPP/wheel/CI，不把实现者历史结果当本次执行。未拒判任何已发现的声明域内行为；全参数收敛证明、旧阶段无关实现、有限特征/FK CCD和完整P6.5不在本次验证范围。该阶段无需第二次fresh审阅；M1核对关闭后进入最终工程。
