# 第十二批：PushT目标反馈、有限换点及连续GPU轨迹

日期2026-11-15；起点2d621fd6。root共享接口/唯一串行执行；原scene/training两个agent隔离纯编码。无新deadline/安装/CI/push/旧套重跑。

## 调研与本批选择

Bauza/Hogan/Rodriguez [CoRL2018](https://proceedings.mlr.press/v87/bauza18a/bauza18a.pdf)用反馈和近似/学习模型处理未知摩擦和压力分布。Hogan/Rodriguez [IJRR2020](https://hdl.handle.net/1721.1/141404)明确处理接触模式切换。Xue等 [ICRA2023](https://arxiv.org/abs/2212.12814)把脱离和换接触面纳入长时推行。采用这些共同的结构：每段重新读真实状态，有限短推，再决定继续/换点/停止。
本批不是上述MPC/GP/TAMP算法的实现，没有训练出的动力学或全局最优保证。先提供可复用目标评分、proposal、有限进度状态和真实路径/执行接口。

## 不交叉实现

- scene树仅新增 manipulation_push_objective.py/test：运输原有界BOX边界模板到当前完整刚体姿态，生成PROPOSAL_ONLY；保原几何strict query不变。参数全绑定，不把小tilt展平写回物理。R正交/右手容差2e-6，当前tilt<=.1rad，仅用于参考提案，随后全3D碰撞检查。
- training树仅新增 manipulation_push_progress.py/test：不可变策略/记录、严格递增step、有限segment/stall/time终止，错误不修改状态。
- root新增 manipulation_push_planner.py/test、run_manipulation_push_feedback.py；旧run_manipulation_control.py增加显式可选动态参考driver，默认旧行为不变。旧actor/Env、真实q/v、模型/PD/Goal不改。

目标评分使用原position/yaw tolerance归一的平方误差。原边界到COM的最大平面半径r定义启发mobility：给单位XY内法向f和Mz，命令长度l时预测Δxy=l*f/(1+(Mz/r)^2)、Δyaw=l*Mz/(r²+Mz²)。记录预测cost下降；实际未改善要诚实停止，不据预测宣称物理。
显式长度(.005,.01,.02)m，stand_off .012m；候选全保留，至多384。运输仅提出目标，原reference_usable不能当当前clear。无positive预测候选就拒绝本段。

## 路径及连续执行

每段从当前接受端点完整q/v出发，所有未控制坐标保持原bytes。若已有上一接触，先沿上一推向反向退出25mm；之后先试直接到新precontact，失败才试围绕物体的两向平面绕行。绕行半径由原边界半径+pusher半径+25mm计算，圆弧间隔<=45度，最多8个中间点；全部实际关节IK/全3D路径重新检查，不仅pusher点。
每段最多12个不同边界candidate，按目标评分次序，各长度按其评分；最多64次IK/段、512次/world，128次path query/world，失败attempt计费并完整保留。预算到限停止，不抹去失败。每次IK仍100迭代/.15关节步长/5e-4点误差，无随机seed。
每phase C1路径最大采样间隔.001s，原filters及其他pair深度1e-4保持；withdraw仅pusher/object允许1mm，entry .5mm，push/hold l+1mm（静态对象参考命令重叠，不是实际物理容差）。每query<=4秒/4097samples，原65536 contact记录限额。
时间由位移/固定速率给定并上界：撤离.25s、接近/绕行各节点.25s、entry .3s、push .7s、hold .25s；首次settle .2s。参考只给原motor PD，绝不改物体位置、施加额外力或重置episode。实际运动可能偏离已查参考，完整接触与误差照录。

正式执行固定world5/6/7依次W1，各唯一新连续CUDA窗最多12000步/12秒和8个完整推段。当前集成为诊断HOST参考规划+CUDA物理，不是无HOST读取W3/W8训练。
每段正载荷pusher合力模的时间积分>0且实际归一cost改善>=.5%才清停滞计数，连续2段停滞停止；原Goal成功立即结束，预算/无路径/模型超域各有独立原因。允许未完成的末段达到12秒hardcap停止，不能把部分段记成完整成功。
同一新episode里保持真实入步速度、所有对象自由度和原Goal；不把上一批state直接写回，也不自动重试同窗。物理失败保留已接受前缀，不重置后挑选好结果。

## 验证与停止

先两新模块与root接缝的独立HOST案例/负控，再三真实窗。同入步完整entry、控制决策/路线绑定、每段预测/实际cost、接触冲量、状态连续、所有接触pair、原Goal逐步评价和checkpoint保存。
消费者不额外推进物理；最后一条也须完整。源/原始失败/首style结果/所有命令保存。全部测试串行；其他实际执行自然结束，不kill。
成功分列：通用接口测试、闭环同轨迹发生、原目标真正通过。不能用cost下降/曾推到/视频冒位姿+IoU+低速+dwell成功。失败后的下一步依据新trace选择模型/路径/控制缺口，不扫描摩擦/质量/阈值。

## Execution binding addendum (before first project execution)

Root is the sole serial execution owner for this batch. Three full new HOST files plus affected runner/driver checks precede three one-shot W1 CUDA windows. The driver starts its monitor from the actual published FP32 endpoint, not native qpos0 decimal reconstruction. Every force integral consumes the preceding accepted entry_q and matching adjacent ticks; model COM is transformed from that exact entry configuration. The 12,000-step cap remains enforced by the physical loop even within a partial route. Planning failure stops before another physical step. At a route boundary, robot reference joints are explicitly clipped to original joint limits only if actual soft-limit state lies outside; projection is saved, actual q/v are never altered. Reference C1 qualification applies within each route; measured reference reanchoring across routes is not claimed to be globally C1. The final Env checkpoint is not a full controller-resume artifact.