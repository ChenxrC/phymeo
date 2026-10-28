# P6.5 固定时间区间载荷重放合同

日期2026-10-28；前置厚度端口8223c74已取得准确提交CI40611577823、Documentation40611577859成功。完整双臂目标不变。本切片解决候选载荷重算的时间/状态/输入所有权；**不是完整接触几何或接触迭代求解器**。

## 冻结接口

`SceneLoadReplay(scene, program, state, inputs, dt, *, channels, settings=None, context=None, registry=None, verifier=None)`。

- scene/program/registry/verifier沿用SceneStepper合同。FP64状态存储；实际tyx仍通过原program执行FP32数值。成功构造后replay拥有内部SceneStepper的关闭责任。
- channels显式声明可叠加载荷的输入字段，例如`{"cloth": ("force",), "robot": ("effort",)}`。字段必须在对应StateFamily注册；不根据状态变量名猜测。调用者声明这些字段是物理输入率，其时间积分单位为该字段单位乘秒（节点N·s；关节按类型为N·s或N·m·s）。不允许把目标位置或数值位置修正声明为物理冲量。
- `evaluate(integrated_loads)`接受域→字段→同形状FP64有限数组；只能使用已声明字段，省略表示零。每次都从构造时的全部状态/history/原输入快照出发，**不是从上次候选出发**。
- 候选输入`effective = original + integrated/dt`，在整个固定dt区间保持；内部已有substeps按同一恒定输入推进，积分总量仍为给定载荷。耦合子步重放通常使用settings.substeps=1；若使用多子步，最后子步失败必须全区间回滚。
- `ReplayTrial(result, integrated_loads, effective_inputs, duration)`：result是原SceneStepResult；integrated_loads记录从实际存储输入回算的`dt*(effective-original)`，全部声明字段都有记录；effective_inputs为独立拥有的输入副本。result.report.accepted只表示本program验收，外部接触/控制/CCD接受条件仍须验收后才选用一次。
- `snapshot()`返回原始SceneState副本；`close`/context manager按SceneStepper关闭，关闭后evaluate拒绝。没有自动提交/累积history或加载旧候选的隐藏操作。

## 单次记账与数值范围

只把输入总载荷换算为平均率并重算一次，不再在末速度上补同一冲量，也不使用`apply_position_load(-old)`撤销旧构型位移。材料预测、XPBD内部乘子、速度重建和机器人质量求解均重新由原program执行；不能用host计算覆盖设备候选。

输入合成在host FP64。非有限/溢出/下溢拒绝。实际输入差回算的每分量误差必须<=`1e-10*abs(requested_integral)`，零请求要求实际差为零；不加基于大背景力的绝对误差地板。很小载荷被大背景输入吞掉必须明确SolverFailure。极端尺度可以拒绝，不承诺全浮点域。

所有输入/输出/history快照独立。沿用StateRegistry全部字段，不仅q/v；StepSettings、绑定问题与能力变化按SceneStepper拒绝。程序为可信插件，外部隐藏物理状态仍不在合同内；持久记忆必须注册history。数值失败返回原始状态和失败report；配置/编程错误传播，资源仍必须释放。输入合成失败在创建候选workspace之前抛出。

## 独立验收

1. 解析节点/平移关节载荷：Δv=p/m，Δq=h*Δv（单子步）；反向总载荷抵消，重复trial不累计；p→0→p应与三个独立新实例相同。
2. 非线性机器人与布材料：重放结果等于从相同原状态、合成输入直接运行原program；上一trial的XPBD乘子和构型不能泄漏。不要求相反载荷撤回前一状态，因为根本不从前一状态开始。
3. 预先给定平面接触的解析摩擦例：法向Jn抵消压向平面的输入、用既有摩擦盘求切向冲量，然后重放整个区间。高μ重力静置不积累g*T*h漂移；低μ按半隐式Euler独立解析解滑动，h/h2比较。此例未检测几何，不冒称自动接触、真实夹爪或布案例。
4. 多域、额外状态、显式history、最后子步数值/close失败、重试、返回数据被修改后再trial均正确；所有workspace关闭。
5. 实际tyx interpret/C++/CUDA双域机器人/节点载荷及重复重放有launch证据；映射host FP64、数值program仍原后端。
6. 对原SceneStepper和P6全部回归无行为改动；安装包新API、uv/pip与工程检查通过。

## 本切片不能替代的后续工作

接触外层仍须决定/收敛载荷及作用点，明确J/M取哪个构型并验证最终材料/动力学残差。返回内层accepted不等于接触收敛。CCD、厚度边缘/侧、弯曲/自碰撞、有限驱动/IK/history与完整双臂递布仍按总里程碑实施；性能仅测量，不优化。
