# 两阶段法向接触的原生边界实验

此页是设计证据，不是生产程序验收。此前仿射净距模块交付5c35200，两项准确CI已success并由1fa44eb关闭。

现有完整 force replay 在返回时已经关闭native工作区。实验用显式wrapper在base velocity收敛、关闭前，通过真实participant施加预先给定末冲量。独立Fraction推导的自由探头和两节点距离链两个非递布案例，NumPy→interpret→CUDA串行各2/2通过。原式和分数oracle见[新设计](../superpowers/specs/2026-10-29-p65-dynamic-normal-design.md)。

最终脚本[scripts/probe_dynamic_normal_split.py](../../scripts/probe_dynamic_normal_split.py) SHA256为9b21867792a8b502c92badf24b652b288d687739eacc6ba8eb457b8070b94dae。
原始记录：[NumPy](../benchmarks/p65-dynamic-normal-candidate-numpy.json)、[interpret](../benchmarks/p65-dynamic-normal-candidate-interpret.json)、[CUDA](../benchmarks/p65-dynamic-normal-candidate-cuda.json)。
运行source126文件hash仍67ae299638326bf2018ee781a3bceadb5f54fbd84a7238f0740d9c27be3a6b60；Python3.12.14、NumPy2.5.4、tyx-compute0.1.0a0/99f6df7c。raw记录运行前后source/script一致及真实设备执行计数。

六例均核对：held输入积分、一次末冲量、bitwise配置不变、阶段前后解析速度、动量/能量、close顺序；原MaterialPathLedger接受pre而拒绝post，证明不能把两阶段账合并冒充原速度重建。tyx误差比较atol2e-6，原ledger tol2e-5未改。CUDA每例robot末总completed32、nodes free4/chain7，末冲量前后计数确实增加；不是只查询CUDA可用性。

初版两次NumPy接线错误各0/2（signature含Fraction、RobotState无copy）只修实验包装，随后初版三模式各2/2。补Ruff与provenance后最终又串行三模式各2/2；raw development_history区分，不把失败记成功或把两轮累加成不同物理案例。最终ruff check/format check通过。

限制：p与η预先给定，无生产自动求解器、自动候选、history/全局事务验收、摩擦、TOI或旋转CCD。单次设计builder dynamic_normal_design_check只读返回nothing important left；未运行探针/数值。下一步按冻结合同TDD实现，不能把本实验记作第17项已完成能力。
