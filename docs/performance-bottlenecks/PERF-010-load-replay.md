# PERF-010：从完整快照重放的成本与Windows C++资源边界

状态：串行端到端观测完成3模式；Windows C++持续运行受阻。细项占比未profiler定位，不优化。

`runtime/load_replay.py::_compose/evaluate`每次合成/校验输入并保存所有权，`SceneStepper`完整重跑；原参与者重建材料工作区和tyx引擎。位置为`backends/tyx_participants.py::_RobotWorkspace/_NodeWorkspace`、`tyx_math.py::TyxMath.call/close`。反复构造/复制/FP64审计是Phymeo集成成本，不能自动归因依赖。

[正式观测](measurements/p65-load-replay.md)：单移动关节＋单节点、900次完成计量。NumPy/interpret/CUDA中位约2.69/29.29/106.03ms，每个tyx重放22次实际launch。规模极小、含主机门禁与完整工作区生命周期，不用于宣称大规模吞吐或Warp差距。

另有已独立确认的依赖问题：[TYX-003](../../tyx_flaw/TYX-003-windows-cpp-library-lifetime.md)。Windows C++显式close后DLL留存，原300次循环出现初始化失败；需要依赖侧生命周期修复和同条件复测。这不是可以通过删验算或统计最快成功前缀解决的问题。

下一步在完整接触outer循环记录trial数量、单次时间与拒绝/接受原因，区分几何、载荷映射、材料、动力学、候选验证和资源；核对是否为重复构造还是依赖加载成本。潜在安全复用只能复用不可变编译工件或无隐藏物理状态的资源，须保持每次从完整初始快照重建、p→0→p、后段失败/关闭全恢复。当前不引入持久状态缓存，不修改tyx pin；C++修复前不能声称持续运行验收完成。
