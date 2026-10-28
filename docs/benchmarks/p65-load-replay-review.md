# P6.5 固定区间载荷重放：独立审阅

一次fresh review，范围`8223c7457781c93269565ae2374f4811b77efc02..1063200`，审阅者`load_replay_final_review`，gpt-6-astra/high。0 Critical、1 Important、0 Minor；未重审此前厚度端口/P6阶段。

## Important：构造后的实际输入可能与账本不同

注册输入类在构造时可量化或限幅。旧实现先按临时数组审计，再构造输入对象，因此tau=2、h=.2、请求积分=.1的临时2.5可被round到2，而账本仍为.1。该路径错误报告载荷已施加，是正确性阻塞，不能延期。

单次修复扩展了相同边界：实际构造后逐字段检查；未声明通道不得被构造器改写；初始state/input和合成输入的注册快照必须保值。主机FP64积分表示门槛未放宽。修复提交`3f23378`；没有修改既有SceneStepper或StateRegistry的全局行为。

4个独立回归在修复前均失败（3项一组、额外状态复制1项单跑）；修复后含旧SceneStepper/机器人集成共129项通过，解释器与新核心46项通过。回归位置为`tests/test_load_replay.py`末四项。原始日志位于本地`.superpowers/sdd/2026-10-28-p65-load-replay/review-red.txt`、`review-state-red.txt`、`review-green.txt`；最终工程结果见本切片验收报告。

此前一次CUDA全量在收到审阅后主动中止，不计作通过。此前两个CPU1324和C++2的运行属于修复前证据；交付只使用修复后全量结果。不发起第二轮审阅，不把本次修复声明为完整接触或夹持验收。
