# PERF-019：自动法向接触的两阶段重放与认证成本

状态：正式串行观测完成，未优化；细项耗时尚未归因。产品4ff19ea，131源码hash `ac777319d0b8fbc03adab92286656a5572df05bca403e913912500908e991060`；脚本hash `338768ffd42c2e7749f38779e20d2563ec56cd435ba2fceef8cc604f58545a75`。

## 观测与口径

Windows、Python3.12.14/NumPy2.5.4、tyx pin99f6df7c；完整环境及执行模式保留在[NumPy原始记录](measurements/p65-dynamic-normal-numpy.json)、[interpret原始记录](measurements/p65-dynamic-normal-interpret.json)、[CUDA原始记录](measurements/p65-dynamic-normal-cuda.json)。全部测试与唯一review的数值探针退出后，按NumPy→interpret→CUDA顺序执行，各3×100计时调用。每模式free/chain各150计时及各3warmup，共900计时/918总调用成功。每次均从固定初态运行，非连续900步轨迹。

前后源码hash与脚本hash一致，记录HEADd2eb887。后续测量期间的git dirty来自前序未跟踪raw文件，不代表产品变化。源码/tests/scripts整体冻结摘要也在最终退出后核对一致。复现命令：`python scripts/benchmark_dynamic_normal_contact.py --mode <numpy|interpret|cuda> --samples 100 --repeats 3 --output <new-output.json>`，不要覆盖已有证据。

| 模式 | 解析场景 | 计时次数 | 中位ms | P95 ms |
| --- | --- | ---: | ---: | ---: |
| numpy | free | 150 | 71.871 | 80.927 |
| numpy | chain | 150 | 367.839 | 411.786 |
| interpret | free | 150 | 269.773 | 300.476 |
| interpret | chain | 150 | 729.893 | 807.753 |
| cuda | free | 150 | 1139.645 | 1232.053 |
| cuda | chain | 150 | 3157.326 | 3342.788 |

包含完整SceneStepper.step、真实held/terminal重放、原式与物理预算审计、诊断、解析和bitwise断言以及workspace关闭，排除factory和最终program关闭。`outer_stage_seconds`保留外层阶段计时，但每项仍混合执行、认证与诊断，不是独立kernel或精确算术耗时。首调用已有缓存，不证明冷JIT成本。

| 模式 | 场景 | held重放 | terminal重放 | completed launch | 上传 / 下载 | 容差 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| numpy | free | 4 | 3 | 0 | 0 / 0 | 1e-08 |
| numpy | chain | 27 | 3 | 0 | 0 / 0 | 1e-08 |
| interpret | free | 3 | 3 | 176 | 0 / 0 | 2e-05 |
| interpret | chain | 13 | 2 | 445 | 0 / 0 | 2e-05 |
| cuda | free | 3 | 3 | 176 | 1382 / 368 | 2e-05 |
| cuda | chain | 13 | 2 | 445 | 3382 / 919 | 2e-05 |

全部实际trial计入execution，terminal post已含pre，不能再次双加。NumPy与tyx的容差/实际迭代不同，以上不构成等精度后端速度比。没有Warp实测、峰值内存、完整递布轨迹或C++长期结果。

## 长套件观测与未证实假设

同一冻结版本完整CUDA回归2216通过，pytest总耗时9148.75s；这是验收耗时，不是上表的性能样本。旧材料低摩擦细化测试在07:27:54至08:53的只读观察窗口未输出结束，worker的CPU计数仍增长；其后测试与全套均成功退出。没有独立单项计时或调用栈证据，不能据此声称死锁、特定kernel瓶颈或tyx实现缺陷，也不因进程活跃就宣称物理推进完成。没有中断重启该进程。

## 代码位置与后续方向

- `src/phymeo/solvers/dynamic_robot_normal_contact.py::_evaluate/_position_metrics/_terminal_energy`：两个阶段完整重算、几何与能量审计。优先统计各trial执行、输入构建、材料认证和接触认证的独立成本。
- `src/phymeo/runtime/terminal_impulse.py::velocity_metrics`及`validation/terminal_impulse.py::_require`：真实末速度冲量和独立跳跃账本；端口响应、快照及精确求解占比未隔离。
- `tests/test_tyx_initial_robot_material.py::test_actual_material_friction_step_refinement`及`tests/test_initial_robot_material.py`的滑动参照：未来按物理步、重放数量和阶段耗时定位长窗口，先观察现有迭代与传输结构，不无理由再跑相同长压力。
- `scripts/benchmark_dynamic_normal_contact.py`：固定free/chain解析负载、计时与断言边界。增加真实网格后需重新观测规模趋势。

候选优化必须保留同域交叉响应、真实后端、反作用一次记账、完整回滚、独立物理预算和精度门槛。本轮只记录；host重放/审计成本不写成tyx_flaw，TYX-003/004独立问题保持开放。
