# PERF-016 材料接触完整重放与主机认证成本

状态：2026-10-29，正式生产outer观测完成，记录瓶颈但不优化。产品0657c98，124源码hash b56bafcac102b696ebf1a180bb2946bd7ab2a8d103b9a3a4b6bffc5a85645dce。完整CUDA1950通过且源码hash不变后，NumPy→interpret→CUDA串行各3×100；各另3次warmup。所有900个计时调用成功并核对解析结果与同模式重复逐位一致。原始记录保留，失败历史见[工程JSON](../benchmarks/p65-material-contact-engineering.json)。

## 测量口径与结果

相同压缩双节点边+质量2的Y移动机器人，每次从相同初态调用完整材料接触step。时间包含所有native重放、原式/物理审计、owned诊断、结果数值断言和workspace.close；不含factory构建及最后runtime.close。未测连续轨迹、冷缓存、峰值内存、Warp或C++长期。第一调用已有缓存，不能称冷JIT。

| 模式 | 300次中位ms | P95 ms | outer迭代/replay | completed launch | 上传/下载次数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NumPy | 149.077 | 159.144 | 24/25 | 0 | 0/0 |
| interpret | 433.378 | 458.930 | 11/12 | 300 | 0/0 |
| CUDA | 2266.466 | 2422.913 | 11/12 | 300 | 2184/588 |

NumPy tol1e-8/FP64计算，tyx tol2e-5/FP32原生计算和FP64存储、主机精确审计；不是等精度CPU/GPU加速比，也不是大规模并行负载结果。三个批次各100个计时样本：中位数NumPy150.396/148.828/148.222ms，interpret436.310/432.552/431.025ms，CUDA2258.899/2288.414/2261.968ms。

来源：[NumPy](measurements/p65-material-contact-numpy.json)、[interpret](measurements/p65-material-contact-interpret.json)、[CUDA](measurements/p65-material-contact-cuda.json)。记录HEAD884019d；NumPy开始时工作区干净，后两模式git_dirty包含前序生成的未跟踪测量文件，生产源码和基准脚本hash前后相同。正式数值与审阅探针已退出；期间仅进行只读状态/文档准备，无并行数值测试。

## 已观察瓶颈及后续方向

成本集中在完整预测和position外层：三模式position阶段中位约136.658/392.005/2071.240ms，predict约4.056/33.224/186.702ms。阶段计时同时包含native调用和认证，并未隔离纯kernel、内存分配或同步成本，不能把全部差值归因tyx。高重放/launch/传输数量是可定位的集成事实；直接原因和占比还需要后续profile。

代码入口：solvers/initial_robot_material_contact.py、initial_robot_node_contact.py的候选细化；runtime/load_replay.py完整native重算；validation/material_path.py与solvers/_material_contact_balance.py独立认证。free响应只是预条件，首个合格candidate之后仍可能在原outer预算内细化，保留全native最佳trial，不能为减少调用删除末gap/物理/事务门槛。

后续先profile重放、传输和精确审计，再研究共享编译/资源生命周期、材料切线或预条件与数据驻留；这些只是排查方向，本轮没有实施。优化必须保留相反输入一次、能量/角动量、持续窗口/步长和晚期失败全恢复证据。未测等价Warp，不声明差距；尚无新tyx实现缺陷证据。

## 早期原型记录

原normal-only2秒窗口h1/16、1/32：NumPy685/934replay，interpret/CUDA322/469；分别32/64步，tol不同。它仅是[候选实验](../development/p65-material-contact-candidate.md)调用计数，不与上表同初态单步吞吐混用。此前并发验证期间的1sample仅harness smoke，不作正式观测。

TYX-003长期C++资源门禁仍开放，TYX-004是解释器cos异常行为差异，均不由本表关闭。
