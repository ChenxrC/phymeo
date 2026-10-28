# P6.5 Ubuntu仿射回归测试修复

937ca594388fe00d02d2d793bfab6513d69918b5的Documentation40654507845成功，但CI40654507850四个Ubuntu CPU组合失败。GitHub API一度限流，公开run页面确认失败；不把此前Windows全量通过当成Ubuntu通过。

70701aa仅改CI诊断：pytest生成本地JUnit，失败注释只允许输出当前公开tests源码里存在的test_函数名；不输出断言、参数、路径或完整日志。完整诊断载荷方案被自动审批拒绝，已改为此有限替代方案。诊断CI40658923536的实际check annotations指向`test_affine_guard_checks_the_sum_path_used_by_robot_plane_positions`。

## 证据与修复

原测试固定四元数和约1e8局部锚点，假定sum路径的实际绝对仿射误差必然超过1e-8。Windows当前框架得到1.56994794e-8，拒绝正确；将四元数第一个分量加1/2个ULP后，误差为9.82261355e-9/4.02720294e-9，原同一测试要求拒绝却得到正确接受。增加这两个邻近框架控制先得到2RED/原1GREEN；这是可复现的测试前提问题。远程runner精确矩阵/BLAS指令未读取，不声称定位到某条特定指令。

1ae0d217仅修测试：用独立Fraction计算当前存储R和实际sum路径误差；必要时只对输入锚点乘2，直到误差确实超过原1e-8，再要求几何门禁拒绝。三种框架都覆盖，原Windows反例不改变；无skip，无容差放宽，无生产代码或积分器修改。

- Windows相关107通过，43.28s。
- 本机Ubuntu3.12.3/Linux NumPy2.5.4的三个修复用例通过，另56未选，3.72s。
- 本机Ubuntu原收集全集1490通过/288未选，706.83s；该集合在测试扩充前收集，不能写成1492全量。它也说明不能把问题简单归因于操作系统名称。
- Linux复现使用已有WSL Ubuntu、hash与requirements/dev.lock一致的Linux NumPy wheel、已有纯Python测试依赖；不是新的干净pip安装门禁。原生/干净安装覆盖仍以既有工程和新准确CI为准。

产品源码仍aef9b257/118文件hash da5b54842e3256ef20b6eb5d27bfdcded77543f656330603f2ce7975d5989123；不重跑未变化产品的CUDA全量/性能/wheel或伪造新计数。修复后的准确head远程CI待验证；最终审阅结果另补。

单次fresh修复审阅：BASE937ca59..HEAD1ae0d217，0Critical/0Important/0Minor。审阅者独立3用例和ruff通过；内存中删除sum路径认证时原orientation_ulp=0用例失败，证实原回归价值保留。+1/+2ULP输入缩放分别2/4，额外价值为前提可移植性，不宣称这两例单独区分sum/matmul。安全注释的合成测试排除了参数/断言/未知名和注入后缀。未重开前阶段物理审阅，未宣称远端修复已通过。
