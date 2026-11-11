# N4a / ROB1d / 通用凸体仿射重建批次

起点：`ff478eaa6ee570562b229491cb25c95e17783740`；2026-11-11。用户授权继续下一批，无截止。按[持续计划](../superpowers/plans/2026-11-11-newton-parity-continuous.md)推进，复用原三位作者的隔离工作树。下表保留原执行前合同，实际结果见后半页；本批已完成执行，N4学习为部分通过，COL仍有四个法向失败。

| 节点 | 难度 | 本批交付 | 验收与停止条件 |
|---|---|---|---|
| N4a 学习与规模 | 高 | 源码可支持的三任务、每任务三种子、固定预算/未训练基线/评价指标；公共学习与规模入口 | 先冻结合同再执行。学习改善与吞吐分开；失败保留全部种子，不临时改奖励或初态。规模按 W1/8/32/64/128/256 记录真实有效转移、内存、复制、同步和 HOST 成本；容量拒绝不丢世界 |
| ROB1d 安装集成 | 中 | 已验公共多参与者能力的源码外 wheel 消费者 | 两个互补案例：机械臂/偏心盒双侧力矩；四类树混合的非零载荷、持续步、恢复后继、局部 reset、错 owner 拒绝。interpret/CUDA 均执行；R03 精度失败不改名为通过 |
| COL 仿射重建 | 中高 | 一个通用 simplex closest-point 算术修复及独立几何测试 | 先测 count1–4、平面/旋转/边界/正权重/双侧见证，再测原六入口及受影响矩阵。保持原输入、容差和资源。Y 的 outgoing-normal 停止问题独立保留；本片不同时修改停止门 |

N4a 作者先交 `p65-mujoco-tyx-n4a-contract.md`，父结合实际任务/learner 源码裁定实现文件表。ROB1d 仅新增 `tests/wheel_consumers/resident_participants.py` 和报告。COL 仅修改 `_tyx_convex_query_kernels.py` 并新增独立测试/报告。已有 N3、机器人积分器及无关 WIP 不在作者范围。

项目执行由父唯一串行持有，作者先纯源码/测试编写。每个实际命令 fresh entrypoint-aware guard；保留首次错误、源码快照、原始退出码和完整输入。源码修改后只运行必要影响验证；同源已绿组不重复。失败不阻塞其他独立方向，也不以改阈值或重复同一尝试代替定位。

三个交接完成后，在最终集成源码上构建一次新 wheel，安装到新源码外 target；核产品成员一致，再运行公共消费者。构建/安装/运行均串行。复用现有依赖，不清理活跃环境，不远程 CI、不 push/main merge、不发包。新增通用修复若失败两次且没有缩小原因，保留反例并拆出未解决条件。

结果将写回本页、各节点报告、[当前入口](p65-continuation.md)和[全景](phymeo-newton-panorama-2026-11-11.md)。本批不代表 G1 完整舞蹈、任意机器人训练或整个 Newton 功能对齐已完成。

## 本批执行记录

- **COL 仿射重建已限域集成**：新六项在 interpret/CUDA 均通过；原两个 HN 失败在两后端通过；其余受影响矩阵各258通过。四个 Y 法向失败保留，不能称全部曲面查询通过。见[正式报告](p65-convex-affine-reconstruction-report.md)。
- **N4a 固定学习已完成评价**：九个种子均完成128×32×8，共294912训练转移。正式三种子汇总：持续接触支撑S通过，中位成本下降26.4%；关节J与分支机器人R分别下降5.6%/8.2%，未达原学习门。S三个策略都优于zero诊断，J仅一个略优、R全未优于zero。保留全部种子，不追加预算或调目标追绿。
- 学习源为isolated `a4c4e2bc`；20原HOST、6新增修复HOST和原三个CUDA预检有明确分组来源。首run31在update7完成后因reset重建设备导致错误的单调计数要求失败，该次缺失的update7完整payload如实保留；新修复按生命周期分计数并覆盖晚失败完整记录，39影响检查8项通过，40/41完整学习通过。29只是无效pytest选项的启动错误，30才是真实预检。
- **N4a规模完成**：parent `06a19790` 在首次scale执行前补足正式循环起点至gzip关闭的实际wall时间；子计时不相加冒完整端到端。43–48静态/新增HOST1通过，49三任务×六规模18pass/530.94s/RAW0。每格一次warmup后测3×32×W转移；W256完整trace吞吐J693.325/S219.108/R560.472 transitions/s，均略低于W128。没有Newton同条件基线；不是256 G1。学习source未被parent计时修改干扰。
- **ROB1d新包通过**：80构建/81安装/82源码-wheel-installed全494产品核对均RAW0。86interpret/87CUDA各完成13次真实调用、21个logical-world steps、53完整typed events；持续双方载荷、恢复后继、局部reset和错owner拒绝均过。工作量由累计收据计算，不用restore后的最终tick。88已安装N4公共评价/学习门/无Torch HOST导入检查通过，没有PPO重跑。
- 原83安装interpret在物理前因Windows `sys.prefix`虚拟环境根路径被消费者拒绝，RAW1保留。启动器v2仅精确移除该根搜索项及已知editable源入口，保留site-packages及全部模块provenance；同一consumer字节随后通过。未动物理或检查阈值。84/85旧job和42旧scale job未执行，不列为测试失败。

证据根：`artifacts/p65-parity-n4a-rob1d-col-reducer-20261111-01`。50/51/52按预声明身份绑定九份完整summary/manifest/压缩证据，J/R的RAW1是原学习门未达，S为RAW0；不是物理运行崩溃。新wheel SHA256 `aca8054872902a050865c4c5b0ac9b083435dd1cc7743d959fe5f115699ce1e5`，1552449 bytes，包产品与7964d1b一致；后续父改动只CLI/test/docs，已核包成员无差异。完整安装证据另保canonical `installed-evidence`，不删除原件或解压旧档。

## 下一批按已测缺口推进

1. **N4b学习（高）**：复用已保存J/R动作、真实转移、PPO/value/timeout记录，先判固定短协议的弱收益来自哪里；再固定后继训练预算/指标。不把九次运行完成叫九次学会，不为追绿挑种子。S成功保留为正控。
2. **规模热点（中高）**：S128已有warmup inclusive profile中，32次物理step约7.97s、snapshot约3.47s，PPO update约0.005s；层次时间相互包含不能相加。先细分物理推进/快照/诊断可复用工作，再声明快速模式及对应reference验证，不静默删检查。内部Torch同步、独占CPU验证耗时和TYX-only显存峰值仍UNKNOWN。
3. **ROB2a与R03（高/有界诊断）**：球关节、第一阶执行器状态、显式传动先冻结多案例公共接口；R03沿已有失败入步做有界PD力相消复现，不阻塞训练和机器人扩域。
4. **COL法向（中高）**：检验当前q/支持下界/同一simplex权重共同决定停止与发布。距离区间正确不保证角度正确，所以是待验证假设，原四Y、资源和误差门不变；同时检验相邻query/patch/trace一致性。连续两次未缩小障碍即保存最小反例并拆分，不盲恢复旧FW门。

本批无新G1长窗、远程CI、push或发包。R03精度、四Y、曲面patch/load、一般mesh/变形CCD、广泛机器人训练和其它物理族仍开放。
