# C1：固定公共场景规模成本

状态：输入、独立标量 oracle 和协议 RED 已冻结，尚未执行。仅测试和测量工具；不修改 A/B 公共场景或材料算法，不给 Newton 速度结论。

产品起点为 `104e49a`（功能基线 `c53e616`）。所有正式作业必须保存实际执行的源码头及文件身份。若父合入 contact history，再按最终 HEAD 测同一矩阵；旧版本与含 history 的结果不得混称同一版本成绩。

## 唯一主矩阵

| 负载 | 每个 world 的真实物理 | W |
|---|---|---|
| free | 1 kg、r=1/8 m 的球从 z=1 m 自由下落，g=−4 m/s²；有限箱在原点，整窗没有材料力。切向初速 ±1/32 m/s。 | 1 / 32 / 128 |
| loaded | 同球与有限箱面的 gap=1/32 m，法向层宽 1/16 m、刚度128 N/m；实际4 N支撑与重力平衡，球有同样切向漂移。 | 1 / 32 / 128 |
| birth | 同球由箱侧棱的 gap=3/32 m、向内速1/2 m/s起步，g=0；进入同法向层、真实加速，正负镜像 world 共存。 | 1 / 32 / 128 |

箱半尺寸为 (1/2,3/8,1/8) m、固定。local body IDs 永远2和9，world keys为17+3i，世界在空间可共址。每项均用原 `ContactSceneBuilder.build(backend="cuda")`、`step` 和接受态 `geometry`；没有改成专用球面岛。每 world 两个形状，W128恰为256上限。当前 all_pairs 的完整表实际为1/2016/32640行，其中同世界物理对只有1/32/128，跨世界过滤仍保留。

固定 h=1/128 s，每窗口32个真实物理步。独立 oracle 自行实现标量半隐式离散：free零载荷、loaded恒4 N、birth的 F=128 max(0,1/16−gap) 与实际速度/位置推进。出生窗第9步入口仍为零压缩，第10步首次施力；不把材料层入口称实体相交或连续TOI。容差冻结于 `tests/scale/scene_scale_cases.py`，不得因规模改变。

## 最小窗口与可测阶段

当前公共 ContactScene 没有 reset。每项只做两个独立32步窗：cold采用新缓存及新scene；warm复用同进程代码缓存，由同一builder重新build初态。构建计时单列。warm意味着代码已热，不冒称新scene的设备缓冲已常驻复用。主矩阵共576实际步；不追加21窗或跨引擎重跑。warm提供整窗总耗时、逐步p50/p95和出生步时间序列，不能把非平稳出生过程当同一稳态样本。

每步分别记录公开 step 和 geometry快照墙钟；两者相加是固定的同步发布边界。后续 oracle、逻辑编码、digest及磁盘归档单列在边界之外。原生TYX计数真实记录kernel、H2D/D2H次数/字节和创建绑定次数。

`scene.stats()["geometry"]` 已有 envelope/broad/midphase/narrow 的真实墙钟差分。它们包含嵌套设备调用，不再与其中的上传/同步/读回相加。独立W1插桩窗才测法向evaluate、刚体step、实际H2D/submit/sync/D2H接口；按嵌套区间分配exclusive墙钟，同时保存inclusive原区间。没有原生GPU事件时device time标未测。正式成绩不启用插桩。

旧4,171束六端读回结果只作原接口事实参照，不重跑、不并入这九项受力统计。最多用原W1输入做一个独立诊断窗；射线没有响应/积分阶段，必须单列。

## 输出、资格与资源

父证据目录：`artifacts/p65-postnight-batch1-20261105-01/c1-scale/`。每窗口一个流式gzip JSON，保留全部公开入步/出步、力/力矩、双侧材料点/权重、路径、求解报告及接受态几何。首个完整pair表绑定一次，之后只存实际变化及完整返回值逻辑SHA；读回器须无损还原，不能删字段来减小输出。每步仍保实际candidate数、接触数和完整状态。逻辑字段身份不受gzip头时间戳影响。

静态存储估计：W128每窗未压缩等价20–80 MiB、gzip约1–10 MiB；正式报告以实际写入字节为准。流式编码只留前后帧，避免持有32份32640行对象表。容量不足则明确拒绝；不提高256上限或截断路径。

先运行新增协议RED节点：
`tests/scale/test_scene_scale_protocol.py::test_cost_harness_preserves_a_real_loaded_step_and_freezes_nine_cases`。
它先通过现有公共真实4 N正控，再因新增驱动缺失失败，不能声称产品本身缺失。后续新增资格涵盖独立32步面/棱/角、刚体共变、world/local-ID、257形状拒绝、拒步状态/时钟/帧保留及同实例恢复、完整gzip解码和变动字段digest。

本作者独占五个新增文件：`scripts/benchmark_scene_scale_costs.py`、`scripts/profile_scene_scale_costs.py`、`tests/scale/scene_scale_cases.py`、`tests/scale/test_scene_scale_protocol.py`、本报告。所有项目执行由父fresh guard串行完成；当前无作者执行权限。
