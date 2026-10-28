# PERF-017：静态距离证书的host精确算术成本

状态：观测，未优化。源快照`6a9a81b`、125文件hash `4079424eeaed5a2ca307c42af4f3a13f014231e5c0e8a743e7c1b2bc31260e04`；脚本hash `a7a33ae6449f6d99e439721e2d5c1131fff6e16dc2b0cafc38c154cbf828683f`。Windows11、Python3.12.14/NumPy2.5.4，host执行；不是tyx实现缺陷。

复现：`.venv\Scripts\python.exe scripts/benchmark_feature_distance_bound.py --samples 100 --repeats 3 --output <new-output.json>`。原始文件[900次记录](measurements/p65-distance-certificate.json)，三个场景串行，各3warmup和3×100计时，900计时/909总调用全部成功。启动前本任务数值/审阅探针均退出；每次核对独立解析距离和重复一致性，退出后核对源码/脚本不变。

| 输入 | 样本 | 中位ms | P95 ms |
| --- | ---: | ---: | ---: |
| 探头/三角板，距离3m | 300 | 0.262800 | 0.320180 |
| 两根偏斜杆，距离2m | 300 | 0.266100 | 0.280025 |
| 最小次正规正距离/退化点凸包 | 300 | 0.362050 | 0.396690 |

计时包含证书计算、解析/重复断言，排除fixture与closest见证生成。首调用分别0.655400/0.273200/0.446600ms，未证明冷缓存，不称冷JIT。未隔离Fraction、KKT和有向开方成本，不能仅由端到端耗时断言哪一步是瓶颈；也未测网格批量、连续轨迹、峰值内存、GPU几何或Warp。

代码位置：`src/phymeo/geometry/feature_distance_bound.py`的精确权重/凸KKT与有向根包围；`scripts/benchmark_feature_distance_bound.py`固定微场景。后续若网格候选数量成为瓶颈，再独立分段计量并观察候选数、整数位宽与极端尺度影响；任何优化需保留原证明和最紧端点合同。本轮不做优化，不把host集成开销记入tyx_flaw。
