# PERF-018：仿射净距认证的区间细分与host成本

状态：观测，未优化。最终126源码hash `67ae299638326bf2018ee781a3bceadb5f54fbd84a7238f0740d9c27be3a6b60`，clean HEAD61ea634；脚本hash `3c87c97d88eec49efb4fededc5241c31dda5e5839d7029df5b16c265daca265c`。Windows11、Python3.12.14/NumPy2.5.4，host几何/有理证明，不是tyx实现问题。

复现：`.venv\Scripts\python.exe scripts/benchmark_linear_sweep.py --samples 100 --repeats 3 --output <new-output.json>`。[原始记录](measurements/p65-linear-sweep.json)三个独立路径串行，各3warmup与3×100计时；900计时/909总调用全部符合解析状态与重复/分区断言。启动前数值和审阅探针均退出，前后源/脚本hash一致。

| 路径 | 状态 | 每次区间数 | 中位ms | P95 ms |
| --- | --- | ---: | ---: | ---: |
| 分离探头 | certified | 1 | 0.369950 | 0.502225 |
| 中途穿越探头 | violation | 1 | 0.856700 | 1.015200 |
| 转向但安全的杆件 | certified | 15 | 22.026250 | 24.581875 |

包含整段认证、解析状态/违例/完整分区与重复检查，排除fixture设置。首调用分别5.512000/.910000/20.473600ms，不证明冷缓存或JIT成本。三种路径工作量/输出不同，不作后端加速比；未隔离closest候选、有理投影、遍历和结果校验的时间，不能仅由端到端数值把差异全归因某一行。

代码位置：`src/phymeo/geometry/linear_sweep.py`的候选/精确投影/有界遍历；`scripts/benchmark_linear_sweep.py`固定路径及测量口径。后续若实际网格候选数量成为瓶颈，先按候选次数/区间深度/有理数位宽分段观测，保持未决拒绝与完整分区合同；本轮不优化，不把host集成成本归于tyx_flaw。未测GPU CCD、FK弧、接触响应、Warp、峰值内存或完整轨迹。
