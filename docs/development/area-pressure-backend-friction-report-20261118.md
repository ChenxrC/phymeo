# I3u：可复用的128点CUDA面积摩擦后端

2026-11-18，codex/exp源码25d2abf3。首次完整4节点（2 HOST、2 CUDA）通过，5.74s；最终Ruff/format通过。将已验证I3r算法迁入可安装src模块，并将每世界样本从8扩到128，尚未接管生产Scene状态积分。

## 接口与边界

- [owned输入](../../src/phymeo/solvers/area_friction.py)：AreaFrictionSample、AreaFrictionProblem，law为final_normal_smooth_v1。
- [CUDA workspace](../../src/phymeo/backends/tyx_area_friction.py)：solve_many，1..32世界、1..6广义DOF、0..128完整样本。
- [kernel](../../src/phymeo/backends/_tyx_area_friction_kernels.py)：同一最终速度共同求Fn/Ft，完整非对称Newton导数、部分选主元、固定24更新和12回溯。HOST只代回验证，不修正速度或力。
- [测试](../../tests/test_tyx_area_friction.py)：独立Fraction制造解、设备字节、完整方程、故障和资源生命周期。

M为完整正定质量矩阵，D为非负对角关节阻尼；applied_force不含该隐式阻尼。完整JA/JB共用同一广义坐标空间。k已包含真实dA，不能再乘一次面积。模块不生成面、不积分q，不修改旧type7/v1观察或friction=none资产。

## 新验证

128个点的位置均不同，两可动物体的六坐标Jacobian同时参与，各点k/phi/mu变化；独立有理算术给已知速度与外力。12点斜法向holdout的法向三个分量均非零，切向非坐标轴。W32混合128点、12点和空世界，末世界为128点；另反序全部128点检查工程等价，不宣称归约次序逐字节相同。

| 调用 | 最大速度分量误差 | 实际上传值的独立质量残差 | 最大Jacobian误差 | 最大Newton更新数 |
| --- | ---: | ---: | ---: | ---: |
| W1 / 128点 | 2.519e-5 | 2.2114e-6 | 3.940e-6 | 3 |
| W1 / 12点斜面 | 4.200e-5 | 1.7963e-6 | 3.506e-7 | 1 |
| W32 / 混合 | 4.200e-5 | 2.2114e-6 | 3.940e-6 | 3 |
| W1 / 128点反序 | 2.516e-5 | 2.2105e-6 | 3.940e-6 | 3 |

门沿[I3r](area-pressure-tyx-coupled-friction-report-20261118.md)：设备1e-5质量尺度、独立代回2e-5尺度；速度每分量5e-5(1+|vref|)、力.002N+.002relative、矩1e-5Nm+.002relative。上述所有结果均未通过另列的1e-10 HOST参考门；无隐藏polish或为追更小误差重跑。

## 容量与事务

32世界/128点注册14个buffer共1,770,624B；每次全上传761,216B、完整诊断下载1,009,408B。8个向量scratch、290条尝试记录未缩减。context/JIT/allocator不计入注册字节。四次成功调用合计kernel约.737s、workspace累计约1.534s，包含首次编译及HOST核验，不用作性能比较。

非法末world（129点、类型/下溢/FP32正定性等）在上传前拒绝。晚world预算失败保留整批last_results并允许明确重试；晚下载错误保留旧发布并使workspace faulted。UNKNOWN_RELEASE仅mock验证所有权。没有静默截断、删点或重复点分刚度冒128点几何。

## 同次证据及下一步

artifacts/area-pressure-backend-friction-20261118-01/06-payloads的11个完整gzip保留原/实际FP32输入、结果、trace、下载字节、独立方程与故障。saved-summary.json只读取已结束输出统计/清单，没有重跑producer/verifier；tested-source-before/after四文件相同。

01首Ruff0；02首format1（两文件排版）；03format修复0；04/05最终静态0；06首次focused4pass/0。一次PowerShell参数绑定错误发生在guard/pytest前，原文本保留，不计额外pytest或产品失败。

下一步是同一真实入步M/J/bias/passive/applied装配，再接共同Qcontact、原生积分和新观察。当前仅可复用数值模块，不等于机器人面积摩擦抓持完成；见[生产F1–F6计划](../superpowers/plans/2026-11-18-area-friction-production-integration.md)。