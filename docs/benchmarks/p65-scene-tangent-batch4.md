# 第四批：真实场景耦合切向与固定凸包复用

2026-11-06，本地已验收。产品提交 `8c4a7db568265a0d587ff38d9c6fb9a2499e3f5a`，基线 `9e1693110880f23245d8d2e780d37316f1d7d21b`。本报告之后只有文档收尾；未推送、未运行远程 CI。

## 本批新增的可用能力

`ContactScene` 和 `RobotScene` 新增显式 `enable_coupled_tangent(registry, stiffness_n_per_m=..., motion_tolerance_m=1e-6, energy_tolerance_j=1e-8, max_frame_turn_rad=.25)`，须在第一次推进前配置。公共 `tangent_history`、`tangent_receipt` 反映同一次成功物理步。例程见 [scene_coupled_tangent.py](../../examples/scene_coupled_tangent.py)。

适配器保留原法向采样的摩擦参数、求积与真实作用点，逐站替换选中旧切力，双方力、力矩与广义力只施加一次相同差量。自由速度包含外载、法向、重力、gyro，以及机器人 PD/bias；切力使用完整跨站点响应 W。真实积分只做一次，随后以原入步材料点和作用点分别重建末态位移，审核逐站功、弹簧能量和释放能量，再共同发布物理、历史、geometry、loads 与 events。失败不消耗接受步。

材料身份来自显式局部锚点、owner、chart 与可选完整 feature 顶点集合；当前实际稀疏支撑必须落在声明集合内。未声明时仍保守重建，不能凭相同 chart ID 自动续接。当前支持 HOST 刚体与普通纯 prismatic 机器人。粒子/柔体、mimic 和 revolute 机器人新模式明确拒绝；自由刚体转动仍须通过实际 exp-map 位移/功门。默认旧摩擦不变，原 mimic 100 g 连续持稳失败没有被本批修复。

独立报告：[切向适配核](p65-scene-tangent-kernel.md)、[材料站点](p65-scene-material-stations.md)、[固定凸包复用](p65-validated-hull-cache.md)。

## 多案例实际验收

| 案例 | 实际结果 | 结论范围 |
|---|---|---|
| 有限双面压载，1 kg / 0.5 N 横向外载 | 24 站点、48 步；真实法向合力约 4 N，0 个滑动站点步；横移 0.197 mm | 弹性黏着瞬态；末切力 −0.946 N，不把它写成已稳态平衡 0.5 N |
| 相同几何，2 kg / 3 N 外载 | 24 站点、48 步；552 个滑动站点步，横移 1.155 mm；末切力约 −2 N | 进入库仑滑移；法向和切向均来自同一真实场景 |
| URDF 同一移动关节、两个 pad | 4 个实际步，含 PD 与外载 | 共享祖先完整响应与关节反作用、公开载荷一致 |
| 有限台–球 contact→zero→reentry | 3 个实际步，接受历史 generation 为 1、空、2 | 弹簧只释放一次，不手写法向或直接删除历史 |
| 偏心转动、注入审核失败、公开回执元数据修改 | 负控均符合预期 | 真实材料运动不满足门时拒绝；物理/历史/观测不部分发布，公开回执不别名污染内部 |

源码外 wheel 的两条 48 步轨迹中，实际材料/施力位移误差最大分别为 1.525e−20 / 9.487e−20 m，能量余额最大分别为 6.439e−20 / 2.304e−19 J。它们是这两个平移案例的观测，不能外推任意旋转、长期持稳或一般物理精度。

## 执行记录与真实失败

完整证据：[本批目录](../../artifacts/p65-scene-tangent-batch4-20261106-01)。全部项目执行由父串行完成，子 agent 只隔离编码及交叉源码审阅。

| 独立组 | 通过数 | 原始收据 |
|---|---:|---|
| 切向适配核 | 31 HOST | 06-adapter-first-functional |
| 材料站点 | 23 HOST | 19-material-functional-fixed |
| 固定凸包复用 | 23 HOST | 18-cache-first-functional |
| 新场景共同推进 | 7 + 1 HOST | 20-scene-functional-complete；23-scene-real-unload |
| 原默认 scene 接缝兼容 | 34 HOST | 17-default-scene-compatibility |
| 缓存设备路径 | interpret 1、CUDA 1 | 21-cache-interpret；22-cache-cuda |

共 119 个 HOST 测试和 2 次设备专项通过。缓存 HOST 组主动排除的一个设备节点已在 interpret/CUDA 各运行；没有把中间失败后已通过的前缀重复计数。最终 Ruff check/format 均 RAW0，17 个 Python 文件中 15 个原 AST 相同，17 个非 import AST 和 import 绑定等价；格式修复未触发同源完整组重跑。元数据篡改负控触发的 NumPy deprecation warning 保留。

首次缺接口/模块 RED、缓存真实重复构造 RED 均保留。材料首次完整组 3 pass / 1 fail 暴露稀疏支撑 (1,7)→(1,5,7) 与完整面身份混用，修复采用预声明完整面，并保留默认重置/越界拒绝。父最初按四角登记，而原采样是每面 12 点的二次求积，故真实组先拒绝缺站点；改登记以匹配原采样，未改变求积或材料律。随后默认 pytest 临时根发生 WinError 5，改用每个 job 全新绝对 basetemp；未更改产品或清理他人目录。公开 receipt 别名有独立 RED，getter 返回拥有的副本后通过。所有首错日志未覆盖。

## 固定凸包复用的同次成本

缓存只复用完整绑定的不变 HullShape 边界/chart。真实球位置、表面搜索、材料见证、窗口几何与实际设备检查仍逐次执行。不同实际表示、移动路径、修订、构造器和工作区均不混用；不缓存接触结论，不称为 GPU 窄相。

以下为同进程 uncached/cached、每边一次 warm + 两个正式样本的公开 step+geometry p50。四档共 24 个完整公开帧；除原四个时钟和三个显式机制计数，状态、几何、历史、事件和全部原 backend 计数一致。完整无损 payload 同次核验，getter 均恢复。

| 模式 | 未复用 ms | 复用 ms | 比值约 |
|---|---:|---:|---:|
| HOST W1 | 6.421 | 3.390 | 1.89× |
| HOST W32 | 191.985 | 86.879 | 2.21× |
| CUDA W1 | 8.159 | 5.163 | 1.58× |
| CUDA W32 | 207.183 | 89.495 | 2.31× |

对应收据 30–33。W32 暖态每次 96 hits / 0 builds。两样本只支持本次有限同输入观察，不支持稳定尾延迟、其他几何或 Newton 性能优越性结论；冷启动/JIT 不混入暖态。

## 源码外交付与剩余工作

由 clean 产品提交 git archive 后，在独立临时源码目录构建 wheel/sdist，离线 no-deps 安装到独立 site；Python -I 从源码外运行公共例程。七个实际模块均来自安装 site，主环境和依赖 pin 未改变。34/35/36 三次命令均 RAW0；[installed-scene-result.json](../../artifacts/p65-scene-tangent-batch4-20261106-01/installed-scene-result.json) 为实际两质量结果。wheel SHA256：`bb6226c65188b2224e548e9f7176aef1bb0cf5a89fec0add24deb0759d50014d`。

本批已完成计划中的有限切片。下一优先项是旋转/铰接与 mimic 的真实运动摩擦、材料特征变化的自动管理，再接可变形材料；性能线继续整条管线的读回和 HOST 窄相，而非重复本次缓存矩阵。原软布双臂两初态尚未完成，不能用刚体样例替代。较大功能差异见[更新总览](../development/phymeo-newton-functional-overview-2026-11-06.md)。
