# 柔性三角面、自接触与 Newton VBD

这批 11 个冻结构型使用同一组米制顶点与原材料 ID。两张独立三角形的最近特征分别落在顶点–面和边–边，另外有共边相邻面及单张有 16 面的折回带。初始速度、重力、结构刚度、阻尼和摩擦均取零。公共参考入口使用 `--lane dd`，按 `--engine newton|phymeo --device cpu|cuda:0` 将四个环境分开运行。[案例库](../../tests/reference_newton/dd/cases.json) · [独立几何真值](../../tests/reference_newton/dd/oracles.py)

### 检测余量与实际作用范围

Newton 1.7 的 VBD 使用两层距离。本批 `particle_self_contact_margin` 为 1/16 m，加 `particle_self_contact_gap` 1/32 m 得到候选距离 3/32 m。检测使用严格小于。一条边–边对按两个原始边索引各产生一个有向记录。程序在容量足够且溢出标志为零后解码原始数组，逐对读回原始行号、材料 ID、实际最近点和重心权重。各顶点或边的最小距离数组也保留在收据中。[官方 VBD 求解器](https://github.com/newton-physics/newton/blob/v1.7.0/newton/_src/solvers/vbd/solver_vbd.py) · [官方网格检测](https://raw.githubusercontent.com/newton-physics/newton/v1.7.0/newton/_src/geometry/tri_mesh_collision.py)

Phymeo 的普通 `ph.detect` 返回零厚有限三角形之间的实际面间距。这里特意将粗筛余量设为 1/4 m，使外侧对照也能取得最近材料点。之后通过 `CollisionLoadSystem.sample_particles` 分别给各张脸绑定材料节点并读取两侧实际节点力。两引擎均使用 128 N/m 的接触刚度。Newton 从 VBD 正常使用的按颜色累积核读节点力，三个选定的校准姿态另保存实际一步状态。Phymeo CUDA 的粗筛和法向力核与其 HOST 几何见证分别记数。[Phymeo 适配](../../tests/reference_newton/dd/phymeo_runner.py) · [Newton 原始缓冲、力与状态读回](../../tests/reference_newton/dd/newton_runner.py)

| 初始构型 | 几何距离 m | Newton VT / 无向 EE | Newton / Phymeo 层力 N |
| --- | ---: | ---: | ---: |
| 顶点对面，独立两布或镜像换序 | 3/64 | 1 / 0 | 2 / 2 |
| 等于作用距离 | 1/16 | 1 / 0 | 0 / 0 |
| 只在探测余量内 | 5/64 | 1 / 0 | 0 / 0 |
| 等于探测距离、超过探测距离 | 3/32、7/64 | 均为 0 / 0 | 0 / 0 |
| 单顶点深压 | 1/64 | 1 / 0 | 8 / 6 |
| 独立两边或不对称镜像两边 | 3/64 | 0 / 1 | 2 / 2 |
| 同一布面合法共边 | 0 | 0 / 0 | 0 / 0，Phymeo 返回 `adjacent` |
| 同张布带远处折回 | 3/64 | 8 / 24 | 下侧合力幅值 64 / 18 |

表中每条 Newton 候选都经独立真值逐项核对。顶点与有限三角形比较面内投影及三条闭边，边–边比较两个有限线段内部的驻点与四个端点–线段组合。所有输入坐标均作为二进制有理数进行比较。原始三角面的无向材料边构成拓扑图，依次剔除共点原语及当前二环设置排除的邻接项。缺少 Newton 候选时，真实几何距离由独立真值和 Phymeo 面见证给出。

### 双方节点与时间

在 3/64 m 的顶点–面基准中，顶点 53 得 +2 N，三角面节点 11、23、37 得 −0.5、−0.5、−1 N。两边异长的边–边留出把 −2 N 按 3/4、1/4 分给一侧，+2 N 按 1/4、3/4 分给另一侧。CPU 和真实 CUDA 都复现了这些材料分配。Newton 经过一个 1/256 s 的 VBD 步后，顶点–面原材料位置沿法向的分开量为 0.000977519 m，边–边为 0.0000858403 m。在仅有候选、无实际力的 5/64 m 姿态，两组节点保持静止。

深压案例的几何和材料分配相同，但刚度函数不同。Phymeo 线性层在 1/64 m 给 `128 × (1/16 − 1/64) = 6` N。Newton 在其正的对数屏障区给 `128 × (1/32)² / (1/64) = 8` N。两引擎的三个面节点分别读得 (−1.5, −1.5, −3) N 和 (−2, −2, −4) N。

单张折回带更能区分力的计数方式。两引擎对目标面间距均复查得 3/64 m。Newton 累加 8 个顶点–面和 24 个无向边–边的原语力。Phymeo 对 9 组最近面片对施力，真实 TYX 法向核处理 30 个材料站点。取下侧节点 100–105，原始 z 向合力在 Newton CPU/CUDA 均为 −64 N，在 Phymeo HOST 为 −18 N、TYX CUDA 为 −18.00000036 N。整带节点力和在各自 0.0003 N 核对界内为零。该构型的每个浅压原语或面片对都处在线性区，差异来自参与相加的几何项。

离散查询、施力距离和推进后顶点位移作为三个独立读数。矩阵采用单步 1/256 s，对 Newton 动态观察只检查实际两侧分开或无作用时维持静止。此矩阵没有连续首触真值。[Newton 原生接触力实现](https://raw.githubusercontent.com/newton-physics/newton/v1.7.0/newton/_src/solvers/vbd/particle_vbd_kernels.py)

四个显式引擎/设备组合均有 11 项通过，共 44 项。Phymeo CUDA 的折回组记录一次法向层核启动、30 个站点、6,656 B 上传和 2,560 B 读回。真实粗筛来自独立 `CollisionWorkspace`，几何见证、材料散力和规范化索引由 HOST 完成。完整节点力、原始索引、设备名、步前步后顶点及源文件 SHA256 位于 `artifacts/p65-night-dd-vbd-20261104-01/dd-{newton,phymeo}-{cpu,cuda}/`，每个目录均保留独立退出码。
