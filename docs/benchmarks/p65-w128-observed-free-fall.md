# W128 异构自由球的逐步 CUDA 状态对照

本机 RTX 4060 Laptop GPU 上，128 个独立世界以各自的初始高度和竖直速度推进，应用每一步都同步读回全部球的位置与质心速度。在三个独立进程配对中，Newton 1.7.0 的稳态应用吞吐分别为 Phymeo 的 **1.402、1.476、1.793 倍**。这是无接触自由刚体的实测，布面承载另有独立的 Phymeo 内部比较。[原 W1/W32 相同发布合同](p65-g2-typed-newton-direct.md) · [原生和 TYX 驱动](../../tests/scale/g2_cross_engine_steps.py)

| 配对及先运行端 | Phymeo 32 步 p50 / p95（ms） | Newton 32 步 p50 / p95（ms） | Phymeo / Newton 核验世界步每秒 |
| --- | ---: | ---: | ---: |
| 1，Phymeo | 10.547 / 14.753 | 7.760 / 8.772 | 370,830 / 519,723 |
| 2，Newton | 10.621 / 13.716 | 7.121 / 8.136 | 374,863 / 553,250 |
| 3，Phymeo | 13.894 / 24.162 | 8.025 / 11.095 | 269,034 / 482,267 |

每世界一个质量 2 kg、半径 1/8 m 的球，重力为 (0, 0, −8) m/s²，步长 1/128 s，碰撞关闭。按登记次序，初始高度从 2 m 起每球增加 1/8 m，初速大小为 (i+1)/16 m/s，正负交替。两端使用同一逆序非连续世界键和同一个应用函数，把每一行按键排序并复制为六列 FP32，再对照离散解析解。Newton 使用真实 `SolverSemiImplicit.step` 双缓冲，Phymeo 使用 `TyxWorldBatch.step` 和 `kinematics()` 的已提交步号。每个进程先测一个物理步，再做 5 个 32 步预热窗和 20 个 32 步计时窗。每端每轮核验的计时样本为 81,920 个世界步。聚合吞吐用全部正式世界步除以正式窗总时长，p95 取 20 个有序样本中的第 19 个。[计时和轨迹保存](../../scripts/benchmark_newton_free_fall.py)

冷步和稳态分列。三轮每端都以空的指定框架缓存目录启动。Phymeo 的第一个物理步为 397.84 / 379.77 / 198.10 ms，Newton 为 24,842.25 / 529.12 / 567.70 ms；第一轮 Warp 的刚体核编译日志记录 24.19 s。建模另耗 Phymeo 383.44 / 381.78 / 262.87 ms，Newton 1,884.57 / 1,136.06 / 955.75 ms。配置缓存为空也可能使用既有操作系统、驱动或编译器缓存。

稳态每轮双方各调用 640 次物理步并显式同步 640 次。Phymeo 实测 640 次 CUDA 核、4,587,520 B 设备回传，其中接受标志为 327,680 B，运动字段和步号为 4,259,840 B，稳态上传和设备新分配均为零。Newton 通过 1,280 次公开数组读取，读回 4,259,840 B 刚体姿态和速度。额外一个窗外 Warp 活动审计步观察到关节求值核、刚体积分核各一次，D2D 一次和 D2H 两次。每世界每步的应用输出两端均为 24 B。

两端各自对全部 801 步逐项验证，位置与速度相对半隐式解析解的最大误差均为零。原始文件保留初态及每个出步，共 802 × 128 × 6 个 FP32。六份 NPY 的文件 SHA-256 均为 `61091f530e3d7064ad81d5ff8da3e65cfaac5810a3b6cfffcc5510042a3cf4d8`。逐个世界另以动量差减重力冲量、位移减末速乘步长、机械能单步变化加 ½m|g|²h² 核验三项离散残差。三轮中最大残差均分别为 **0 N·s、0 m、0 J**。第 801 步时，世界键 17 的 z 与 v_z 分别为 −189.02392578125 m 和 −58.0625 m/s。[完整逐比特和力学核验程序](../../scripts/check_heterogeneous_free_fall.py)

可在各自解释器运行 `python scripts/benchmark_newton_free_fall.py --engine <phymeo 或 newton> --device cuda:0 --phase measure --worlds 128 --asymmetric-starts --warmup 5 --samples 20 --expect-empty-cache --cache-root <新缓存目录> --out <新目录>/report.json --trajectory-out <同一目录>/trajectory.npy`。Phymeo 命令另加 `--phymeo-observation kinematics`。父树 `artifacts/p65-night-multiface-and-newton-20261105-01/n1-r{1,2,3}-w128-{phymeo,newton}` 保存六份实际轨迹、全部窗口和初始化记录。
