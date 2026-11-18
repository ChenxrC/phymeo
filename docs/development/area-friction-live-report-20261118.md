# 当前场景的面积摩擦：B1/B2/B3 完成报告

更新：2026-11-18，codex/exp。承接[三批计划](../superpowers/plans/2026-11-18-area-friction-production-integration.md)。本轮没有改 dev、push、安装或重建 wheel。

## 新增能力

公开 MuJoCoTyxHeterogeneousAreaPressureScene 现在可以显式选择 final_normal_smooth_v1，在同一步中完成当前几何、全部积分点的法向和摩擦共同求解、原生运动积分、状态与力共同提交。没有 CPU mj_step、历史轨迹回放施力或先法向后摩擦 kick。

```text
ModelSpec：有序 body/material + mu/vs + law/精度版本
 → 当前 accepted q/v → body FK → 全部当前有限面与真实 dA
 → 同一设备 owner 的 M/J/bias/passive/applied/D
 → resident pack → 完整 Fn/Ft 非线性共同解 v*
 → 同一原生 q/WXYZ/local-omega/Kahan-clock 积分尾段
 → actual v、物理 M 加速度、A=M+hD 方程、全力与全矩核验
 → 原子提交 q/v/clock + 新摩擦 receipt/设备记录
 → v2 混合观察、检查点、selected reset、恢复与重演
```

法向仍是 Fn=max(0,-k(phi+h*vn))；切向 Ft=-mu*Fn*vt/sqrt(vt²+vs²)。k=dA*g、phi=-p/g，完整积分点保留；微滑属于正则化模型，不声称精确静摩擦或全球接触势能守恒。

## 为什么改了积分接缝

原 I3x 四个历史入口支持“存储 Q 交回原生 solve”，但连续 live 轨迹暴露其边界：质量范数可掩盖小惯量方向的速度误差，重新从近似 Q 解出的 FP32 vnext 与共同求解器 v* 不完全一致，进而影响最终摩擦和力矩。

第一处诊断增加 live 内部逐分量 A^-1 residual 停止条件；双区窗口随后在第 5 步仍被实际力矩门拒绝。按原计划的备选实现，共同速度现在直接进入原生坐标尾段，旧路径仍走原模式。没有放宽共同残差 2e-5、速度 5e-5、力 0.002 或矩 1e-5 的原工程门。

物理 qacc 仍由 M^-1(base+Q) 得到；integration_acc=(v*-v0)/h。两者及 A*integration_acc=base+Q 仍独立检查，阻尼只计一次。原 q、WXYZ、local omega、时钟代码共用且只执行一次，失败候选不翻 accepted bank。

## B1/B2 结果

唯一最终组合 run23：29 passed / 25.80 s / RAW_EXIT=0。包括：

- HOST law 声明与版本：9；实际 CUDA 场景：2。
- CUDA 双侧和生命周期：4；HOST 读回协议：8。
- 准入 HOST：5；真实 CUDA 多法则分桶与末候选恢复故障：1。

实际场景覆盖 mu=.5 / mu=0 / 旧 normal-only / 空接触 D40；三步推进、上离卸载及有效空面。保留完整物理 M、双方点 J、入步外力/阻尼、原生坐标对照和实际发布载荷。

双侧用当前 hinge 与 moving plate 的完整 JA/JB，核双方广义反作用、虚功与世界力矩。native-only 桶保原路径。selected reset 只变所选 world，控制相位、检查点重演的数值字节保真；新 owner 配置身份单独核验。

故障覆盖真实 pack/common-solve 结束后的错误状态传播、最后桶下载、整批提交前、提交后的观察，以及所有 replacement 已完成后最后一个 restore 报错。前提交失败保原物理/力/receipt/geometry；后提交观察失败仅重试观察；restore 失败保原 owner，随后同 capsule 可重试。伪造 Fn/Ft/v/solver status、错 mu/vs/材料身份均拒绝。

新 AreaFrictionEntryReader 在真实混合设备 frame 上解码，并与 accepted receipt 全样本匹配。v1 reader 明确拒绝 v2；旧法向行的新增字段是零填充，不捏造相对滑速。

## B3 连续轨迹与规模结果

最终 source 的 run16 双区窗口 1 passed / 6.20 s，run25 其余连续与规模 4 passed / 62.22 s。原资产、载荷、h、摩擦参数与误差门保持；仅规模测试在首次执行前为三个尺寸一致声明既有 64 MiB 显式预算。

| 实际案例 | 步数 | 观察到什么 |
| --- | ---: | --- |
| 双接触区、三个可动物体 | 40 | 完整双方 J 与共同方程、实际 Fn/Ft/矩、一次原生运动共同通过 |
| center-drop 自由体 | 150 | 冲击、卸载、分离、再次接触；均从同条接受轨迹识别 |
| offset-support 自由体 | 150 | 持续支撑与偏心转动；未观察到上述冲击/分离事件，明确保留 NOT_OBSERVED |
| 同桶滑动与释放两 world | 各20 | 支撑20步、至少10步有实际滑动、累计摩擦功为负；释放世界首步有面但零力，后19步为空面 |
| 同模型 1/4/32 world | 每组3 | 全部数值输出与同次单世界逐字一致；world/config 身份分别核验，不拿它们冒数值差异 |

独立 compatibility run26：20 passed / 16.34 s，覆盖原 standalone solver、历史适配、resident pack、原生 bridge 与 body wrench 五个文件。以上54项是本轮相关功能与兼容范围，不是全仓库验收。

### 整步成本

下表测量完整 Scene.step + 观察刷新/读回 + integration_snapshot，包含当前几何、装配、求解、积分和生产核验；不包括测试的独立 oracle 与证据写盘。沿用现有 JIT cache，第一步不声称冷缓存。后续仅两步，因此是定位下一阶段工作的有限成本记录，不是稳定性能排行榜。

| world数 | owned场景初始化/s | 首个测量步/s | 后两步/s | 后两步吞吐/world-step·s⁻¹ |
| ---: | ---: | ---: | --- | ---: |
| 1 | 0.0704 | 0.4804 | 0.1446 / 0.1474 | 6.85 |
| 4 | 0.0719 | 0.6566 | 0.4295 / 0.4565 | 9.03 |
| 32 | 0.0979 | 3.4722 | 3.2398 / 3.3167 | 9.76 |

共享 fixture 声明另计0.1169s。后两步中的 Scene.step 分别为1世界0.1370/0.1419s、4世界0.4206/0.4498s、32世界3.2282/3.3067s。大部分时间落在 Scene.step；尚未拆分几何、求解及核验各自成本，不能先归因于几何，也不能宣称达到训练规模效率或快于Newton。

| world数 | solver manifest payload/bytes | observation payload/bytes |
| ---: | ---: | ---: |
| 1 | 95,752 | 24,540 |
| 4 | 379,276 | 98,088 |
| 32 | 3,025,500 | 784,536 |

这些是已声明设备缓冲字节，排除HOST、MuJoCo模型、CUDA上下文、JIT与allocator额外开销，不能当作整机显存。步间无allocation增长，关闭后本场景预算的current/reserved归零。

### 真实证据查看器

查看器只读取本轮保存的完整证据，展示当前几何、裁剪面、Fn/Ft、实际接受的运动和载荷曲线，不重新运行物理。[打开离线查看器](assets/area-friction-live-20261118.html)：默认选择最终run25的冲击轨迹，可切换偏心支撑、双区、滑动和卸载。蓝箭头为法向力，橙箭头为摩擦力，绿色为实际裁剪面；拖动滑块比较力和运动。run13/14的PASS保留为旧积分路径的历史尝试，不属于当前版本验收；当前最终轨迹来自run16/25。失败尝试单列；不将未提交候选画成运动。画面允许减少帧、箭头与面片用于显示，原始数据不删，全部帧的载荷趋势和全样本合力保留。

## 后续顺序

1. 扩大自由度及同世界 native/area 共同方程，先给定异构机器人、自由物体和多接触案例；不得把两个独立solver串接称为共解。
2. 对已经保存的整步成本做分段分析，再决定设备几何、装配或核验哪个值得优化；同模型、同输出比较才讨论收益。
3. 将新v2面积摩擦观察接到真实训练输入。新磁盘law封套、一般有限边缘、变形体散射分别有独立准入和验收，不用单个任务替代它们。

## 使用与边界

在原 model 声明上显式设置 area_friction_laws，按 area_pressure_pairs 顺序传入 AreaFrictionPairLaw(identity, mu, slip_speed_m_s)，再用原 Scene 入口。body-local 几何资产仍可复用；旧 asset bundle v1 的 friction=none 不可直接恢复为新物理模型。

范围：CUDA，每桶 1..32 world、1..6 DOF、完整点≤128（三点三角积分实际≤126）、pair≤16、body/geom≤128；冻结入步几何/J，逐片 g>0。新 law 的同世界仍是 area-only，拒绝 BALL/activation、limits/dry、equality/tendon 和剩余 native pairs。旧 native-only 与 normal-only 可以在别的桶同场景存在。

当前几何仍由 HOST 或设备提出拓扑加 HOST 精确复核；不称全设备碰撞、面积 CCD、29 关节摩擦机器人、任意有限边缘或同世界一般混合约束。g=0 且 p>0 仍需显式拒绝，不能加 epsilon 或删面。FFT/Kamino 与通用自动算法选择仍是独立路线。

## 原始证据与失败记录

本轮根目录：`artifacts/area-friction-b1-live-20261118-01`。保存每次实际命令、fresh guard、stdout/stderr、RAW_EXIT、源清单与完整 gzip 输入/输出。run15/22/24/28/32/34/36/41 在预检发现其他执行而未启动项目；不是测试失败。

首轮语法/导入检查及格式检查失败保留；修正仅导入/排版。run06 NumPy 标量转换、run07 oracle 对 FMA 的零值等式、run09 测试控制集合问题均保原日志。run11/12 实际速度不一致、run14 实际矩不一致是结构修复的原始证据，不改写为通过。run21 对最终 36 个 Python 文件确认 34 个 raw AST 同、2 个仅顶层导入顺序同绑定，其余改动之前的测试结果按各次源码说明。

Ruff run30 和 format-check run31 均 exit0，覆盖查看器schema修复前的36个Python文件。run29前一次父进程使用错误的 PowerShell 数组传参，在 wrapper 和项目执行前被参数绑定拒绝；随后改为结构化request文件，原记录保留，不算产品测试失败。

查看器run39首读把continuous统计对象当作window名称，报TypeError；修复为按两个已有schema分流，run40实际生成成功：14个world/尝试视图、10个原始payload、979,286 bytes，未执行物理。1280×1000与390×844浏览器均无横向溢出；播放/暂停/滑块/放大、0/20失败禁播、4/40失败前缀及全部成功案例切换已检查。console唯一错误是临时静态服务没有favicon的404，无脚本错误；仅用于检查的本机服务已关闭。run38最后两文件证明：viewer排版raw AST同，scale仅顶层import排序；随后viewer schema修复按run40和实际浏览器单独验证，不冒AST等价。截图与检查记录保存在同一evidence目录。

验证边界：schema修复后的查看器已经由run40真实生成与实际浏览器检查通过。其单文件Ruff/format复检在run41及后续有界等待中被外来长时执行占用串行槽，未启动，因而没有新的静态通过声明；本轮不会与该外来执行并行，也没有遗留后台队列。下一次串行槽空闲时可只补此文件静态检查，物理组和已通过的其余35文件不重跑。
