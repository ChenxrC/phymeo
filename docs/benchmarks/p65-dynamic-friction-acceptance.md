# P6.5 两阶段自动摩擦：实现与验收进度

状态：生产实现6df5b14已提交、实现报告cb209b8已交付；唯一fresh审阅已完成0Critical/0Important/1Minor，M1已通过合同澄清与4项定向控制/27项事务回归闭环，最终串行验收已全部完成（session27196、完整core311/nativepip各1945通过/362未选，665.54/657.52s；完整CUDA2307通过6196.34s、Ubuntu新核心80通过223.65s、有界CPP1通过23.15s，全部exit0）；质量、134源码外wheel和新nativepip安装smoke通过。正式NumPy/interpret/CUDA各300计时/306总调用已独立核对，共900/918成功；准确产品远程SHA/CI仍待交付核验。不是完整H1/P6.5完成。134源码hash `d18e557c49b9711647d7b5218f9706739227eab84073c9a6a309e6276c44b33d`。

## 框架能力与边界

`DynamicRobotFrictionContactProgram`使用显式owned摩擦映射，自动求held法切冲量和terminal末法切冲量。支持全prismatic机器人、全动态distance材料或None、固定朝向无限平面、零半径及零恢复；原normal程序和旧collider的μ0合同不变。案例仅配置和调用。

初态closed行的held摩擦容量μp、使用真实pre滑速；初态open行held切向为零。terminal对初态closed行用μη，对新建立行用μ(p+η)，在真实post滑速验收。多面共享响应经完整原生重放，实际world精确原式、材料/机器人账本与物理预算共同接受。该分阶段定律不声明总冲量末态Coulomb或统一最大耗散。

完整pre状态/路径/history与选中held一致；末作用点使用当前x1，真实live末冲量一次，不能把P+E塞回held载荷。初态几何重新分类，history只记录不warmstart；分离清零，后子步或已有best后发生真实失败仍全帧回滚。

## 现有证据与版本

- 最终相关CPU483通过、11未选，394.21s，含76个新CPU例和旧受影响集；不是整分支全量。
- 新actual CUDA11通过1020.29s，覆盖两解析、两面/生命周期、T0.5两μ两步长重建、FP32拒绝与真实计数。其后仅新增非法collider构造类型guard；不能称该次运行包含后加guard。
- 最终源码actual interpret核心7通过、4未选，12.10s；早版interpret完整8通过229.31s含设备T0.5窗。NumPy窗口为T2，设备窗口为T0.5，范围不混写。
- 独立支撑/滑动、材料斜撞、非平行A/B、共享祖先、三维三角、变换顺序与故障事务已有控制。容量中间舍入、旋转纯切向的负normal表示、非法对象构造均经历真实RED修复；立即GREEN的新增测试不称RED。
- 源码有限smoke与九文件ruff通过。首次运行完整installed-smoke入口被既有源码位置保护拒绝，后runpy仅调用新增函数；这是实现阶段source smoke；最终pipeline已另外完成134源码外wheel与安装smoke，不能混为同次执行。
- 父NumPy→interpret→CUDA最小基准各2计时+6warmup成功，源和脚本hash前后相同，仅验证基准程序，不作为正式性能样本。

精确日志和版本在[工程记录](p65-dynamic-friction-engineering.json)；[唯一审阅](p65-dynamic-friction-review.md)已结束，M1回归已核对关闭。成功帧累计所有真实trial，含未选best；既有失败report清空diagnostics，不提供部分launch总数，故障调用/关闭/回滚由独立探针验证，不冒称失败性能已计量。成本观测入口见[PERF-020](../performance-bottlenecks/PERF-020-dynamic-friction-contact.md)。

## 本地最终证据

同一冻结摘要ea6e07e3…下16步骤全部exit0，pipeline session27196已结束。只读汇总器独立核对所有退出码/pytest摘要、134源码外wheel逐字节一致，以及三模式raw计数、顺序、物理预算、launch完整性、源码脚本前后hash和中位/P95。wheel SHA93e8d659b438afee09a2591312be7b27ba9846c3e9b191e3c26df5bd1cacd88a。

原始数据：[NumPy](../performance-bottlenecks/measurements/p65-dynamic-friction-numpy.json)、[interpret](../performance-bottlenecks/measurements/p65-dynamic-friction-interpret.json)、[CUDA](../performance-bottlenecks/measurements/p65-dynamic-friction-cuda.json)。每模式support/chain各150计时、各3warmup；不把固定初态重复称连续轨迹。性能只记录，未优化。

## 剩余交付

本地验收和正式计量已完成；提交push专用分支后核对准确SHA与CI，才关闭本阶段交付框。之后继续有限特征/切换和FK保守CCD、弯曲自碰撞、有限单爪承重/低μ滑落、目标IK/history与真实双臂递布和同源Three.js。按用户最新[分层选测](../development/p65-validation-policy.md)减少重复旧回归，整个流程步长/网格/重复/扰动/事务门槛不减。NUM-005、TYX-003/004保持开放，未测Warp不称差距。
