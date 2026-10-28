# P6.5 固定区间载荷重放验收

本切片产品代码`3f23378`。新增`SceneLoadReplay`和`ReplayTrial`：候选积分载荷转换为整个区间内恒定平均输入，每次都从完整原始state/history/input重算，保留机器人和布材料的原求解阶段。它是动态接触所需的重算基础，**不是自动接触外层、单爪或完整双臂验收**。

## 数学与事务证据

- 自由节点/平移关节：独立解析Δv=p/m、Δq=hΔv；多个子步共享总积分输入，不重复计冲量。载荷p→0→p与独立新实例一致。
- 给定平面、m=1kg、法向压力20N、切向g=9.81m/s²、T=2s，h=.01/.005s。μ=.6静摩擦为零末速度和零累积位移；μ=.2按半隐式Euler解析滑动，a=5.81m/s²、v=-aT、q=-a*h²*n(n+1)/2。它排除了只修末速度所致g*T*h漂移，但几何和法向载荷是解析预设，不能叫作检测到布接触。
- 双杆非线性机器人：独立平面质量矩阵求解总关节载荷，核对q/v；布XPBD独立解检查材料位移/乘子，重复trial不泄漏上一轮乘子。实际tyx两域路径验证非零执行计数，未用host候选覆盖。
- 注册额外状态/history、输入副本、返回对象被改写、最后子步预测/预算/资源关闭失败均覆盖；拒绝返回完整原状态，重试与新实例一致。
- 大背景吞掉小积分、溢出/下溢、注册构造器量化/联动字段/复制副作用，在workspace之前拒绝。账本记录host FP64实际输入差，不等同于证明FP32 program或外层接触已收敛。

## 独立审阅

见[审阅记录](p65-load-replay-review.md)：一次fresh review，0Critical/1Important/0Minor，构造后实际输入问题经4个RED→GREEN回归修复；相关新旧测试129项通过。先前未完成或修复前全量结果不计入最终验收。

## 后续必须完成

外层仍需从接触几何/材料速度求出并收敛物理载荷，确定M/J的构型，验证最终原方程残差、反作用一次及每域能量。再接保守CCD、弯曲/自碰撞、有限驱动单爪与低摩擦滑落、控制/IK/history、双臂递布和同源Three.js。不得把本页解析平面例或原P5显示说成完整夹持演示；完整步长/网格/扰动/重复/回滚门槛仍有效。

## 修复后工程结果与未闭环项

[机器记录](p65-load-replay-engineering.json)保存源码/包hash；[性能报告](../performance-bottlenecks/measurements/p65-load-replay.md)保留失败，不只列最快成功样本。

| 验证 | 实际结果 |
| --- | --- |
| 全量CUDA | 1605 passed，468.37s |
| Python3.11核心 | 1328 passed / 277 tyx deselected，338.64s |
| native-pip Python3.12核心 | 1328 passed / 277 tyx deselected，337.31s |
| 新核心＋interpret | 46 passed，2.38s |
| 新actual C++短功能测试 | 2 passed，42.42s |
| uv构建＋源码外全新pip wheel | 109文件字节完全一致，新API及求解smoke通过 |
| ruff / 公共tyx边界 / compileall / uv、pip check | 通过 |
| NumPy/interpret/CUDA串行计量 | 各300，共900次全部成功 |
| Windows C++300次长期计量 | **失败，未验收**，WinError1114；TYX-003 |

[TYX-003](../../tyx_flaw/TYX-003-windows-cpp-library-lifetime.md)独立公开API复现表明显式close+GC后DLL仍线性留存；需要tyx修复后重测。没有更改依赖pin、没有静默CPU替代或关闭物理门禁。功能/安装验证已完成，C++长期运行这一外部阻塞保持开放；不能把整个切片描述为所有后端完整闭环，更不能标完整P6.5完成。可继续NumPy/CUDA动态接触与几何的独立开发。

远程CI在本报告推送后按准确head另核对；旧8223c74两项CI均成功，但不能替代本次新head。
