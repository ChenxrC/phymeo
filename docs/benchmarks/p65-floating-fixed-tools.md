# 浮根固定工具：冻结输入与验收

状态：AUTHORING / UNEXECUTED。起点 4392249；首 tests-first 55ecf6c。作者未执行 import、测试、数值脚本或静态检查。父已保存 69-floating-fixed-entry-red：旧默认 fixed 拒绝先通过，新 collapse_fixed_joints 参数产生有效 TypeError；1 failed / 0.70 s / RAW1。不重跑首红。

公开入口是 load_floating_urdf(..., collapse_fixed_joints=True)，默认 False。新资产保留原 root q 的坐标轴和原始链接材料坐标，沿真实物理组汇总正质量、COM、COM 惯量；不制造标量关节、不使用 weld/pin、不改积分器或接触力律。仅 HOST 场景；未宣称浮根 CUDA、mimic、pads、任意非对角根惯量或浮根 interval CCD。

## 输入与独立真值（执行前冻结）

F01 文件 floating_fixed_composite.urdf：根 2 kg、I=(1/5,3/10,2/5)，固定工具 1 kg、I=(1/10,1/10,1/10)，安装偏移(1/4,0,0) m。聚合 M=3、COM=(1/12,0,0)、I=(3/10,53/120,13/24)。工具原点 F=(0,3,0) N 的组 COM 矩=(0,0,1/2) Nm；零速零重力时 a_y=1 m/s²、alpha_z=12/13 rad/s²。一步门保持 COM 漂移与 root-origin 位移分开。另连续 1/4 s 外载窗口不重置；整体轴循环/平移留出。解析 FP64 门为 1e-11 量级。

F02 文件 floating_fixed_probe.urdf：
- 根 m=2、I=(.2,.3,.4)；固定 mount m=.5、I=(.02,.03,.04)，相对根 x=1/4。
- arm 在 mount 上以 z 转轴连接，m=1、COMx=1/4、I=(.05,.075,.1)；关节范围[-1.5,1.5] rad，力矩帽16 Nm、速率帽4 rad/s。
- carriage 沿 arm x 滑动，关节原点 x=1/2，m=.5、I=(.025,.025,.025)；范围[-1/8,1/4] m、力帽32 N、速率帽2 m/s。
- tool 固定在 carriage 的 x=1/8，绕 z=π/2，m=.25、I=(.01,.02,.03)；有限盒尺寸(1/8,1/4,1/8) m。无质量 witness 在原 tool 局部 x=1/32。
- 根组 M=5/2、COMx=1/20、I=(11/50,71/200,93/200)。活动组数2，物理体数3；全部5个正质量原部件参与独立世界惯量/质量矩阵/功率求和。
- 姿态留出(q_R,q_P)=(0,1/8)、(.37,-1/16)、(-.51,3/16)；第二姿态整体xyz轴循环并平移(.5,-.25,.75)。材料点=(1/32,-1/64,0)；外力由局部(-1,2,.5)旋转得到，驱动=(1/4,-1/2)。期望由五原部件及串联几何构造，不读取产品 M/J/bias 生成 oracle。

F03 被动弹性窗口（物理回退，不称主动抓持）：
- F02 q=(0,1/8)，root COM 初速(0,1/4,0)，角速/关节速率0。
- 独立自由球 m=1 kg、r=1/16，位于 tool 的世界 +Y 有限侧面外 1/1024 m；未旋转时球心(1,1/8+1/1024,0)。
- RobotContactMaterial k=2048 N/m、d=0、μ=0；重力、外载和全部驱动为0。
- T=1/4 s，h=1/2048 与1/4096；全程实际推进，不回填位置、不重置。不按输出调初态/阈值。
- 要求开始无载→正载→反弹离开，记录真实 root/关节/球轨迹、全部 site 力和共同作用点、冲量、双方总矩/瞬时功率及事件时刻。
- 比较同物理终点的 h/h2：位置差≤5e-4 m、平移速度差≤5e-3 m/s、角速度/关节速率差≤1e-2（rad/s或m/s）、累计球冲量差≤5e-3 Ns。这是有限窗口工程精度门，不是一般收敛定理。
- 自由球动量变化与完整入步冲量门1e-10 Ns；机器人+球实际末态总线动量残差门5e-4 Ns，并分别报告，不将入步广义力平衡冒作末态严格角动量守恒。瞬时作用反作用/共同原点矩与功率按1e-10尺度检查，不称能量守恒。
- 反向登记与整体刚转留出同输入同物理结果。若未出现冻结阶段或跨门，保留实际失败，不用时间标签代替测量。

F04：无正质量组、非对角根、无效惯量、fixed控制/mimic、错误site/shape owner明确拒绝；合法无质量标记为正控。两个build和外部数组无别名。第二参与者或晚期事件失败时 q/v、time、history/events全部保持，修复重试与独立实例一致。

## 源码边界和后续

作者仅六个 NEW 路径：新资产模块、专项测试、两份URDF、公开例程、本报告。父维护旧 loader opt-in 与浮根 contact_view 转换桥。sites 使用真实 body_index 0=root、i+1=child；collider.link_index 保留资产 -1=root、i=child。root 工具转换后仍为 root-local，不是静态 world。

Newton 仅引用已验收的 pinned 1.7.0 机制边界；本片未运行新的原生对照。旧 fixed-collapse 惯量反例不丢弃，后续新参考应保留原生 fixed joints 与显式 free root。能力/测试/窗口结果待父独占执行后补录。


## 交接候选（仍未执行）

源码检查点 64e2c74；父在纯文本审阅中指出 collider 重复 key 可蒙混覆盖，已先提交 dd9c3a1 的精确反例并修为“唯一且集合相等”。同时验证 root site 原点/轴及真实 body/metadata 数量；q 和 −q 的同一 identity 旋转保留。该发现属于执行前审阅，不标成已执行 RED。首入口 RED 只有前述69。

专项文件现在覆盖：独立 F01 首步/连续外载和实际 root 工具两侧接触；F02 三组姿态的五部件 oracle、固定旋转圆柱几何（仅几何）；F03 四个完整窗口（h、h/2、反向登记、整体变换）；F04 构造/物性/控制/所有权/迟发事件/第二参与者拒绝及重试。实际 collection 数量与 PASS 数待父执行。公开例程只跑 F03 探头窗口；F01 独立连续外载在专项测试内。

例程输出完整逐步入态/末态、全部接触点与双侧材料地址、实际力/COM矩/功率、事件入步和提交时间、自由球冲量、真实线/角动量残差和动能。动能及角动量为观测诊断，不添加未授权“通过预算”；不会把功率等式说成整个离散算法严格守恒。JSON写入在全窗口之后；正常数值但阶段门失败时仍保存全输出并返回2。运行异常仍由父守卫保原异常，未声称可恢复中途失败的完整JSON。

父接好两旧hook后，建议先跑轻量组：
python -m pytest -q tests/test_floating_fixed_tools.py -k "not a03"
再跑4个冻结窗口：
python -m pytest -q tests/test_floating_fixed_tools.py -k a03
公开归档例程：
python examples/floating_fixed_tools.py --dt 0.00048828125 --out <新的绝对路径.json>
h/2用0.000244140625；反向/变换分别用 --reverse-registration / --transformed。这些是待执行命令，作者没有运行。

本树没有合并父；父旧 loader/contact_view、最终style、必要兼容及wheel仍由父负责。当前交付为 CODE_READY_TESTS_PENDING，不是功能已验收。
