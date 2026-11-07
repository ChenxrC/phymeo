# 显式作用点增量切向模型：短矩阵与验收边界

状态：**TEST_AUTHORING_ONLY / UNEXECUTED**。2026-11-08，仅新增本报告与 `tests/test_tangent_application_reference.py`。父唯一持有项目执行槽并修改旧产品接缝；作者没有执行 import、collection、数值、pytest、静态检查或格式化，没有重跑原四个抓取窗口。此文不构成 application 模式通过声明。

## 已有证据与这次改变

原 material 模式证据保持：280 G01 在 594 接受步后因实际材料功被拒，281 G02 在 618 步后因材料/作用点运动差被拒，282 G03 在 580 步后因实际材料功被拒；283 G04 Panda 完成 3006 步、1.87875 s，原八门通过。不能因 G02/G03 预期“不持稳”而把执行拒绝称负控通过。

父 297 从各原 retained 状态重建一个失败请求，七公开字段先精确匹配，错误精确复现且拒绝后不发布。G02 切力/储能/功全零，材料—作用点差 1.008421122e-6 m，而作用点预测误差约 8.3846e-8 m。G01/G03 的作用点误差约 9.82e-9 / 4.54e-8 m；原材料能量残差为负，量级约 1.11e-8 / 1.02e-8 J。这是坐标区分的依据，不能外推新模型一定持稳。

新合同由父 `497fb13` 的原计划 T3 冻结：`motion_reference='material'|'application'`，默认 material；场景只能 t=0 配置一次。低级 prepare、密封 trial、receipt 与每 episode 的 update 都记录模式。retained history 即使储能为零也不得改变模式；真实结束后新 generation 可以从零选择另一模式。

application 模式使 J/W、力矩、弹簧增量与功对应同一对入步作用点；用实际末姿态运输它们。原 1e-6 m 作用点预测门、1e-8 J 逐站及总功门仍然有效。真实材料位移与材料功保留为诊断，不能改名成作用点位移/功，也不能重新加入模型的相等门。旧 material 的所有门保持。这里只是逐步作用点增量模型，不是 Newton 跨帧 sticky anchor，也不是整个刚体积分器的精确能量守恒定理。

## 冻结专项

| 节点 | 独立输入与必要断言 |
|---|---|
| a01 | 原有限层 ContactScene：两面各 2 N、24 个实际站点、中心自由体法向合力/矩为零，采样不推进；随后调用新公开 keyword，短步核实际动量、mode 与 history。 |
| a02 | 同一有限夹持，默认、显式 material、application 各 3 步；完整物理 state/frame/geometry/events 一致，history/receipt 除明确 mode 字段逐项一致。 |
| b01 | 原 F05 真实 2 N/12 站入步 sample；两 owner 同时按 Ry(1/64) 与固定平移运输。材料面初始差 (0,0,1/1024)，独立预期为 `(sin(theta),0,cos(theta)-1)/1024`；共同作用点相对位移为零。application 无伪切力/储能/塑性功，material 仍拒绝。该节点是有限几何运输与低级模型审计，不宣称制造姿态被动态场景接受。 |
| b02 | 实际站点身份配单站规定平移，h=1/1024、free_vx=1/16、kt=200，独立 f=-200/16384；材料位移另加 1/1024。实际作用点功与材料功分别保存，只有对应模型的功入 spring balance。 |
| b03 | 两站 f=-200/16384 N；相反 0.95 um 误差逐站超 1e-8 J 而总和抵消，等向 0.75 um 误差逐站不过门而总和超门。两种均必须拒绝，且运动误差均小于原 1e-6 m。 |
| b04 | mu=0，真实正 N，作用点预测精确而 surface/app 差非零：力/功/储能/塑性均零；另给真正 application 超门，仍必须拒绝。 |
| c01–c02 | 既有弹簧应变 1/4096、kt=200，E=100/4096^2 J；缺席或 N=0 只释放一次，再入为 generation 2；mu=0 清能量亦只记一次释放。 |
| c03–c06 | 非法模式；正/零储能下 retained mode 改变经 prepare 和直接 store.commit 都拒绝；trial mode/预测篡改；合法原请求纯重试；旧省略 keyword 与六参数 update 构造保持 material。 |
| d01 | 原实际单 RobotScene Rz 工具，原 h=1/4096、Iz=.02、tau=.01。以真实声明锚点的 K=200 sum(x²+y²) 独立核首步 theta=h²tau/(Iz+h²K)、反力矩及实际有限旋转作用点位移。 |
| d02 | 原 F05 合法 fast tilt：初速 5 rad/s、h=1/128。捕获真实候选确认 application 自身预测误差超门；两次同一请求拒绝，七字段完整保留，无以材料差代替真正 application 反例。 |
| d03–d04 | 实际作用点审核成功后注入故障，七字段不发布；撤销注入后同请求与独立参考场景完全相同。场景 t=0 二次配置及步后改模式仍拒绝。 |

e01–e02 另以纯协议记录验证新 driver：application 下不同坐标的两项相等检查须从 checks 中移除，三个真实差值完整保存在 motion_diagnostics_m；history 与 receipt 的模式不匹配须令 checks.motion_reference=False，不自动改标签。这两项不执行物理窗口。

这些低级规定运动测试不提供第二积分器。真实动力学只由原 ContactScene / RobotScene 推进；F05 资产、质量、原控制与几何不变。没有长窗、Panda 完整流程或设备性能宣称。

## 交接与实际执行记录

- tests-first `ae1d65279474ab087200f2e2f18549bb621272ba`；完整短矩阵 `eccfa744e2655d94b4fa4227662f380ba792df7f`，均来自原独立分支，没有 merge 父产品。
- 父已完成首公开 RED：305 / 父导入 `06e26be`，1 failed / 0.90 s。真实原 normal 两侧各 2 N、24 站及不推进前提成立；唯一失败是 configure 不接受 `motion_reference`。该首 RED 不重跑。
- 父另已报告 `2095a2f` 四个旧产品接缝及 312：首 a01 与受影响六套共 101 passed / 40.67 s / RAW0。原门未改；这是首节点与相邻旧行为的验收，不是后续新增矩阵已通过。
- 其余专项截至作者交付 **未执行**；手工列举预期 29 参数实例，最终收集数以父真实 collection 为准。所有源码修改、格式和执行收据由父后续记录，不能把作者文本检查写成通过。

父可直接运行新单文件，不需要先执行旧四窗：

```text
<project-python> -m pytest tests/test_tangent_application_reference.py -q --basetemp <fresh-absolute-path>
```

实际七字段是 state、reading、geometry、history、receipt、events、last_frame；以原公开完整序列化比较，不仅比较位置或摘要。原真失败、raw 输出和严格门不覆盖。短矩阵通过后，父才决定原保存拒绝请求与受影响新窗口的同输入 application 对照；本报告不给这些后续结果预判。


## 父实现与专项执行（2026-11-08）

2095a2f只修改四个旧共享接缝：合法mode验证、trial密封、receipt/episode保存、prepare与direct commit均拒绝retained切换；两个scene桥传递原mode。默认material保全部旧门。4b88571给原完整窗口driver增加显式keyword/CLI和保存绑定；application的不同坐标差只列diagnostics，不伪造True，相同坐标的运动/逐站及总能量门保留。

305有效RED继续复用；306/307首静态失败为1 import和3格式，308/309修正、310/311通过。312首a01+受影响六旧套共101通过。完整作者7d5dfd2导入后313/314首静态为1 import和2格式，315/316修正、317/318通过；319剩余28个新实例+13受影响驱动协议共41 passed/1 deselected/2.11s/RAW0，未重跑已绿a01。新专项合计29实例全部通过，仍非完整新模型抓持结果。

320对297三个已保存失败candidate仅做独立算术：相同原力/预测/真实末位移，改用共轭application坐标计算功门；G01/G02/G03的总残差分别5.010761311e-11、0、-2.101073790e-9 J，逐站最大8.041320017e-11、0、1.513352046e-9 J，均在原1e-8内，application运动门也通过。**0物理步、未切换/重新解释旧history、未调用solver或制造accepted状态**。它只关闭反例中的坐标差异解释；新模型必须从t=0形成自身历史的321–323完整窗裁定。原G04 material通过继续保留。


## 原初态完整窗口与工程精度边界（2026-11-08）

321 G01/application：1420接受步/1.42s，真实双侧获取、20mm抬升与100ms持稳达标；释放阶段因原总功门拒绝，完整释放仍未通过。322 G02/application：1681步/1.681s/RAW0，零摩擦按预期滑落、实际释放并重新台面支撑，原八门全部通过。323 G03/application：597步/.597s，获取阶段同总功门拒绝，不冒完整超载负控。原G04 material 3006步完整通过继续复用。参数、控制器和预算未改。

324对321/323精确失败入步做单请求诊断，实际application历史、全部七字段原样，原异常复现且完整回滚。G01/G03的application运动误差分别2.440675769e-8/4.833710781e-8 m，逐站功残差均小于1e-8 J，总功残差却为+1.0029494141177266e-8/-1.1653069883974057e-8 J。两者均失败于原total gate；没有材料点误差冒作用点误差，也没有发现释放重复计账。保留为离散化与容差政策的开放项，不靠改夹力、h或单例阈值通过。用户允许工程精度损失；后续应统一评估相对/绝对功审核或实际非线性修正，当前不阻塞C6独立功能开发。无新接受物理步、无完整前缀重跑。
