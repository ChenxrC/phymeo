# 第十批：同相位抓持载荷诊断与 PushT 有限边界候选

日期：2026-11-14。以61840202为起点；原两作者隔离纯编码，root唯一串行执行。执行登记见[本批计划](p65-grasp-wrench-plan-20261114.md)；上一批实际轨迹与操作结果见[第九批报告](p65-manipulation-route-report-20261114.md)。

## 抓持：新增可复用测量和保存轨迹消费者

产品作者合入028b3629，格式及消费者64a6b206。

- `phymeo.control.manipulation_wrench.measure_object_contact_wrench`只消费公共accepted-entry NumPy数组、显式object body/同入步世界COM、0..8个分组geom。完整接触坐标系变换force/couple，反号作用到body1，按`couple + (point-COM) × force`得到COM矩。
- 逐contact、group、other、total全部保留。table等非pad接触不丢弃；detected但inactive的记录明确零载荷。非零质心不可用body origin代替。输出不可变元组，不归一化或修补坏frame，不推断力闭合、动力学平衡或抓稳。
- 14个新HOST节点首次通过，涵盖独立力/矩手算、反作用、非零COM、力偶、旋转/平移、condim1/3/4/6、分组、坏frame/mask/count/所有者、padding、溢出、不可变性。Ruff通过；首format-check失败后纯formatter，两个文件AST完全相同。
- `scripts/measure_manipulation_wrenches.py RUN OUTPUT`读取一个完整W1诊断目录，使用后条feedback_source与前条SuccessFrame/contacts逐项绑定。原模型XML和trace SHA校验；只编译模型读取COM偏移，无mj_forward/step。每个记录对原已发布world force/couple和双侧body-origin矩独立核对。2e-6门仅用于FP32发布转换核对，不是新物理容差。
- 原world1和world3各2399条、共4798条公共entry首次完整消费通过；每世界最后第2400条因没有后续完整feedback_source明确未覆盖，未补造。末步另在下面14点同状态诊断内。完整gzip JSONL与摘要保留。

| 保存消费者 | entry数 | 最大逐接触力转换差 / N | 最大矩转换差 / Nm | group法向和差 |
| --- | ---: | ---: | ---: | ---: |
| world1 | 2399 | 6.6174e-8 | 3.1929e-9 | 0 |
| world3 | 2399 | 2.6483e-7 | 9.4642e-9 | 0 |

消费者命令：

```powershell
python scripts/measure_manipulation_wrenches.py <original-W1-run> <fresh-output>
```

默认命名object/left-pad/right-pad对应原诊断场景；消费者当前只支持保存完整feedback_source的单世界轨迹。公开测量模块本身支持显式row、多世界数组和任意声明分组，并不自动从名字识别机器人。

## 十四个固定实际状态的 native 对照

预先固定world1/3各1100、1300、1500、1700、1900、2100、2400。使用原入步q/v、实际ctrl/qfrc/xfrc恢复CPU scratch，调用mj_forward一次求同输入结果；封禁mj_step/step1/step2，前后q/v/time不变。完整保留全部contact，按geom pair汇总比较force和COM矩，未挑最近点作代表。

13个快照的最大pair合力分量差不超过2.655e-6 N；world1第1900步例外：左pad合力分量差0.0180993 N，COM矩差0.000255542 Nm。该点两边均8contacts，左pad两个点枚举顺序不同，位置差约5e-8 m。**顺序不同是观察，尚未证明它是力差原因**；不能称所有14点均精确匹配，也不据此将抓持失败归咎于单个数值差。

这些是同状态输出比较，不是CPU/GPU整轨迹一致、完整native验收或新CUDA窗口。

## 已定位的控制风险

| 场景 | 实际观察 | 下一控制设计应处理的内容 |
| --- | --- | --- |
| world1盒 | COM在pad局部x约24.2–27.6mm，pad半宽12mm；接触出现在pad宽度边缘。COM局部z从约-11mm降至-39.6mm，底缘-40mm；相对tool绕y转动从1100步约-0.154rad至末步0.402rad。 | 有限pad覆盖、接触力臂和相对转动；不能只追TCP位置或两指法向力。COM投影越界是风险证据，不是不存在任何稳定摩擦抓持的证明。 |
| world3胶囊 | COM局部z从-48.3mm降至-76.3mm；逐接触退到pad下缘，末步三个剩余接触都在约-39.9mm。两指法向力仍非零，未阻止持续下滑/旋转。 | 先筛查实际表面与pad纵向覆盖，再设计姿态/搬运过程。法向力充足不能替代相对低速保持。 |

world1在2100步pad竖直合力约1.1756N、world3约1.1789N，接近各自1.1772N重量，但此前建立的角速度/滑移可以持续；瞬时合力近重量不等于稳定。未扫描摩擦、质量、PD或solver参数，未降低原Goal。world0原短窗抓球成功复用；本批尚未增加抓放成功任务数或共享网络训练结果。

## 证据

父目录：`artifacts/mujoco-tyx-grasp-wrench-20261114-01`。
- 01-fixed-native-wrench与fixed-native-wrench.json：14点一次新native查询，RAW_EXIT0。
- 02 Ruff0，03首format-check1，04format0，05原AST对比0，06HOST14pass/1.20s/0。
- 07消费者首Ruff仅1项import排序失败，08format0，09final Ruff0，10保存消费者0。一次wrapper的-p参数歧义发生在pytest启动前，改显式参数数组后才启动06；不是产品测试失败。
- `public-consumer-01/world-{1,3}`：完整4798条测量与摘要。已有原轨迹、物理运行及已绿组均未重跑。

## 保存消费者审查补足

独立文本审查发现published NaN可能被max(0,nan)隐藏，以及任意--object可能借用任务对象的quaternion。16f6c9ea补完整world/body wrench及finger-load精确shape/finite，保存CLI明确只准绑定的任务object并核object_key。新增14个HOST首次通过，含非零旋转COM正控、NaN/Inf/可广播坏shape、同位置异对象、identity/phase/tick反例。11format、12Ruff、13HOST均0。4798条正常保存消费是64a6b206阶段的结果；之后仅增加拒绝守卫及上述14项针对验证，未重复旧轨迹。

## PushT：有限外边界、半径遮挡与真实COM

作者32e6a890正常合入c950fb2c；最终纯格式及测试数据生命周期修正b1a521f6。新增：
- `phymeo.control.manipulation_push.enumerate_native_planar_push_contacts`；
- `tests/test_manipulation_push.py`。

覆盖同一native body完整1..2 BOX（共同平面轴和Z区间）的外边界，包括重叠、相接、相同、包含及分离矩形。坐标格取并集外边界，消除内部缝，完整保留几何所有者；不用整个并集AABB替代T/L凹形。native xipos作为COM，原三T的local COM y=.0065，不用body origin或面积质心。

球/竖直胶囊的中心或轴段必须和BOX共同Z区间相交；仅caps与顶部相交的情形明确unsupported。pusher center保原高度，真实side-contact point单独保存。每条外边默认.25/.5/.75及法向进入，完整边×分数×角度有界积，最多128；包括blocked候选都返回，不丢失败尝试或扩大预算。

接触圆盘和完整直线进入段逐BOX解析距离检查：segment/slab交集、两端到矩形、全部角点到segment。保留1e-9m查询容差、原始距离和原因，不称精确CCD、全机器人路径或动态接触响应。返回单位内法向的几何wrench（含COM力臂），不是预测实际摩擦力、yaw或任务成功。输入完整MJB≤8MiB及原q/声明绑定，model/q不变，结果冻结。

### 验证与首次失败

14Ruff0；15首format-check1；16formatter0；17两个AST完全相同。
18首HOST在前8个节点后，native对照的临时MjData接触视图越过owner生命周期，Windows access violation，**RAW_EXIT=-1073741819**。完整原日志保留。只修测试辅助函数：在MjData有效期间立即复制float(dist)，不返回借用视图；原几何、负控、预期与产品算法未改。19Ruff0；20最终完整 **54 passed / 1.41s / RAW_EXIT0**，无skip。

54项包含真实native几何比较、凹角半径遮挡、端点清空但中段穿角、原三T、全部外边/所有者、真实COM及反面积质心、yaw/平移、sphere/capsule、unsupported高度/倾斜/遗漏geom、全128积与超预算、输入与结果不可变。

### 原三个机器人端点集成（21运行）

按执行前登记，原world5/6/7各一次固定24候选；候选构造不调用forward/step。每个usable候选仅原q0单seed先precontact，再仅从收敛的precontact参考求contact。原100迭代/.15步长/5e-4m门与joint limits不变；完整FREE对象坐标、model bytes不变。

| 原世界 / 构型 | 外边 | 候选 | 几何usable | 两个端点IK均收敛 |
| --- | ---: | ---: | ---: | ---: |
| 5 / RR | 8 | 24 | 22 | 10 |
| 6 / RRR | 8 | 24 | 22 | 10 |
| 7 / PPPR | 8 | 24 | 22 | 14 |

全部72候选、拒绝原因和对应IK尝试保存于`push-original-qualification.json`，RAW_EXIT0。这34个是**有限候选中的已达端点**，不是机器人整段无碰撞、动态推行或Goal成功。未收敛候选仍保留，不宣称全局不可达。

## 本批结论与下一节点

新增82个HOST节点（14测量、14保存消费者、54PushT），完整4798条旧entry消费与原三机器人72候选/34端点对接入。没有新增GPU物理窗口、视频、安装、训练、push或CI。父及两原作者本批执行完成释放，既有无关WIP保留。

1. 优先接 **PushT完整接近路径→真实短推→目标推进**：从已保存候选按原Goal方向/名义力臂做固定有限排序；完整机器人路径检查后仅选合法参考进入一次预登记短推。检查真实同入步接触位置、物体位移/yaw与COM力矩，再延长同轨迹朝原目标推进。名义单位wrench不当真实动力学oracle。
2. 抓持补 **pad有限覆盖与相对姿态约束**，以这次实际边缘/滑移证据设计一组有界参考；优先原6R胶囊路线，保留4R自由度限制。仅取得新几何依据后再新CUDA窗口，不扫描摩擦/质量/PD。
3. world1第1900步0.0181N native差异保留为独立未闭项；需要解释该状态时只做同输入有界诊断，不把它扩成无截止精度攻关，亦不阻塞与之独立的PushT路线。
4. 稳定抓持后实际搬运/释放/再支撑，扩八世界与holdout，再接真实语言编码和单共享网络优化。原目标仍是GPU异构训练；当前新增的是可复用控制/观察连接，不能称已经训练成功。

本批收口收据：`artifacts/mujoco-tyx-grasp-wrench-20261114-01/final-batch-receipt.json`，SHA256 `5eb228d73a1214be52616cb9a09b20ccf9635e7ef0a5588a35c9e710cb82f456`。记录21条实际命令原始退出、允许的formatter变更、6个最终交付文件与其已测快照一致、72个本地文档链接存在、fresh matching execution=0。其余Node单列，不称整机无进程。原三份device-convex-patch无关untracked文件保留。
