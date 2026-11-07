# C10 源码外公开消费者：冻结作者声明

状态：**AUTHORING_ONLY_UNEXECUTED**。依据父 `fd57c9b` 的 C10 计划“源码外消费者纯作者补充”。仅新增 `tests/scale/resident_floating_contact_wheel_consumer.py` 与本报告；此前 d221065 四文件不变。父唯一执行槽，本线不 import、数值求值、测试、静态、构建或安装。

固定公开任务：`python -I -B` 运行复制到 checkout 外的单文件消费者。CLI 为 `--mode interpret|cuda --output ABS --wheel-site ABS --expected-package-root ABS --expected-wheel ABS --forbidden-checkout ABS`。输出新目录必须不存在；cwd为脚本目录且不在Git checkout。只显式 `import phymeo as ph` 公共API，无 repo tests、参考模块、私有backend或HOST补状态。

## 固定输入与验收顺序

1. W4顺序 `(503,91,1207,6)`。503=F01 root m2/I.5、q_z=3/32、球r1/8、有限z箱、k64/d0/mu0；91=F02 motor原root2+Px child1/I.25、q/v0、motor3、child球r1/8对有限x箱；1207=C9原ordinary P（m2、offset(1,0,0)、轴+y、q1/8/v1/16、COM外力(0,2,0)、零PD/ff）；6=C9原FreeSphere（m2/r1/16、p(0,0,2)、v(1/16,0,0)、g(0,0,-2)）。后两world无接触绑定。
2. h1/1024，原W4连续40步；tick19完整checkpoint恢复后21步。输入、entry、所有state/FK/diagnostic/receipt、材料与入/末candidate物理字段完整重放一致；epoch、bank与累计计数另核，不删除原字段。
3. 两个真实active bank各一容量0拒绝：bank0暂停503、bank1暂停91；邻world正常推进，足额原plan重试。若为进入下一bank需要对齐，只作一真实合法推进并单列保存；不restore/reset来伪装连续窗。此段开始于原40+21之后。
4. F03另一个独立batch/world503：root2/I.5、Px siblings m2/1/I.25、offset x=±7/64、两球r1/8、k64/d0/mu0、motor0，40步。必须 `tree_dynamic_pairs=True`，首两力−2/+2N、(a_root,qdd1,qdd2)=(0,-1,+2)，净线动量。
5. 完整合法材料表的 unsupported sampling、真正跨batch owner拒绝、owned旧回执不变、重复close和plan/batch存活字节归0。

独立数学是本文件自包含的root+Px森林：每个真实COM的Jv/Jw、全M、Coriolis偏置、独立有限球–面/球–球法向律及一次半隐式漂移；不读设备M/J/force生成唯一期望。same-entry5e−6/5e−5、M/方程2e−5、PD5e−5、40步2e−4沿原合同。root惯量/child惯量均原各向同性值；没有新增转轴、浮根affine、摩擦或CCD域。

## 预声明文件与记录

固定输出 `resident-floating-contact-wheel.trace.json.gz` 与 `summary.json`。无损完整payload上限256MiB；保存全部原输入/绑定/law、实际接受前缀、失败类型与traceback，不以摘要替代完整轨迹。summary分别计W4 forward40/replay21、F03 forward40、两个capacity bank及额外真实对齐步，不把拒绝当接受步，也不报性能倍率。

安装来源核验沿C9规范并完整记录：actual wheel SHA/字节、direct_url真实本地wheel路径（若installer提供hash必须相等）、全部installed phymeo成员与wheel精确字节/成员SHA、dist-info关键元数据、全部已加载phymeo模块origin/__file__/package search path及实际wheel成员字节；物理窗前后都核完整成员。不得仅看到某个__file__就声称全安装同源。

声明先以 4937e39 冻结，消费者完整作者交接见后节。此文不是执行收据，不提前接受C9/C10。最终执行、真实设备名/计数、所有失败与结果由父在C9正式接受后同一新wheel环境保存。

## 完整作者交接（未执行）

消费者已完成本轮固定任务，实际公共入口为 `ph.FloatingRobotTree`、`ph.TyxWorldBatch`、`batch.prepare_contacts`/`step_contacts`、`snapshot`/`restore`、`tree_kinematics`/`tree_diagnostics`/`summary`、plan 的 `query`/`last_accepted`/`last_status`/`stats` 与正常 close；没有导入仓库参考或推进私有函数。自包含 NumPy 数学只支持本次声明的 root+Px 森林，先以三个 literal 完整质量块/力/加速度核验公式，再从独立初态推进，并另核每个真实入口的响应。

W4 每行含原 controls、完整入口 state/FK/energy/候选、实际 acceptance、全部末态/diagnostics/receipts/status、完整材料/法向律、两类实际功、独立真值和原 public stats。19→21 重放比较全部物理字段，保留而单列 epoch/资源/累计计数。局部容量拒绝明确保存两个 plan 的原/末 accepted；被拒 world 全态/FK/能量保留，其余三个 world 真推进，之后原足额 plan 以被保留状态重试。F03 单独保存40步、逐侧 COM wrench、完整 M/J 与净线动量。所有 root checkpoint 以原 anchor + `local_origin_m` 核对 world origin，不假设 root 初始位置为零。

所有应用层断言仍为原门：same-entry `5e-6/5e-5`、完整M/方程 `2e-5`、PD `5e-5`、窗口 `2e-4`；`1e-10 m` 只核原表面/入口作用点的记录构造与 stored R/p 运输。材料功、共同作用点功、入步功率乘h及动能变化分别记录，不另设能量守恒通过。F03总线动量除原总质量形成COM速度残差，使用原FP32速度门。

预定成功计数为 W4 forward40、replay21、capacity bank0/1各一次及各一次足额retry，真实alignment最多2步；F03 forward40。各窗口是独立公开批；不把query次数、被拒world或恢复步骤算作物理接受数。完整 payload 无损 gzip 后同时保存压缩/未压缩SHA和字节；超过上限则 `resource_stop`，不发布截断成功；IO错误单列且不谎称已回滚。硬件名称由父同次外部运行收据绑定，脚本只记录公开真实mode/分配/上传下载/launch等字段，不用型号常量冒设备探测。

父待 C9 正式接受并完成 C10 产品接缝后，复制本单文件到新的 checkout 外目录，以新wheel实装目录运行，例如（路径均替换为父实际新目录）：

```text
python -I -B resident_floating_contact_wheel_consumer.py --mode interpret --output <ABS_NEW_OUT> --wheel-site <ABS_INSTALLED_SITE> --expected-package-root <ABS_INSTALLED_SITE/phymeo> --expected-wheel <ABS_ACTUAL_WHEEL> --forbidden-checkout <ABS_PARENT_CHECKOUT>
python -I -B resident_floating_contact_wheel_consumer.py --mode cuda --output <ABS_OTHER_NEW_OUT> --wheel-site <ABS_INSTALLED_SITE> --expected-package-root <ABS_INSTALLED_SITE/phymeo> --expected-wheel <ABS_ACTUAL_WHEEL> --forbidden-checkout <ABS_PARENT_CHECKOUT>
```

两 NEW 清单：`tests/scale/resident_floating_contact_wheel_consumer.py`、本报告。此轮只做源文本/API对照与Git范围检查；没有import、运行数学、collection、pytest、AST、lint、formatter、build/install或native。**UNEXECUTED / AUTHORING_RELEASED**；执行结果、独立源码审阅和安装验收仍待父，不先称C10接受。
