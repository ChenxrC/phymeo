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

当前只冻结声明；后续同一授权补消费者实现。此文不是执行收据，不提前接受C9/C10。最终执行、真实设备名/计数、所有失败与结果由父在C9正式接受后同一新wheel环境保存。
