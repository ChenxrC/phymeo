# BALL-MC 独立文本审阅（2026-11-12）

状态：TEXT_REVIEW_ONLY / NO_PROJECT_EXECUTION。审阅者未修改产品或测试，未执行 import、collection、数值、pytest、lint、formatter、build 或安装。本文不构成运行通过结论。

## 审阅快照

- 作者树：C:/Users/38347/.codex/worktrees/p65-h1-capacity/phymeo。
- 主体全文快照 4b747b66713d46b86139740998777616a9c59d18；随后核 b0c0721bce3e80ea637f444c92161fada0f009af 只删除 consumer EOF 一个空行。
- 父共享接缝按读取时工作文件审阅，父 HEAD 4b3452792421123a3d63971a316cd7bc96921ada。父四文件当时包含未提交修改；下面 hash 绑定实际内容，不把 HEAD 当作这些 WIP 的完整源码身份。
- 作者一行 admission 修改须与父两 device 实参接线共同集成；只合入 allow_normal_contacts 准入并不足以实现组合。

| 文件 | 读取内容 SHA256 |
| --- | --- |
| child src/phymeo/backends/mujoco_tyx/_tree_model.py | 552c61bab7a152c856bad51105578b63bc25eb0bc8ea8e6e9501257a5632854c |
| child tests/test_mujoco_tyx_ball_motor_contacts.py | 522308ae435c756df3452aad078f6983f20a51ba5ae32a04ef25e181e4146c01 |
| child tests/test_mujoco_tyx_ball_motor_contacts_transactions.py | f4dab2043b4c4b0ee3a061bd3acf74002eaee3cad46bcec71c0e77a4beb6ff40 |
| child tests/installed/check_ball_motor_contacts.py，b0c0721b | eba291ae4a9d4a8d1647342706eb2c2b149e537a27a0d812be4c2262093d82af |
| parent src/phymeo/backends/_mujoco_tyx_runtime.py | 972a72608c8829a4be3bf3ec868739db4aa6f6bb69d81078c39f106dcbbdb141 |
| parent src/phymeo/backends/mujoco_tyx/_normal_device.py | 0b14187b89a080c86aa794ed8e6e8fa85d4e7bcc7304e25a8b95fea07b75a8e3 |
| parent src/phymeo/backends/mujoco_tyx/_constraint_device.py | 1bb420a7da58c5ad985ec7e55fa2cf89ef2bb5234788f2dd13daca6b9ded1c3a |
| parent src/phymeo/backends/mujoco_tyx/_model.py | f1a07d99c21080671dd6bbf4041881fea8dcd315a041b9088547a5953f4de3af |

## 唯一具体 finding 与处置

P2，NEW consumer continuous() 的 finally 直接 runtime.close()：如果 advance()/compare 已失败，close 再抛异常会覆盖 primary，最外 main 只能记录 cleanup。可达条件是一次真实检查失败加一次关闭失败；不涉及物理公式，也不能据此声称普通轨迹已失败。

同文件 lifecycle() 已有正确的 primary + cleanup 保全模式。已报父并由原作者负责最小修改；要求保原异常对象和 traceback、只尝试一次 close，不更改 fixture、oracles、调用数或求解预算。

闭环：只读核 8cb938c1fdb0596f35cd5dcd8063a5b1b3ee192a 的窄增量。continuous 在 finally 首先捕获 sys.exc_info()[1]，close 只尝试一次；cleanup 失败时以 BaseExceptionGroup 保存原 primary/cleanup 对象，close 成功时原 primary 自然传播。最终 consumer SHA256=039e7816ff53fed9ef8ff108564bf22d56ea0b544d6a2ba1bca09b7b7a4de505。本 finding 已文本闭环；未实际注入或运行。

## 完整力与运动接线

- TyxNormalDevice / TyxConstraintDevice 都通过原 TyxTreeDevice 初始化，获取相同 CSR row_offsets、dof_indices、moment 与 transmission_fields；两处 mechanics 调用现在传真实 enabled/数组。
- 原 fixed scalar 未启用 transmission 时，五实参仍分别为 flag0、parents、parents、gravity、body_force 别名；FILTEREXACT 仍以 nu=0 后调用原 activation stage。没有增加一套力学计算。
- kind3 分支仍传空 tuple，未访问 tree-only transmission 属性，浮根 mechanics ABI 未变。
- 原 mechanics 先完整累计各 actor 的 generalized force，再加入 applied；normal / pyramid QP 使用同一个完整 M、bias、passive、applied。接触力随后进入总 RHS，原积分核输出真实 q/v；未发现独立逐点施力或重复 motor 加力。
- base download 保存 transmission_fields，contact wrapper 只覆盖自己的 constraint / smooth / contact 输出。runtime 先为全部 world 建好 candidate、contact/constraint、transmission 和 typed receipt，最后统一 publish；最后一个 transmission 发布错误没有先提交前面的 world。
- 能力名称区分 joint-motor 与 scalar，同时继续限制 condim1/3、无 joint limits/dry、无 direct BALL activation/FREE composition。没有通过名称扩展 G1 或全部 BALL 域。

上述判断还对照了既有 _tree_kernels.py、_tree_device.py、_joint_transmission.py、normal/constraint plan 的 base 转发；未重新审查或重跑已有几何/active-set 算法。

## 案例与负控覆盖

六个 authored 连续案例保留非零原始来速；native same-entry 与独立连续 CPU 轨迹分别保存、分别比较，CPU forward/step 在 TYX step 周围被哨兵禁止。所有 q/v 比较均接在实际 accepted/counter 断言之后。

完整 transmission 检查保 CSR 地址、全部 BALL 三列（含零列）、标量 clamp/disable 顺序、native dense moment、actor velocity 和两种 motor power 表达。MC3 的 scalar qpos 地址不同于 DOF 地址，避免把四元数存储当标量位移。MC6 检查 disabled raw force=0 后再 positive force clamp，不假定 disabled 就最终 force=0。

接触检查包括完整几何双射、当前 J/R/aref/force、native EFC owner、两侧 force/COM torque、回投到同一个 generalized constraint force。MC5 还显式要求两条分支实际运动、双方实际 contact moment 非零；不是只比较一个刚体的末状态。

负控确实从 native 可构造 XML 指向目标能力：BALL/scalar activation、FREE、限位/dry/spring、椭圆锥/condim4/6、非法 BALL affine / JOINTINPARENT、33 actors，以及 17 possible pairs / 40 static slots。测试核具体 issue.code 并拦截 device allocation，不仅断言任意异常。

两 solver 路径各有真实 warm step 后的末 world status、完整 download、末 world transmission publication 和已经构造/验证 restore candidate 的晚故障；检查完整 publication、checkpoint bytes 和 revision 不变，再与独立 clone 的实际后继比较。源码外 consumer 另有 checkpoint A/restore/B、clone、partial reset 后两实际步和完整未选 raw clock/mirror 保持。

consumer 只运行 MC1/MC3 的连续窗及 MC3 lifecycle；其成功调用数手工为 56 batch calls / 112 worldsteps 每 backend。其它四连续案例和 fault injection 属源码测试范围，不能据安装消费者结果替代。51 个源码节点也只是本轮作者枚举，必须等父实际 collection/execution。

## 结论与剩余边界

除上述消费者双错记录 finding，审阅范围内未发现新增电机/contact 接线、native 地址、完整力矩或事务发布阻断。范围限定为这些实际源码快照；真实 CUDA frontend、收敛、数值门、构造 XML 和安装来源仍由父唯一串行验证确定。

不要求重跑旧完整套。父已有的少量 scalar/FILTEREXACT/no-contact 兼容检查足以针对变更接缝；本审阅不增加模型、预算、时间窗、容差或未授权域。
