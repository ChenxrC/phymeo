# M2-B2：持久入步状态与真实接触记录

日期：2026-11-14。产品提交：`994b084248dc920afb99adcbc44748ed07a275ee`。
[实施计划](../superpowers/plans/2026-11-14-language-conditioned-gpu-manipulation.md)；
[公共用法与边界](p65-heterogeneous-manipulation-io-usage-20261114.md)。

## 本批交付

设备在 step prepare 内从真实输入 bank 和本次接触求解结果 D2D 保存记录。
每字段拥有两个独立 bank，与物理 q/v 的 active 索引相同；全部桶成功后，原物理提交同时接受记录。
失败的 advance、最后桶验证或重试不会覆盖已接受记录。
没有从 HOST receipt 重建并上传每步接触，未改动力学、摩擦律、求解器预算或物理误差门。

覆盖 legacy T0/T1、T3 normal、T4 constraint 的统一记录：
T4 从 contact_map 解出实际 candidate slot，保留 compact contact index 与 slot index 的区别；
limits/dry 约束行不伪装成接触。condim1/3/4/6 的六维 contact-frame wrench 完整保存，
包含 torsion/rolling couple。legacy body IDs 来自真实 geom_bodyid，不假定平面总属于 world body。

这是第11节 B2a 持久记录、B2b 路由归一，以及 B2c 的存储生命周期部分。
**公共 workspace 的 entry/body/world-wrench 输出、Torch 接触副本、SuccessFrame 消费和 B3 Env 尚未完成。**
本批保存原生入步 q/v；后续须从这份 q/v 复用 FK 得到同相位 body 状态，不能使用已完成 B1 的端点 body 数据混配过去载荷。

## 内部记录布局

各字段均按 world 行排列。C 是既定 contact capacity，C=0 仍有一个全零 sentinel 槽。

| 字段 | 单 bank 形状/内容 |
| --- | --- |
| entry_q / entry_v | FP32 [W,max(8,nq/nv)]，真实 native 入步状态及零 padding |
| entry_act | FP32 [W,max(1,na)]，真实入步 activation；无 activation 时全零 |
| entry_clock | FP32 [W,4]，input high/compensation、output high/compensation |
| entry_meta | int32 [W,6]，valid/input_tick/output_tick/detected/active/overflow |
| entry_contact_ids | int32 [W,max(1,C),8]，valid/active/geom1/geom2/body1/body2/feature/condim |
| entry_contact_data | FP32 [W,max(1,C),20]，distance/point3/frame rows9/contact-frame wrench6/include_margin |

frame 的三行是接触轴；wrench 的后三项是接触力偶，未擅自转成某个 body origin 的世界力矩。
成功空接触是 valid=1、count=0；初始或 reset 后无记录是 valid=0。
controls-only revision 不产生新的物理输入帧。tick 使用非负 int32；溢出拒绝提交。
这些字段是内部 ABI，不是供应用绕过公共 workspace 直接依赖私有桶的接口。

## 生命周期与资源

- 14个持久状态数组和必要静态路由数组进入原 allocation manifest、跨 owner 预算、alias 与 UNKNOWN_RELEASE 管理。
- partial reset 只清所选 world，其他行的状态、补偿时钟与历史记录字节保持。
- clone 复制所选实际 accepted q/v/act/clock 与 entry，保留乱序 world 映射。修正此前从 HOST mirror 重打包会丢失补偿时钟低位的路径。
- checkpoint 保存真实 accepted-entry 字节；restore 在分配替代设备前检查布局、tick/时钟、输入状态、身份和同次 receipt 的一致性。
- 旧 v1/v2 胶囊缺少扩展时恢复为显式无历史记录，保留 output tick，下一实际 step 才产生新有效记录。测试使用删除新扩展的 synthetic legacy fixture，未冒称重跑旧 producer。
- 大 entry 字段按 C-order 展平分块保存，每块不超过原 1,048,576 元素；小数组格式保持。
  原128MiB胶囊与逐数组限制未放宽。此修复避免 W1/C65536 或 W256/C256 的合法布局仅因新 entry 数组触及旧单数组门而不能保存。
  两个大布局只有 HOST codec 资格，不是新增 GPU 规模资格。

显式 reset/clone/checkpoint 边界仍允许 HOST 字节迁移与读回检查；
普通 step 的记录生成保持 D2D，但原求解器诊断及本批 metadata 状态检查仍有 HOST 读回，未声称零往返。

## 实际验证

父 /root 唯一串行执行，每次命令 fresh entrypoint-aware guard。
现有 Python3.13.3、MuJoCo3.3.7、NumPy2.4.6、pytest9.1.1、TYX0.2.0a1；
Phymeo 取父 src，TYX 取已验证 site313。未安装、更改 pin、运行远程 CI、全仓库或新 wheel。

| 组 | 实际结果 |
| --- | --- |
| 原资源预算/分配/关闭完整文件 | run06：24 passed / 6.83s |
| 原 checkpoint 完整文件 | run07：47 passed / 52.37s |
| 原异构事务完整文件 | run08：24 passed / 58.76s |
| 新 W8 全路由/排列/W1/非空胶囊 | run12 前4节点通过；后一个力偶 fixture 断言失败，原始结果保留 |
| 新力偶、reset、clone/activation、legacy、损坏与晚失败 | run17：10 passed / 31.51s |
| 新大数组分块/损坏拒绝 | run16：5 passed / 1.35s |
| 最终13文件静态检查 | run18 lint、run19 format-check 均通过 |

共114个不同验证节点通过，其中19个新增：7 interpret、7 CUDA、5 HOST codec。
run12 的4个已绿 W8 节点没有重复执行；修复只改独立力偶 fixture，W8输入与函数保持。
原95项兼容在分块扩展前完成；后续5项codec及新W8/activation胶囊验证覆盖最终包装/恢复连接，
不将全部95项冒称在后续相同字节快照重跑。

新增物理测试覆盖八个真实不同路由/构型、稀疏IDs与正反排列、同GPU W8和独立W1字节相同、
输入/端点区分、有效空记录/非激活检测/实际受载、非首FREE与activation、同桶partial reset、
乱序clone、完整capsule及下一步重放、corrupt q/tick在设备分配前拒绝。
独立 native MuJoCo oracle 检查同入步几何及 wrench；原几何门与 force rtol/atol=3e-3保持。
普通step期间测试封锁向 entry 数组的 HOST upload。失败注入发生在真实 advance 后和全桶 prepare 后。

## 首次失败与修正

run12 原力偶正控设置了同时平移和自旋，native 与 TYX 均返回零 angular wrench；
接触地址、记录与 native 比较已通过，失败是“必须出现非零力偶”的作者初态假设。
保留源、stdout、完整payload和首错。未改产品求解器或降低阈值。

修正保留原两球、质量、接触参数、Y转角0.02和 contact0→slot1：
按实际接触中点 r=(.14 cos(a),0,-(.199+.14 sin(a))/2)，
设置 v=-omega×r+(0,0,-.005)，并用 R^T omega 写 native local angular。
condim4以世界Z自旋检验torsion，condim6以世界X转动检验rolling。
首步 native 与 TYX 的指定非零分量均需>1e-6；第二步继续比较完整实际载荷，不假定切向抵消自动持续。
该修正来自运动学与保存反例，没有扫参数或修改验收物理容差。

首lint/format仅导入排序和排版修正，原退出码均保留。
一条文件修改命令曾因自动审批服务 Bad Request 未执行；随后拆为较短命令重新获准。
归档首尝试发现纯HOST codec未使用tmp_path，pytest未创建其basetemp；修正目录预期后继续核验已有副本，未重跑测试。

## 证据与续作

证据：`artifacts/mujoco-tyx-entry-record-20261114-01`。
19条命令保留环境、原始退出码、日志、执行前后源码清单；13个最终文件完整快照。
85个新保存文件共14,554,390字节原样归档，所有源/目标SHA与size相同，无重解压/字段删除。
`final-evidence.json` SHA256：
`5b211515b43a3d54f8203919af1aff4e13db1fb68b71f78729227b6a33aa5a1e`。
fresh execution guard为空，所有本批执行槽和作者均已释放。只获源码资格，未发布或push。

下一批完成第11节B2c公共entry观察：从保存q/v复用FK，输出同入步body状态，
由原frame/wrench明确转换世界力、双侧反作用和以body origin为参考的力矩。
再完成B2d SuccessFrame按input tick消费一次，以及B3目标/episode/reset/checkpoint。
真实抓取/放置/PushT、自然语言编码、共享策略训练及新安装交付仍待完成。
