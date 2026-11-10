# MuJoCo-TYX 原生自由根树：本机集成验收

2026-11-10，产品/测试快照 `1cb47675023b54af8e7a0644f3afc8f2c6e627cb`。T2新增**一个原生自由根加0–29个hinge/slide关节**，最多nq=36、nv=35，完整保留静态和welded身体。解释、真实CUDA、公共Env/PolicyRunner和新wheel源码外消费者均已通过。本批要求模型没有可能接触对；没有把29关节动力学通过写成G1足地支撑通过。

## 新能力

CPU MuJoCo只负责原模型编译；真正步进由TYX geometry、mechanics、solve三个kernel完成。每步重新计算body pose/COM、全部COM Jacobian、完整质量矩阵、重力与速度bias、实际外力和驱动，再分别求physical acceleration与Euler隐式阻尼加速度。偏心根COM、非球惯量、根–关节及跨分支耦合、末身体载荷均保留，没有冻结初始M或只计算前8维。

使用原生root-origin世界线速度、root-local角速度和WXYZ。四元数按局部角速度右乘增量；q与v独立分配，scalar qpos/dof地址分别使用。世界COM wrench经完整Jᵀ回传，armature/damping、scalar springref、gear和stateless affine执行器沿原语义处理。旧T0自由球与T1固定根按来源结构显式分派，没有失败后换后端。

同次入步量在 `state.last_step.dynamics` 中只读返回，端点snapshot的detached CPU观察仍明确分开。CPU/Warp/HOST动力学与NumPy求解被禁用时，实际TYX step仍通过。溢出、限位越界和真实设备工作后的晚回读失败不发布任何世界；原请求重试与独立clone一致。

## 独立多案例与实际结果

[预登记案例和工程门](../development/mujoco-tyx-t2-cases.md)包含非球自由陀螺、偏心根、单子关节与welded工具、7/16/29关节分支。每backend六主案例共136步，CPU从相同实际存储初态独立连续推进；同入步另外比较FK/J/M/bias/加速度。没有用参考状态重置TYX，也没有按输出调整容差。解析陀螺动量与能量、偏心根J/M/力矩作用点及双向受力耦合另有独立检查。

| 原始任务 | 实际结果 | 覆盖 |
|---|---|---|
| 01-environment | exit0 | Python3.12.14、MuJoCo3.3.7、TYX0.1.0a0，实际模块来源 |
| 02 / 06 / 07 | exit0 | 首次Ruff、最终Ruff与format，精确7个Python文件 |
| 08-floating-interpret | 20 passed，16 deselected / 114.88s | 解释节点、四构造拒绝及公共ENV |
| 09-floating-cuda | 16 passed，20 deselected / 9.43s | 所有CUDA节点及公共ENV |
| 10-adjacent-t0-t1 | 10 passed / 1.69s | 旧slider/free-body/偏心摆双backend及四T1准入边界 |
| 11 / 12 | exit0 | 新wheel构建、源码外无依赖安装 |
| 13 / 14 | 各exit0 | 源码外解释/CUDA的完整浮根ENV消费者 |

所选节点无skip；deselected只表示backend分组。36个T2节点首次物理执行全部通过，无数值修复。03首次format检查exit1（四文件），05仅格式化后最终静态检查通过；04-import-fix任务文件未执行。原首次结果和命令均保留。

ENV用nq=9/nv=8的偏心自由根双关节模型、稀疏世界4/31，独立CPU连续对照12步，每3步更新策略，验证动作保持、readonly动力学和不同q/v维数。单世界reset后未重置世界的第13步与clone逐位相等；专项W1/W4另核被重置世界与fresh world的真实下一步。wheel消费者复制同版ENV用例，各执行26个主runtime world-steps加1个clone world-step。

## 证据与交付

原始命令、guard、前后源码、logs、28个同次JSONL及wheel在 [T2证据目录](../../artifacts/p65-mujoco-tyx-t2-20261110-01)。完整动力学按预定首/中/末步保存，所有主轨迹保存q/v/控制；没有为报告重跑producer。

[最终纯文件核对](../../artifacts/p65-mujoco-tyx-t2-20261110-01/final-audit.json)覆盖13个完成命令、62项当前源码绑定、28个保存文件以及466个wheel产品/资源成员；源码、wheel和实际安装目录全部一致，0 mismatch，fresh执行guard匹配0。wheel SHA256：`6f828b167fd7605ca892d460f3e1372dad24611a5ce5ac30400095e3e1fdd3ce`。LICENSE/NOTICE/AUTHORS随包；消费者核所有已加载PhyMeo模块都来自新安装目录。

三个原作者分别完成产品、独立测试、源码语义审阅，父完成共享接缝和唯一串行执行。原K1三个WIP保留；无远程CI、push、main合并、依赖pin变更或周期任务。

## 边界与后续

本域上限128个完整body、35速度维；不支持可能接触、free/ball弹簧、ball/equality/tendon、复杂激活/传动、dry friction、非Euler等未迁移语义。scalar限位仍是strict-interior拒绝，没有native stop求解。NumPy观察与host调度保留，不声称全GPU训练或性能超越Newton；两个测试组耗时也不是公平性能基准。

下一项为[扩展路线T3/T4](../development/mujoco-tyx-expansion-plan.md)：有限形状产生多个真实接触行，通过完整JM⁻¹Jᵀ共同求解，再接native摩擦与限制。多案例先覆盖双侧响应、偏心点、边缘撤离、接触容量和失败共同发布，随后以G1持续支撑作为组合留出。整个P6.5及Newton功能对齐仍未完成。
