# MuJoCo-TYX 固定根旋转树：本机集成验收

2026-11-10，产品/测试快照 `71ea1dc0be56ce077812e04099b4249498c969c3`。T1完成解释、真实CUDA、Env/PolicyRunner及源码外wheel验证，新增**固定根1–8DOF hinge/slide混合树**。入口仍是CPU `MjSpec.compile` → 编译记录 → `MuJoCoRuntime(mode="cuda")`，默认TYX，显式 `implementation="warp"` 才走Warp。

## 新能力与独立依据

每个真实TYX物理步重新计算FK、COM Jacobian、完整M、速度/重力bias、passive、applied及加速度。保留同body多关节、分支、welded身体、偏心COM、非主轴惯量、anchor/ref、armature、阻尼、弹簧与stateless affine驱动。`last_step.dynamics`是同次设备入步量，physical M与Euler `(M+hD)` 加速度分开；snapshot仍明确使用detached CPU端点观察，不参与求解。

八类原生XML案例包括偏心摆、非零速度双摆、R/P混合、共祖分支、7/8关节、welded工具、弹簧/affine驱动。CPU oracle独立编译原XML，逐步比较动力学，同时从初态连续推进短轨迹，未将参考q/v写回TYX。每backend主矩阵264步。全部[预登记误差门](../development/mujoco-tyx-t1-cases.md)保持；新增弹簧休止值、负gear和潜在碰撞拒绝在首次物理执行前登记。

另覆盖W1/W4稀疏ID、selected reset后真实下一步与clone一致、实际设备工作后的晚回读失败全批不发布及重试、限位越界拒绝、CPU/Warp/HOST求解入口禁用。公共环境用两关节树推进12步，每3步更新策略，核动作保持、变化的M及单world reset。

## 实际结果

| 原始任务 | 结果 | 范围 |
|---|---|---|
| 04-clamp-both | 6 passed / 5.27s | T0 disabled-group之后的force clamp，解释/CUDA |
| 13/14 final static | Ruff/format均exit0 | 本批精确8个Python文件 |
| 15-tree-interpret | 21 passed，17 deselected / 17.42s | 17解释节点及4构造诊断 |
| 16-tree-cuda | 17 passed，21 deselected / 8.33s | 全部CUDA节点 |
| 17-tree-adjacent-t0 | 8 passed / 1.58s | 原纯平移/自由体力矩双backend及4项相邻拒绝 |
| 18/19 wheel | build/install exit0 | 新wheel，无网络或依赖重装 |
| 20/21 wheel consumers | 各exit0 | 源码外两路径各24 batch、48 world-steps、72 kernels |

所选节点无skip；deselected是明确backend分组。T1共有38个独立节点，物理比较首轮通过，没有数值修复或容差调整。旧完整T0组没有重跑。

保留02格式、09单处import空行、10六文件格式检查的原exit1，随后修复和最终检查exit0。07/08 job文件未执行。04实际先于03运行，guard确保串行；03只格式化clamp测试，没有改变已测kernel。T0修复源自独立源码审阅：disabled group置零后仍统一force-range clamp，与全局禁用actuation不同。

## 证据和交付

原命令、前后源码、raw exit/log、guard和wheel位于 [`artifacts/p65-mujoco-tyx-t1-20261110-01`](../../artifacts/p65-mujoco-tyx-t1-20261110-01)。`saved/`保存两backend共30个同次JSONL：全部主轨迹q/v/控制、预定首/中/末完整动力学和事务负控。没有再运行producer取数。

纯文件IO的[最终核对](../../artifacts/p65-mujoco-tyx-t1-20261110-01/final-audit.json)检查67项源码绑定、19个已结束命令、9个wheel产品/归属成员及30个保存文件，0 mismatch。wheel SHA256为 `ba00d4fce66295ef8ddebfa7c765f4fdc5ecf5e8fc30ce56c03305faf89a820`。两安装包消费者核所有已加载Phymeo模块均来自新site；LICENSE/NOTICE/AUTHORS随包。

原产品、测试和源码语义审阅分开，父完成共享入口、集成及唯一串行执行。没有远程CI、push、main合并或pin修改；原K1三个未提交文件保留。

## 边界与下一批

T1最多128个完整body、8个scalar DOF，要求没有可能原生接触对；含hinge时走新T1，纯平移/居中各向同性自由体继续T0。有限关节限位只做strict-interior拒绝，没有native stop求解。

下一项是[扩展计划T2](../development/mujoco-tyx-expansion-plan.md)：原生free-root树及0/7/16/29标量关节容量，先关闭根原点线速度/局部角速度、偏心根COM、非球惯量、完整根–关节M/J/bias及四元数积分。再依次做T3/T4多接触与native摩擦、T5 G1支撑。NumPy观察和host调度仍在链上；本批不宣称全GPU训练、性能超越Newton或整个P6.5完成。
