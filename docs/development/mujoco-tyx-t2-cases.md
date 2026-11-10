# MuJoCo-TYX T2 独立多案例与工程门（预登记）

2026-11-10，父基线 `4ab18bd580b5be77dd6037283cafa1509412016c`。状态 **AUTHORING_ONLY / UNEXECUTED**。本文件先于新产品运行登记，作者不import、collection、probe、pytest、静态检查或构建；父持唯一执行槽。产品实现与测试作者分开，测试使用原XML独立CPU MuJoCo3.3.7，不从TYX的M/J构造oracle。

## 固定案例

所有案例h=.001s、Euler，非零root-origin世界线速度及root局部角速度，非identity单位WXYZ、真实scalar初速。初态四元数先用FP64归一满足原构造合同，连续CPU参考从实际runtime存储q/v开始，之后独立推进，不把CPU结果写回TYX。

| 案例 | 预定输入 | 步数 |
|---|---|---:|
| gyro0 | m=2.6kg，COM=0，diaginertia=(.13,.19,.23)，gravity/load/a/D均0；非球自由刚体 | 32 |
| offset0 | root COM=(.07,-.04,.03)，fullinertia=(.13,.19,.23,.006,-.004,.005)，m=2.6；世界COM wrench和native root广义力 | 24 |
| root-child1 | 偏心根+hinge子体+massive welded工具，root a=.03/D=.04，scalar a=.025/D=.08/K=.6/springref=-.12；gear=-2/affine驱动 | 32 |
| mixed7 | 7scalar，二叉分支、每第三joint为slide，其余hinge，旋转零位/local anchors/refs；最后body负载 | 24 |
| tree16 | 同生成规则16scalar，完整22维M；静态marker使free root ID>1，末关节/末身体载荷 | 12 |
| tree29 | 同规则29scalar，完整35维M与36维q，保全部welded/static身体；末关节/身体载荷 | 12 |

root-child1/mixed7/tree16/tree29前置无joint的massive marker与末尾welded tool，防止把body数当DOF数或假定root ID=1。子body质量=.6+.05*(i mod 4)，COM=(.035,-.02,.025)，fullinertia=(.025,.03,.035,.001,-.0006,.0008)，父偏移按joint编号固定；轴循环xyz，间隔一层旋转零位。source模型禁用全部可能geom碰撞对，不以初始ncon=0代替准入。

## 固定门与独立检查

| 量 | rtol | atol |
|---|---:|---:|
| body pose/COM/rotation | 3e-5 | 4e-5 |
| world linear/angular COM Jacobian | 2e-4 | 5e-5 |
| physical M（原native列SI） | 3e-4 | 5e-5 |
| bias/passive/applied | 4e-4 | 4e-4 |
| physical / Euler integration acceleration | 2e-3 | 4e-3 |
| 连续q：m/rad/无量纲WXYZ各原分量 | 0 | 1e-4 |
| 连续v：m/s及rad/s | 0 | 1e-3 |
| 时钟s | 0 | 2e-6 |

每步以实际入步q/v独立`mj_forward/fullM/jac`比较，同一控制独立`mj_step`连续推进。前/中/末保存完整oracle/receipt/input，全部步保存原控制和q/v；失败步补完整。每个输入的native qacc_smooth与独立M解也交叉核。原M与M+hD不能混为一谈。

gyro0另用解析M=diag(mI,Ibody)、bias=(0,omega×Iomega)、角加速度检验，不调用产品oracle；32步无外力线动量误差<=2e-5，世界角动量/转动能相对初值误差<=5e-4（绝对基数max(norm, .1)），作为Euler短窗门而非精确守恒。offset0另由显式四元数旋转/COM lever arm构造完整root Jacobian与M，并核力矩作用点。

真实coupling另核root施力引起非零child加速度、child/末body受力引起root反作用，预定差值>1e-4。容量覆盖要求nq/nv、全部body、末列/末行，不能截断或只跑前8DOF。

W1/W4稀疏ID、不同初态/控制、selected reset/clone后下一步；实际download之后故障全batch不发布并原请求重试；实际step禁CPU/Warp/HOST/NumPy求解；FP32 overflow不替换任何控制；strict scalar limit越界全拒；30scalar、ball、free spring、潜在接触等显式拒绝。所有case和门在实际执行之前固定，不在观测输出后追改。

本矩阵不是G1支撑/接触/摩擦验收，也不是29关节长窗性能保证。Parent另做Env/PolicyRunner和新wheel源码外连接。首次执行结果由父单独追加，不倒灌作者历史。

## 父实际验证追加（2026-11-10）

以上预登记输入、步数和容差未变。产品/测试 `1cb47675` 的完整解释20项、CUDA16项首轮通过（含父ENV用例），相邻10项及源码外新wheel两路径通过。原作者UNEXECUTED描述保留为历史；当前结果和完整边界见[正式验收](../benchmarks/p65-mujoco-tyx-floating-tree.md)。
