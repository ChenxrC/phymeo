# MuJoCo-TYX T2：原生自由根树源码交接

**CODE_READY_TESTS_PENDING / UNEXECUTED，2026-11-10。** 作者复用原support隔离worktree，分支 `codex/mujoco-tyx-t2-source`，基于已验T1父 `4ab18bd580b5be77dd6037283cafa1509412016c`。本批只新增三产品模块和本文。没有import、collection、pytest、数值probe、lint/formatter、compile/build/install/profile、native或CI执行；Git文本检查不是项目测试。父保共享runtime/dispatch和唯一项目执行owner。

## 模型与接口

`compile_floating_model(mj, model)` 返回 `TyxFloatingModel(kind=3)`，从原native编译字段构建静态表，不使用某姿态的M/J/bias作为一般模型。域是一个world-child free joint加0–29个后代hinge/slide，`nq=7+n`、`nv=6+n`，保全部body ID、massive welded与world下静态身体，最多128body。root body可以不是ID1。ball、多free、独立活动树、可能接触等明确拒绝。

- `root_body`和继承的`body`均为原root ID。
- `joint_ids/joint_data`按**DOF**地址建表：前3行为world平移，后3行为root-local旋转；6之后每行保原scalar qpos/dof/body地址、kind、axis、anchor与ref。native标量qpos地址是dof+1，原身份未重新编号。
- `body_joint_range`仅遍历scalar行；root range为空，由free分支直接处理。`ancestors`保6个root列和各scalar列，静态身体的整行零。
- `vectors` shape `(9,nv)`：原D、scalarK、scalar qpos_spring、armature、strict-interior上下限，限位按DOF地址；root a/D六列完整保留。free-joint stiffness不走scalar公式，显式`free-spring`拒绝。
- `TyxFloatingDevice`沿T1构造参数与download schema。向量/矩阵动态宽度为`max(8,nv)`，q buffer直接拥有父传入的`max(8,nq)`副本，固定math scratch仍为8×3；不在每步用host计算root力学或补填结果。

父需在原_model路由T2并扩展runtime的nq/nv打包、scalar限位qpos地址、kind3设备和dynamics receipt分派。原T0自由球与T1树路由继续保持，不能在旧路径失败后静默换方程。父还需追加包NOTICE的T2文件映射；本文未修改旧文件。

## 每步三pass及原生坐标

1. **geometry。** free qpos前三位直接是root body origin world坐标，后四位WXYZ。仅在scratch归一化用于R，原始input bank不改。root origin world velocity为v前三列；world omega为`R @ v[3:6]`。在native qacc=0时root origin acceleration=0、world alpha=`R(omega_local×omega_local)=0`。随后按完整body/joint顺序递推子树和COM，保全部离心、Coriolis及alpha运输项。
2. **mechanics。** root平移J为world基；root角列Jw为R各列，Jv为该列叉乘`COM-root_origin`。子树scalar列沿原T1完整链规则。装配 `M=Σ(mJvᵀJv+JwᵀIworldJw)+diag(armature)` 与 `bias=Σ[Jvᵀm(aCOM-g)+Jwᵀ(Ialpha+omega×Iomega)]`；包括所有massive welded成员和所有根–关节交叉项。世界COM wrench经完整Jᵀ回传native generalized force；root局部角广义力不是未旋转的世界torque。
3. **solve。** physical M与`M+hD`分别dense Cholesky，rhs为applied+passive-bias，原D只在passive中计一次；两份实际加速度分别读回。先更新原生v，再更新root world平移和scalar q；WXYZ使用归一化输入四元数**右乘**local angular increment，并归一化candidate。原生MuJoCo输入归一化与Warp0.0.2输出归一化的FP32差异交独立工程门验证，不声明FP64逐位相等。

scalar spring使用qpos_spring而非FK ref；affine驱动使用raw q×gear，保ctrl clamp、gain/bias、disabled-group置零之后force clamp、最后gear投影/聚合。root qfrc_applied与全部body world COM力/力矩均参与，未偷偷替换为PD或HOST动力学。

## 事务、边界与观察

三个kernel仅写candidate state/scratch；晚回读或receipt失败前不交换active bank。继承publish/close资源协议。非法quat、非有限、Cholesky坏pivot、strict-interior越界均保非零status，父负责全批拒绝，不做accepted-state修复。坏pivot分支中的1只避免失败计算再次除零，非正定结果不会被发布。

`download()`与T1一样给出输入q/v/time、body origin/rotation/COM、world COM J、M、bias、passive、applied、smooth acceleration和integration acceleration，保真实nq≠nv。snapshot的detached CPU端点forward继续由原父runtime明确分开；这些新kernel不调用CPU/Warp/HOST求解或NumPy linalg。

范围不包括free spring、复杂传动/激活、ball/equality/tendon、fluid/gravcomp/dry friction、非Euler/非默认模拟flags、可能接触及限位stop求解。native合法却不在本域的来源需明确拒绝，不能删原字段后称同源支持。29scalar容量不是G1持续支撑通过。

## 纯author检查点

`ec1e9458`新增static model；`9594c1e3`新增真实TYX kernels/device。工作文件SHA256（父格式化/验证后应另绑定新版本）：

| 文件 | SHA256 |
|---|---|
| `_floating_model.py` | `4fe13394058e8352d9763524972d86cf3a88a63b0134e767c787a29ae2fcdadc` |
| `_floating_kernels.py` | `b9190e51ee76331f90b4ea6a24416665a5721fb41b805e7d40a9096a7129dad8` |
| `_floating_device.py` | `31b6a88a4e91020c514208b9667411bbc41e0b2d7b93511892c2dab59d5cb305` |

源码全文纯文本回读，独立测试作者另写原生CPU/解析多案例；所有实际执行、独立保存、集成与wheel验证待父完成。不预报测试数量或物理通过。

## 固定来源与归属

参照固定MuJoCo-Warp0.0.2的`_src/smooth.py` kinematics/com_vel/crb/rne及scalar transmission、`passive.py`、`forward.py` Euler/_advance，另以CPU MuJoCo3.3.7的free FK/J、native quaternion integration与passive语义为准。三新模块是TYX数组/世界COM完整装配改写，沿用Apache LICENSE与原Newton归属头部，追加PhyMeo修改归属；没有通过改import包装Warp执行。前两者与父T1实现共享公式和低级schema/helper，T0/T1源码字节未修改。

## 父集成验证追加（2026-11-10）

父完成共享路由、独立nq/nv打包、原生限位地址和receipt连接，并补包内NOTICE。产品/测试 `1cb47675` 已通过解释20、CUDA16、相邻10项和新wheel源码外双路径；作者阶段hash保留为历史，最终源码由实际命令快照绑定。详[正式结果](../benchmarks/p65-mujoco-tyx-floating-tree.md)。本次未扩到可能接触或G1支撑。
