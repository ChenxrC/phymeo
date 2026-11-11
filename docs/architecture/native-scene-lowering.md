# 自有场景到原生模型的编译切片

`builders.mujoco_scene.compile_mujoco_world` 将现有 `ArticulationWorld` 编译为
`CompiledMuJoCoWorld`，属于专用 solver 适配，不增加第二套模型/状态，也不导入 Newton、
Warp 或 MuJoCo-Warp。resident 继续使用同一声明与原 `compile_scene`，不需要原生模型。
本入口完成冷编译；PhySolveRuntime 已可消费完整结果。P2PE Stage 的原生资产切片
见[装配与初态](native-assembly.md)，两个入口均未取得 C ABI 资格。

## 输入与物理语义

显式传入 `MuJoCoWorldProfile(timestep, friction, joint_limits, contact_dim=6)`。
它选择原生 Euler、dense Jacobian、Newton 约束算法及 pyramidal 接触；此 Newton 是
MuJoCo 求解算法名称，不是 Newton Python 引擎依赖。

- friction 三项分别为滑动、扭转和滚动。contact_dim 可为 1/3/4/6；维度不足却传入
  非零摩擦会失败，不能静默忽略。使用原生默认 solref/solimp，不从 resident 材料拟合。
- 控制使用原生 affine PD：`kp*(target_q-q)+kd*(target_v-v)+feedforward`，整体受
  effort cap 约束；这不是 resident 的隐式 bounded-PD 算法。PHYMEO cap=0 表示零驱动，
  不等于 P2PE 的不限力 sentinel；P2PE 转换尚待显式处理。
- joint_limits 决定是否生成原生软限位，关闭时仅保留资产边界语义。未支持单侧限位。
- 现有声明只允许单 articulation 与有限静态环境。此切片沿其 self_collision=false
  语义，仅生成有显式 material 且 group/mask 双向匹配的活动机器人–静态环境 pair。
  material 数值不被当成原生接触参数。固定根–静态物体、静态–静态和机器人自碰撞
  不进入此声明的响应；没有改变原生文件入口及其原过滤行为。

支持普通固定/自由根树、root-only 刚体、球/胶囊/凸 hull、圆柱/椭球，以及
FoldedURDFAsset/Panda 的具名 frame。mimic、self_collision=true、初始外部 body load、
未支持几何显式拒绝。多参与者同 world、动态环境、复杂执行器和任意 mesh 仍须扩展。
Panda 验证以 `load_panda()` 的公共资产为输入，不宣称原始 URDF 所有标签已承接。

## 编译绑定版本 1

结果的 `binding_version=1` 描述本编译绑定，不代表完整引擎执行合同版本。

| 字段 | 含义与 owner |
| --- | --- |
| compilation | 原有 MuJoCo 编译快照，拥有 MJB/资源；不借源文件路径 |
| world / instance / declaration_identity | 声明身份；原生地址只在本次编译有效 |
| body_ids | 公共物理树的 body 名到 native body；折叠后物理轴可能是主惯量轴 |
| frames | authored frame 到物理 body 的局部变换，保留固定关节合并后的具名 frame |
| joints | 公共 joint 顺序对应 native joint/qpos/DOF/actuator 地址；不假定深度优先顺序相同 |
| shapes | `source_identity=(asset或environment域, source_key)` 与 native geom 地址；还需 world/instance 限定 |
| initial_qpos/qvel/ctrl | 拥有的只读初始数组；qpos 与 DOF 维度独立 |

浮根 public q 使用原点 XYZ/XYZW；public v 使用世界 COM 线速度、世界角速度。
原生 qvel 转为原点世界线速度和根局部角速度，包含偏心 COM 的叉乘项。
普通固定根第一层关节已在世界坐标中，生成根 body 时显式逆变换，不能重复施加根位姿。
折叠资产的碰撞点从 authored link frame 转到物理 owner frame；圆柱/椭球轴同步转换。

`fresh_native_data()` 从 MJB 创建独立 model/data，并写入三个初始数组，不执行 forward。
仅把 `.compilation` 交给 Runtime 仍会丢失该声明的非零初态/控制，必须直接传完整
CompiledMuJoCoWorld。MJB 的 qpos0 是结构参考，不能为设置初态改写它。

冷编译复杂度为 O(B+J+S+R×E)，R/E 为机器人/环境 shape 数；显式 pair 数影响原生容量。
不创建运行时、设备数组或 Graph，不增加每步复制/同步。批量同构运行必须复用模板编译，
不能在热路径逐 world 调用本函数。最终部署和性能不由本编译切片资格代替。

## 验证入口与剩余工作

`tests/test_mujoco_scene.py` 使用 FP64 FK/COM、质量矩阵及 CPU 原生模型作编译对照，
覆盖初始控制、固定/浮根、树排序、多 shape、域身份、过滤、owner 隔离与拒绝路径。
FP64 pose/mass 绝对门限 1e-10，原生 FP32 hull 顶点几何为 2e-7 m；resident 小样继承
原 FP32 用例 2e-7/3e-5 门限。所有门限在相应候选对照前定义，没有放宽历史机器人门限。
现有 `test_mujoco_compiler.py`、`test_scene_compiler.py` 同时回归；CPU/CUDA 分开记证据。

完整 B1 仍需 P2PE Stage 转换、多参与者表达、所需机器人导入能力及同构模板绑定；
B2 再消费完整初态/控制、接入显式碰撞排除和原生 Host，关闭 Panda/FANUC 求解失败。
本切片不修改 codedev、TYX 或六 CUBIN，不是机器人 step/forward/reset 通过结论。

2026-11-11 B1a 实测：源码三个模块 91 通过；新 wheel 安装后在源码目录之外、
禁止 Newton/Warp/MuJoCo-Warp 导入的进程中复测 91 通过。resident CUDA 选择集
5 通过，并额外禁止 CPU MuJoCo 导入；没有 skip。该选择集含不同树/自由根、
真实接触反作用和 armature/damping，但不替代 resident C ABI N=1/8/64 资格。
pip check、限定文件 Ruff 与 diff 检查通过；全仓文档仍为 479 个历史缺失目标，
与 B0 相同。独立代理完成只读复审，两项身份/摩擦发现已修复并补回归。
测试初轮的 mj_fullM 调用、Panda API/电机额定值 fixture 错误已修正，没有放宽容差。
原始日志、安装路径与包散列保存在独立实验根 `evidence/b1-*`；本节点可本地提交，
不能据此声称完整 B1 或整条消费者链路通过。
