# BALL-S：球关节树中的标量限位与干摩擦

本文说明当前公共接口与限定功能。示例仅按源码核对，尚未单独执行；
本批有限源码、事务和双后端安装资格已完成，详见[验收报告](p65-ball-scalar-constraints-report-20261113.md)。
下方额外用法示例仍只是源码核对示例，不冒充冻结矩阵中的另一次执行。

## 可以组合什么

固定根 BALL/HINGE/SLIDE 树，或单 FREE 根下的 BALL/HINGE/SLIDE 树，可以组合：

- HINGE/SLIDE 的软上下限位；不是把位置强行裁到范围内。
- native DOF 干摩擦；不是粘性 `damping`，也不是额外的被动力近似。
  XML 的 `joint frictionloss` 会由 MuJoCo 编译到 DOF；BALL 对应三个旋转 DOF，
  发布记录使用各自的 native DOF 身份。所有系数必须有限且非负。
- JOINT 传动的 NONE 或 FILTEREXACT 执行器，包括 BALL 三分量 gear、共享目标关节。
- 只有关节约束的模型，或同时具有原 primitive condim1／pyramidal condim3 接触的模型。

有标量限位或正干摩擦时，即使没有接触对，也使用原 T4 联合约束求解。
这一域要求源模型 `cone="pyramidal"`；不能用无接触来绕过该要求。
BALL 自身的角度限位、BALL 弹簧与 BALL affine bias 仍不在该域内。

## 从 XML 创建并推进

构造参数是 `implementation="tyx"`，不是 `backend="tyx"`；`backend` 是结果中的描述字段。
`mode="interpret"` 使用 TYX 解释路径，换成 `mode="cuda"` 使用 TYX CUDA。
`mode="cpu", implementation="native"` 则是另一条原生 CPU MuJoCo 路径。

下面是无接触的固定根 BALL + HINGE 示例。XML 编译在 CPU 完成；随后状态推进走选定的 TYX 路径。

```python
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.backends.mujoco_runtime import (
    MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime,
)
from phymeo.backends.mujoco_checkpoint import MuJoCoTyxCheckpoint

xml = """<mujoco>
  <compiler angle="radian" autolimits="true"/>
  <option timestep=".001" gravity="0 0 0" integrator="Euler"
          solver="Newton" iterations="128" tolerance="1e-10"
          jacobian="dense" cone="pyramidal"/>
  <worldbody>
    <body name="ball_body">
      <joint name="ball" type="ball" limited="false" frictionloss=".2"
             armature=".003" damping=".04"/>
      <inertial pos=".01 0 0" mass="1" diaginertia=".03 .04 .05"/>
      <body name="tip" pos=".12 0 -.1">
        <joint name="hinge" type="hinge" axis="0 1 0"
               limited="true" range="-.25 .25" margin=".03125"
               frictionloss=".2" armature=".003" damping=".04"/>
        <inertial pos="0 0 -.02" mass=".5" diaginertia=".012 .014 .016"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <general joint="ball" gear="1 -.5 .25" dyntype="filterexact" dynprm=".004"/>
    <motor joint="hinge" gear="-.5"/>
  </actuator>
</mujoco>"""
compiled = compile_mujoco_xml(xml, require_asset=False)
starts = {
    3: MuJoCoInitialState(q=(1, 0, 0, 0, .24), v=(.1, -.05, .03, .02), act=(.05,)),
    19: MuJoCoInitialState(q=(1, 0, 0, 0, -.24), v=(-.1, .03, -.02, -.02), act=(-.05,)),
}
with MuJoCoRuntime(
    compiled, implementation="tyx", mode="interpret",
    world_keys=(3, 19), initial_states=starts,
) as runtime:
    runtime.set_controls({3: MuJoCoControl((.1, -.05)), 19: MuJoCoControl((-.1, .05))})
    accepted = runtime.step(.001)  # 必须等于 XML timestep；返回逐 world 接受映射。
    state = runtime.integration_snapshot()
    receipt = state[3].last_step
    print(accepted, state[3].q, state[3].v, state[3].act)
    print(runtime.capabilities["domain"], receipt.raw_contact_fields["counts"])
    for row in receipt.constraints:
        print(row.constraint_type, row.source_id, row.dof, row.side_or_edge, row.force)

    capsule = MuJoCoTyxCheckpoint.from_bytes(runtime.checkpoint().to_bytes())
    runtime.step(.001)
    runtime.restore(capsule, expected_revision=runtime.checkpoint_revision)
    runtime.step(.001)  # 从保存的真实积分状态、时钟和控制继续。
    runtime.reset((19,))  # 仅该 world 回到构造初态；之后可继续实际推进。
```

FREE 组合需在树上增加唯一 FREE 根，并按 native 完整 qpos/qvel 顺序提供初态：
FREE 是 7 个位置坐标／6 个速度，BALL 是 WXYZ 四元数／3 个局部角速度；
不能把 qpos 地址当 DOF 地址。控制向量按编译后的 actuator 顺序，FREE 本身不能作为 JOINT 执行器目标。

## 区分声明容量与每步实际力

存在标量／干摩擦行的 BALL-S `capabilities` 提供：

| 字段 | 含义 |
| --- | --- |
| `domain` | fixed/free、joint-only 或 mixed-contact，以及 FILTEREXACT 的组合域 |
| `native_dry_dof_ids` / `native_scalar_limit_joint_ids` | 声明的完整 native 行来源 |
| `dry_constraint_rows` / `scalar_limited_joints` / `candidate_slots` | F、L、静态原始候选槽 S |
| `contact_dimensions` | 声明的接触维度；joint-only 为 `()` |
| `conservative_efc_bound` | F + 2L + 各原始槽的行数；condim1 为 1，condim3 为 4 |
| `contact_capacity` / `constraint_capacity` | 本 runtime 请求的硬容量；不是本步实际使用量 |
| `constraint_capacity_policy` | 实际活动行溢出拒绝，静态上界只作诊断 |

`last_step.raw_contact_fields["counts"]` 是该已接受步的检测接触数、活动接触数、实际 EFC 行数；
`last_step.constraints` 保存完整实际行的 J、R、aref、上下界、力和身份。
静态限位声明不意味着每步都有两个活动限位行，也不能用静态上界替代运行时容量检查。
无正干摩擦、无限位的旧模型保留原能力标签，不应假设上述 BALL-S 专属字段总存在。

`last_step` 同时保留完整入步状态、执行器力、约束力、接触双方 COM 力矩及动力学记录；
有 FILTEREXACT／完整 JOINT 传动时还含相应 actuation／transmission。
`integration_snapshot()` 读取已接受积分状态和同次收据；`snapshot()` 另有端点 CPU 观察计算，
其端点接触不能拿来替代上一步实际求解的接触力。整批失败不会部分发布某个 world。

当前保留 fixed nv8/nq10、FREE nv35/nq45、body128、actor32、activation32，
及 geometry128／pairs16／rawslots32、contact128／EFC512 上限。
显式碰撞 excludes、多个 FREE 根、新形状／condim4或6、BALL 角限位仍拒绝。
本功能不新增刚柔接触或异构扩容，也不表示 G1 完整长舞、持续支撑或所有机器人组合已经验收。
