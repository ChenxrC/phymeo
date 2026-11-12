# 距离材料耗散 API 用法

D6-A提供可选的**轴向材料速度耗散**：在原弹性候选上按声明边顺序执行一次
隐式速度子步，再与原物理状态共同发布。它不是全局速度缩放，不修改接触法则，
也不保证布面所有振型都衰减。

当前已完成父任务的源码**功能资格**；原严格精度失败单独保留，
源码外安装资格仍待验。下面代码是用法示例，本文未执行示例。

## 声明参数：单位和稳定身份都必须明确

~~~python
from dataclasses import replace
import numpy as np
import phymeo as ph

def with_distance_dissipation(world, coefficients_ns_per_m):
    # world 是已有合法 DeformableNodeWorld。
    # 原 constraints.indices 是存储下标；参数绑定的是稳定 point ID。
    parameters = ph.DistanceDissipation(
        world.point_ids[world.constraints.indices],
        np.asarray(coefficients_ns_per_m, dtype=np.float64),
    )
    return replace(world, dissipation=parameters)
~~~

每条边一个c，单位 **N·s/m**；有限且非负，并能表示为正常或零FP32，
不能把非零小值下溢成零。原float64声明被保留并进入模型身份。
参数拥有不可变数组；E<=1536，边顺序、方向和完整数量必须与原约束完全相同。
不能排序、漏掉零系数边，或用存储下标冒充稳定ID。共享节点ID可重复出现，
同一条边重复声明也会按顺序分别执行。

参数是注册时的模型声明。改变c应创建新的world/owner或新scene声明，
不要原地修改已注册参数、设备buffer或accepted状态。

## D1：在持久节点 owner 中使用

~~~python
def one_material_step(world, coefficients, force_fp32, h, *, mode="cuda"):
    damped = with_distance_dissipation(world, coefficients)
    # force_fp32 是原 StepInputs 合同要求的 N×3 float32 外力。
    with ph.TyxDeformableBatch((damped,), mode=mode, trace=True) as batch:
        receipt = batch.step(
            {damped.key: ph.StepInputs(force_fp32)},
            h,
            expected_revision=batch.revision,
        )
        state = batch.snapshot()[damped.key]
        phase = receipt.worlds[damped.key].dissipation
        return state, phase
~~~

示例假定至少一个c>0；全零时返回原DeformableWorldStep，没有dissipation字段。
实际应用应在owner作用域内持续step，使用公共checkpoint/reset/restore及revision，
无需每步重新创建owner。耗散位于standby候选，失败不改accepted bank；
下一物理步读取真正接受的耗散后速度。

active域增加设备系数、重建速度副本和32列trace；预算可在创建runtime前调用
TyxDeformableBatch.requirements取得。原memory_limit_bytes仍覆盖已声明的
device＋owned HOST数组储备，不是进程RSS上限。

## D5：通过已有场景声明透传

对已经合法的deformable_normal场景替换其材料world即可，保留原资产、
collider/surface安装、稳定身份和完整pair材料表：

~~~python
def add_scene_dissipation(declaration, coefficients_by_world):
    if declaration.contact_response != "deformable_normal":
        raise ValueError("requires the explicit deformable_normal scene route")
    original = declaration.deformable
    if set(coefficients_by_world) != {w.key for w in original.worlds}:
        raise ValueError("one complete coefficient declaration per world is required")
    worlds = tuple(
        with_distance_dissipation(w, coefficients_by_world[w.key])
        for w in original.worlds
    )
    return replace(declaration, deformable=replace(original, worlds=worlds))

# declaration、forces_by_world、h由应用原模型给出。
# damped_scene = add_scene_dissipation(declaration, coefficients_by_world)
# with ph.compile_participant_scene(damped_scene, mode="cuda") as scene:
#     receipt = scene.step(forces_by_world, h, expected_revision=scene.revision)
#     phase = receipt.node_receipt.worlds[world_key].dissipation
~~~

ParticipantDeformable会拥有并校验完整声明，包括新增耗散字段，
通过原D5 lowering交给同一个D3共同事务；不要另外创建D1 owner后顺序step。
当前仍限原D5的每world一个ordinary单body运动球/胶囊与一个节点模型，
不是任意关节机器人、多个刚体或MuJoCo模型与节点的通用联合求解。

原NormalLayer的接触damping单位是s/m，和这里的c不同，不能直接互换。
当前接触力继续按真实入步几何/速度采样，不在耗散后偷偷重算本步接触力。

## 读取phase和固定支座反力

active step是DeformableDissipatedWorldStep；phase中主要字段：

| 字段 | 含义 |
| --- | --- |
| reconstructed_velocity | 原弹性积分/投影后重建的FP32速度 |
| dissipated_velocity | 新接受速度，必须与accepted state绑定 |
| trace | 每条边实际法向、rate、冲量、前后速度及h/c/质量的32列设备记录 |
| node_impulse | 按设备实际j*n累积的requested材料耗散冲量，包括固定端 |
| node_velocity_impulse | 实际速度差除以注册stored inverse mass；固定端为零 |
| kinetic_change / ideal_work / rounding_work | 每edge实际动能差、理想离散功和有符号舍入差 |

质量口径是D1注册后的stored FP32 inverse mass。requested impulse和实际速度冲量
不强制相等；两者差异不能被抹去。原弹性算术审计应对reconstructed_velocity，
不能继续将耗散后的accepted velocity当作旧位置重建结果。

D3 receipt.worlds[world_key]保留原material_impulse（全部ordered弹性行），
并仅在active时增加damping_impulse和damping_step。
固定反力为原外力/接触载荷、elastic impulse、damping impulse三者负和。
固定点速度保持零，但其reaction impulse不能清零。
不要把整步外载、弹性、接触和材料耗散的能量混成“总能量必须下降”。
能量字段是原始记录，不是允许求解器增加能量的credit。

## None、全零和合理校准

- dissipation=None保持原数值路径、row类型、设备分配和launch。
- 显式全零也不执行耗散或新增设备buffer；但声明进入source/checkpoint身份，
  因而不能与None模型互换checkpoint。active row才有phase。
- 每条边每物理h仅执行一次，与弹性迭代数无关；完整网络是有序一阶分裂，
  不是同时求解所有边，也不是对edge重排不变。

c应来自材料或模态的物理标定。在线性单轴弹簧近似下，可用
c=2*zeta*sqrt(k*m_eff)作为量纲明确的起点，m_eff=1/(wi+wj)；
它不是任意网格/接触/大变形的统一临界阻尼公式。
compliance=0的硬约束不能用k=1/compliance推导此参数。
应先用独立单边/振子校准，再声明应用模型；不要扫描c直到旧失败窗口变绿。

该算子对瞬时刚体平移/转动速度是轴向零模；平膜某些面外速度的一阶轴向rate
也可为零。没有弯曲/剪切耗散自由度时，增加轴向c不保证这些模态衰减。
它不支持运动规定支座；非零c的零长度边明确拒绝。
有限转角整体积分误差与瞬时客观性应分开观察。

## 如何理解当前资格

父已完成独立kernel、D1连续/事务及D3球/斜胶囊短耦合的源码功能验证。
D6 phase原硬门、完整FP32弹性算术核、连续参考及共同发布要求保持。
但两个原strict失败仍保留：

- run104：旧理想总动量2e-5门失败；同次保存算术闭合将差异定位到原弹性舍入。
- run112：旧弹性重建速度2e-5门失败。

因此应记录为**限定功能资格＋已知严格精度失败**，不能写成全部原精度门通过。
源码外安装仍待验；当前没有持续稳态支撑通过结论，也没有将原D4失败改写为成功。

细节见[设计合同](../superpowers/specs/2026-11-13-distance-dissipation-design.md)、
[作者接口与资源报告](p65-distance-dissipation-report.md)。
本文只新增使用说明，未执行示例或修改产品/旧验收记录。
