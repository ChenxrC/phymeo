# 当前状态驱动的面积压力连续场景：I2b

日期：2026-11-17；分支 codex/exp；实现提交 d5b3dec901fd57bb8cd3ae7c3d027f7357bd84fb。本批没有修改 dev。
已实现每步从真实 TYX 已接受状态重新生成有限接触面，再进入共同动力学求解。

## 直接看结果

![实际 CUDA 连续轨迹](../../artifacts/area-pressure-current-scene-20261117-01/figures-final/current-area-contact.png)

图的读法：

1. 左上：负压入量是分离，越过零后进入压力层，随后再次分离。
2. 右上：接触力随压缩增长、再卸载到零；两条线是两个步长。
3. 左下：能量下降主要是后向欧拉数值耗散；这里没有标定物理阻尼。
4. 右下：另一个偏心转动关节案例，接触区从无到有、面积改变后消失，
   关节角速度由真实压力力矩改变。没有人为指定这些运动轨迹。

这里的“两个可动体”是两个有真实质量的滑动关节体，不是通用自由 6DOF
刚体覆盖。转动案例另外检验真实 hinge 与另一滑动体的共同响应。

## 本批结果

22 个新测试通过：16 HOST 协议节点、3 interpret 节点、3 CUDA 节点。
六条连续轨迹完整保存，共 1280 个实际物理步；另外的 FK 故障/重试步
不混进这个轨迹计数。HOST/interpret 首轮 19 passed /45.29 s，
CUDA 首轮 3 passed /18.58 s。没有动力学失败、调参或重跑旧绿套。

| CUDA 双滑动体 | h=0.5 ms | h=0.25 ms |
|---|---:|---:|
| 首次正接触力 | 10.00 ms | 10.00 ms |
| 最后接触步末 | 29.50 ms | 29.75 ms |
| 峰值法向力 | 10.6590 N | 10.9931 N |
| 最大压入量 | 1.1843 mm | 1.2215 mm |
| 分离后相对速度 | 0.17670 m/s | 0.18794 m/s |
| 相对机械能损失 | 21.94% | 11.70% |
| 最大总动量误差 | 1.47e-8 kg m/s | 2.42e-8 kg m/s |

峰值力/压入量的两步长差约 3.04%，接触结束时间差 0.25 ms，均通过
执行前固定的工程门。相对能量上漂低于 4e-13；粗/细步长耗散各低于
预设 30%/18%。这不是能量精确守恒，也不意味着已达到连续时间解。

偏心转动 CUDA 案例峰力 11.1370 N，接触面积在 0–450 mm² 之间变化，
角速度从 2 rad/s 变为约 1.21699 rad/s。该案例检查了随姿态变化的
压力深度、真实点 Jacobian、广义力和同一步力律；没有声称一般转动
接触的全局能量定理或精确终点几何收敛。

## 实际连接

新增 BodyAreaPressureAsset 和 MuJoCoTyxAreaPressureScene：

```text
body-local 完整压力体 + body-local 有限刚面
    ↓ runtime 锁 / revision / timestep
当前 accepted-bank TYX FK
    ↓ 世界坐标资产变换
I1 HOST tetra–triangle 裁剪与真实 dA
    ↓ 每个 world / pair 的新 stamp
I2a 设备 Jacobian、共同求解、实际积分
    ↓ 原有物理原子提交
状态 + 实际载荷 + 本次几何记录
```

- 不读取 last_step 中上一入步的位姿；也不使用 CPU MuJoCo 推进或 FK
  代替设备状态。FK 读回的 q/v/补偿时钟与 accepted mirrors 逐项核对。
- HOST 做网格变换与裁剪；动力学仍为 TYX interpret/CUDA。
- 每个资产只接受 full 零边界压力体，local 制造场明确拒绝。
  测试使用四 cell 的完整压力体，有限小刚片位于底 cell 的正梯度区域；
  没有删除其他 cell 来绕过范围判断。
- 所有世界先完成几何生成，随后只提交一次完整 runtime step。
  某个世界生成失败或物理失败时，旧状态和旧 last_geometry 保留。
- last_geometry 是最后成功物理步的**入步几何**，不是新终点几何。
  外部 control/reset/step 改变 revision 后，此属性返回空，下一步重新生成。
- 锁内准备 FK 可改 candidate scratch，但不改 accepted bank 或旧 owned receipt；
  本批不承诺失败后调试 scratch 字节不变。

调用方式：

```python
runtime = MuJoCoTyxRuntime(
    compilation, mode="cuda",
    area_pressure_pairs=(("pressure", "plate"),),
)
asset = BodyAreaPressureAsset(identity, pressure_body_local, rigid_face_local)
scene = MuJoCoTyxAreaPressureScene(runtime, (asset,))
for _ in range(step_count):
    scene.step(h, expected_revision=runtime.checkpoint_revision)
    state = runtime.integration_snapshot()[world]
    geometry = scene.last_geometry[world]  # 对应 state.last_step.input_time
```

资产自身保存节点/cell/face/material 身份和不可写数组；场景每步生成新
世界坐标值。原生 pair 排除、容量、法向力共同发布等复用 I2a，本批未
改旧求解器、原有碰撞路径或 dev 分支。

## 独立检验

双滑动体 m=(0.5,1.25) kg，初始间距 2 mm、相对靠近速度 0.2 m/s，
无重力/关节阻尼，两个 h 均连续推进到 0.08 s。完整压力体底 cell
p=2e7*z，刚片面积 .00045 m²，因此本域 K=9000 N/m、约化质量 5/14 kg。

每个实际入步，用独立标量几何与 BE 闭式预测该步力和速度；没有调用
裁剪器或面积行装配器当期望值。另核三点各 F/3、共同原点力矩、
总动量、相对能量、分离后零吸引力。位移域始终小于预设 4 mm，因此
压力场适用区不是依据运行结果重新挑选的。

协议组使用 fake runtime，只验证所有权/变换/元数据/事务，不冒充
物理或 FK oracle。真实连续组另包含设备 accepted-q 读回篡改：
在实际物理推进前拒绝，旧状态、旧几何保留，随后正常重试成功。

## 证据与验证边界

原始记录：artifacts/area-pressure-current-scene-20261117-01/records。
trajectory-manifest.json 保存六文件的原路径、SHA256、size、步数与摘要。
source-before.json 在动力学执行前保存源绑定，归档记录时确认无变化。
所有首次 lint、执行 guard、command、stdout/stderr 与 RAW_EXIT 均保留。

绘图只读取这些保存记录，不重跑物理。第一次图的空白坐标范围作了显示
调整，最终图单独保存；没有改变数据。Ruff/format-check 最终通过。
环境沿用已验证的 .venv-training-cuda313；无安装、远程 CI 或性能比较。

## 仍未完成与下一步

这是**小型 full 压力资产、有限刚面、无摩擦的连续连接**，不是完整通用
Hydroelastic/CCD。接触仍按入步采样，步内出生可延迟到下个采样；没有
宣称严格不穿透。I1 的正有效梯度域和资源限制全部保留。大幅转动至
不支持域必须拒绝，不能翻法向或丢掉那部分接触。

本批尚未验证持续外载下的静态承载、多处面积接触耦合、摩擦、面积设备
观察 ABI 或 checkpoint/clone。下一批优先持续外载与两处接触的功能
案例，然后补面积观察/生命周期；性能阶段再移植几何到 TYX，避免把
HOST 原型称为全 GPU 碰撞。

与 FFT 的关系不变：这条运行时是局部压力场法向接触，不能自动代表
FFT 的非局部弹性材料响应。先前未通过资格的力表没有接入，未声称
“自动选算法”或在运行时混合 Kamino/FFT。

[冻结计划](../superpowers/plans/2026-11-17-area-pressure-current-scene.md) ·
[I2a报告](area-pressure-tyx-step-report-20261117.md) ·
[总体架构](contact-model-architecture-20261117.md)
