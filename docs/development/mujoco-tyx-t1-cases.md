# T1 固定根旋转树独立案例（预登记）

状态：**AUTHORING_ONLY / UNEXECUTED**。2026-11-10，基线父 `af3a78545d00cfbfa18d4efeb5da324352923a81`；`codex/mujoco-tyx-t1-tests` 独立新增测试，不改产品或旧测试。原 support 作者负责产品，父唯一执行。这里的参数/容差先于任何执行固定；不是已测成功，也不在运行后追调质量、初态或门。

## 域和公共量

固定根1–8个 scalar hinge/slide，支持同 body 多 joint 按 native 顺序复合及 massive welded body。所有物理几何 `contype=conaffinity=0`，无可能接触；h=.001s，原 native Euler，非零真实入步 q/v，质量 .4–2kg、偏心 COM、各向异性惯量。无 free root、ball、tendon/equality、activation 或碰撞能力声明。关节有限域沿原 strict-interior guard，跨域拒绝，不把停止力响应当已实现。

`receipt.last_step.dynamics` 的 body_position/rotation/com、jacobian_com(linear/angular world)、mass_matrix、bias(C+gravity)、passive(-Dv-K(q-qpos_spring))、applied(actuator+qfrc+COMworld wrench投影)、smooth_acceleration 均来自同次入步 actual TYX readback。phase=`tyx-euler-input-dynamics`，time=input_time，保存input_q/input_v。连续smooth a=M^-1(applied+passive-bias)，Euler integration acceleration另核(M+hD)^-1同一rhs；不混 physical M/implicit mobility。

## 固定矩阵

| 案例 | 固定非平凡因素 | 步数 |
|---|---|---|
| pendulum | 2kg、COM x=.4m、Iyy=.07、armature=.11、D=.17、ref=.125；独立解析 M=.5 与重力bias，实际world COM力矩 | 24 |
| double | 两摆真实非零 qdot；完整Coriolis/bias、a、implicit damping | 48 |
| mixed | 同body hinge+slide后接hinge，offset anchors/refs，gear2/-1/.5 | 48 |
| fork | 两分支共享祖先、不同质量、world wrench施到单分支；完整M/J交叉项 | 24 |
| seven | 7铰链非主轴fullinertia/COM、旋转零位、local anchor/ref；最后body受力 | 24 |
| welded | 2 joint、额外massive静态tool与固定marker；tool COM力/力矩经真实祖先J传入 | 48 |
| eight | 8关节边界、最后关节/身体、完整列保留 | 24 |

除解析摆外，motor有ctrl/force双clamp，纯stateless直接传动。CPU oracle由同一原XML独立编译，不能从作者T1输出生成期望矩阵。每步以实际receipt入步q/v独立`mj_forward/mj_fullM/mj_jac`核分层量；另保从原初态连续`mj_step`的完整短窗，不把任何oracle q/v写回TYX。

## 预先固定的FP32工程门

| 量与SI单位 | rtol | atol |
|---|---:|---:|
| FK/body COM(m)、rotation(dimensionless) | 2e-5 | 3e-5 |
| COM J（native各列对应SI） | 2e-4 | 5e-5 |
| M（按native hinge/slide对应SI） | 2e-4 | 3e-5 |
| bias/passive/applied（Nm或N） | 3e-4 | 2e-4 |
| smooth/integration acceleration（rad/s²或m/s²） | 5e-4 | 8e-4 |
| 连续24/48步q（rad或m） | 0 | 5e-5 |
| 连续24/48步v（rad/s或m/s） | 0 | 3e-4 |
| 时钟(s) | 0 | 2e-6 |

这些是模块级原生同律工程门，不是修改旧物理预算。解释/CUDA独立参数选择；缺可选MuJoCo/TYX只在依赖边界明确skip，CUDA实际执行不借CPU替代。

## 失败、隔离和适量保存

W1/W4稀疏keys/不同原初态与controls；selected reset后未reset worlds的**下一步**与clone相同，而不只snapshot相同。晚download/decode故障在candidate工作完成后注入，整个batch无q/v/tick/控制发布，再原请求重试；CPU/Warp/HOST物理入口禁用，snapshot允许明确标记的detached CPU观察。

每例保原XML/参数、全部步q/v与controls摘要；预定 step 1、中点、末点保存完整oracle和同次actual dynamics/receipt/input/storedcontrol，失败所在步也完整保存。文件只写本test tmp_path，不新跑producer取数，不造大规模冗余轨迹。父另负责Env/PolicyRunner与源码外wheel consumer，避免本测试复制该连接任务。
