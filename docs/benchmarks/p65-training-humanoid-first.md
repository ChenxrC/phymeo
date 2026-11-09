# 训练接口与 G1 驱动策略：首批结果（2026-11-09）

用户新优先级已落在[主线计划](../superpowers/plans/2026-11-09-training-humanoid-priority.md)。[K0 默认距离/见证已验收](p65-device-convex-query-acceptance.md)，提交 `b3f4cf3`；优先级文档提交 `625e715`。

## 通用训练入口

新增 `phymeo.training.ResidentVectorEnv`，借用已有 batch，task 提供 controls/observe/evaluate。每次一个真实 batch physical step；提供 batched action/observation/reward/terminated/truncated、独立 episode ID/length、保留终止前 final observation，再自动复位已结束世界。借用必须覆盖整个 batch，防止未列出的世界被暗中推进。

拒绝非法动作时不推进物理；物理已接受后 task/readback 失败，会显式抛 `VectorEnvStepError(physics_accepted=True)` 并要求全部复位，不能把已发生的步装成原子失败。结果数组 bytes-backed 不可写。wrapper 不拥有或关闭 batch。

实际 TYX **interpret 5 passed / .57s，CUDA 5 passed / 1.54s**。真实 W1/W32 自由球受力、终止状态留存、独立自动复位、超时、非法动作和 late observer failure 均有测试；目录 `artifacts/p65-training-env-20261109-01/`。这是 NumPy/HOST observation 边界，尚无零拷贝设备张量、随机化复位、PPO 学习更新、W256/W1024吞吐或大规模训练接受声明。

## G1 资产与策略

选 Unitree G1，官方 `unitree_rl_mjlab` commit `1425b15f73bd4095f0df53709d7c389c3eb9e790`。官方 ONNX 输入 `[1,154]`、输出 `[1,29]`；动作引用 `dance1_subject2` 共6574帧、50Hz。部署配置有29关节映射、目标缩放/offset、PD gains。权重、NPZ、模型、全部35个网格及原URL/hash在 `artifacts/p65-g1-research-20261109-01/`；它们不进入产品包。

公开复现入口：

```powershell
.venv-humanoid/Scripts/python.exe -B examples/g1_policy_reference.py --assets artifacts/p65-g1-research-20261109-01
```

该例明确是 **MuJoCo 独立参照**，依赖隔离环境中的 MuJoCo3.3.7 / ONNXRuntime1.22.0 / PyYAML6.0.3。策略接收真实状态和动作参考，产生有限力矩，仿真推进后再观测；不发布参考 root trajectory。

完整官方 `scene_g1.xml` 和全部物理网格实际运行 **20秒 / 10000步**：50Hz策略、500Hz物理、1000份真实状态；未触发 root height<.35m 或倾角>1rad 的跌倒检查。根高度 .6692–.7799m；关节跟踪RMSE平均 .1060rad、最大 .1975rad。此参照不是能量/接触力或跨引擎一致性验收。

[打开实际状态回放](../../artifacts/p65-g1-research-20261109-01/g1-dance-reference.html)。回放只从已保存 q/v 做 FK 绘制骨架，无新物理步，无动作参考替换。桌面截图和UI检查保留；播放器/拖动/8秒位置检查通过，无文档横向溢出。

## Phymeo 仍未跳舞：具体失败和下一依赖

[通用 MJCF 导入](p65-floating-mjcf-frames.md)已验证真实29关节资产与框架映射。第一次 Phymeo HOST 扭矩/地面接触尝试保留真实入步速度，只接受 **2步 / .004秒**，第三步因关节越界拒绝；完整请求、入口状态、力矩、轨迹前缀和栈保留在 `phymeo-first-attempt.json`。没有靠改角度限位、冻结腿或覆盖状态追绿。

该 primitive 模型未包含官方执行器的反射惯量等项，官方 `g1_constants.py` 明确按电机/减速比计算 armature。现有设备浮根还限16关节。**下一步是通用 actuator inertia/damping/friction 语义与全 M 动力学对照，再扩17/29关节设备容量、足地持续支撑与策略迁移。** 尚未完成因果隔离，不能断言仅补一个 armature 就能跳舞。

原生胶囊模型在独立 MuJoCo 也于 .428秒失稳，说明同一网络不能忽略 plant/config 差异。另一个早期20秒参照移除了部分物理网格，只保留为受限诊断；完整模型17号任务的20秒结果才用于上面的参照结论。所有原失败保留。

当前完成：K0、训练接口首片、G1资产协议与完整模型策略参照。当前未完成：PPO学习、实测大规模、Phymeo稳定站立/舞蹈、29关节设备支持。各状态独立计，不能相互借用通过。
