# K0 收束后的主线：大规模训练与 G1 舞蹈（2026-11-09）

用户要求立即收束 K0，将大规模训练提高优先级，并选真实人形机器人、找到驱动策略使其跳舞。此计划覆盖旧八模块严格串行顺序：**训练接口与规模能力提升到 K0 后的最高优先级；以 G1 暴露通用框架缺口**。不将单个人形案例当全部引擎验收。

## 已有基础与真实缺口

K0 默认距离/见证已限域验收，见 [报告](../../benchmarks/p65-device-convex-query-acceptance.md)。已有设备宽相、普通/affine 固定根与浮根动力学、选择复位与多世界事务。完整设备接触集合和设备持续摩擦尚未验收。

当前设备 ordinary/floating tree 限制最多 16 个标量关节；G1 常用模型为 29 或 23 个关节。不能裁掉真实活动关节或固定基座后称完整人形验证。HOST 浮根模型无这一 16 关节门，可作资产/策略语义参照，最终训练主线仍为 TYX。

## 调研结论与固定资源

选择 **Unitree G1 29-DoF**。官方 `unitree_rl_mjlab` 同时提供模型、`dance1_subject2` 动作、预训练 ONNX 网络及部署配置，支持动作模仿训练与 MuJoCo 验证；与重新编造舞蹈轨迹相比，已有控制定义可独立核对。

- [官方训练/验证说明](https://github.com/unitreerobotics/unitree_rl_mjlab)：向量任务、PPO、motion tracking、策略导出。
- [官方部署配置](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/1425b15f73bd4095f0df53709d7c389c3eb9e790/deploy/robots/g1/config/policy/mimic/dance1_subject2/params/deploy.yaml)：50 Hz 控制、29 关节排列、目标缩放、PD 参数、观测顺序。
- [BeyondMimic 原实现](https://github.com/HybridRobotics/whole_body_tracking)：动作追踪训练与参考表示。
- [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)、[RSL-RL](https://github.com/leggedrobotics/rsl_rl)：设备批量仿真与训练接口参照，不替换 Phymeo 产品后端。

本机已将官方 commit `1425b15f73bd4095f0df53709d7c389c3eb9e790` 的模型、ONNX/外置权重、NPZ 和 YAML 保存于 `artifacts/p65-g1-research-20261109-01/`，`sources.json` 记录原 URL/hash。仅本机研究；上游权重/动作再分发许可尚未确认，不将其直接打包进产品。

驱动策略：真实仿真观测 → 官方输入顺序与坐标约定 → ONNX 推理 → 缩放后的关节目标 → 有限 PD/力矩 → 物理步 → 下一次观测。动作帧只用于策略参考，不能将参考根轨迹直接写成发布状态。先做独立 MuJoCo 短参照核策略/观测，再做 Phymeo 接入；参照成功不能代替 Phymeo 舞蹈成功。

## 开发与验收节点

| 节点 | 难度 | 交付 | 验收 |
|---|---|---|---|
| T0：向量训练接口 | 中 | batched action/observation/reward/terminated/truncated、显式 seed、选择 reset、最终观测、episode 计数；先复用已有设备 batch | W1/W32 多案例；复位一个世界不改其它世界；独立终止/超时；相同 seed 可重现；拒步不偷偷推进策略历史 |
| H0：G1 资产与策略协议 | 中 | 完整 29 关节、惯量、关节名/轴/限制、根 COM、动作与观测顺序；策略输入/输出检查 | 官方配置映射逐项一致；有限输出/额定力矩；零动作/站立及策略原生短参照；明确 quat 顺序与控制/物理 dt |
| H1：人形容量与接触 | 高 | 将 TYX 关节布局、scratch、FK/M/b/J、接触绑定从固定16扩展为有界声明容量；足地支撑/摩擦 | 17/29关节正控，容量失败与回滚；根与关节反力；站立、屈膝、重心变化，不伪固定根/外加托举 |
| H2：真实舞蹈闭环 | 高 | 官方预训练策略的 Phymeo 适配、逐步真实状态与力回放 | 先站立稳定，再连续舞蹈；记录跌倒、跟踪误差、足滑、穿透、关节/力矩限额；选定窗口完整通过才称成功 |
| T1：训练闭环 | 中高 | 与 PPO 训练器兼容，保存/加载策略、归一化状态、随机化与 resume | 在至少两类小任务上观测到学习而非只推理；同 seed 恢复；策略更新后可独立评估；不要求首批从头训练舞蹈 |
| T2：大规模 | 高 | 从 W32 到 W256，再按显存扩到 W1024/更大；变长接触/选择读回；设备张量互操作按能力明确 | 实测端到端 env-step/s、策略时间、传输量、峰值显存；稀疏/稠密与混合 episode；完整输出和容差相同，不能把 W32 当大规模已完成 |

T0 与 H0 的独立文件可并行作者；本次不另建助手。全部 import/测试/推理/训练/build 严格单槽，父为唯一 owner。H1/H2 所需 K1/K2 接触部分按实际依赖推进，其余新形状/传感/新物理族后置。

## 避免空转与完成定义

每节点交公共入口、可运行命令、多案例/失败测试和实际结果。连续两轮没有缩小失败或增加可调用能力，停止扩写，处理最小阻塞。使用合理工程精度，不要求精确接触/能量恒等，不以放宽测试或运动学播放替代真实物理。

完成应分别报告：K0、向量环境、训练更新、实测规模、G1模型导入、策略原生验证、Phymeo站立、Phymeo舞蹈。各项独立计状态，不能互相借用通过。
