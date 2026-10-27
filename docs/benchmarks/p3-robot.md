# P3 机器人 CPU 参考验收

2026-10-27，冻结实现 `1b0d1c7d1b5200a3d92458f9075d3d38230102ba`，
包源码 SHA256 `e46b55d5aa80353b32cd777c67219fc404ac8aab58c6fb4847ac94596acdf294`。
P3 的实现、独立数值、完整轨迹和本地工程检查已完成；P4/P5、整分支独立审阅及最终 CI 收口继续推进。

## 实现与边界

- 固定基座、拓扑有序树、每个连杆一个旋转/移动关节；所有权完整的 FP64 q/v、输入及局部状态族。
- 世界 FK/点雅可比、COM 质量二次型、世界惯性载荷与重力/科氏偏置、外载荷投影及稠密有门禁求解。
- RobotPointPort 保留共享祖先和重复点的完整响应；冲量不推进 q。统一 SceneStepper 掌握子步与全帧事务，末子步/资源退出失败恢复原 q/v。
- NumPy CPU 参考；不宣称优化 CRBA/ABA。不支持浮动基座、闭环、关节限位、接触驱动或新机器人 GPU 算法；P5 添加接触，P6 处理新设备链路。

## 独立物理证据

单杆和两杆解析质量/重力/科氏项、移动轴极坐标方程、几何差分能量、功率对偶、共享祖先/重复点交换均测试通过。
机器人新增测试共 102 项，Python 3.11 和原生 pip 3.12 分别通过。

固定 MuJoCo 3.3.7 仅在独立环境调用公开 API；关闭接触/阻尼/armature/限位，明确质量、惯量和世界 COM 载荷。
XML 编译顺序与本项目拓扑顺序通过关节名称映射，不假定深度优先数组相同。
[机器可读记录](p3-mujoco.json)包含版本、脚本/锁/源码哈希、种子、阈值和逐类最大误差。

| 机构 | 样本数 | 质量矩阵最大绝对差 | 偏置最大绝对差 | 加速度最大绝对差 |
| --- | --- | --- | --- | --- |
| 单杆 | 32 | 4.17e-17 | 4.45e-16 | 7.11e-15 |
| 两杆 | 32 | 3.33e-16 | 1.78e-15 | 4.97e-14 |
| 混合旋转/移动分支树 | 32 | 3.55e-15 | 8.89e-15 | 2.14e-14 |

全部 FK/COM/旋转、M、bias、qacc 满足 atol=5e-10、rtol=1e-9。
S1 初态 q=(0.35,-0.2)、v=(0.1,-0.05)，零输入，h=1/2000 s，共 1000 步。
与同积分 MuJoCo 的最大 q/v 差为 9.71445e-17 / 8.88178e-16，低于 1e-7；
本实现重复运行的 q/v/能量/时间数组逐元素一致。轨迹 SHA256：
`415f77b6510b76ecc55cbfda2f119296b7c474911e66ce79afa6b7cff7bf6809`。

完整本地数据在 `artifacts/p3-oracle-1b0d1c7/`：XML、96 个输入/输出样本、外部和本实现完整 q/v、S1 数值运行记录。
仓库保存小型摘要；复现命令如下，输出目录必须不存在：

```powershell
.venv-mujoco/Scripts/python.exe scripts/compare_robot_mujoco.py --output artifacts/p3-oracle-new
```

安装步骤见[隔离环境](../development/environment-and-ci.md)。以上是同模型数值一致性，不能据此宣称任意时间步的连续物理误差小，也不代表 MuJoCo 功能对等。

## 工程和性能

完整实际 CUDA 回归 `pytest -q --require-tyx --tyx-mode cuda`：933 passed，260.65 s，无跳过（测试耗时，不是性能测量）。
新 wheel 的 70 个 Python 文件逐字节匹配源码；源码外新原生 pip 环境的原场景和机器人解析 smoke 通过，未导入 MuJoCo/tyx/Torch/Warp。
ruff、文档、公开依赖边界及编译检查通过。核心依赖锁保持不变；新增独立 oracle 锁和 CI 作业。

冻结源码后结束测试，再依次串行执行完整对照和[1/2/8 自由度性能](../performance-bottlenecks/measurements/p3-robot.md)。
没有进行性能优化；重复计算的候选开销见 [PERF-004](../performance-bottlenecks/PERF-004-robot-dense-reference.md)。
原 tyx 路径回归通过，本阶段未发现可独立定位的新 tyx 缺陷；未做 Warp 对照。

冻结提交的 [CI](https://github.com/ChenxrC/phymeo/actions/runs/40509615510)核验时仍在运行，
[Documentation](https://github.com/ChenxrC/phymeo/actions/runs/40509615496)已成功。最终交付必须核验最终 SHA 的全部检查。
