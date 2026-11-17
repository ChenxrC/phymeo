# 面积接触：观察与生命周期结果

2026-11-17，`codex/exp`。实现提交 `891669825e5ad79aff2580a3daa93ebd6ac10a0b`。dev 未改，未 push。

本批让已经接通的面积压力动力学能够保存、恢复和克隆，并提供相位明确的只读接触观察。它解决的是复现实验和批量环境的生命周期，不改变接触力方程，也没有把 HOST 裁剪迁移到 GPU。

## 新增能力

| 接口 | 行为 |
| --- | --- |
| `runtime.area_pressure_snapshot(keys=None)` / `scene.observe(keys=None)` | 返回不可修改的 HOST 快照，不启动设备 kernel、不重新裁剪、不做 CPU 物理步 |
| `runtime.checkpoint/restore/clone` | 面积接触可以复用原有精确设备状态事务，保存原 quadrature、实际 FP32 行、完整载荷与历史 |
| `scene.checkpoint()` | 在 runtime 胶囊外绑定完整本地压力资产、刚面、连接关系、材料身份、压力和 full 标志 |
| `scene.restore(cp, expected_revision=...)` | 先检查资产与历史一致性，再完成替代设备验证与原子提交；成功后清除旧几何缓存 |
| `scene.clone(world_keys=...)` | 新设备 owner；支持乱序和子集世界，历史身份保留，下一步重新生成当前接触面 |
| `scene.reset(keys=...)` | 仅重置选定世界的状态与历史；空选择是 no-op |

观察同时保留输入相位的 q/v、接触力和输出端点 q/v。`applied_control` 是上一成功步实际使用的存储后外载，`current_control` 是下一步将使用的当前控制，两者可以不同。力矩分别提供共同世界原点、输入机体原点及输入 COM 三种参考点；不能将不同参考点的力矩直接相加比较。

原始 quadrature 与实际 FP32 转换、raw/typed 约束行、双方力和力矩、共同运动相互绑定。恢复前检查原 FP32 准入、样本身份、相位、force/motion 定律及全部广义约束载荷。旧 `configuration_id` 保留为历史来源；新步使用目标运行时的新身份。原 native accepted-entry ABI 不扩充为虚假的面积设备记录，area 胶囊也不接受缺少该完整设备入口的旧格式。

## 实际验证

所有项目命令由 root 唯一串行执行；两原 agent 仅纯编码或源代码审阅。现有环境，无安装、远程 CI 或旧长期窗口重跑。

| 组 | 结果 | pytest 用时 |
| --- | --- | --- |
| 新生命周期 interpret | 6 passed | 19.10 s |
| 新生命周期 CUDA | 6 passed | 10.69 s |
| 同 MJCF、不同面积 pair 顺序拒绝 | 2 passed（两模式） | 2.77 s |
| 受新 schema/checkpoint 接口影响的面积接触 | 31 passed | 7.40 s |
| 原生精确 checkpoint 连续性：四模型×两模式 | 8 passed | 18.52 s |

共 **14 新节点 + 39 受影响/兼容节点**通过。新生命周期前两组按 mode 分开，因此各自 deselect 另一模式；合并覆盖全部声明节点，无 skip。最后增加 pair 顺序负控仅执行新增两节点，未重跑已绿组。最终七个 Python 文件 Ruff 与 format check 均通过。

关键结果：

- 双面积持续支撑在第 30 步保存，继续到第 80 步；恢复后再走同样 50 步，全部物理数组逐字节一致。重新运行产生的 configuration/revision 身份单独比较；恢复瞬间原历史 stamp 必须完全保留。
- 双环境的乱序 clone、单环境子集 clone、独立继续和源 owner 关闭后继续通过。重置一个环境不清除另一个环境的载荷历史。
- 控制更新后，历史 applied load 不变而 current control 更新；观察没有增加 kernel、同步或传输计数。旧快照及其数组保持不可写。
- 改变压力资产、调换同模型 pair 顺序、重算 checksum 后篡改 force/quadrature/stamp/raw row/phase，或删除 accepted-entry，全部拒绝；没有替换已接受状态。
- 最后设备 readback 验证后的故障注入保持状态、revision、receipt、checkpoint bytes 和 scene 几何缓存；之后合法恢复及继续成功。
- 偏心 COM 的真实 hinge/slider 步验证三种力矩参考点，世界原点处双方力和力矩相消。

原始证据：[本批目录](../../artifacts/area-pressure-lifecycle-20261117-01)。保存两模式第30步胶囊及31–80步 q/v/广义接触力数组；逐步完整对象一致性断言在同次测试内执行。保存源前后哈希与最终源绑定。首次 Ruff 的 import 排序失败及修复保留；一次 PowerShell `-p` 参数绑定错误发生在 Python 启动前，随后使用显式参数数组。没有产品测试失败，也未调物理参数或放宽原误差门槛。

## 使用

```python
from phymeo.backends.mujoco_area_pressure_state import MuJoCoTyxAreaSceneCheckpoint

frame = scene.observe()[world_key]
checkpoint_bytes = scene.checkpoint().to_bytes()
# scene 在别处继续运行后，恢复到上述存档：
checkpoint = MuJoCoTyxAreaSceneCheckpoint.from_bytes(checkpoint_bytes)
scene.restore(checkpoint, expected_revision=scene.runtime.checkpoint_revision)
scene.step(h, expected_revision=scene.runtime.checkpoint_revision)

branch = scene.clone(world_keys=(world_key,))
try:
    branch.step(h, expected_revision=branch.runtime.checkpoint_revision)
finally:
    branch.runtime.close()
```

胶囊沿用有大小界与 checksum 的可信应用存档语义，不是身份认证格式；scene 存档绑定资产但不打包资产文件。恢复到另一 scene 时需构造相同资产。策略、RNG、任务目标仍由各自环境层保存，不在本胶囊中。

## 下一批

I3b 优先把面积载荷接到持久设备观察：固定布局及双方身份、输入/输出 tick、有效性、容量和 reset/restore/clone 生命周期，在同一 accepted bank 提供批量读取，验证 HOST 对照与跨环境隔离。之后再单独设备化接触面生成并测整步成本。

当前仍是 full 压力体与有限刚面、弹性无摩擦法向律、HOST 几何 + TYX 动力学。没有新增设备面积 ABI、GPU 裁剪、面积摩擦或 FFT 非局部材料；这些不因保存恢复通过而被宣称完成。后续继续按[检测与动力学架构](contact-model-architecture-20261117.md)推进。