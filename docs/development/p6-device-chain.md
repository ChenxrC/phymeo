# P6 设备链路运行与验收

当前为实现检查点，完整阶段尚未验收。进度以[记录](../progress.md)及[冻结计划](../superpowers/plans/2026-10-28-p6-tyx-chain.md)为准。环境按[uv/pip说明](environment-and-ci.md)安装；tyx固定SHA为`99f6df7c0fd8fdeef77f97df83914da27b64b7ee`。

## 执行边界

公开入口为`TyxRobotParticipant`、`TyxClothParticipant`、`TyxTetrahedralParticipant`、`TyxArticulatedCouplingProgram`和`make_tyx_articulated_runtime`，均可从`phymeo`导入。基础包导入不载入可选tyx运行时；实际执行缺依赖会失败。

显式选择`interpret`、`cpu_cpp`或`cuda`；没有设备失败后的CPU替代。模型、状态、输入与调度沿用原合同，`storage_dtype=float64`，`compute_dtype=float32`，`host_validation=float64`。FK/J/M/bias、StVK、节点预测与重建、接触几何、响应与载荷更新使用tyx公开kernel；主机负责静态构造、串行调度、阈值验收和记录。设备结果只扩宽存储，不由主机重新计算替换。

本轮串行适配每次kernel分配、同步、上传/下载，尚未优化。计数是数组操作次数，不是字节数，详见[PERF-007](../performance-bottlenecks/PERF-007-p6-device-boundaries.md)。极端尺度的FP32材料有明确拒绝边界，详见[NUM-003](../legacy-issues/NUM-003-p6-fp32-material-scale.md)；tyx语法缺口见[TYX-002](../../tyx_flaw/TYX-002-chained-comparison.md)。

## 可复现命令

在已安装环境中执行，下列目录必须不存在。Windows若默认临时目录不允许编译缓存写入，可将`TEMP`和`TMP`显式设到工作区内`artifacts/tmp`。

```powershell
python -m pytest tests/test_tyx_math.py tests/test_tyx_robot_math.py tests/test_tyx_node_math.py tests/test_tyx_participants.py tests/test_tyx_articulated.py -q --require-tyx --tyx-mode cuda
python examples/articulated_device.py --scene S2 --mode cuda --frames 200 --output artifacts/p6/device/S2
python examples/articulated_device.py --scene S2 --mode cuda --frames 200 --output artifacts/p6/repeat/S2
python examples/articulated_device.py --scene S2 --mode reference --frames 200 --output artifacts/p6/reference/S2
```

对S3、S4重复后三条命令，再执行：

```powershell
python scripts/verify_articulated_device.py --device-root artifacts/p6/device --repeat-root artifacts/p6/repeat --reference-root artifacts/p6/reference --output artifacts/p6/verification.json
python scripts/measure_articulated_device.py --input artifacts/p6/device --output artifacts/p6/performance.json
```

`reference`是显式NumPy FP64独立对照，使用同P6设置，不计入设备证据。CI只在基础环境运行短reference场景与消费者测试；本机真实CUDA验收另记，远程CPU CI成功不代表GPU通过。`uv run --no-sync python`可替换以上`python`；原生pip环境使用该环境自身解释器。

## 冻结门槛和限制

S2/S3/S4均200帧、每帧两子步、h=1/4000s、位置/速度各最多200轮，P6容差2e-6。独立审计穿透≤5e-6m、闭合速度≤5e-5m/s、关节动量平衡≤5e-5kg·m/s、配对载荷≤5e-8尺度；固定点必须精确不变，材料域维持原范围。旧P5的FP64门槛不变。

CUDA两遍所有保存数值数组完全相同；同设置FP64对照检查每帧及每个子步，q≤5e-4m、节点位置欧氏误差≤1e-3m、速度≤2e-2m/s，每个节点域至少5个实际接触子步。独立消费校验配置、模式、来源、非零成功launch、全部子步和载荷，不信任自报摘要。该机制针对可信生成器中的错误，不是针对恶意重造全套轨迹的密码学证明。

性能脚本必须单独串行运行：恢复完整轨迹第100帧，每次执行完整step，先一次首调用、2次预热，再3×20计时调用，保存两个子步与下一帧的精确一致性、审计质量、计数和失败原因。独立审计在计时区间外；生产校验和诊断包含在step时间内。没有等价Warp测量，不声明差距或加速比。

P6仅迁移已有机器人/节点与球面无摩擦接触合同。双臂、多指夹持、摩擦、平台接触和递布验收由P6.5另行落实，不能用本阶段短实验替代。
