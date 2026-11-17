# I3r：共享 DOF 摩擦的独立 CUDA 参考

2026-11-18，codex/exp 源码57840124；首次完整6节点（5 CUDA＋1 HOST mock资源）通过，4.22s。最终Ruff/format通过；只有导入排序和排版修正，无数值失败、参数或误差门调整。它推进了[HOST共享DOF参考](area-pressure-coupled-friction-report-20261118.md)，还没有接入生产面积Scene。

## 已验证的实际链路

独立 `TyxCoupledFrictionWorkspace.solve_many` 接收整批owned原FP64问题，完整准入后转换并保存实际FP32输入。设备计算M+hD、自由速度、最终法向和平滑切向力、完整非对称Newton Jacobian和部分选主元消元。固定24更新×12回溯，保存最多290条尝试/最终记录。HOST只对实际设备输出作FP64代回验证，不求更精确速度、不替换设备力。

全批成功后才替换immutable `last_results`。末输入非法在上传前拒绝；晚world数值预算失败保留上一批并可全量重试；晚结果封装失败不发布前缀；CUDA上传/运行/下载异常使实例faulted。释放沿用DeviceResources，包括UNKNOWN_RELEASE；该故障仅mock，不破坏真实设备。

## 数值范围与结果

FP32设备门为 `||A^-1/2 R|| <= 1e-5(1+||vfree||A)`；独立实际上传值/实际输出代回门为2e-5倍同尺度。与原FP64参考速度的门为每分量5e-5(1+|vref|)，力为.002N+.002relative，矩为1e-5Nm+.002relative。I3q的1e-10门另列，没有将两者混为一谈。

| 固定输入 | 出步速度 | 实际上传值的独立质量残差 | 相对原参考最大速度分量误差 |
| --- | --- | ---: | ---: |
| 支撑滑动 | (0.016006995, 0) | 5.57e-7 | 7.87e-7 |
| 卸载分离 | (0.02, 0.300000042) | 2.11e-8 | 4.17e-8 |
| 法切共用一个DOF | 0.0239073355 | 3.15e-7 | 4.44e-7 |
| 两点已知解 | (0.014996974, -0.001999998, 0.300037622) | 2.28e-6 | 3.76e-5 |
| 两点无摩擦 | (0, -0.090290122, 0.174612448) | 4.62e-9 | 1.55e-8 |
| 无接触 | (0.02, -0.0080000004) | 0 | 4.47e-10 |

上述非空案例均未满足I3q严格HOST门，仍满足本次预先声明的FP32门。无接触在实际FP32输入上的残差为零，但与原FP64输入仍存在表示差异。所有这些结果都原样保存，没有为追求更小误差再跑或补迭代。

六例逐个W1与反序W6的原始数值数组逐字节一致。制造例独立完整Jacobian最大误差3.11e-8，保持明显非对称。单独非对角质量例实际发生一次行交换，不能用Cholesky代替Newton。W32/nv6/ns8把原两点无摩擦各四分同位置刚度，保持前三DOF物理响应，额外DOF零接触；这验证容量与隔离，不冒新的32种复杂机器人场景。

该W32单次记录：802944B注册设备buffer、54656B上传、748288B完整诊断下载；整调用约.403s，其中设备kernel+sync约.127s、HOST核验约.211s。含首次该形状编译/诊断，不是暖性能比较，也不声明快于HOST或Newton。设备context/JIT/allocator不在注册buffer字节内。

## 证据与下一步

- [独立workspace](../../experiments/area_pressure/tyx_coupled_friction.py)、[kernel](../../experiments/area_pressure/_tyx_coupled_friction_kernels.py)、[独立测试](../../tests/test_area_pressure_tyx_coupled_friction.py)。
- `artifacts/area-pressure-tyx-coupled-friction-20261118-01/validation-summary.json` 保存命令、原退出、最终源前后相等与12个同次JSON清单/hash。全部raw原/packed输入、FP32状态/载荷/trace、独立误差及失败诊断均保留。
- 01首Ruff1（两处imports）、02首format1；03导入排序0、04排版0、05首次完整6pass/0、06/07最终静态0。未重跑已绿组。

下一接线需明确新显式接触律、生产实际M/J/外力、同一步q积分、完整载荷观察与版本恢复。当前实验不更新q、没有碰撞检测、只允许冻结J线性gap，也不支持一般输入必收敛或精确静摩擦。现type7和资产friction=none不变，无法据本报告声称机器人已用面积摩擦抓持。
