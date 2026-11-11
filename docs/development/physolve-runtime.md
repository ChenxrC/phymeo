# PhySolve 原生运行时实验

**资格状态：未通过机器人验收，不可用于消费者部署。** 本批简单模型通过，但
机器人原生对照存在超过既定 1e-6 门限的数值差异。B3 修复两处 TYX 初始化
标记缺陷后，接触点几何回归通过、原 FANUC CUDA 719 在完整矩阵中不再出现，
但 FANUC step 后速度全零，与 CPU 参照不符。未缩小验收范围或放宽门限；详见配套 P2PE
`docs/phymeo-integration.md` 与实验 evidence。本入口保留用于定位，不能据可调用
step/forward/reset 认定整条物理链路正确。

此分支提供 `PhySolvePayload` 与 `PhySolveRuntime`，将 `compile_mujoco` 的完整
MuJoCo 编译快照送入独立 PhySolve Host。Newton、Warp、MuJoCo-Warp 不参与导入或
执行；`physolve_tyx` 保留 MuJoCo-Warp 算法来源告知，其外围内核通过 TYX 编译。
六件求解 CUBIN 和 Host 二进制由显式路径及 SHA-256 指定，不随 PHYMEO 自动下载。

## 依赖和调用

使用 `requirements/physolve-runtime.txt` 专用清单，不混用原 MuJoCo extras。
限定 Windows x64、标准 CPython 3.13.13、MuJoCo 3.8.0、TYX 0.1.0a0、
PhySolve TYX frontend 0.0.0a5 和本批验证的 sm_120 载荷。
`PhySolvePayload(native_path, cubin_directory, native_sha256, cubin_sha256)`
接收一件 Host 和六件 CUBIN 的冻结散列；这不是正式 SDK 安装 manifest 的替代品。

```python
from phymeo import PhySolveRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco

record = compile_mujoco(robot_path, require_asset=False)
with PhySolveRuntime(record, payload=payload, nworld=8) as runtime:
    runtime.step(qfrc_applied=forces)  # float32 可表示的 [8, nv]，N/N·m
    state = runtime.snapshot()       # 显式同步的独立只读 host 快照
    runtime.forward()               # 更新派生结果，不推进 qpos/qvel/time
    runtime.reset(mask)              # bool[8]；不传则全量恢复
```

URDF 不自动生成 actuator：`ctrl` 使用原生 `[N, nu]`，外力使用 `[N, nv]`。
坐标保留 MuJoCo 原生 q/DOF 顺序，自由根四元数 WXYZ。P2PE 的 XYZW、public name、
joint force 和 reaction wrench 契约需要下一阶段显式适配，不能直接暴露原生数组冒充完成。
`xpos/xquat/qfrc_*` 是 solver 最近一次求值结果；step 后不追加 CPU forward。

## 所有权与失败

每个实例独占模型、两个持久 data bank、两个 Host handle 和一条 TYX CUDA stream。
每次 step/forward/reset 从已发布 bank 复制积分状态到候选 bank，验证完成才发布。
只复制 time、qpos/qvel、act、warmstart、控制/外力、equality 和 mocap 积分字段；
派生量由 Host 计算。循环遍历固定字段，不逐 world 调度。
模型只在准备时转换；CPU MuJoCo forward 仅用于初始 data 准备。

当前 API 是同步 direct 验证接口：有明确的 stream completion、容量和有限值 host 检查，
快照包含 D2H，host 控制写入包含 H2D，reset mask 创建临时 device selector。
复制工作随 N×状态宽度增长。这些成本公开保留，不能作为 P2PE 高频 executor 或性能结果。
后续应与 P2PE execution epoch gate 和 device 检查接合，避免第二套事务与同步。
关闭失败保留未释放资源可重试；进入关闭状态后拒绝再次推进。

当前限定 Euler、Newton 约束算法、dense Jacobian、同构 world；不接受插件、flex、
userdata 或原生 sensor。内部条件 Graph 服务约束求解，不代表整步 CUDA Graph 已资格化。
外部 stream、异步/D2D 接口、任意状态写入、传感器、整步 Graph 与 P2PE C ABI 尚未接入。
`require_asset=False` 保留完整原生碰撞/接触数据，不意味着 resident articulation lowering
已支持所有 geom、equality 或 actuator。

## 正式测试入口

`tests/test_physolve_models.py` 比较 Panda/FANUC 全部可枚举原生数组、标量及 option，
并验证 q/v 映射。MJB 不保留 MjSpec signature，测试明确验证其为 0；其他字段不放宽。
`tests/test_physolve_runtime.py` 使用实际 Host/CUDA，覆盖 N=1/8/64、自由落体、外力、
机器人单 bank 原生对照、forward/reset、接触与失败发布/清理重试。

设置 `PHYMEO_PHYSOLVE_SOURCES` 为显式来源 JSON 路径。它包含 `native_host`、`cubins`、
`robot_assets.fanuc`；同级 `evidence/payload-baseline.json` 包含 `native_host.sha256`
及六件 `cubins` 的 name/sha256。无配置时 GPU 用例明确 skip，不算通过。
测试输出和逐字段报告保留在实验根 evidence；与旧 Newton 转换存在的差异必须在 P2PE
场景编译适配阶段处理，不能据相同 nq/nv 认定原始 URDF 与旧转换模型物理等价。

最终安装环境复测使用 fail-fast：43 通过、1 失败、13 项未继续执行；失败为 Panda
首次 forward 的 qacc 对照，最大差 296.59686，容差仍为 atol=1e-6、rtol=0。
已安装 TYX wheel 的相关七模块 101/101 通过，但不足以覆盖下游碰撞/约束组合。
Panda 指尖接触点另有约 0.413 m 几何越界证据，保持失败，不作为舍入误差豁免。

后续 B3 安装环境矩阵：50 通过、9 失败，无 skip。Panda/FANUC N=1/8/64
均实际执行，Panda 的独立 step/forward/masked/full reset 三项通过，CPU/CUDA
指尖接触点对照通过；六项原生对照与三项 FANUC 运动检查仍失败。
FANUC 的同配置 CPU 参照确实运动，不能删除 motion 检查让静止结果通过。
