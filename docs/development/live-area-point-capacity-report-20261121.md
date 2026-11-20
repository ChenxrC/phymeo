# 正式 CUDA 步进中的面积接触点容量

2026-11-21，在 `codex/exp` 验证。

`MuJoCoTyxModelSpec.constraint_capacity` 现在决定每个世界可容纳的面积点与原生标量约束行的总数，声明范围为 0 至 512。正式面积摩擦路径原来另有 128 点限制，使声明 256 的场景也无法使用第 129 点。入口现在按模型容量一次分配面积点、双方点雅可比和逐点输出。同一个求解步使用全部有效点。零容量保留一个物理占位，但有效点数仍须为零。收据与检查点恢复使用同一个总量检查。[设备分配](../../src/phymeo/backends/mujoco_tyx/_allocation_manifest.py) · [设备求解入口](../../src/phymeo/backends/mujoco_tyx/_area_friction_device.py) · [恢复时的校验](../../src/phymeo/backends/mujoco_area_friction_state.py)

## 三组实验

| 入口 | 输入与观测 | 实测结果 |
| --- | --- | --- |
| 正式异构 `Scene.step`，由当前状态自动生成接触面 | 两个可动滑块的同一块受压三角面，粗网格产生 6 点，细网格产生 216 点。独立计算逐点载荷、合力、力矩和两侧速度 | 两种网格均与解析解一致。设备观察读回了最后一个点的受力。恢复序列化检查点后，下一步的状态和观察逐值复现。另一个容量 215 的场景拒绝 216 点，先前的受载帧仍可读并可继续步进 |
| 正式异构 `batch.step_area_pressure`，提供接触面 | 将同一块侧面分成 81 个互不重叠的三角形，得到 243 个受摩擦面积点。足底另生成 4 条 MuJoCo 原生地面约束行 | 容量 247 时，侧面法向合力 9.396905 N、切向合力大小 4.697000 N、地面竖直支撑 34.478206 N。全部 243 点均有正载荷和非零切向力。两关节出步速度、原生行力、逐点力、动量残差及功率通过独立双精度方程核验。容量改为 246 时，上一帧与检查点完整保留，重新提交较小接触面可继续 |
| 原始八自由度 RR Push-T 的 CUDA 入口打包 | 取原记录 1.255 s 的推力峰值姿态和 1.767 s 的撤离姿态，重复使用同一组 256 槽缓冲 | 前者 237 个真实面积点的双方雅可比逐点通过独立 `mj_jac` 对照。后者有效点数为零，即使输入保留旧点字节，输出点与两侧雅可比仍全部清零 |

三组分别覆盖自动几何、带原生地面响应的混合动力学，以及真实 RR 输入的设备传递。RR 的 237 来自原记录峰值姿态，先前实验连续轨迹中最多有 238 个点。512 是可声明的容量上限，本次动力学实测达到 243 个面积点加 4 条原生行。[自动几何案例](../../tests/test_live_area_point_capacity.py) · [混合动力学案例](../../tests/test_damped_coupled_mixed_area_friction.py) · [RR 两时刻对照](../../tests/test_pusht_resident_area_pack.py)

## 复现与范围

在仓库根目录使用原 CUDA 环境，设置 `PYTHONPATH` 为仓库根目录与 `src`，依次选择以下 pytest 文件或节点。该批共 14 项不同用例通过，13 个变更 Python 文件的 Ruff 检查和格式检查通过。每次命令、标准输出及原始退出码保存在本机 `artifacts/live-area-capacity-20261121-01`。混合步的数值记录见 `06-dense-mixed-tmp/test_dense_side_friction_share0/dense-mixed-load.json.gz`。

| pytest 选择 | 用例数 |
| --- | ---: |
| `tests/test_tyx_area_friction_entry_pack.py` | 4 |
| `tests/test_pusht_resident_area_pack.py` | 1 |
| `tests/test_live_area_point_capacity.py` | 2 |
| `tests/test_damped_coupled_mixed_area_friction.py::test_dense_side_friction_shares_capacity_with_live_ground_rows` | 2 |
| `tests/test_mujoco_area_friction_wide.py::test_current_wide_scene_and_checkpoint_replay[8]` | 1 |
| `tests/test_damped_coupled_mixed_area_friction.py::test_coupled_native_and_area_capacity_failure_preserves_both_dofs_and_retry` | 1 |
| `tests/test_damped_coupled_mixed_area_friction.py::test_coupled_mixed_route_preserves_absent_ground_and_empty_area_limits` | 3 |

原始 RR 场景的整段推动与撤离仍由实验入口完成。正式路径下一批应让八自由度状态和同一步生成的原生接触、限位行共同进入已有设备工作集，并将推杆与 T 块的当前几何积分点绑定到该状态。先用已保存姿态逐行核对面积力、桌面反力、关节力矩与出步速度，再用同一控制窗口对照物体轨迹和分离时刻。[原始模型的五姿态与连续基线](rr-pusht-entry-working-set-report-20261120.md)
