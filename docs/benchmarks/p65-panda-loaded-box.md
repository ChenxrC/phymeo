# 经典 Panda 抓取非立方载荷

公开例程 `run_panda_cube` 现在接受三个局部全尺寸、质量、名义闭合开度、面内初始偏移及指垫相对物体质心的抓取高度。第一轴沿夹紧方向，第二轴沿另一水平边，第三轴沿竖直方向。均匀长方体的三个主惯量分别为 m(b²+c²)/12、m(a²+c²)/12、m(a²+b²)/12。原 50 mm 立方体仍使用 100 g、29 mm 闭合命令和同一张有限水平桌面。[例程](../../examples/panda_cube.py) · [物理程序](../../src/phymeo/experiments/panda_cube.py) · [原方块对照](p65-panda-rigid-grasp.md)

## 同一轨迹的夹持与滑落

物体长宽高为 46 × 58 × 40 mm，初始位置沿世界 X 偏移 1 mm。指垫摩擦系数固定为 0.65，每边四个材料采样点。120 g 物体的三个主惯量为 (4.964、3.716、5.480)×10⁻⁵ kg·m²。三组手臂的 8×7 个关节目标逐元素完全相同，闭合名义开度和质量按表变化。每组从静止独立推进 3,600 个 0.625 ms 子步。在 1.35 s 量实际净空及位移，用 0.95–1.35 s 整窗检查双指持续承重和台面卸载。[冻结输入和断言](../../tests/test_panda_loaded_boxes.py)

| Phymeo HOST | 120 g，闭合 24 mm | 120 g，闭合 44.8 mm | 950 g，闭合 24 mm |
| --- | ---: | ---: | ---: |
| 1.35 s 物体最低点离台净距 | 58.268 mm | −0.134 mm | −1.071 mm |
| 1.35 s 实际 X 位移 | 77.063 mm | 0.151 mm | −0.135 mm |
| 搬运窗双指较小法向力的最小值 | 5.407 N | 0 N | 0 N |
| 搬运窗桌面峰值法向力 | 0 N | 1.177 N | 10.885 N |
| 最终放置目标误差 | 0.488 mm | 74.849 mm | 75.143 mm |
| 最后 50 ms 平均桌面力 | 1.17720 N | 1.17720 N | 9.31948 N |
| 其余机械臂形状至载荷的最小保守轴向净空 | 3.947 mm | 6.880 mm | 4.724 mm |

三个窗口的关节限位违反量和松指后指垫力均为零，物体步进的动量残差最大为 7.42×10⁻¹⁸ N·s。负净距表示柔顺桌面支撑时的压入。120 g 强夹抓放达到全部原门槛；弱夹和重载都实际发生过双侧夹压，随后由桌面承担物重。[强夹轨迹](../../artifacts/p65-panda-loaded-box-20261104-01/pytest-height/test_asymmetric_box_loaded_and0/polymer-strong.json) · [弱夹轨迹](../../artifacts/p65-panda-loaded-box-20261104-01/pytest-height/test_asymmetric_box_loaded_and1/polymer-weak.json) · [重载轨迹](../../artifacts/p65-panda-loaded-box-20261104-01/pytest-height/test_asymmetric_box_loaded_and2/dense-strong.json)

### 首轮保留的放置反例

把 40 mm 高物体放在原固定桌面，同时直接沿用 50 mm 立方体的绝对指垫高度，得到质心上方 17 mm 的夹取。120 g 强夹时物体实际离台 58.946 mm、横移 77.508 mm，放下时却曾转动超过 8 rad/s。其余机械臂形状的最小轴向净空为 −32.252 mm，末段平均桌面力为 1.20536 N，与 1.1772 N 重力相差 0.02816 N，原容差为 0.025 N。上表三组均按实际质心上方 12 mm 生成同一手臂轨迹；三组采用同一固定桌面、物理子步和原始放置门槛。[首轮完整轨迹](../../artifacts/p65-panda-loaded-box-20261104-01/pytest-box/test_asymmetric_box_loaded_and0/polymer-strong.json) · [首轮退出与断言](../../artifacts/p65-panda-loaded-box-20261104-01/03-box-host-windows/stdout.log)

## Newton 1.7.0 独立受载步进

参考驱动只接受初始化和 8 个控制结点的 JSON。它检查 URDF 文件摘要，直接从 Newton 的 URDF 导入器取得机械臂，添加具有相同六自由度、质量与三轴惯量的长方体，以及同尺寸同姿态的有限桌面。原生模型的九个活动链接受质量均与 Phymeo 相符。为了对应两条独立受力滑轨，导入后将右指的 mimic 关闭，使用相同对称命令；外部反馈控制承担驱动，因此关节的隐式目标力、附加转子惯量和被动阻尼均设为零。初始坐标只赋值一次。此后每步从 Newton 自己的 q、v 重新计算同一套惯性补偿、限幅伺服与目标采样，机械臂和物体继续由 Featherstone 的原生 `step` 推进。[独立参考驱动](../../tests/reference_newton/robot/panda_loaded_box.py) · [纯初态和控制输入](../../artifacts/p65-panda-loaded-box-20261104-01/pytest-height/test_asymmetric_box_loaded_and0/polymer-strong.inputs.json)

两侧完整的原生指垫盒与载荷、有限桌面与载荷共三对形状通过显式名单进入 Newton 碰撞管线。接触点由 Newton 的刚体流形生成。指垫有效法向刚度/阻尼为 1200 N/m、1.2 N·s/m，台面为 2200 N/m、3 N·s/m；摩擦系数分别为 0.65、0.45。Featherstone 使用每对形状材料的平均值；该运行使用有效切向黏性系数 1000 N·s/m、Huber 参数 0.04、零几何 margin 和零角阻尼。Phymeo 的滑速正则化是 0.04 m/s，并使用每指四个材料点与物体八个角点。两种切向本构和接触点选择各有自己的实现。此处 Featherstone 公开提供 `BODY_PARENT_F` 观测，逐接触法向力由实际原生行和该 solver 的弹簧阻尼律重建，物体总非重力合力另外由接受状态的动量变化测得。[本机环境及可用观测](../../artifacts/p65-panda-loaded-box-20261104-01/02-native-capabilities/stdout.log) · [Newton 1.7.0 接触核](https://github.com/newton-physics/newton/blob/v1.7.0/newton/_src/solvers/semi_implicit/kernels_contact.py)

| 相同非立方体，120 g，Newton CPU 各自推进 3,600 步 | 强夹 24 mm | 弱夹 44.8 mm |
| --- | ---: | ---: |
| 1.35 s 净距 / X 位移 | −1.156 m / −3.800 m | 60.644 mm / 75.581 mm |
| 搬运窗双指较小法向力最小值 / 桌面峰值 | 0 / 0 N | 0.226 / 0 N |
| 最终目标误差 | 13.233 m | 20.406 mm |
| 末段桌面法向均值 / 动量给出的非重力竖直合力均值 | 0 / 0.00152 N | 1.23994 / 1.25044 N |
| 其他结构最小保守净空 | −37.518 mm | −20.718 mm |

强夹原生轨迹在 0.544375 s 时物体质心已离开有限桌面的水平范围，之后落下。弱夹在搬运期抬起物体，最终位置和结构净空未达完整放置门槛。两组的 RAW_EXIT 均为 0，含义是原生计算完成了 2.25 s。这个参考设置在两组均未通过与 Phymeo 相同的完整物理验收。Phymeo 的弱夹、重载对照一直留台，不能外推到这一原生接触设置。[Newton 强夹全轨迹](../../artifacts/p65-panda-loaded-box-20261104-01/native-strong.json) · [Newton 弱夹全轨迹](../../artifacts/p65-panda-loaded-box-20261104-01/native-weak.json)

本机 Newton 环境检测到 CPU 与 RTX 4060 Laptop CUDA。该批负载步进用 CPU。MuJoCo 和 mujoco-warp 两个可选包均缺失；实际创建 `SolverMuJoCo(use_mujoco_cpu=True)` 返回缺少后端的 ImportError。Featherstone 同环境的自由刚体在 1 ms 单步读到 v_z = −0.0098100 m/s。[依赖和实际步进检查](../../artifacts/p65-panda-loaded-box-20261104-01/02-native-capabilities/stdout.log)
