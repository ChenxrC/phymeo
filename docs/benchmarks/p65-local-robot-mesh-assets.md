# 本地机器人闭三角网格资产

普通 `load_urdf(path)` 与 `load_mjcf(path)` 可以把本地三角 OBJ 注册为有材料面和连杆归属的闭实体。孔槽和凹角随原关节 FK 运动。两种描述使用同一个 4 面凸四面体和 20 面 L 形棱柱，后者有 12 个原始位置 ID。[URDF](../../tests/reference_newton/robot/mesh_assets/mesh_arm.urdf) · [MJCF](../../tests/reference_newton/robot/mesh_assets/mesh_arm.xml) · [L 形实体](../../tests/reference_newton/robot/mesh_assets/models/bracket.obj)

## 可加载范围

- URDF 的 `mesh filename` 取 XML 目录下的相对路径，默认尺度 (1,1,1)。MJCF 的 `asset/mesh` 以作者指定名称或文件 stem 注册，`geom mesh` 在完成碰撞掩码筛选后取数据。`compiler meshdir` 可选，其相对路径以 XML 所在目录为根。文件须落在相应资产目录内。绝对路径、越界路径、包或网络 URI、其他文件格式和不支持的语义属性会给出明确错误。仅用于显示的 URDF visual 或 MJCF 双零掩码 geom 可以引用不供物理使用的资源。[文件读取和命名解析](../../src/phymeo/builders/_robot_mesh.py)
- 支持 OBJ 的有限三维位置、已写出的正或负位置索引、三角面及常见的纹理/法线角点索引。位置 ID 决定材料拓扑；显示分组、材质和光滑度记录不会触发额外文件读取。注册时复用普通碰撞系统的闭壳校验，拒绝开边、反向或重复面、塌缩、折叠和自相交等输入。当前普通注册上限为每个网格 128 个三角面，并沿用碰撞模型既有的 8,192 个顶点总预算。错误包含原闭壳原因及碰撞体所属 link 和名称。[原生闭壳校验](../../src/phymeo/geometry/_closed_mesh.py) · [独立正负控](../../tests/test_robot_mesh_assets.py)
- 三个尺度分量均须有限且非零。奇数个负分量造成镜像时，逐面反转绕序，保持外向法线。MJCF 的 mesh 参考帧先从源顶点减去 `refpos`，施加 WXYZ `refquat` 的逆旋转，再作逐轴尺度；随后施加 geom 的局部姿态和所属 link 的真实 FK。`geom size` 在 mesh 实例上不改变尺寸。显式原语拟合、内联 mesh 和网格简化属性均有独立的域边界。[MuJoCo 网格定义](https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-mesh) · [Newton 1.7 MJCF 导入器](https://github.com/newton-physics/newton/blob/v1.7.0/newton/_src/utils/import_mjcf.py) · [Newton 1.7 URDF 导入器](https://github.com/newton-physics/newton/blob/v1.7.0/newton/_src/utils/import_urdf.py)

## 同一资产的实际几何

固定基座的 marker 使用 (0.125,0.25,0.5) 尺度。活动 bracket 的尺度为 (0.125,0.25,0.25)，局部绕 Z 旋转 90°，位于 (0.125,−0.25,0.0625) m，再随绕 Y 运动的关节变换。两种描述保持活动体 2 kg 的显式质量与同一惯量。Phymeo 的固定根键均为 (0,0,0)；活动 URDF 碰撞键为 (0,1,0)，MJCF 为 (0,1,1)，后者的前一个元素是仅供显示的 geom。

| 球半径均为 15.625 mm | Phymeo 普通闭实体查询 | Newton 1.7 实际导入网格的有符号球心距离 | Newton 原生刚性接触行 |
| --- | --- | --- | --- |
| L 凹槽内的空区 | separated，球皮净空 31.25 mm | +46.875 mm | 0 |
| L 实体腿内的球 | mesh_contains_round，球皮距边界 46.875 mm | −62.5 mm | 0 |
| 材料边界切入 | boundary_overlap，穿入 7.8125 mm | +7.8125 mm | 2 |

表内对照来自 URDF/MJCF、q=0 和 q=atan2(3,4) rad、Newton CPU 与实际 CUDA 的 8 个组合。Newton 根形状/刚体 ID 为 0/0，活动形状/刚体 ID 为 2/1。使用设备 FK 和 finalize 后的真实 `shape_source_ptr` 逐面读取顶点，两个格式都保留根 4 面和活动 20 面；活动顶点与解析世界位置最大偏差为 0.102 µm。球心距离通过对应设备上的有符号三角网格查询取得。接触行另由 Newton CollisionPipeline 发出。内含球的深度由签名网格距离辨识；它既未穿过网格表面，也没有产生接触行。[CPU 原始读数](../../artifacts/p65-night-robot-obj-assets-20261104-01/05-native-cpu-witness/stdout.log) · [CUDA 原始读数](../../artifacts/p65-night-robot-obj-assets-20261104-01/06-native-cuda-witness/stdout.log)

### 实际关节运动读回

固定世界球心 (0.25,0.0625,0.09375) m，向已加载的铰链施加 29/64 N·m 的恒转矩。在原一阶半隐式积分中，该惯性的解析关系为 qₖ=k(k+1)/1024 rad。原质量及关节状态逐步积分，10 个已接受步均通过对应角度的 FK–网格查询；第 8 步开始切入固定球，第 10 步 q=0.107421875 rad、间隙 −12.43976 mm。这是驱动与碰撞观察试片，球接触力没有反馈到这次关节动力学。HOST 上新专项 39 项通过；相邻原球、盒与曲形注册的 4 项通过。真实 TYX CUDA 在非轴角姿态运行 1 次 FK kernel、4 次碰撞候选 kernel，再由精确闭网格几何给出凹槽与切面分类。[实际受力步](../../artifacts/p65-night-robot-obj-assets-20261104-01/11-host-final/stdout.log) · [TYX CUDA](../../artifacts/p65-night-robot-obj-assets-20261104-01/12-tyx-cuda-fk-candidates/stdout.log)

### 原生接触锚点的 0.100 mm 偏差

在 8 个组合中，切面球的几何球皮间隙为 −7.8125 mm，Newton 的实际接触距离为 −7.91246 至 −7.91249 mm。原定的 0.060 mm 对照阈值全部失败，首次严格断言已保留。以 q=0 的 CPU 为例，导入网格的 margin 为零，接触行中的网格 margin 亦为零，球 margin 为 15.625 mm。两个发布锚点的法向投影差为 7.712543 mm，减去球 margin 恰得到 −7.912457 mm。切面球的输入中心为 Y=132.8125 mm，而公开接触球侧锚点为 Y≈132.7125 mm，网格侧锚点 Y≈125.0000 mm。该差额位于发布的锚点，尚未用单因素实验隔离 Newton 内部其余构造步骤。[原阈值失败](../../artifacts/p65-night-robot-obj-assets-20261104-01/03-native-newton-cpu/stderr.log) · [含锚点的参考程序](../../tests/reference_newton/robot/mesh_assets/native_readback.py)

## 复现

```powershell
python -m pytest -q -m "not tyx" tests/test_robot_mesh_assets.py
python -m pytest -q --tyx-mode=cuda --require-tyx tests/test_robot_mesh_assets.py::test_mesh_from_device_fk_uses_real_registered_candidates
python tests/reference_newton/robot/mesh_assets/native_readback.py --device cpu
python tests/reference_newton/robot/mesh_assets/native_readback.py --device cuda:0
```

前两项使用项目 Python，后两项使用 Newton 1.7.0 / Warp 1.17.0 的独立环境。原生程序分别保存真实数据和原 60 µm 是否达标的布尔值，可独立检查该未对齐项。
