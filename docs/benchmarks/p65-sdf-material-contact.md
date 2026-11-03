# 已登记刚体距离场与柔性材料的离散查询

`sample_sdf_contacts` 观察一个刚体场在给定姿态下与粒子、带半径粒子、有限零厚三角面的关系。它在同一批输入中按世界、碰撞掩码和排除表筛选材料，保留每个形状及原材料节点的 ID。下例可在仿真步的当前状态读取场观测。[公共接口](../../src/phymeo/geometry/sdf_material.py)

```python
field = ph.SdfFieldPose.bake(
    rigid_mesh, vertices_local_m, shape_xyz=(33, 33, 17),
    origin_m=(-2.5, -2.5, -1), spacing_m=(0.15625, 0.15625, 0.125),
)
with ph.TyxSdfSampler(field.grid, mode="cuda", capacity=1024) as sampler:
    pose = field.at_pose(translation_m=body_position_m, orientation_xyzw=body_quaternion)
    batch = ph.sample_sdf_contacts(
        model, pose, {cloth_face.key: cloth_face_world_m, bead.key: bead_center_world_m},
        rigid_vertices_world_m=rigid_mesh_world_m, sampler=sampler,
    )
```

每张三角面的输入顶点次序沿用其注册的 `vertex_ids`。每个样本同时保存局部及世界位置、材料重心权重、所在格、场值、半径修正后的间隙，以及可得的局部/世界梯度。梯度非零时，`projected_surface_estimate_world_m` 给出沿梯度的一阶零值面估计。刚体实际三角面号和材料权重由闭体几何查询提供。输入是单时刻顶点，时间轨迹需要另行查询。[闭体与整布面](p65-closed-mesh-cloth-triangle.md) · [规则格及精度界](p65-signed-distance-grid.md)

### 整面判据

HOST 在材料重心坐标中把三角面与实际相交的格单元裁剪，处理公用格界、点及线段。每个单元的三线性多项式限制到三角域后总次数至多为三。十个三次 Bernstein 系数的最小值给出整个小三角域的下界，必要时继续细分。已访问的真实材料点给出最小间隙的上界。一个材料点间隙不大于零即可确认场接触。只有所需区域均在有效格内、完整覆盖域的下界严格大于零时，整面才为 `clear`。当覆盖不全时，上界仍可确认局部命中，下界留空。[单元裁剪与完整界](../../src/phymeo/geometry/_sdf_material_triangle.py)

| 构型 | 独立整面真值 | 批查询结果 |
| --- | --- | --- |
| 单元内的斜面，三条边场值均至少为 +1/16 m | 面重心 −1/48 m | `contact`，3 个原材料权重均为 1/3 |
| 同一斜面，将常数项换为 13/32 m | 整面最小值 +7/96 m | 细分后 `clear`，关闭细分时为 `field_precision` |
| 仅边中点接触，其余点场值严格为正 | 边权重 (1/2, 1/2)，间隙 0 | `contact`，另一个共址世界被筛除 |
| 各向异距格中的分片线性场 | 四张面最小值为 0、3/8、−29/280、−29/280 m | 前两项上下界相等，镜像后两项均由相邻两个 FP64 数包围 |
| 三角面部分超出场的定义域 | 完整面没有可用的下界 | 已知区域有负值时 `contact`，只有正样本时 `unresolved` |

分片线性留出的独立判据枚举三角域边界与正交象限分界线的交点，直接求分片仿射函数的全域最小值。镜像两例的轴对齐包围盒含 160 / 140 个格，实际覆盖 25 / 24 个格，各采样 159 个材料地址。每批默认最多检查 4,096 个候选单元、评估 16,384 个材料点，细分深度为 2。容量耗尽、窄带外、域外和精度不足各有独立原因。[解析几何与资源验收](../../tests/collision/test_sdf_material.py)

### 场与源实体

普通 `SdfFieldPose(key, grid)` 允许导入任意外部场，`source_geometry_status` 为 `unbound`。`SdfFieldPose.bake` 保存烘焙时的闭网格注册、材料顺序及原局部顶点。给出当前刚体顶点后，查询逐一核对注册与当前姿态下的顶点。核验通过才把烘焙标量误差及姿态舍入界计入源几何间隙，缺失或不匹配时返回 `pose_unverified`。整个流程保留独立的 `field_status` 与 `source_geometry_status`。

复用原 Newton 1.7 原生纹理的三种资产和 16 个自由点，将每一点登记为带独立 ID 的材料。凹 U 槽心的场距离为 +1 m，源下界 +0.566987 m，两壁内为 −0.5 m，源上界 −0.066987 m。薄壁中心的粗场为 +0.109375 m，源界 [−0.1071314, +0.3258814] m，因此源实体关系未决。同点的 Newton 纹理曾实测 −0.0062540 m，解析实体值为 −0.015625 m。Newton 当时的 16 个原点与 96 个差分邻点沿用既有实测。[同点纹理读数及薄壁细化](p65-signed-distance-grid.md)

### 真实批采样

HOST 负责模型筛选、材料地址、完整范围证明和源资产核对。TYX 的常驻规则格核计算每个选定点的三线性标量、梯度及逐行状态。完整 HOST 证书使用原始格节点值，设备读数保留 FP32 求值结果。RTX 4060 的材料混合批实际计算 8 点，包括整面 7 点和带半径粒子 1 点。容量设为 3，实跑 3 次核启动和同步，上传常驻格 64 B、逐点输入 128 B，读回 160 B。另一世界的材料未触发采样。[设备材料批验收](../../tests/collision/test_tyx_sdf_material.py)

首次回归在基线因缺少场注册入口失败。最终 18 项 HOST、TYX interpret 和真实 CUDA 专项均通过，四个专属 Python 文件通过 Ruff 检查和格式核对。原始日志、源文件 SHA256、完整间隙和设备传输计数保存在本地 `artifacts/p65-night-sdf-material-20261104-01/sdf-material-signed-*`。
