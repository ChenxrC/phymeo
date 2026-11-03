# 由柔性材料特征施加实际节点力

`DeformableFeatureLoadSystem` 对离散 VT 和无向 EE 分别施加一个无摩擦法向层。输入采用已有的 `CollisionModel`、`ParticleAttachment` 与 `NormalLayer`。两个材料点的速度以各自的原节点权重插值，沿第一侧到第二侧的分离法向求相对速度，按同样的权重把等反力分配到双方节点。[法向力层](../../src/phymeo/solvers/deformable_feature_loads.py) · [材料特征](p65-dd-material-features.md)

```python
loads = ph.DeformableFeatureLoadSystem(
    collision_model, material_bindings, ph.NormalLayer(1/16, 128.0),
    search_distance_m=3/32,
)
with ph.CollisionWorkspace(collision_model, "host") as work:
    frame = loads.sample_particles(state, work)
    result = ph.SemiImplicitSolver().step(
        particle_model, state, frame.step_inputs(external_forces), dt,
    )
    if result.report.accepted:
        next_frame = loads.sample_particles(result.state, work)
```

也可给 `sample_particles` 传入已求好的 `features=`。复用要求同一碰撞模型、完全相同的全部材料面位置、两距离阈值和拓扑环数。改变速度仍会在本次采样中重新求站点速度。已有观测失效、搜索或输出容量不足、活跃特征没有分离法向及力核返回无效数值时，先前成功的 `last_sample` 保留。节点力只读，`step_inputs` 返回独立的力数组。

### 折回带的双侧节点

相同的 16 面布带通过原面片力系统测得下侧节点 100–105 的 z 向合力为 **−18 N**。可选特征力系统实际把 8 个顶点–面和 24 个无向边–边的力写入节点数组。下侧合力为 **−64 N**，上侧节点 112–117 为 **+64 N**。32 个特征各贡献 2 N，6 个平行最近区间使站点数为 38。整张布的合力和关于世界原点的合力矩均在验收精度内为零。调换粒子存储行号后，按材料 ID 读回的节点力相同。[冻结的原生对照](p65-dd-vbd.md) · [实际节点力与功率回归](../../tests/collision/test_deformable_feature_loads.py)

原生 Newton VBD 的下侧合力同为 −64 N。逐节点分配还能分辨平行区间的处理方式。以节点 100 和 101 为例，本层端点均分给出 −12.6 与 −14.9 N，Newton 的原始读数约为 −11.6 与 −15.9 N。

两个平行边的最近点构成线段时，本层以两端各一半的配额进行积分。每个端点都读取两侧各自速度，得到一对相反的力。这一约定在静态均匀载荷下等效于作用在最近区间的中点。平行区间上的速度变化由上述两端求积约定处理。

顶点–面独立例得到顶点 +2 N、三角面三个节点分别 −0.5、−0.5、−1 N。不对称边–边留出在两侧分别使用 (3/4, 1/4) 和 (1/4, 3/4) 的材料权重。对一组平行边施加非均匀节点速度后，两端的实际法向相对速度为 +0.5 和 −1 m/s。阻尼参数 0.5 s/m 时，站点分力为 1.5 和 3 N，总机械功率 −2.25 W。节点速度与节点力的内积之和独立复核相同的功率。1/64 m 的深压顶点继续遵循当前线性法，力为 6 N。

### 真实推进、能量与设备

16 面折回带所有节点质量均为 1 kg，重力和外力为零。用原半隐式粒子求解器推进 1/1024 s 后，HOST 从真实速度差得到下/上两组冲量 −0.0625 / +0.0625 N·s。两组平均分离量增加 0.00002034505 m。新姿态仍有 32 个受力特征，弹性储能由 0.5 J 降至 0.4982228493 J，节点动能为 0.0009000206 J，实际外功为 0 J。

同一状态经真实 TYX CUDA 积分，得到两组冲量 −0.0625 / +0.0625000007 N·s，平均分离量增加 0.00002034567 m，重测储能为 0.4982227990 J。较小的 VT 例经过 1/256 s，间隙由 0.046875 m 增至 0.04691696167 m，法向力由 2 N 降至 1.9946289063 N。独立外力对照给每个节点同向 0.25 N，读取位移后得到外功 5.72205×10⁻⁶ J、材料力做功 8.39233×10⁻⁵ J 和步末动能 4.48227×10⁻⁵ J。

真实 GPU 路径用 TYX 筛选面组合并求每个站点的法向力度与向量。HOST 根据原材料地址求最近点、插值速度和分配节点力，随后 GPU 粒子求解器更新速度及位置。折回带初态和推进后的两次法向力核共处理 70 个站点，法向力工作区累计上传 13,312 B、读回 5,120 B。若 FP32 无法分辨已证实的正压缩，系统给出精度失败并保持上次成功记录。[实际 GPU 受力与积分验收](../../tests/collision/test_tyx_deformable_feature_loads.py)

原红测先实际读取原系统 −18 N，随后因缺少可选系统入口失败。最终 HOST 14 项、TYX interpret 4 项和真实 CUDA 4 项通过，四个专属 Python 文件通过 Ruff 检查与格式核对。原始节点力、起末状态、再查询及输入 SHA256 位于 `artifacts/p65-night-dd-feature-loads-20261104-01/dd-feature-forces-{host,interpret,cuda}/`。
