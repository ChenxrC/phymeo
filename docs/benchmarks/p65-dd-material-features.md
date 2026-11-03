# 柔性三角网格的材料特征查询

`ph.query_deformable_features` 对同一时刻登记的零厚三角面，返回搜索距离内的全部顶点–面和唯一无向边–边材料对。普通 `ph.detect` 仍可用于整面最近距离和面片对载荷。这个独立入口适于需要逐原语计数、材料地址或两侧实际位置的调用者。[材料特征接口](../../src/phymeo/geometry/deformable_features.py)

```python
with ph.CollisionWorkspace(model, "cuda") as work:
    frame = ph.query_deformable_features(
        model, current_face_poses, work,
        search_distance_m=3/32, contact_distance_m=1/16,
        topology_rings=2, feature_capacity=32768,
    )
```

每个节点由 `(world, body, material_id)` 寻址。VT 的第一侧是顶点，第二侧是材料三角面。EE 的每组端点和两条边均按材料地址排序。一个共享边特征只返回一行，`support_face_pairs` 列出允许它出现的注册面组合。最近点对 `closest` 保存两侧世界坐标与材料权重。平行边有一段最近点时返回两个极端点，两侧必须使用同一个插值参数。在严格不到搜索阈值时才保留。`inside_force_distance` 另按作用阈值判别。距离为零时没有分离法向，`force_eligible` 为假。该入口返回离散几何观测。

每个已注册三角面在同一世界、同一体内连接其材料顶点。环数 1 排除自关联顶点和共端边，环数 2 再排除一条材料边邻居，之后每个环数增加一步图距离。测试覆盖仅边中点相近、近乎平行但唯一的最近对、两个物体重复使用相同的节点号、不同世界共址、单面和整批镜像换序。相同材料节点的位置发生冲突、重复材料面、第三张面接到同一材料边、三点共线均有显式失败。超过原语候选预算时给出已知候选数下界。若所有原语已核验但输出容量不足，返回实际需要的条数与空列表。[独立几何与容量回归](../../tests/collision/test_deformable_features.py)

### 16 面折回带

16 张面共有 120 个无序面组合。宽筛保留 48 个，材料级包围盒进一步排除后，HOST 对 49 个原语求有限最近距离。结果为 **8 个顶点–面和 24 个无向边–边**，其中 6 个平行边对有非零长度的最近区间。CUDA 工作区实际完成两次宽筛核启动。到该次查询结束，累计上传 3,044 B、读回 864 B，同步一次。

复查复用了 Newton 1.7 已保存的 CPU 和真实 CUDA 数据。每端各有 8 条 VT 和 48 条有向 EE 记录。测试逐一检查原材料身份、两侧重心权重和两侧空间点，并对区间上的两组量使用同一个插值参数。所有 112 条记录均由返回的 32 个物料特征覆盖，坐标与权重分别在 3×10⁻⁶ m 和 3×10⁻⁶ 以内。两份原始 JSON 的 SHA256 随被抽取的记录保存。[实测原始条目](../../tests/reference_newton/dd/native_fold_features.json) · [Newton 与普通 Phymeo 载荷对照](p65-dd-vbd.md)

32 个特征的距离均为 3/64 m。用此次对照的作用距离 1/16 m 和线性刚度 128 N/m 作逐特征只读折算，得到 64 N。普通面片载荷原先对 9 组面片施力，实测为 18 N。这个算例验证了计数差异，实际受力仍由调用者选择的载荷系统发布。

真实 TYX 仅生成同世界、未排除且包围盒相交的面组合。HOST 用实际材料坐标验证有限顶点–面与边–边，比较有理平方距离和阈值，并给出距离的保守浮点包围。设备故障会明确报错，调用前普通检测最后一次发布结果仍可读取。[设备分工与误差边界回归](../../tests/collision/test_tyx_deformable_features.py)

原基线因缺少公开入口返回 1。首轮 HOST 的镜像 EE 校验曾用局部编号给两个物体排序，已修改测试适配为公开的完整地址顺序，几何、阈值和容差均沿用冻结值。最终 HOST 21 项、TYX interpret 2 项和真实 CUDA 2 项通过，3 个专用 Python 文件通过 Ruff 检查与格式核对。收据在 `artifacts/p65-night-dd-features-20261104-01/dd-features-{host,interpret,cuda}`。
