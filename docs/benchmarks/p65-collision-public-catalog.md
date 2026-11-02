# 通用碰撞检测 · 15 族案例的三端验收

2026-11-02 在集成分支的同一份产品代码 `5aa4518` 上，通过公开的 `phymeo.detect` 验收了登记表的全部 74 个构型。每个构型有自己的碰撞世界，所有形状在同一模型、同一次查询中登记。三个后端分别为 HOST、tyx interpret 和实际 CUDA。[原始登记表](p65-general-collision-cases.json) · [可执行的混合回归](../../tests/collision/test_public_collision_catalog.py)

| 案例族 | 构型数 | 此次核对的行为和适用范围 |
| --- | ---: | --- |
| RR01 球–平面、球–球 | 5 | 固定法向半空间及平移球的首触、分离、显式过滤。[结果](p65-collision-m2-round.md) |
| RR02 胶囊–胶囊 | 6 | 有限轴、端帽、相切、重合轴和并列最近点。[结果](p65-collision-m2-round.md) |
| RR03 凸体–凸体 | 5 | 锐边满维凸体在静止或平移下的面、边、线段接触区间。[结果](p65-collision-m2-convex-hulls.md) |
| RR04 旋转胶囊–球 | 3 | 有符号定轴转动、途中首触与整段分离，另有端帽擦边及球独立平移回归。[结果](p65-collision-m2-rotating-round.md) |
| RR05 凸体–凹闭合网格 | 5 | 连通有向闭合网格的凹槽、穿壁及实体包含，保留双方材料身份。[结果](p65-collision-m2-closed-mesh.md) |
| RD01 球–布面 | 5 | 平移球与三顶点独立仿射运动的双侧零厚面，含布顶点全在球外的情形。[结果](p65-collision-m2-sphere-face.md) |
| RD02 胶囊–布面 | 5 | 平移胶囊、完整有限轴与端帽，对双侧非退化零厚面给出最近区间和扫掠首触。[结果](p65-collision-m2-capsule-face.md) |
| RD03 有限圆边板–布 | 5 | 正面、背面、侧边、圆角、绕行与整面覆盖，有点、线段及面积见证。[结果](p65-collision-m2-rounded-plates.md) |
| RD04 球–软体外表面 | 5 | 有向四面体外面及其归属。内部共享面陷阱按外表面合同返回 `surface_clear` 并且净距为正。[结果](p65-collision-m2-tetrahedral-surfaces.md) |
| RD05 变形布面–凸刚体 | 5 | 可变形三角面进入刚箱或一般凸体的首触、材料截面、真实边界与支撑方向。[结果](p65-collision-m2-hull-surfaces.md) |
| DD01 异体布面 | 5 | 双侧零厚仿射三角面与点–面首触，异体可复用局部顶点号。[结果](p65-collision-m2-triangle-pairs.md) |
| DD02 布边–布边 | 4 | 边内部相触、换序、分离与首触时间留出。[结果](p65-collision-m2-triangle-pairs.md) |
| DD03 同体布面 | 5 | 合法共享边和异处自接触由材料地址区分。[结果](p65-collision-m2-cloth-adjacency.md) |
| DD04 布–软体外表面 | 5 | 零厚布及受支持的恒定派生法向厚布与四面体外边界相交。[结果](p65-collision-m2-tetrahedral-surfaces.md) |
| DD05 软体–软体 | 6 | 四面体实体包含、交叠和外边界首触，原塌陷负控明确返回无效几何。[结果](p65-collision-m2-closed-volumes.md) |

### 同一输入的结果

- 164 个形状，共 13,366 对；101 对进入窄相。跨世界的配对均在测试中逐对核对过滤原因。74 个构型对应 82 条目标配对记录，因为部分构型有多处测点或多个软体单元。
- 三端按照原登记顺序分别保存完整目标配对。记录含实际返回的状态、时间区间、材料权重和适用的区域或实体数据。三份记录的 UTF-8 长度都为 225,641 字节，逐字节比较相同，SHA-256 均为 `41a13d9dcb40bde6cc3e51d779fedd50156f59e82bcc762d62a9ad9219631431`。
- 每个 tyx 后端完成 2 次宽相启动、39 次特征窄相启动和 145 次凸投影启动。CUDA 路径实际同步并下载输出。窄相使用设备候选与 HOST 对原几何的复核；本次计数对应此输入和现有全对实现。
- 四个新增模块的本地提交依次为 `60dd001`、`2b5664e`、`26b34c5`、`5aa4518`。最后一组的公共碰撞回归为 HOST 258 项、tyx interpret 和实际 CUDA 各 32 项。表中混合模型在这组回归之后单独执行，各后端均通过。

### 复现

在项目环境中运行以下三个命令，每条使用独立的绝对临时目录。原运行使用固定 tyx 提交 `99f6df7c0fd8fdeef77f97df83914da27b64b7ee`、工作树 `src`、单线程和关闭自动 pytest 插件加载。

```text
python -m pytest tests/collision/test_public_collision_catalog.py::test_complete_public_catalog_host -q -p no:cacheprovider
python -m pytest tests/collision/test_public_collision_catalog.py::test_complete_public_catalog_device --require-tyx --tyx-mode=interpret -q -p no:cacheprovider
python -m pytest tests/collision/test_public_collision_catalog.py::test_complete_public_catalog_device --require-tyx --tyx-mode=cuda -q -p no:cacheprovider
```

原始记录在 `artifacts/p65-collision-m2-public-catalog-local-20261102-01`。三个命令收据为 `05-host-pytest-202547219`、`06-interpret-pytest-202603923` 和 `07-cuda-pytest-202651013`。同次保存的 `catalog.json` 公开每个构型的完整目标记录。比较收据 `all-families-parity.json` 的 SHA-256 为 `ac4a83aed4216dea05a0a83b7ad603b582b0f1c19262f2e6a612154dacf6aa9f`。

M2 的验收范围是上表列出的几何与运动。M3 接下来对照这份全对基线测量 SAP/BVH 的筛选效果、更新成本及更多连续运动。M4 独立验证三族的力和运动，再进入单爪与双臂应用回归。[后续计划](../superpowers/plans/2026-11-02-p65-tyx-collision-pipeline.md)
