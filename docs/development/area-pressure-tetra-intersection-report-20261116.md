# 三维压力体—有限刚面裁剪：结果与下一步

2026-11-16，codex/exp。测试源码 **9640a1deccd5311b8708cb1407b4764f21b9db8e**。两个新完整HOST文件首次 **45 passed in 2.67s**，RAW_EXIT=0；没有功能失败、参数调整或阈值修改。资格为 **HOST_STATIC_INTERSECTION_ACCEPTED**。

## 本次新增能力

上一批需要调用者直接给接触三角形。本批从三维体网格和有限刚面出发，自动裁出交集，再交给已接受的面积压力端口积分。

[公共资产与查询](../../src/phymeo/geometry/area_pressure_intersection.py)提供 TetraPressureMesh、RigidPressureMesh、intersect_area_pressure。低级[裁剪模块](../../src/phymeo/geometry/_area_pressure_clipping.py)使用存储FP64的精确有理数进行半空间判定和交点计算，最后检查并发布FP64三角形。没有新增依赖。

- 四面体内采用节点压力的分片线性插值，保存完整世界压力梯度。
- 刚性三角形自身边缘参与裁剪；输出真实三维面积，逐片保留法向、双方身份及来源cell/face。
- 相邻单元共享面只发布一次；普通跨单元接触保留两侧有效面积。
- 所有tet对和rigid face对检查共形交集；重复、覆盖、非匹配交叉与退化输入明确拒绝。
- 压力体区分 full 和 local。full要求外边界压力为零且存在正压力内部节点；local仅供局部制造场对照，不能当作完整物理资产。
- 点/线接触返回空面；不支持的梯度和容量溢出整体报错，不能静默删片或发布部分结果。

当前上限为每mesh128节点、32tet、64rigid faces、2048候选对、1024输出三角。HOST全对检查和Fraction实现用于小型正确性基线，尚未做GPU或大规模性能优化。

## 七个同次保存的可视案例

[打开离线三维查看器](../../artifacts/area-pressure-tetra-intersection-20261116-01/results/index.html)，可拖动相机、滚轮缩放。建议先选择“完整边界零压力体”，再看“有限刚面切边”和“共享面只计一次”。

蓝线表示压力体网格，灰面表示输入的有限刚性面，暖色表示程序生成的接触面，圆点是实际面积积分点。颜色为每个输出三角形的平均压力。**旋转的是视角，页面没有物体运动仿真。**

| 固定案例 | 实测面积 m² | 主要验证 |
| --- | ---: | --- |
| 水平截面 | 0.2812500000 | 自动生成三角，Fz=0.0703125 N |
| 倾斜截面 | 0.3535533906 | 真实斜面面积，Fx=Fz=0.0625 N |
| 有限刚面切边 | 0.2651650429 | 有限边界减少面积与承载，Fx=Fz=5/96 N |
| 共享面 | 0.8660254038 | 两个候选只计一个面，重复计数1 |
| 跨两个单元 | 0.6562500000 | 两cell均保留，Fz=21/128 N |
| 完整边界零压力体 | 0.1250000000 | 四cell资产，有限刚面只截有效区域，Fz=1/16 N |
| 两个不同方向的面 | 0.0188610435 | 各自法向参与合力和原点力矩 |

每例同时验证完整六分量世界原点wrench，并保留独立解析期望。普通尺度比较为执行前固定的 rtol=atol=1e-10，原I0内部检查未改。完整输入、输出、求积点、dA、压力、梯度、来源ID均在同次[七份JSON](../../artifacts/area-pressure-tetra-intersection-20261116-01/results)中；查看器只消费保存数据，没有重跑几何生成。

其余测试覆盖任意tet顶点顺序、节点表/单元重排、刚面循环顺序、一般旋转平移下的面积/力/力矩协变、不可变副本、非法网格、边界退化、梯度门及资源上限。32tet×33rigid faces产生1056片的固定负控正确拒绝，未截断到1024；精确正面积但FP64发布塌缩的微小片正确报告publication_failure，未伪装为空接触。

## 验证与证据

源码和测试由原两作者隔离编写：源码作者未读新测试，测试作者未读新实现；父全文文本审阅后正常cherry-pick。父是唯一执行owner，逐命令fresh guard，原Python3.13.3/NumPy2.4.6/pytest9.1.1环境，thread1/noautoload/nousersite/nobytecode。

9条实际命令：环境0 → 首Ruff1（两处I001）→ 首format1（四文件）→ 导入修正0 → format0 → qualifiedAST0 → 最终Ruff0 → 最终format0 → 首完整HOST0。首错日志保留。三个文件raw AST同，area_pressure_intersection.py仅顶层import次序不同；全部非import AST和import集合相同。

[验收收据](../../artifacts/area-pressure-tetra-intersection-20261116-01/acceptance-audit.json)核9条命令、11个源码/配置/模板绑定、7份同次保存数据、查看器与预览。SHA256：96c0c729fb1635eb4054adf647de96d67399566fb72996f0d762c15dca98fbed。三个无关untracked文件未改；执行结束matching=0，两作者和父槽释放。此后只新增本报告、计划状态和查看器提交，四个数值source/test保持测试时字节。

[测试](../../tests/test_area_pressure_intersection.py)和[网格准入测试](../../tests/test_area_pressure_mesh.py)是本批唯一运行的功能文件；未重跑旧I0/V00–V06、产品、外部Newton或CUDA组。浏览器七例切换、相机拖动/缩放及积分点切换检查完成，无页面错误；本地查看服务已关闭，离线HTML仍可直接打开。

## 限制与下一步

这完成 I1 的小型HOST静态几何生成，**尚未接入MuJoCo-TYX实际运动、共同约束、设备生命周期或训练**。法向由刚面winding给出，必须满足 g=grad(p)·n>0；合法full体的一次宽面查询也可能包含g≤0区域，此时整包拒绝。恰落共享面时还要求两侧完整梯度精确一致，梯度折点尚不支持。local标记、这些准入限制及调用者相位元数据不得被后续入口当作完整动态证书。

下一批 I2 应固定零摩擦法向的最小共同步：双自由体、偏心力矩和单关节响应，使用本批实际生成的面积面；明确同pair仅走一种接触模型，验证实际更新速度而不是仅看力映射。随后处理关节阻尼对应的真实速度响应。具体接缝和 M 与 M+h*Cjoint 的差异见[集成裁定](area-pressure-3d-integration-decision-20261116.md)。若上述g域不足以覆盖目标，应直接在接触律/离散处裁定，不增加几何特判反复追绿。

V03外部双软球面积/收敛FAIL保留；本批不冒Drake/Newton交叉验证通过。稳定dev仍76a240d8，FFT/Fourier没有新增运行结果，T-smoke共享训练保持独立待接。本批[冻结计划](../superpowers/plans/2026-11-16-area-pressure-tetra-intersection.md)及原验收门不变。
