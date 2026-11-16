# I1：三维四面体压力场与有限刚面裁剪

2026-11-16，codex/exp；基线11b19a62。上一批[面积端口](../../development/area-pressure-3d-port-report-20261116.md)已接受，本批只做新的几何生成，不重跑旧组。按提交检查技能保存小提交。

## 目标与范围

实现通用小型共形 tetra pressure mesh 与有界 oriented rigid triangle mesh 的真实三维相交，输出已有 AreaPressureSurface。先固定静态多案例；不做 MuJoCo-TYX 共同动力学/GPU/CCD/体变形。不修改已接受 I0、native contact、pin 或旧测试。

资料：[Drake 用户指南](https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html)、[ContactSurface 接口](https://drake.mit.edu/doxygen_cxx/classdrake_1_1geometry_1_1_contact_surface.html)于本日核对：一软一硬接触面位于刚性面，压力来自体内 PL 场；法向指向受力的柔顺体。以下有界资产/去重/错误约定是本项目设计，不冒外部引擎复现通过。

## 1. 文件、所有权和执行

原 scene 作者只新增：
- src/phymeo/geometry/_area_pressure_clipping.py：有界 exact-from-FP64 低级几何。
- src/phymeo/geometry/area_pressure_intersection.py：资产、网格准入与公共裁剪。

原 training 作者独立新增：
- tests/test_area_pressure_mesh.py
- tests/test_area_pressure_intersection.py

两作者只合同/旧代码/文本编码，不读取对方新实现或测试，不执行任何项目命令。root 独占后续环境/静态/新完整HOST，逐命令 fresh guard，所有计算/渲染串行。父维护文档与离线查看器；不新开 agent/分支、不覆盖旧WIP、不安装。旧3份untracked及作者旧PNG/review保持。

## 2. 固定公开 API

以下位于 area_pressure_intersection.py；除 keyword-only 字段外按列表顺序：
- TetraPressureMesh(node_ids, world_vertices, cells, nodal_pressure, cell_ids, *, pressure_domain="full")。
- RigidPressureMesh(node_ids, world_vertices, faces, face_ids)。
- intersect_area_pressure(pressure_mesh, rigid_mesh, *, identity, stamp) -> AreaPressureIntersection。
- AreaPressureIntersection 为 frozen 结果，字段 surface: AreaPressureSurface，pressure_domain: str，candidate_pairs: int（全部cell×face），polygon_count: int（去重后正面积凸多边形），shared_face_duplicates: int（已检查合格但由其他cell拥有的正面积多边形数）。

node_ids/cell_ids/face_ids 为唯一非空str（单ID<=128），各节点必须使用；同坐标的不同node ID拒绝。world_vertices(N,3)有限FP64 owned bytes-backed；cells(K,4)/faces(F,3)是全局节点表的row index，整数不得bool，不得越界/重节点；保存为owned不可写int64。nodal_pressure(N,)非负有限FP64；同节点只有一个压力，PL跨共享面连续。复制前先检查形状/容量。接受array-like但不任意iterator。

tet输入顶点排列任意：det非零即可，体积取abs(det)/6；这是静态体单元，不提供“变形未翻转”证明。rigid winding 决定 B→A 法向，不得自动翻面追求正g。identity/stamp复用I0，body A compliant/B rigid，元数据是调用者绑定不是状态证书。

pressure_domain仅 "full"/"local"：
- full：所有incidence=1边界面节点压力精确为0，至少一个内部节点pressure>0。首域为小型共形体复形；不是体积弹性或一般manifold拓扑证书。
- local：同样进行全部几何共形性检查，只免除边界零压力和内部正压力。必须在返回结果保留local标记；仅局部制造场/裁剪参考，未来物理入口不能当full使用。

异常 AreaPressureIntersectionError(ValueError) 带 category 字段。固定类别 invalid_input / invalid_mesh / unsupported_gradient / unsupported_shared_face_gradient / resource_limit / publication_failure。g不支持不能返回部分surface。普通非法类型也统一 ValueError 或此子类。

## 3. 有限资源与算术

上限：每个mesh节点128；tet32；rigid faces64；候选pair2048；每裁剪polygon<=7点；最终I0三角<=1024（严格继承）；所有空输出形状与元数据仍完整。空 rigid faces/vertices 或空tet不接纳；无交集由query产生空surface。

精确谓词和交点采用 stored FP64 的 Fraction exact ratio，无数值probe/迭代调整。每个输入scalar的分子/分母各<=256 bits；每次有理 +-*/ 的结果各<=4096 bits，越界整体 resource_limit。固定有限几何循环，无无限精化或回退epsilon。不要新增大型 interval/证明框架、第三方依赖或把这个HOST路径宣称适合大GPU规模。

浮点只在发布到I0时使用：分母/非零面积/边界归属先由exact算术决定。坐标、压力、gradient转FP64后必须有限、正面积且通过原I0一致性门；失败 publication_failure，不放宽、不夹负压力或静默漏面。

## 4. 资产准入与裁剪

每对tet都检查闭交集恰为shared-node simplex。候选交点来自双方各6edges裁入对方4个重心半空间后的端点；包含关系/共面也覆盖。对所有候选，用tet精确重心坐标检查“不属于shared-node ID集合”的权重均为0；无共享ID时交集必须空。拒重叠、不同ID的共面接触和非匹配边界，不能只检查体积正或跳过shared-face对。

完整face incidence不能超过2；共享三角面两对顶点必须在两侧。几何检查不把不相交的多个component误判失败。重复tet、不同ID坐标重合拒绝。节点压力共享使跨face相等，不另接受每cell互相矛盾的压力表。

Rigid mesh 每对三角也检查闭交集恰为shared-node simplex。可用双方edge与另一个triangle所在平面相交；共面edge再裁3个重心半空间。拒重复面、正面积共面覆盖与非shared交叉。共享边方向相反，最多2face owners；开放/不连通表面允许，不声称是有向闭合刚体体积。

对每个cell×face，从完整有限rigid triangle依次裁四个tet重心半空间。去重和去共线，点/线交集返回空，先识别无面积再检查g。保留rigid winding，凸polygon扇形三角化。source cell/face ID保全，输出triangle ID以二者及fan index无歧义编码（可canonical JSON哈希，<=128）。遍历按稳定cell/face ID排序，polygon起点按exact world坐标字典序规范化；同输入顺序重排结果确定。压力按tet重心插值，gradient按完整节点差解析求解。

内部shared face上的正面积polygon可能被两个cell各发现一次：
- 两边g均须严格正；任何一边不支持则整包拒绝。
- 两边完整world gradient必须精确相同；不同则 unsupported_shared_face_gradient。这是明确的首域限制，不能任意owner选择掩盖不同刚度。
- 合格时仅字典序最小cell_id发布一次，另一份只计 shared_face_duplicates。
- 普通跨cell cut仅共享边，不删任何正面积piece；空/线/点片不能因相邻cell g<=0误拒。

已有I0的g=grad(pA)·n>0门原样保持。完整合法压力体仍可能有一次query超出该门，见下面的full负控；不能称“合法资产一定可接触”。

## 5. 独立固定案例与门

所有力矩关于world origin，m/Pa/N/Nm。测试作者手算oracle，不读新实现、不调用旧几何producer作oracle。允许使用已接受I0 quadrature/row map作为真实消费者；不跑I0旧测试集。仅比较几何/面积/压力/合力矩及身份，不强制某个fan三角数。

局部unit tet T=conv(0,ex,ey,ez)，p=z；必须 local：
1. z=1/4，n=ez，有限rigid三角覆盖截面。A=9/32，F=(0,0,9/128)，M=(9/512,-9/512,0)。
2. x+z=1/2，n=(1,0,1)/sqrt2，覆盖截面矩形 x,y∈[0,1/2]。A=sqrt2/4，F=(1/16,0,1/16)，M=(1/64,1/96,-1/64)。
3. 同斜面，有限rigid三角(0,0,1/2),(1,0,-1/2),(0,1/2,1/2)。A=3sqrt2/16，F=(5/96,0,5/96)，M=(17/1536,1/96,-17/1536)。

双local tet加入conv(ex,ey,ez,(1,1,1))，仍同一p=z：
4. rigid恰为shared face conv(ex,ey,ez)，n=(1,1,1)/sqrt3。单计A=sqrt3/2，F=(1/6,1/6,1/6)，M=(-1/24,1/24,0)，duplicates=1。
5. z=1/4切过两cell，A=21/32，F=(0,0,21/128)，M=(33/512,-33/512,0)。必须含两cell source。

full资产：unit tet内部加c=(1/4,1/4,1/4)，p(c)=1，其余p=0；4cells，p=4min(x,y,z,1-x-y-z)：
6. rigid三角在z=1/8，vertices(1/8,1/8,1/8),(5/8,1/8,1/8),(1/8,5/8,1/8)。A=1/8，F=(0,0,1/16)，M=(7/384,-7/384,0)。邻cell只线交，不可误拒。
7. unit local tet内两不相交面：H=(1/8,1/8,1/4),(1/4,1/8,1/4),(1/8,1/4,1/4)；S=(1/4,1/4,1/4),(3/8,1/4,1/8),(1/4,3/8,1/4)。总A=(1+sqrt2)/128，F=(5,0,11)/3072，M=(79,-44,-47)/98304。g分别1、1/sqrt2。
负控宽水平面切full体：有正面积g=0/负区域，必须unsupported_gradient。

补充功能边界：vertex/edge only、分离空集；g=0、反winding；shared-face梯度不同、任一侧不正；重复/overlap/nonmatching tet、相邻节点不共享、刚面重复/共面部分覆盖/非shared交叉；full边界非零拒绝而local允许；不可写/输入修改无影响；有限caps/资源拒绝。重排cell/节点表同步重排indices/rigid循环顶点以及单cell通用刚体旋转平移：A'=A,F'=RF,M'=RM+t×RF。不需要同一已过组反复执行。

普通尺度门 rtol=atol=1e-10（几何发布后含平方根的工程比较），原I0内部1e-12及压力一致性门不变。错误应可归因；最多一次明确功能修正，不改固定参数/验收门追绿。

两个新完整HOST文件在同次运行。7个主正控用tmp_path各保存唯一JSON，文件名固定 horizontal.json / oblique.json / finite.json / shared_face.json / cross_cells.json / full_body.json / multiple_faces.json。内容包含原asset/rigid输入、identity/stamp、完整输出三角/pressure/gradient/source IDs、I0完整quadrature、计算area/wrench及独立expected值。父只读这些保存记录核对与生成离线可视化，不再调用producer求结果。失败/负控保pytest日志；本批不要求视频。

## 6. 收尾与下一步

源码/独立测试作者提交前全文文本自审，明确UNEXECUTED释放。root验证后记录真实结果、源绑定、原失败和提交，并推exp；dev不改。I1成功只代表小型HOST静态几何生成，下一I2仍需actual step、全joint/free-body响应和pair模型归属；g-kink、full3D大规模/动态/设备等不支持项继续明列。


## 7. 实际收口

2026-11-16：测试源码9640a1de。两新完整HOST首次45pass/2.67s，七个固定解析案例及完整保存数据通过，无功能修补。首两I001和四文件排版修复保留raw，qualifiedAST通过；未运行旧组。浏览器七例交互已检查，父/作者执行槽释放。见[报告与离线查看器](../../development/area-pressure-tetra-intersection-report-20261116.md)。仅HOST静态几何，下一I2接同一步实际双方运动，full/local与g限制继续保留。
