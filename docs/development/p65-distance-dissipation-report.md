# D6-A 距离材料耗散：接口与有限资格

当前裁定：**源码及包外 FUNCTIONAL_PASS，原 strict 精度失败保留**。
下文作者 UNEXECUTED 段保留为历史；当前以文末父资格记录为准。

状态：CODE_READY_TESTS_PENDING / UNEXECUTED。本文是原固定
[设计合同](../superpowers/specs/2026-11-13-distance-dissipation-design.md)
的四文件作者交付，不是实际通过报告。原 D4 稳态支撑 FAIL 保持。

作者基线61001c1c；分支codex/p65-d6-source-20261113。
tests-first e70078a6，schema/kernel/observer 7c02bd98，
原语/能量审计与空edge补足38129027。父拥有共享接线和唯一项目执行槽；
作者未import/collection/pytest/数值/probe/lint/format/build/install。

## 1. 精确接口

[deformable_dissipation.py](../../src/phymeo/backends/deformable_dissipation.py)
不导入TYX、不创建runtime：

- DistanceDissipation(ordered_point_ids, coefficients_ns_per_m)：
  拥有不可恢复写权限的bytes-backed int64(E,2)和float64(E)，E<=1536。
  ID非负，每边两端不同；重复边按原声明顺序有独立语义，不排序或去重。
  c原值finite/nonnegative；其FP32表示必须finite、normal-or-zero且无非零下溢。
  保留原float64系数，不用FP32值替换声明身份。
- DistanceDissipationStep：schema=distance-dissipation-step-v1，
  所有字段严格按合同，均拥有不可变payload。
- make_dissipation_step(parameters, native_edges, weights,
  reconstructed_velocity, dissipated_velocity, trace)：
  weights明确为D1注册后实际stored FP32 inverse mass；不是未舍入FP64质量。
  原point_ids[constraints.indices]精确身份由父注册时绑定；
  此observer另核所有边中的stable-ID/native-index一一对应与原顺序。

phase数组：

| 字段 | dtype / shape |
| --- | --- |
| ordered_point_ids | int64 E×2 |
| coefficients_ns_per_m | float64 E |
| reconstructed_velocity / dissipated_velocity | 各float32 N×3 |
| trace | float32 E×32，不把padding算成材料边 |
| node_impulse / node_velocity_impulse | 各float64 N×3 |
| kinetic_change / ideal_work / rounding_work | 各float64 E |

node_impulse逐边读取trace24:27实际device RN32(j*n)，提升FP64后按+i/-j累积，
包括固定端。没有用FP64重新乘j*n替换device请求。
node_velocity_impulse为实际(final-before)/registered inverse mass；固定端为0。
两者不会被强行设为相同，实际更新舍入单独保留。

## 2. 真实TYX核与唯一物理时间

[_tyx_distance_dissipation.py](../../src/phymeo/backends/_tyx_distance_dissipation.py)
包含：

~~~text
copy_dissipation_velocity(n, velocity, saved_velocity)
dissipate_distances(count, h, edges, coefficients, weights, x, v, trace, status)
~~~

copy核逐元素原样保存velocity，不作算术。父只在active域调用copy与dissipate，
新增两launch；None/显式全零不分配、不launch，由父共享代码保证。

dissipate为单lane原edge order的一次pass；每条仅消耗一次h，
不乘elastic sweep数、不更新position/lambda、不作global drag。
方向按scaled norm构造，保持旧checked FP32 _add/_mul/_div原语。
实际g2=RN32(dot(n,n))参与denominator：

~~~text
b = RN32(h*c)
A = RN32(wi+wj)
denom = RN32(1 + RN32(RN32(b*A)*g2))
j = RN32(-RN32(b*s_before) / denom)
vi[a] = RN32(vi[a] + RN32(RN32(wi*j)*n[a]))
vj[a] = RN32(vj[a] - RN32(RN32(wj*j)*n[a]))
~~~

fixed端不写；两端固定且静止合法零冲量。非零c的零长边拒绝，
不使用原弹性world-axis fallback。c=0跳过方向构造，但仍完整记录
endpoint pre/post、h/c/weights；其length/rate/normal/impulse/g2为0、denom为1。

status约定0成功；1非法h/count、系数或非有限中间量；2非零c零长度；
3非法inverse mass或运动固定端。任何非零都拒整候选。
原status非零不继续后继边。失败可以留下standby prefix，绝不意味着发布成功。
native索引/数组shape由父静态准入；核不以动态越界读代替输入校验。
未涉及edge的fixed节点仍由父完整候选检查和observer的全N固定速度门覆盖。

32列保持合同：0:3 n；3入步rate；4 j；5实际后rate；
6:9/9:12端点前v；12:15/15:18端点后v；18 h；19 c；20 wi；21 wj；
22长度；23denom；24:27实际requested impulse_i；27g2；28:32零。

## 3. Observer边界及能量

observer从原重建速度copy出一个只用于验证的chain：
逐行pre必须按原FP32字节接上前一行，post沿原trace串接，
最终完整N×3必须逐字等于实际传入dissipated_velocity。
它核标量原语、requested impulse、每endpoint更新、fixed不变及reserved列；
任何核对计算都不返回给设备、不生成替代candidate速度。

每edge使用trace实际stored n/g2/j/rate及注册质量分别记录：

- ideal_work = j*s_before + .5*(wi+wj)*g2*j²；
- kinetic_change来自实际stored pre/post端点速度，仅正质量端；
- rounding_work = kinetic_change-ideal_work。

这不是独立oracle，也不是能量credit。没有把负耗散跨edge抵消正误差、
没有以总能量门掩盖错误或改正常状态。

## 4. 内存口径

父active域新增native payload精确为12N+132P，P=max(1,E)：
重建v scratch12N；coefficients4P；trace128P；复用原status。
None/全零无这些设备分配。

HOST常量原参数24E+padded coefficient4P。单phase receipt payload
72N+176E：两个velocity24N、两impulse48N、trace128E、IDs16E、
系数8E、三个energy24E。原last receipt/new receipt及其他并存由父原
三份receipt保留口径计费，不作为本文件隐藏的零成本。

建议observer临时payload保守上界128N+1024E+1024 bytes：
- current chain12N，两计算中impulse48N，三个计算中energy24E；
- 参数FP32 cast、shape/finite/ID关联验证的array暂存；
- 单row的FP32 relative/scalars/requested/primitive核对以及FP64能量短向量；
- owned bytes构造时与上述计算数组并存，最终owned字段属于receipt储备。
该界有意留出NumPy短数组同时存活余量，不把96N+256E当未经核实的精确峰值。
父另保原device readback/candidate储备；Python对象、dict、allocator/runtime/RSS
和caller持有历史包仍不在native/owned-array硬cap承诺内。
本报告没有内存profile或吞吐结论。

## 5. 本lane独立测试

[test_distance_dissipation_kernel.py](../../tests/collision/test_distance_dissipation_kernel.py)
手工33节点：11 HOST，22TYX（每个指定backend各跑22）。
未collection取数；实际收集及结果由父报告。

- 两自由质量1/2kg、正反速度、h=1/128和1/256；
- 非共线两edge链及反序，原顺序逐段参考，不宣称全网络BE；
- 固定支座反力和两端固定；
- 非共面刚体瞬时translation/angular零模、旋转/translation/boost/storage permutation；
- 五点膜z=0合法轴向零模与z=1/8解析瞬时系数，完整pass另用ordered参考；
- 非finite/零长/fixed moving/非法h/overflow等真实kernel状态；
- 零c无虚构方向、E0 padding无额外材料时间；
- 参数所有权/FP32表示/容量/错误shape；
- observer完整tracechain、reserved/参数/impulse/g2/denom/final损坏及不可变。

测试将真实inputs、完整raw trace/status/v和copy结果在首assert前写NPZ。
运行器创建真实TYX runtime/CUDA buffers，资源在finally统一close；
没有CPU物理fallback或伪kernel输出。

独立FP64 oracle不调用产品observer，固定候选几何逐edge解隐式子流。
速度/impulse固定atol2e-6、rtol2e-5。FP32 primitive检查只使用实际stored操作数；
斜边(1,1,1)在执行前手推选定，用于保证stored g2不可假设恰1，
不是从输出筛选几何。

逐edge能量上界从两次乘法及一次加法各half-ulp传播，加实际
stored s/g2相对FP64点积差；FP64审计自身另用gamma32×绝对操作数和界。
包含subnormal舍入区间下界，不加入任意固定能量容差或跨edge抵消。
原语遇实际非零underflow仍按旧helper产生非finite并拒候选。

## 6. 尚待父集成与结果

D1/D3 shared subtype、phase资源/身份/checkpoint、共同事务、固定反力、
None/zero完全原路径由父负责。原tests作者另负责真实D1连续振子/多world/
memory/latefault/reset/restore；原audit作者负责D3耦合与包外消费者。

本lane只四NEW，未修改共享D1/D3、原弹性核、原NormalLayer或任何旧验收门。
这是有序一阶速度分裂及轴向耗散，不是全网络联合隐式求解，
不保证平膜中心normal零模衰减、不支持运动规定支座、
不声称原D4两稳态窗因此合格。新的3s应用窗必须等本模块及短耦合实际验收，
且另行预登记；本报告不启动该执行。

## Parent source qualification, 2026-11-13

The imported author snapshot was UNEXECUTED. Root has now connected D1 standby
velocity, resource accounting, subtype/checkpoint validation, public exports,
and D3 separate damping impulse/fixed reaction in the four shared files.

- 102 kernel interpret:33pass (11 HOST+22 TYX);103 CUDA:22pass/11 HOST deselected.
- D1:104 original strict pair momentum FAIL, 105 other26pass,106 two 1s
  oscillators pass,110 four pair cases pass complete saved arithmetic acceptance.
  Interpret unique current matrix34 includes6 HOST declarations;111 CUDA29pass,
  excluding those6 and including the actual UNKNOWN_RELEASE failure boundary.
- D3 direct:112 original strict elastic-velocity FAIL;116/117 functional4pass
  per backend including late actual candidate failure/retry and restore/reset.
- Run107 audits only the failed saved pair prefix: its raw momentum defect is
  original elastic rounding, damping contribution0 there; no exact physical
  conservation claim. The original strict failures stay FAIL, with thresholds
  unchanged. New same-candidate damping and all original continuous engineering,
  force, arithmetic, energy and transaction checks remain hard gates.
- First static96/97 failures are retained; seven local assigned lambdas became
  equivalent defs, imports/format fixed, two unused test imports removed.
  100/101 final12file checks pass; later narrow consumer/test static receipts
  accompany the explicitly documented functional qualification.

D6 is optional ordered axial dissipation, not global drag or full network BE.
The old None/zero numerical path remains tested bitwise. Current scope is
SOURCE_FUNCTIONALLY_VERIFIED with a retained old elastic precision limitation;
installed qualification and public D5 damped assembly follow separately.
No sustained-support, full cloth, Newton performance or whole-stage claim.

Public D5 integration now has two additional asset sphere/capsule successors
per backend:121 interpret2pass,122 CUDA2pass. They bind c changes into scene
identity and carry complete damping phases through common step/query staleness/
checkpoint/restore. Run120 was a test reference metadata omission before the
first step (root_initial), repaired by binding the original asset root state.
Source and installed qualification remain distinct; no full old suite rerun.

## Parent installed qualification, 2026-11-13

Product ec0e2997, outside wheel-02: run127 compared all532 product members
source/wheel/installed with zero mismatch. Wheel SHA256
48cc7e3938a7ba511ebce7e2efe04dfbc5b9b4d8163ab71edb6f91ba1932fb14.

Runs128/129 (interpret/CUDA) each completed22 calls /44 worldsteps,20 checked
contact sites,4 restores,4 resets and2 stale rejections. Both return
FUNCTIONAL_PASS with strict_elastic_precision=FAIL_RECORDED:20 original
same-entry elastic reconstruction violations per mode, maximum velocity error
0.00012538650431448417 /0.0001253864511099323 m/s respectively.
The independently checked new damping phase, complete stored arithmetic,
continuous engineering gates and common publication remain hard requirements.
No old strict threshold or result is rewritten.

The installed consumer exercises D1 pair/chain and D3 sphere/inclined-capsule.
Dedicated damped D5 declarations passed source121/122 (two cases per mode);
the ordinary D5 installed compatibility runs132/133 are distinct and do not
constitute an additional installed damped-D5 matrix. This is a reusable
material operator, not yet a successful long loaded membrane application.
The separately declared D6-B application is being evaluated at this checkpoint.
