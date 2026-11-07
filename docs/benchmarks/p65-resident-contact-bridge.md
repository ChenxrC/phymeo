# C6 HOST finite-contact bridge

当前已由父实测并限域接受；结果、失败保留与范围见本文末节。下方最初作者交接的UNEXECUTED标签属于历史。

状态：**AUTHORING_COMPLETE / UNEXECUTED**。本片只新增 `_resident_contact_bridge.py`、`test_resident_contact_bridge.py` 与本文；作者没有执行 import、collection、数值脚本、pytest、lint、format 或设备命令。父已保存公开入口 325 的有效 RED，本片不重复。父后续执行结果须另记，不能从本文推出 HOST/CUDA 已通过。

## 接口和职责

`ContactShapeFrame` 保存原 CollisionShape、逐 material ID 的实际 world 点、真实 stored FK origin/R、group、COM 和入步 COM/angular velocity。所有数值为独立 owned、不可写 FP64；Mapping 复制并只读，静态形状的 reference 为声明 world 顶点均值。R 仅按 det>0、maxabs(R.T@R-I)<=5e-5 检查，原值不修正；`rotation_orthogonality_error` 明列误差，不把该域当物理精度。

`sample_tree_contacts(*,world,frames,materials_by_pair,candidates,group_count,site_capacity)` 复用已有 `contact_patch_sample` 与 `sample_patch_loads`。每个明确 canonical pair 只取一个旧 patch，原 quadrature weight 在旧 law 内乘一次。每个共同 load point 的速度严格为入步 `v_COM + omega × (p-COM)`。动态一侧按实际 COM 累加 full N×6 力/矩，static 侧给等反力及绕其明确 reference 的矩；它不是整棵树固定根的全部支撑反力。

`ResidentContactSite` 保原 ContactSite 的双侧 surface、material IDs/weights/world skin offsets、normal/gap/common point/weight，原 RobotContactMaterial，双侧速度/力/reference。`ContactLoads` 保存完整 sites、wrench、static_reactions、原 frames 和 world。采样时四个实际位移及两类 work 均为 None。

`finish_contact_loads(loads,candidate_frames)` 不重采法向律、不推进、不发布：以 entry R/p 的 FP64 solve 逆映射原 surface/common anchors，再用实际 candidate R/p 运输，static 位移为零。两类功分别按 `F_first·d_first + F_second·d_second` 保存。candidate 速度不回填本步力。`finalized=True` 仅表示这次运输完成，不代表物理状态已 accepted。失败不修改原 loads；可用原包重试。

## 范围、来源和资源

首片仅 sphere、有限 capsule、unrounded full hull；pair 必须同 world、不同物理 body，恰一侧 dynamic、一侧 static，friction=0。非法几何、deep/tie 或旧查询不能表达的输入按原异常传播，不转为分离。group_count 沿原 1..16，shape/vertex 总帽沿原 MAX_SHAPES/MAX_VERTICES。没有隐式全局配对表。site_capacity 为 0..4096，按完整几何站点计，零力站点仍占容量；超帽 `ContactSiteCapacityExceeded.count/limit` 保留实数。空 candidate 给 full N×6 零；空 frames 只接受空材料与空 candidates。

同 group 的 FK origin/R、COM/v/omega 在每个 packet 内必须完全同源。candidate 必须保全 shape/key/material IDs/radius/group 和 static 原几何；local 声明点的重构以 1e-10 m 检查记录一致性。该门**不**施于跨态 COM：实际设备 COM 和 p/R 各自有 FP32 舍入，父 plan 负责原 model/group/source proof，桥不把 COM 重新由 p/R 推算或替换。1e-10 不是 FP32 物理精度或新能量守恒门。

父负责 `_resident_contact_plan.py` 与旧 batch/tree/collision hooks：真实设备 entry/candidate、源/epoch/tick、FP32 上传/广义散回、PD/惯量/积分、局部容量状态和最终单次 swap。桥为 HOST 几何/法向载荷，不称设备窄相、CCD、moving/moving、affine/floating 或持续摩擦已经接通。

## 冻结短测试

| 组 | 输入与独立预期 |
|---|---|
| a01–a03 | 半宽1/8 box 中心 z3/32，有限台顶z0，k64得总2N、4点归一；球–球/有限胶囊轴部/端点各2N，越端零；c4、入步vz=±1/8 得1.5/2.5N。 |
| a04–a05 | 同group两球各1N、COM x1：总Fy2/Tz−2且相反z力臂取消；偏心球 x1/2、omega_z1/2 的共同点速度(1/128,1/4,0)，c4得到Fy1/Tz1/2。 |
| a06–a08 | canonical owner 换序、整体轴循环和平移；绕z=.37的真box/非box截锥2N；动态有限capsule两真实端点各1N，分离仍两几何站点但零力。 |
| b01–b06 | 零力容量、完整站点计数/重试、空域full零、world/owner/law/group/candidate/资源负控、原invalid/deep异常与原law故障传播、输入不变。 |
| c01–c07 | 所有权/不可写；真实Rz(cos3/5,sin4/5)运输，材料功33/40J与作用点功13/16J分别记录；合法stored非精确正交矩阵必须用solve，非法R拒绝；source/局部点/static变化拒绝与重试；独立舍入COM不被二次重建；nonfinite/维度错误。 |

以上是有限包构造/力与运输用例。大角度运输测试不是一个已执行的物理积分步，也不证明一般转动接触能量守恒。实际树长窗、W1/W4、真实 interpret/CUDA、故障共同发布由父与另一作者的 C6 设备专项验证。

父建议专项命令（本作者未执行）：

```text
python -m pytest -q tests/test_resident_contact_bridge.py
```

tests-first `fc1b9dc`；schema-only checkpoint `f09bc26`。实现 source 绑定以最后交付提交为准；需要父当前已有 finite patch/law 模块，作者按授权未 merge 父。

## 2026-11-08 父实测

当前 **HOST_VERIFIED_AND_C6_INTEGRATED**。原作者UNEXECUTED为历史交接标签。340对完整本文件专项实际50 passed/1.18s/RAW0，原import/format首错与最终静态记录保留；产品只style整理后随C6同源使用。C6解释74分组、实际CUDA74、相邻15、同次112完整payload和两backend源码外wheel消费均完成，完整范围/原始任务和失败记录见[父正式接受](p65-resident-tree-contacts.md#2026-11-08-父执行与正式限域接受)。桥仍是HOST有限几何与载荷/锚点运输；设备运动、容量拒步及共同发布由原batch负责，不扩到affine/floating/CCD/moving pairs或持续摩擦。
