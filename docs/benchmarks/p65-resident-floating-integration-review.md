# C9 浮根驻留集成：限定文本审阅

状态：**SOURCE_REVIEW_RELEASED / 无作者执行**。审阅基准为父 `4f8db2b245f2a0da5936d8c7317401a4b1cdaa61`（前置 style `2a09e70`）；直接只读父工作树和该提交差异，不合并父、不改产品或冻结测试。随后核对父 `403634a` 时，以下五个 OLD 文件及 floating storage 相对该基准没有新差异。本文只新增此报告，没有 import、collection、数值、pytest、静态、构建或设备运行。

## 范围与结论

审阅 OLD `src/phymeo/__init__.py`、`backends/_world_step_layout.py`、`tyx_world_batch.py`、`_resident_collision_plan.py`、`_resident_contact_plan.py`，并读 NEW `_resident_floating_storage.py` 的 stage/check/commit/borrow/close 协议以及必要 binding/schema。没有重新审整套力学核、旧 C8 或已经通过的旧测试。

**未发现本次五个 OLD 挂接中的阻断错误。** 发现一项新作者测试的注册配置错误，已报父并在父 WIP 作最小修正，见下节。该结论不等于完整 C9 接受，尤其不涵盖后续算术修复的最终设备结果。

- **维数与登记**：`_world_step_layout.py:226` 按类型编译 floating spec；`tyx_world_batch.py:106` 把 kinds9/10 分入唯一 floating sidecar。初始化 snapshot 明确分别传 root、joint、full applied 与四块 control，没有套旧固定树 n维状态。浮根 body/velocity/configuration/motor 维数不互换；公开导出位于顶层 `phymeo`，没有增加 backends 包的 eager device 导入。
- **单次发布与恢复**：`tyx_world_batch.py:456–553` 先验证/上传全部持有命令、在 dst 完成各 storage 的推进/状态核验与完整 contact receipt，再在543行统一换 active、增 epoch。`_resident_floating_storage.py:430–446` 的 check 只 stage 状态名/有效性，finish仅赋值；中途异常清空诊断而不发布状态。restore在 batch:695–738 先完整校验各域、写备用 bank、同步检查，再一次换 bank/commit held。`_device` 在 batch:295–308 尝试排空失败提交；若无法同步明确 unavailable，后续必须从已保存checkpoint重建，不能声称损坏设备仍可读取。
- **混合 contact**：`_resident_contact_plan.py:97–101` 仍拒 floating binding。未绑定浮根在189–194行先因无 contact world跳过 affine `.physical` 访问；356–375行按其 `_held_drive/_held_effort/_held_loads` 保存 stale-attempt 依据。batch:508–514 让它走原 floating stage_advance；contact:514–515 合并同次 pending status，receipt只为实际 contact worlds构造。因此没有把无接触浮根挂成 fixed/affine法则，也没有额外推进一次。
- **资源所有权**：batch先登记原runtime和每个实际buffer的 ExitStack回调；floating storage:204–244 用同一 `_buffer` 分配并最后登记自己的close。半构造异常返回 batch:237–238 的close，已分配buffer仍由原stack释放；不存在半初始化 sidecar先进入注册表的路径。batch:777–786 用独立ExitStack依次关闭所有plans，finally再释放原资源，plan失败不跳过其它plan；floating close不再次free原stack的buffer。
- **实际 group/owner**：floating binding:73把资产 root=-1/child=i 映成 body0/i+1；折叠工具仅在75–80行应用一次 authored site变换。collision plan:399–421 对floating使用B而非n，按实际full parent图判邻接，并以(world,body)与(world,group)双向唯一映射拒别名。True域先检查same-body/mask/excluded再拒未过滤的直接父子；False保原显式excluded门。跨world不会形成pair；静态body不能冒动态group。
- **接受态宽相**：batch:317–359 在同锁/原runtime内借当前 active 的root/joint/FK和global ticks；floating bounds入口使用实际storage row与group，未用slot冒row。collision:705–740核完整stamp/tick、有限ordered bounds、过滤flags、registered-domain和capacity，均在 `_publish_query` 前完成。same-tick restore通过epoch使旧结果失效；close清除floating kernel/static buffer引用但保旧owned结果自身。

## 具体 finding：测试缺少原默认域排除

父格式化后的 `tests/scale/test_resident_floating_collision.py:308` 原为 `CollisionModel(tuple(shapes))`，该 folded composite 节点同时登记root marker与tool两个dynamic shapes，随后调用默认 `tree_dynamic_pairs=False`。`_resident_collision_plan.py:478–483` 在False分支先要求每个同world dynamic pair显式excluded；`collision_shapes.py:190–199` 不自动增加同body exclusion。于是节点会在实际包围前拒绝，不能用它报告材料变换失败。

已向父给出最小修正：只给该测试model加 `tuple(combinations(tuple(sources), 2))`，保默认行为、输入几何、物理、容量和期望不变。父当前WIP正是此一行差异。它是作者fixture配置修正，不是C9产品回归；此处只完成文本闭环，不预写该节点实测通过。

## 父执行证据与尚未验证的算术修复

父反馈：P03 interpret445/CUDA443完整通过；453 HOST15通过。完整新增interpret454随后在P01第二步 `FK_NOT_FINITE` 拒绝，保一接受步前缀及原retained字段。作者未执行这些命令，也未解压或重算payload。

父456从原拒绝请求读实际scratch：第二步root/FK/twist有限，仅energy为NaN，vx约±3.308722450212111e−24 m/s。原 `_tyx_numeric._mul` 会把非零乘积绝对值低于FLT_MIN转成NaN，因此极小速度平方不是可接受的近零能量，而成了整步拒绝。父先仅把非负动能项改为weight-first原生FP32乘积，保最终finite/非负门；457原P01接受第二步后，第三步又因其它小量乘积 `NONFINITE` 拒绝。原454、456、457均保留，不能倒写为已绿。

对此只作协议层意见：明确的 **floating-only FP32 FTZ内部算术** 与原“公开输入normal32/zero、候选finite”合同可并存，不需要修改旧通用helper、fixture或容差。仅有限subnormal结果归零；NaN、overflow、非正SPD pivot、PD/full residual及坐标/角域失败仍须拒绝。生成的root/joint存储需保持可被原checkpoint输入规则恢复，不能接受自身restore拒绝的非零subnormal状态。派生energy两bank及restore走同一规则，不能以改速度来消除能量下溢。weight-first避免先v²丢失可被大mass放大的项；不把半质量先FTZ后再乘速度。

此局部工程算术不承诺所有极端尺度/巨大h下与实数解的统一误差界，也不建立新的中间量证明门。父正在实施并继续实际矩阵，**本报告没有把拟修复或源码审阅当作已完成interpret/CUDA能力验收**。浮根contact、affine浮根、CCD和持续摩擦均仍不在C9接受域。
