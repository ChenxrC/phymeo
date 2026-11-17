# I3k：同一入步的通用接触合力与力矩

2026-11-18；codex/exp，源码7a5feee5。新增8 HOST手向量与1真实CUDA生命周期节点通过，61原生诊断和8面积观察受影响兼容节点通过，合计78节点。dev不变。

## 能做什么

新 `BodyContactWrenchReader` 借用公共heterogeneous workspace。按world key、body名字和显式入步参考点，汇总该body的全部native/area接触。可按对方body名字分组；其余接触仍计入total。不依赖某个任务Scene、独立FREE对象或固定手指命名。

```python
from phymeo.control.body_contact_wrench import BodyContactWrenchReader

reader = BodyContactWrenchReader(
    workspace, area_identities={
        model: tuple(asset.identity for asset in assets)
        for model, assets in area_scene.assets.items()
    },
)
measurement = reader.measure(
    current_frame, world_key=41, body_name="object",
    reference_point_world=(0.0, 0.0, 0.0),
    group_body_names=("support-body",),
)
```

参考点是调用者明确绑定的同入步世界点，接口不自行推断COM。结果只有接触合力/矩，不包含重力、驱动、阻尼和外载。readback是明确的HOST诊断消费，不称设备零拷贝策略特征。

原生记录保留真实intrinsic couple和geom身份；面积记录保留材料、sample/source身份及真实dA，没有虚构geomID。面积力已是N，不再次乘dA；A正/B反，矩由已发布世界原点矩按M_ref=M_origin-reference×F平移。有效性标志表示矩不可用时，选中body的完整wrench明确拒绝；无关body/world的合法缺省不阻止当前body读取。

## 共用验证与防漏载

`AreaPressureEntryReader` 将材料/有序pair、样本目录、真实面积字、native meta/clock/body owner、mask/padding、A方力、FP32矩表示和重复bodypair检查抽为共用层。返回owned immutable样本；原SuccessAdapter复用它，保持原digest次序和按pair数值聚合。顶层每次measure只做一次workspace.readback，不二次下载或混帧。

实际模型声明与workspace导出选项分别检查：area batch漏开area导出时构造拒绝；native batch选择扩展area导出时接受全零bank。旧native-only wrench/slip函数遇选中area声明会明确拒绝，避免原来的静默少算。无area旧调用保持。

## 有判别力的证据

手算同body同时受native力(1,2,4)N、真力偶(.5,-.25,.75)Nm，以及总3N的面积力。独立预定总F=(1,5,4)N，原点矩=(-3.5,-1.25,-2.25)Nm；参考点(.2,-.3,.4)m后矩=(-.3,-.85,-3.55)Nm，均通过。真实.02m²三角3→12点细分保持积分结果；四个body的共同原点反作用合计为零。

新CUDA采用I3j已固定的三个model/world，source实际2步、fresh成对恢复后1步，共3个真实step调用。分别核object、support、两finger，与实际typed area receipt及独立native公开world-wrench字段对照；selectedreset无旧载荷，未reset world保持，恢复重放一致。该短窗不是新承重稳态、抓持成功或训练结果。

负控包含错误材料/目录/tick/owner、native-area重复pair、末world故障、数值/padding/矩伪造、真实FP32乘积超范围的矩缺省；空/重名body在读回前拒绝。所有失败不修改输入或物理。

## 运行记录

根目录 `artifacts/area-pressure-body-wrench-20261118-01`：
- 04原native诊断：61passed/1.17s，RAW_EXIT0。
- 05受影响面积reader：8passed/41.14s（interpret/CUDA各4），RAW_EXIT0。
- 09新完整：9passed/20.77s，RAW_EXIT0；同次CUDA完整gzip为105428bytes，SHA256 35a38cba00dfbd390d444709c858fcf3886bde4c9ec8cf935592f964c7c9ab9f。
- 10/11六文件Ruff与format-check通过；09执行前后六文件SHA相同。05中的generic/角色adapter/旧guard文件之后未改；后续排版仅new BodyReader和new测试。
- 首源码与测试Ruff只有import排序，均在相应数值执行前修复；无数值失败或参数调节。
- 文本审阅在执行前补齐invalid mask、实际area声明缺导出、空/重名body及native全零area导出的兼容边界。

下一按[07:10计划](../superpowers/plans/2026-11-18-area-pressure-through-0710.md)展示真实多world连续运动与同入步接触量。当前异构语义encoder/PPO仍是计划，旧同构PPO只消费q/v；不能据本批诊断称网络已接通或学会操作。
