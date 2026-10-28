# 材料路径阶段单次独立审阅

BASE `f259e690946c612a93a793989ee1e5b50d3995e8`，reviewed HEAD `0bf06195b02b00a8791c5382127d63e3cf5371e6`。一次fresh审阅：0 Critical、1 Important、0 Minor。不派第二次审阅。

## Important：观察未使用的非有限中间量造成新拒绝

actual interpret下，两固定节点`[[0,0,0],[2,1.2e-38,0]]`，rest2、alpha/lambda0、零速度外力；关闭记录可正常预测/投影/重建，残差0。开启记录却在投影时报non-finite FP32 kernel result。原因是方向中的checked除法产生NaN，但零分母让旧kernel跳过所有实际投影；新增trace将这一未使用中间量列为需有限的输出，改变了接受行为。另一大尺度例`[1e30,1e-20,0]`也复现。两个原输入分量均可正常FP32表示；不是依赖缺陷。

执行者先记录5RED：两项零分母表示检查、两项direct实际设备例、一个完整固定workspace例。修复保持原kernel数学及全局非有限输出门禁不变：零分母时direction三列为显式**无施加方向的零哨兵**；非零分母才记录实际使用方向。NumPy同一表示，18列schema不变；不伪造单位法向、不更改物理状态。此规则已写入设计和helper文档，后续ledger必须核对weights/alpha才能识别跳过，不能将零方向当接触法线。

修复后核心30＋actual interpret17共47通过(.77s)，核心＋CUDA47通过(1.66s)。原5RED均包括其中；正分母coincident仍原X方向，普通多边路径/所有权/后段失败仍通过。最终全量与wheel见[工程验收](p65-material-path-acceptance.md)，修复前1824/两核心1522/wheel120证据不冒充修复后结果。

## 未评估范围裁定

- 材料/接触共同认证、原角动量损失修复、转换与fixed host恢复认证：确为后续P6.5必需工作，本切片不声称完成。
- TYX-003和C++长期：仍开放，无依赖修复不重复压力；有限C++不代替长门禁。
- 工程全量/串行计量：执行者收齐后记准确版本与计数，不能提前算通过。

以上为本项目记录输出表示缺陷，不新增tyx_flaw或Warp差距声明。
