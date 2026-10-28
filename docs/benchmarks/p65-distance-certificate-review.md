# P6.5 距离证书唯一审阅与修复

一次fresh审阅`distance_certificate_final_review`，范围`bab34e4757ba2a469bcbb13cc52a4f30d75164ca..0ba866c2d3500ea47bd807eff51bfe420d0b40c7`，包含生产API、独立测试、安装smoke及计量脚本。结论0Critical、0Important、1Minor；全部探针已退出，无第二次审阅。原始工作记录保留在本地SDD ledger，公开验收见[工程记录](p65-distance-certificate-engineering.json)。

## M1 已修复：冻结记录中的可变method

原构造器只判断method与固定字符串相等，0维/单元素NumPy字符串数组也能通过并保持别名；调用方普通修改数组即可改变冻结记录。正常认证函数总传入内置字符串，几何数值未受影响，但公开标量/所有权合同被破坏。

先加入6条RED：0维/1维数组、NumPy字符串scalar、可变相等对象及错误类型；`2838c90`要求内置str后再比较固定值。错误内置字符串仍ValueError，非str为TypeError。6GREEN，相关89通过；后续最终两CPU相关各174、Ubuntu89、actualinterpret/CUDA各2、重新构建的源码外wheel通过。没有放宽数值门禁。

主agent另核对到actualtyx测试缺标记：修复前`-m not tyx`收集91含两原生用例，`6a9a81b`只新增tyx标记后收集89/91且排除两原生用例，再显式执行两后端各2通过。此为测试选择修正，不计为第二次review或生产数学修改。

## 独立证据与未裁定项的处理

审阅者实际运行83host/2未选，并用seed652029生成90个有理几何探针检查独立活动面最小值包围和端点相邻性，全部通过；另直接复现M1。审阅者未重跑设备测试，不能把实现者结果改称审阅者实测。

- 动态接触、厚度、法向、history与CCD：接受范围边界，继续留在H1后续，不以静态证书关闭。
- CUDA几何与原生精度：此处host有理几何；最终薄集成由主agent核验实际设备候选，未宣称GPU几何。
- 旧材料接触与长期CUDA/C++：不重开旧审阅，TYX-003未修不重复压力。
- broader CPU、源码外wheel、正式性能与CI：由主agent独立完成并按版本记录，修复前全量不冒充最终全量，远程结果另核对。
- 故意object.__setattr__/直接__dict__绕过：不扩展冻结dataclass合同；M1是普通合法输入对象别名，已修。
- 同时多个无效输入的异常优先级：设计未规定，保持单项类别合同，不新增无关API承诺。

最终产品只修标量method边界，两CPU完整回归已有前版证据；最终相关用例及重新打包充分覆盖修复，准确远程CI仍需独立通过。测试时间不作为性能数据，性能只采串行独立脚本。
