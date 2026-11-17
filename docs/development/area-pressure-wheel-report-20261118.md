# I3t：源码目录外的面积接触 wheel 验证（2026-11-18）

新面积路径已从本地 wheel 安装后运行成功。消费者在源码目录外使用两个模型、三个稀疏 world ID，执行四次真实 CUDA 整批步，完成资产重载、检查点恢复重放、选择 reset 和公开双方载荷读取。没有调用 tests/experiments 或依赖 parent/src。

这是一项分发与生命周期边界验证，不重新证明I3s长窗，也不代表生产摩擦已接入。该 wheel 绑定构建时7b7a00ef的生产源码；后续新增模块须单独验证新包，不能倒灌本结果。

## 实际构建与隔离范围

- 从Git登记的工作字节冻结573个Python文件、7个包内资产，以及pyproject.toml/README.md。没有复制未登记WIP、tests或experiments。
- 已有Python3.11的build/setuptools执行wheel/no-isolation；已有CUDA313环境仅以--no-index、--no-deps、--no-compile安装本地wheel到全新target。未下载/升级依赖。
- wheel、冻结stage、构建时当前src、安装target的580个文件逐字节相同；检查精确文件集合、重复/非法路径、ZIP RECORD的每项SHA256与size。MuJoCo派生代码的LICENSE/NOTICE/AUTHORS和Panda四个资产均包含。
- 外部消费者复制到Temp独立目录，Python -I并清PYTHONPATH。启动前拒绝已导入Phymeo及其他sys.path上的Phymeo候选，显式插入安装target；最终362个已导入Phymeo模块全部位于该target。
- 依赖复用原环境：NumPy2.4.6、MuJoCo3.3.7、tyx-compute0.1.0a0，Git pin99f6df7c0fd8fdeef77f97df83914da27b64b7ee。记录原.pth和依赖路径；因此是独立产品安装测试，不冒“全新依赖环境”。

wheel：phymeo-0.1.0a0-py3-none-any.whl，1,888,957字节，SHA256 **5a0c5a8886c12af8bc5b378b49dcb4b4ebd245065b84ad06b50e7afb4d51fe75**。

## 四次真实整批调用

h=.001s，world顺序(41,7,305)。41/305使用I3s FREE方块和完整压力资产，初态分别为qz=.00926425/v0及qz=.0095/vz=-.02。7为I3l原生垂直sphere、m=.30kg、D70。XML、完整资产和初态都内联在[独立消费者](../../scripts/verify_area_pressure_wheel.py)，未把实验fixture打入产品。

| 顺序 | 行为 | 独立检查 |
| --- | --- | --- |
| 1 | step1，随后保存scene checkpoint | 三世界实际运动；area正dA/Fn，native接触实际有载；同入步载荷和状态标签 |
| 2 | step2 | 同一资产与连续输入推进 |
| 3 | restore(step1)后重做step2 | 恢复历史及重放所有raw数值数组、q/v/time/tick逐字节相同；只将应更新的configuration身份单独核验 |
| 4 | 只reset305后step | 41/7历史与状态保留，305清历史并回原初态；最终ticks=(3,3,1) |

每次measure_many只有一次public readback，原生与面积载荷独立按接触坐标/双侧符号/力臂重建。公开world0、bodyorigin、COM参考点的F/M互反；dA以原设备两个int32字重建FP64并与原quad字节一致，积分面积等于完整有限面。没有把归一权重当面积或重复乘dA。

消费者总计9.1711秒，4个physical_calls及4个completed_calls，RAW_EXIT=0。此时间含编译/初始化/记录，非稳态benchmark。

## 混合几何的真实工作量

可选TYX拓扑建议+HOST精确复核，capacity64：4batch calls/4launches、192raw candidate pairs、165culled、6device polygons、21fallback pairs。实际仍有HOST fallback，不能称全设备几何。原生sphere世界没有面积记录，面积世界没有重复原生接触载荷。默认几何选择未更改。

## 保留的诊断

首次PowerShell封装将裸-I误匹配为InformationAction/InformationVariable，在脚本和build执行前被拒；改显式ToolArgs数组后首次实际build成功。首次包审计已完成文件字节核对，随后对METADATA错误地假定LF而失败；原文件使用合法CRLF。修复为标准邮件header解析、保留原metadata字节后完整包审计通过。没有修改产品或重新构建追绿。

消费者首Ruff仅import排序，首次format需排版；两项修正后静态检查通过。父补充tyx的真实distribution名tyx-compute，使Git pin来源正确记录。唯一实际消费者运行即通过，无物理失败/重跑。

## 证据与后续

证据：[area-pressure-wheel-20261118-01](../../artifacts/area-pressure-wheel-20261118-01)。source-manifest.json、01-build/02-install、03原审计失败、04最终包审计、11-outside-consumer原日志、wheel副本和consumer-output完整保存；原Temp仍保留。

完整消费者events.jsonl.gz为65,640字节，SHA256 a1edae704e27bde81f4b1f20be02e4574b110f9748021b6146ca084b2238e1ca。资产2,574字节和checkpoint135,416字节另有SHA收据。源消费者提交f3bacfd8。

后续I3u解决独立摩擦参考8点容量与experiment依赖，使设备求解器成为可打包模块；随后仍须从真实同入步M/J接入、共同积分、观察版本和真实摩擦多案例，不能以wheel能导入代替这些功能验收。
