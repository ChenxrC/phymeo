# 直接球电机接触与资产设备边界来源：2026-11-12 实际结果

状态：**SOURCE_AND_INSTALLED_QUALIFIED**，本批关闭，无新截止。用户授权“继续下一批”；执行计划见[本批计划](../superpowers/plans/2026-11-12-parity-motor-contact-asset.md)。三名原作者在隔离分支并行编码/文本审阅，父承担共享接线和全部串行执行。整个P6.5/Newton对齐仍未完成。

## 交付的公共能力

1. **BALL-MC：直接球关节电机与接触共同求解。** MuJoCo-TYX的normal、pyramidal constraint路径使用已存在的完整JOINT moment表，实际执行器力、广义力、接触响应与运动同次发布。六组多模型覆盖法向/摩擦、球与标量关节混合、共享执行器、两个运动分支、禁用与力限幅；保留独立同入步native和连续native轨迹对照、双方COM力矩、恢复与失败事务。
2. **M1-A：原资产几何直接成为设备边界查询来源。** ordinary、affine、floating、floating-affine四类from_asset保留原模型、shape、graph、group、材点与完整facet身份，经真实accepted设备FK、gather/BVH/距离核发布。原OBJ mesh、程序triangle、球/胶囊及fixed-tool变换有独立多案例。未把mesh改成凸包，未从HOST上传当前坐标伪装设备来源。
3. **M2设计：只完成真实变形生产者的调研与分片。** [设计](p65-m2-producer-bank-design.md)核对现有TYX临时数组生命周期和固定Newton 1.7 State实现。下一步先做M2-P持久accepted/candidate节点状态与提交/lease，再做M2-Q查询；本批没有M2产品或执行资格。

BALL产品提交257b9448；资产产品与最终参考/消费者提交d362799fbec81082a45b1a580a5194b5b621f92f。独立审阅见[BALL-MC](p65-ball-mc-independent-review-20261112.md)与[资产](p65-resident-boundary-asset-source-review.md)。M2设计19738ca0不计功能通过。

## 实际源码验证

| 范围 | 同次执行 | 结果 |
| --- | --- | --- |
| BALL-MC两完整新文件 | 015-ball-full | **51 passed / 54.46s**，含interpret与CUDA，无skip |
| BALL原受影响入口 | 016-ball-compat | **6 passed / 12.94s**，三个旧节点各两后端 |
| M1-A HOST | 023-asset-host | **2 passed**，确认新factory准入不放开旧convex compiler |
| M1-A interpret | 024、027、028 | 原37个节点分别13+5+19通过，已过同源节点不重跑；两次首错见下文 |
| M1-A CUDA | 029-asset-cuda | **37 passed / 59.51s**，完整TYX选择 |
| 旧M1 direct/static兼容 | 035、037 | interpret **8 passed / 16.98s**；CUDA **8 passed / 33.01s** |

HOST/tyx选择中的deselection是预登记后端分组，不是跳过失败用例。没有把源码节点数换算为整个框架完成百分比。

保留的首错及修复：

- BALL首Ruff/format为import顺序和排版，修后013/014通过，没有数值修复。
- 资产首Ruff/format为import顺序及单行语句排版；最终检查通过。消费者首排版失败也保留，033/034最终静态通过。
- run024固定MJCF capsule参考地址次序错误。fixed importer的原材点IDs对应end/start，floating importer对应start/end；NEW reference显式分开，产品、XML和误差门不变。
- run027 floating-affine wrong-model的预期拒绝文本错误。原graph token包含model，既有前置guard更早拒绝；测试检查该family的准确阶段，未改产品检查顺序。
- run036兼容命令basetemp误用未创建的035-cuda目录，setup报WinError3，无物理步。037只修运行路径。所有首失败raw均保留。
- 两份消费者审阅发现的双错保全、独立原资产参考、BVH stop字段检查均在首次安装执行前修正；不伪称这些为运行失败。

## 一个统一wheel与四个源码外消费者

040一次构建、041安装、042文件parity均RAW_EXIT0。使用已有依赖，不安装或更新TYX/MuJoCo/Torch，不借CPU MuJoCo替代TYX物理步。

- 产品HEAD：d362799fbec81082a45b1a580a5194b5b621f92f。
- 完整516个产品成员的source/wheel/installed字节一致，issues=0；不是所有历史功能在本包重新执行。
- wheel：phymeo-0.1.0a0-py3-none-any.whl，1,648,842 bytes。
- SHA256：ebfd8b396f88d47e2be0ccd712198c0ad1260b214cfa62e4fdc2d78b0dce9544。
- 运行目录、消费者脚本及安装site均在checkout外；bootstrap删除原editable source搜索项，全部已加载Phymeo模块来自本次site。BALL各320个、资产各302个模块的实际来源均满足该检查。

| 消费者 | interpret / CUDA | 实际范围 |
| --- | --- | --- |
| BALL-MC | 043 / 044，均PASS及RAW_EXIT0 | 每后端112worldsteps：normal两world×24步、mixed两world×24步、生命周期8batchcalls×2world。每份120events完整typed gzip |
| M1-A | 045 / 046，均PASS及RAW_EXIT0 | 每后端6worldsteps、3complete queries、24facet rows、2stale拒绝。原OBJ/URDF来源与真实运动同轨迹；每份23events完整typed gzip |

四次安装执行首次均通过。完整gzip包含原输入、实际物理状态/力或设备packet、独立对照和失败/恢复前缀；消费者保存逻辑bytecount/SHA与压缩bytecount/SHA，父归档仅复制压缩表示并核物理SHA/size，没有再次运行producer/verifier或全解压。

证据根：artifacts/p65-parity-motor-asset-20261112-01。新installed-wheel-01归档61文件，完整wheel/scripts/results逐文件匹配；installed-close.json SHA256为436e6037bce48df55455b33effd1e268eef59fa0721ad87a3cd4d1b21d7ca118。源测试原日志、前后快照、首静态失败与source-staging/parity收据均留在该证据根。没有重复旧批审计、恢复旧session或改证据Junction。

## 准入边界与下一片

BALL-MC限fixed BALL/HINGE/SLIDE、nq10/nv8/nu32、直接BALL的JOINT/NONE/FIXED/BIAS_NONE、na0及原primitive condim1/pyramidal3。未开放direct BALL activation、FREE+BALL、limits/dry/spring/equality或任意组合；原BALL-A/C的标量activation资格仍按原范围复用。

M1-A是四类资产到TyxWorldBatch M1的连接。**ParticipantScene逻辑participant到其另一组物理bank的映射仍未接到M1**；不是零HOST、M2、CCD、closed-solid内外判断或mesh接触力。原M1全部资源/发布语义和过滤保留，局部query通过不能当作载荷通过。

下一批可并行纯编码的两个优先节点：

- 场景接缝：ParticipantScene→M1，先冻结逻辑world/participant、内部机械row/group、原asset身份和多bank lease/共同epoch；多机器人/工具、反registration次序、恢复、旧结果与晚失败共同覆盖。
- M2-P：按已完成调研冻结持久节点producer的数组预算/API，复用原ordered距离核，独立案例必须真实改变边长；先通过持久状态与事务，再接M2-Q。不要把这份设计计作已实现。

父仍独占共享接缝和串行执行。N8后继只分析原saved梯度/Adam历史，不新增采样或训练；更广机器人组合按原十组合同另选。G1、J/R原学习门、R03和曲面touch-SE仍开放。未push、remote CI、发包或清理环境。
