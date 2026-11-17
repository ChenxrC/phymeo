# I3u/v：新摩擦模块的源码外安装验证

2026-11-18，冻结源825902af。新wheel含577个Python与7个资产，全部584个产品文件在父源、冻结stage、wheel、独立site逐字节同；完整RECORD通过。未改依赖、未发布/dev合并。

wheel phymeo-0.1.0a0-py3-none-any.whl为1907446B，SHA256 a778eceac233150d6c0b8755ee37cd522aaa43ea74dc320a9988dd256c69b549。含I3u三模块和I3v适配器。已有Python3.11无隔离/网络build；现CUDA313以--no-deps/--no-index/--no-compile装全新target。

## 实际消费

[外部消费者](../../scripts/verify_area_friction_wheel.py)复制到checkout外，Python -I/清PYTHONPATH，拒目标外phymeo来源。依赖仍用既有环境，不称全新依赖安装。

使用I3u接受的固定128个不同点，仅一个新CUDA求解：capacity1、6DOF、128点、24更新/12回溯。RAW_EXIT=0，13组完整结果数组与接受结果逐字节相同；同时对本次实际FP32输入/输出独立代回Jacobian、Fn/Ft、双方矩和残差。工程门通过，严格HOST门仍false。输入压缩SHA256 0f4db08ce5908f55b0e0dd1820727052e30c8084878089918bf6121945d30321。

workspace55332B，上传23788B、下载31544B、kernel一次；kernel约.0567s、solve约.1021s、consumer总计2.3434s（含导入/来源检查），不是性能对比。

## 证据与范围

artifacts/area-pressure-friction-wheel-20261118-01保source-manifest、07-build/08-install/09-package-audit/10-consumer日志与RAW_EXIT、package-audit、完整consumer-output-01和wheel。外部stage/site保留。

证明的是可安装的独立backend；I3v逐字节打包，实际入口验证见[I3v](area-pressure-entry-friction-report-20261118.md)。本消费者没有q积分、生产Scene/type7或原生混合约束。

[I3t混合场景](area-pressure-wheel-report-20261118.md)仍绑定原wheel；新源相对其版本只新增四个摩擦模块，旧产品文件未改。不把旧结果倒灌新wheel，也不重复长窗。
