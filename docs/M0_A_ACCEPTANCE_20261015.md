# M0-A Host 验收摘要

日期：2026-10-15，Asia/Shanghai。只审计，不运行旧TP4 orchestrator、NPU算子测试、
计时、Profiler或M1；不改生产kernel，不安装/升级框架，不改inference-map。

## STATUS

**Host checks完成；历史custody部分核验；M0-B BLOCKED / NOT APPROVED。**
个人main已`pull --ff-only`至`633ea9943d34f534fae9b102244d17c00000e457`。
上游remote main=`cb095ea9dc335507ffdf7aedfb5d761fe6b5d88c`、PR14 HEAD=
`11aca1de03129d6f4f56d846864af20ab35eaee8`，均已用`ls-remote`核实。

复用现存上游对象库创建独立detached sparse worktree；仅代码、小JSON/JSONL和摘要落盘，
约412KiB，未下载/展开大型CSV，也未初始化submodules。首次`--no-checkout`后索引尚未
填充，host脚本exit4提示缺少测试文件；用`git read-tree -mu HEAD`完成该新worktree的
sparse初始化后，HEAD精确匹配、工作树干净，再运行正式host checks。失败本地记录保留。

## ENV

| 项 | 本轮核验 |
| --- | --- |
| 设备 | 1个visible Ascend910B2；is_available=True；设备标识/原始日志只留本地 |
| CANN /driver | 9.1.0 /26.0.rc1；分别读OPP version.info和driver version.info |
| Python /torch /torch-npu | 3.12.13 /2.10.0+cpu /2.10.0.post4；导入成功，不代表算子ABI已验证 |
| vLLM | 0.23.0+empty；`/vllm-workspace/vllm`，`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`，clean |
| Ascend插件 | 0.23.0；`/vllm-workspace/vllm-ascend`，`5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`，clean |
| Triton /Ascend distribution | 3.5.0 /3.2.2；导入origin为system site-packages，未切到历史fork |
| Container digest | UNAVAILABLE，环境未注入；未推断或编造镜像SHA |

原preflight脚本查找的两处legacy version.cfg未提供版本内容，故另存当前真实版本文件和
distribution metadata。`LD_PRELOAD/FunctionLoader`警告已保留，未诊断为失败。
个人仓库、上游main、PR14 worktree、两个installed golden checkout在编辑前均clean；
historical fork对象本地已存在，审查只用`git show`，未切换或修改那些工作树。

## TESTS

`m0_preflight.sh` exit0；`m0_host_checks.sh <PR14 sparse checkout>` exit0，
三项实际执行通过：attribution CPU tests（13项）、custody CPU tests（2项）、capture `--self-test`。
它们不证明历史CSV真实有效、设备正确性、capture执行等价或性能复现。

Custody index有28项：**12个JSON/JSONL大小及SHA全部匹配；16个kernel_details.csv
共597659552 bytes未materialize、未哈希、未重聚合**。未核验集合为
`arm{A,B}_w{1,2}/rank{0,1,2,3}/kernel_details.csv`；rollback CSV本来就是local-only，
未作为已验证输入。完整custody verifier保持NOT_RUN，不能沿用历史index的`resolved`作本轮PASS。
6个请求窗口按历史`_slo_per_request`重新聚合，完成数/TTFT/TPOT与6个agg JSON一致。

只从已核验的小输入复算w2：A/B/rollback TPOT中位数31.893/31.478/31.938ms，
B相对A为-1.301%，未达2%服务门槛；rollback drift约+0.141%。这是历史profiled数据，
不是本轮测得服务收益。consumer +2.33585%仅复查派生manifest算术，**未从CSV独立重算**。
manifest里的`service_path_ruling`仍写旧consumer +16–26%，与matched verdict不符，
属于stale narrative，不能解释成此次硬件归因。历史runner dirty/hash和token-ID oracle缺失不可补写。

## SOURCE DRIFT

固定历史fork：vLLM=`e4ce33646f2ef1781289e6dc651fad0d00177c55`，Ascend=
`52f923884b9adaedfc2e764428207d0448227549`，Triton=
`ef6c292108651ca88aa8c5f59ecfc1849693a71f`；对象均本地存在，未当作installed版本。
11个相关源码文件已作内容SHA/diff审查；不是全树语义/ABI等价性证明。

**确认的源码链，而非已运行的dispatch：**

1. 当前Qwen3Attention.forward与历史AST相同：qkv_proj→split→Q/K head RMSNorm→RoPE→attention。
2. 插件注册AscendQKVParallelLinear；无量化、无custom parallel override时，经linear method
   到`torch.ops.vllm.unquantized_gemm`→`F.linear`。额外通信/量化分支不能混当该baseline。
3. vLLM `compilation/backends.py`通过platform动态取得GraphFusionPassManager；Ascend
   compiler的fusion_pass_compile调用它，`fuse_qknorm_rope`默认true。
4. QKNormRopeFusionPass仅注册BF16、head_dim128、eps1e-6/1e-5的匹配模式。
   命中后只把consumer部分替换为DeviceOperator；**没有把QKV GEMM合进该primitive**。
5. 910B2使用BaseDeviceAdaptor，调用`torch.ops.vllm.qkv_rmsnorm_rope`，注册到Triton
   `ops/triton/linearnorm/split_qkv_rmsnorm_rope.py`。该文件与历史逐字节相同，SHA256=
   `ca226aac8c5559904646765e3f358dcdd394c8d8af58f5102338676844312ae1`。
   A5的SIMT分支不是本次910B2路径；实际graph match数/launch仍NOT_RUN。

历史fork fusion pass对所有不同attention shape注册，当前版本只取第一个Attention层；
mixed target/draft形状准入不能假设相同。Qwen3/linear/loader/config/dispatch文件也有内容差异；
consumer字节相同不证明不同Triton/CANN下生成代码、positions时序或replay正确性相同。

**NZ policy不可直接复用：** 当前源码无`VLLM_ASCEND_QKV_NZ_MODE`或旧load-time hook。
历史adapter patch在精确pinned两个目标文件上`git apply --check` exit0，
在当前golden插件上exit1（model_runner_v1.py hunk105不适用）；仅check，未apply。
当前generic weight_nz_mode0/1/2仍在：910B2 BF16/FP16仅mode2转换，但这是一般权重策略，
不是QKV-only B1，不能偷换。历史hook按max_num_seqs≤8、shape、dtype/no-bias作load-time
guard，不是逐step M选择；未来须核对capture padding、M16和bias fallback，以及实测单份驻留。

## BLOCKERS

- Container digest、目标Qwen3-32B checkpoint/trace身份未知，未加载模型。
- 16个CSV与rollback原始profiler未核验；历史执行dirty/hash和输出oracle缺失不可修复。
- 当前B1没有可直接使用的旧policy，patch不兼容；不在本轮移植或改生产kernel。
- 精确输出oracle/tolerance、replay覆盖、预算及advisor审批尚未冻结；只有单卡，M0-C仍缺TP4。

## NEXT GATE

保持`approved_for_hardware: false`。先完成剩余host/provenance决策、审查隔离环境中的
兼容方案及完整oracle/manifest，提交Issue #20请advisor批准后才能进入单卡M0-B。
本轮不代发审批、不执行M0-B/C、不开展M1，也不把INCONCLUSIVE改成机制NO_GO。

本地收据（不提交）：`m0_receipts/preflight_20261015T002354Z`、
`host_audit_20261015T002410Z`、`source_audit_20261015`；校验文件SHA及artifact hashes见
[manifest](M0_MANIFEST.yaml)。原始设备/主机日志、详细diff及历史请求内容全部留在本地。
