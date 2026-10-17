# 研究状态与论文地图（2026-10-14）

2026-10-17 owner 决策更新：[最终快速筛选](M0_B_FINAL_SCREEN_20261017.md) 的两个 M=8
进程均控制 INVALID。暂停当前 Small-M/NZ 主路线，不再追加 Stability 或融合开发；
这不是机制证伪。未建立独立可量化 >=5% 融合空间，下一课题转向 Ascend MLP
Materialization（本轮未启动其硬件实验）。以下历史研究问题与证据保持原样。

## 1. Problem

目标：在 Ascend 910B2 的 Small-M decode（历史 Qwen3-32B TP=4，单 rank QKV `X[M,5120] @ W[5120,2560]`、BF16、`M=1/2/4/8`）中，验证 GEMM producer 与 **现有 fused consumer** `split_qkv_rmsnorm_rope` 之间是否存在可消除的执行开销（materialization、synchronization、scheduling）。原始关注点是整体 `GEMM → Consumer`，不是重新实现已经融合的 split/norm/rope。

### 已知事实与源码依据

1. [主研究合同](https://github.com/intellistream/ascend-small-m-gemm/blob/main/RESEARCH_CONTRACT.md) 和 [owner handoff](https://github.com/intellistream/ascend-small-m-gemm/blob/main/docs/2026-10-14-mingxuan-handoff.md) 明确：旧 blanket NZ 和 standalone no-copy 已关闭，新原语尚不存在；冻结 M0→M1→M2。
2. [历史 r5 设计与数据](https://github.com/intellistream/ascend-small-m-gemm/blob/main/docs/source-evidence/2026-08-05-qkv-only-policy-design.md)：910B2、QKV no-bias、M≤8 时 `f_linear_nz` 相对 ND 存在约 1.10–1.19× 的**组件级**收益；bias/M≥16 不可自动准入。
3. [PR #14 @ 11aca1d](https://github.com/intellistream/ascend-small-m-gemm/pull/14) 已关闭未合并；08-27 的同运行 3-arm capture 中 w2 median TPOT ND 31.893 ms、NZ 31.478 ms（-1.30%）、rollback 31.938 ms；低于冻结 2% 服务门槛。每臂 32/32 完成，但**无 token-ID/oracle**。捕获 runner 的 dirty SHA 缺失，源输入 custody 校验不等于重现了捕获代码环境。
4. 旧跨日期分析观察到 consumer +24% 左右的回退；08-27 同运行仅约 +2.34～2.36%，因此**不能**把 +24% 或 +2.3% 固定归因给 NZ。残余原因仍未识别。
5. [固定 consumer kernel](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/52f923884b9adaedfc2e764428207d0448227549/vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_rope.py) 已在 Triton 执行 Q/K head-wise RMSNorm、RoPE 和 Q/K/V 输出；[fusion pass](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/52f923884b9adaedfc2e764428207d0448227549/vllm_ascend/compilation/passes/qknorm_rope_fusion_pass.py) 负责 graph 匹配。**配置启用不代表每次实际命中**。
6. [vLLM-Ascend PR #16699](https://github.com/vllm-project/vllm-ascend/pull/16699) 报告了 RoPE positions 的 UB/Scalar 读取时序竞态；截至本笔记审查时该修复仍未合并。必须包含跨 tile、多次 replay correctness。
7. [vLLM-Ascend PR #15416](https://github.com/vllm-project/vllm-ascend/pull/15416) 已合并 KDA projection 的异步 stream overlap，表明“跨算子 overlap”是现实的替代设计，不必预设 single-launch fusion 最优。

### 理论推论，尚非硬件结果

`Y=XW`，FLOPs = `2MKN`。假设每次完整读取 BF16 权重，weight bytes = `2KN`，算术强度近似 `M FLOP/byte`（忽略 cache、其他 traffic）。

假设恰好消除中间 `Y[M,N]` 的一次 GM write+read，则 bytes saved = `4MN`，相对 weight traffic = `2M/K`；在 `M=8,K=5120` 时仅 **0.3125%**。因此“省一份 QKV tensor 的 HBM 流量”本身不足以预测巨大加速。若存在实际收益，可能来自同步、dispatch、layout 转换与 Cube/Vector 利用率；均需 profiler 验证。

RMSNorm 依赖每个 head 的 `sum_j y_j²`，`y_j` 又依赖完整 K 轴归约。Fusion 要求 K-reduction 先完成，并将**完整 head** 的结果安全交给 Vector，不能直接用任意 GEMM partial tile 作最终 norm。

### 待证假设（互不混同）

- **H1 materialization**：存在真实且可避免的 GM write/read 或 TransData。
- **H2 synchronization**：AIC/AIV handoff 或 graph-boundary wait 占据关键路径。
- **H3 scheduling**：在保证正确性的前提下，tile overlap 能减少 idle 而不增加更多 MIX 控制成本。

M0 先定位 H1/H2/H3 的可测空间；不能凭旧两组总耗时推断任一成立。


### M0-A 静态审计发现：历史派生判定的叙事残留（2026-10-14）

[PR #14 matched window2 的 manifest](https://github.com/intellistream/ascend-small-m-gemm/blob/11aca1de03129d6f4f56d846864af20ab35eaee8/results/qkv_same_run_matched_20260827/window2/manifest.json) 明确给出：`split_qkv_us` A=765649.8 us，B=783534.2 us，即 +2.33585%；consumer <+10% 为 PASS；TPOT 31.893→31.478 ms（-1.30123%）对 ≥2% 为 FAIL。

但同一 manifest 的 `mechanism.service_path_ruling` 仍称 `split consumer +16-26% >= +10% limit`，来自旧的 selector `qkv_path(..., context='service')` 静态 `reason`，**与本次 matched verdict 不一致**。这属于派生摘要的 stale narrative，不能当作本次 consumer 因果归因。新 M0 必须以原始 trace、窗口指标和显式判定公式为准；将“执行代码来源”“证据 custody”“实验判定”和“机制解释”分别记录。

## 2. 高价值相关工作（供学习，不宣称复现）

| 论文 / 源码 | 信号 | 本课题必须回答的对比问题 |
|---|---|---|
| [ClusterFusion, NeurIPS 2025](https://github.com/xinhao-luo/ClusterFusion) | H100 ClusterReduce/DSM 扩展 QKV→Attention fusion | Ascend 是否具有有竞争力的 on-chip producer/consumer handoff？ |
| [FlashFuser, HPCA 2026](https://arxiv.org/search/?query=FlashFuser&searchtype=all) | 跨核通信 / 融合边界 | 目标数据流依赖什么新于已有 fusion 的调度？论文正式来源待固定 |
| [VTC, OSDI 2026](https://github.com/ADAPT-uiuc/VTC) | VirtualTensor/index mapping 避免物理数据移动 | 纯 layout-aware consumer 能否达到同样效果？ |
| [MPK, OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/cheng) | persistent mega-kernel + task graph | 执行图优化与 single primitive 谁更合理？ |
| [XY-Serve, ASPLOS 2026](https://doi.org/10.1145/3760250.3762228) | Ascend Meta-kernels / virtual padding | 小 M 与动态 shape 在真实 NPU 服务中如何表现？ |
| [RedFuser, ASPLOS 2026](https://github.com/alibaba/redfuser) | 级联 reduction online fusion | 是否能改变 K 轴归约与 head-norm 的依赖？ |

文献还可在 [awesome LLM inference systems / kernels-compilers](https://github.com/sad-and-bad1231/awesome-LLM-inference-systems/blob/main/topics/kernels-compilers.md) 查看。优先精读 ClusterFusion 源码、固定 Ascend consumer、Ascend C MIX/Cube/AIV 文档，其次 VTC/MPK。**新颖性不是“做了 fusion”，而是可验证的 Ascend-specific winning/losing boundary。**

## 3. 当前状态

- 完成：公开源码审查与研究命题拆解、历史结果边界、初始阅读池。
- 未完成：新的 container/device provenance、单卡 per-rank 硬件复核、TP=4 同运行服务复核、raw output correctness、融合原语及 M1/M2。
- 下一步：执行 [M0_PROTOCOL.md](M0_PROTOCOL.md)；原始证据与新硬件结果须分开存放，不做事后阈值选择。
