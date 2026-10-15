# M0 — Reproduction & Attribution Protocol（Draft，待 advisor 审核）

目标：**用真实 910B2 严格复核旧 QKV-local NZ 机制，识别 GEMM→fused consumer 的真实损失边界**，在合理的重复预算下形成 M0 GO/NO_GO。M0 不开发新融合原语；不能因一次负结果或未确认的归因直接换题。

## 0. Frozen historical identity / gates

上游 [Issue #20](https://github.com/intellistream/ascend-small-m-gemm/issues/20)；主合同 [RESEARCH_CONTRACT](https://github.com/intellistream/ascend-small-m-gemm/blob/main/RESEARCH_CONTRACT.md)。历史分支 `feature/ascend-small-m-gemm-qkv-fusion-m0 @ 11aca1de03129d6f4f56d846864af20ab35eaee8`；运行引用 submodules：

- `vllm-hust=e4ce33646f2ef1781289e6dc651fad0d00177c55`
- `vllm-ascend-hust=52f923884b9adaedfc2e764428207d0448227549`
- `triton-ascend-hust=ef6c292108651ca88aa8c5f59ecfc1849693a71f`
- adapter: `adapters/vllm_ascend/qkv_nz_capture_policy.patch`，须验证 patch hash、可应用性和真实运行位置。

历史约束：只有 no-bias、BF16、Qwen3-32B TP=4 per-rank `W=[2560,5120]`、`M=1/2/4/8` 有 no-copy 准入证据；`M=16` 和 bias 作 fallback 负对照。旧服务候选 `VLLM_ASCEND_QKV_NZ_MODE=1` 对照 baseline `=0`，不能偷偷启用全局 `weight_nz_mode=2`。

保留上游冻结门槛：consumer delta 上界 +10%、allreduce per-launch median +5%、至少两个一致窗口、TPOT candidate ≥2% 改善、baseline→candidate→rollback drift <2%；M1/M2 门槛见上游合同。**不为了通过门槛调参。**

## 1. M0-A：Host / provenance（不占 NPU）

1. 固定 `main` 与 PR #14 SHA，清点两者 diff；历史 PR 未合并到 main，**不能**只跑 main 里的 `run_static_checks.sh` 就声称旧 M0 已复现。
2. 检查 PR #14 的 `qkv_m0_attribution.py`、`qkv_same_run_capture.py`、`qkv_m0_verify_custody.py` 与 CPU tests；读取 `results/qkv_same_run_matched_20260827/manifest.json` 和 raw custody index。
3. 在独立、可恢复 worktree 运行 CPU regression/custody tests。验证现存输入的 SHA 与 re-aggregation，保留原始 expected vs observed diff。
4. 明确原 08-27 runner 的 dirty-state/hash 缺失**无法事后修复**；新 NPU 捕获需新建完整 provenance，并与历史结果分开。

## 2. M0-B：Single-NPU per-rank 组件实验（真实硬件）

一张 910B2 先验证**目标 TP rank 的同形数据流**，不加载完整 32B 不等于四卡服务：
- B0：official ND QKV GEMM + **现有** `split_qkv_rmsnorm_rope`。
- B1：QKV-only NZ/no-copy + **同一个 consumer**。保证同 checkpoint/tensors、相同 graph capture、无额外常驻 weight copy。
- M=1/2/4/8；M=16、bias 走 baseline fallback；实际 head geometry 和 cos/sin 参数从 pinned model/config/consumer 获取，不猜。
- 输出 oracle：先 QKV、再 Q/K/V 对照 official chain / 明确的 FP32 reference（容差由 pinned op tests 确定、测量前冻结），记录 shape/dtype/strides、finite、max abs/rel、replay determinism。变更有 rounding 可能，**不能把相同 token 数当作正确性**。
- 计时：NPU event/graph replay 与 profiler trace 分开采集；每一 arm 保留独立重复。正式 attribution 至少 10 个独立 allocation/repeat（和上游 M1 方法一致）；同时保存每次 median、p5/p95 和环境/频率探针，避免百万 kernel launch 的伪独立样本。
- 收集真实 dispatch ID、launch count、GEMM latency、consumer latency、GM/MTE/UB/AIC/AIV wait、显式 conversion、peak/resident memory。**不能把 total kernel time 或固定 kernel 名等同关键路径**。
- 归因方向：若 consumer delta 不稳定，记录未识别；若 H1/H2/H3 没有可验证空间，不因“局部 GEMM 赢”而启动 M1。

## 3. M0-C：TP=4 matched graph-mode serving（需四张兼容 NPU）

单卡结果不得冒充 M0-C。以 Qwen3-32B、相同模型/trace/scheduler/graph config、同一四卡 allocation 完成 **A/B/A rollback**，其中 A=ND，B=QKV-local NZ。
- 原协议的两个窗口：w1 lazy/capture（8 requests），w2 steady（24 requests）仅作为历史可比性窗口；另将**不带 profiler 的服务计时**与 profiler attribution 分开记录，避免将 profiled-only TPOT 视为真实服务 SLO。
- 建议预注册最少 5 个独立 A/B/A triplets，保持固定 request order、seed、arrival/concurrency，并保存所有 triplet 与逐请求记录；不能用单窗口几十个相关请求计算独立重复置信区间。若预算不足，标注结论为 descriptive-only。
- 补充逐请求 `prompt_sha256`、output token-ID hash（安全时可保存 token IDs）、finished、TTFT、decode-only TPOT、output tok/s；比较 baseline/candidate token 序列，不仅是 output length。记录被拒绝的回放、差异请求和任何重试。
- 用 **同一 arm / window** 的 kernel profiling 去核对 consumer delta 与 TPOT；此前历史 +24% **未在 same-run 复现**，+2.3% 仍不能单次因果认定。
- drift gate <2%（历史标准），若不满足直接 invalid，不通过筛选掉不利结果制造胜出；保存所有失败重复及其原因。

## 4. 判定（不要提前切题）

- **M0_GO_TO_M1**：新捕获 correctness / provenance / matched execution 有效；复现/解释了组件与服务边界，且发现可归因、可量化、足以合理支撑 M1 5% 组件与 M2 2% 服务目标的目标开销。然后另行做 exactly one native primitive。
- **M0_NO_GO_THIS_MECHANISM**：在预注册有效覆盖与重复下，目标边界无可识别优化空间，或所有可能节省小于测量波动/必要控制成本。保留原数据、停止 QKV+Consumer Fusion；**这不推导所有 Small-M 优化不可能**。
- **M0_INCONCLUSIVE**：设备/容器不匹配、provenance 不完整、oracle 失败、设备 drift 超限或缺少四卡服务测量。先修复测量，不把不确定性当作 NO_GO，不换题。

## 5. 交付与权限

所有 raw receipts 保存在**本地受控位置或显式选择的私有仓库**；本仓目前为 PUBLIC，只发布去标识的 manifest/schema 和结论摘要。

第一份 M0 manifest 按 [M0_MANIFEST.yaml](M0_MANIFEST.yaml) 填写并提交给 [Issue #20](https://github.com/intellistream/ascend-small-m-gemm/issues/20) 审核；**advisor review 是正式实验准入条件，不是每次本地诊断的设备使用许可**。正式准入尚未批准，不能伪造审批字段。

Owner 明确授权后，可以在已分配的单卡上运行有界 `DIAGNOSTIC_ONLY` smoke，
不要求导师审批链接。当前范围以 manifest 的 diagnostic_permission.cells 为准：
seed0、M=1/8 的 B0/B1 正确性及每臂最多5次 replay 的短 profiler，
以及 M=16 无 bias、M=1 有 GEMM bias 的正确性 fallback。
Owner 随后明确授权 M=1/8、seed0 各一组独立进程 B0/B1/B0 的 chain timing pilot：
10 warmup、5 trial x100 replay，不并行启动 profiler，不扩展 seed/shape。
Event 和 Wall median 的前后 baseline drift 任一 >=2% 都标记整组 INVALID，
保留原数据且不重试替换；只有 correctness/Graph/source/resource 失败停止后续硬件。
这不是 FORMAL approval，也不是正式统计检验或 M1/M2 GO。
诊断仍须通过源码身份、设备空闲/归属、
correctness、资源预算和失败即停止检查；失败记录不覆盖。诊断成功不等于正式准入。
个人仓库可按 Owner 要求持续提交、push 脱敏记录；真实原始设备日志仍留本地。
