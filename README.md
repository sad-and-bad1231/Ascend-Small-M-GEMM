# Ascend Small-M GEMM — Research Notebook

个人研究笔记与 M0 实验合同。**当前公开仓库**；不存放未公开数据、凭证、私有服务器地址或设备日志。

- 上游课题：[intellistream/ascend-small-m-gemm](https://github.com/intellistream/ascend-small-m-gemm)
- Owner：[@sad-and-bad1231](https://github.com/sad-and-bad1231)
- 当前阶段：**M0 / evidence reproduction**。已有单卡诊断证据，尚无新融合原语或正式性能结论。
- [研究问题、论文及已有证据](docs/RESEARCH_STATE.md)
- [M0 实验协议](docs/M0_PROTOCOL.md)
- [M0 manifest 草案](docs/M0_MANIFEST.yaml)
- [M0-B 单卡对照合同与 CPU-only 准备](docs/M0_B_CONTRACT_20261015.md)
- [M0-B 单卡诊断结果与原始失败边界](docs/M0_B_DIAGNOSTIC_20261015.md)：M=1 seed0 正确性、Graph Replay PASS，未跑性能 Benchmark。
- [M0-B 真实 dispatch / 转换归因](docs/M0_B_DISPATCH_20261015.md)：M=1/8 独立 NZ 路径已观测；短 trace 不是性能收益。
- [M0-B 非正式 timing pilot](docs/M0_B_TIMING_PILOT_20261015.md)：两组正确性通过，但 B0 rollback drift 超限，均为 INVALID；未建立有效收益。

## 目标与验收

验证 **Ascend 910B2 Small-M QKV GEMM → 已融合 Q/K RMSNorm+RoPE consumer** 是否存在可复现、可消除的执行边界成本。按上游合同顺序执行：

- **M0**：固定代码与环境、复现旧 NZ / no-copy 对照、补齐输出 oracle、归因 consumer，给出 GO/NO_GO。
- **M1**（M0 准入后）：一个 native QKV+consumer primitive；100% correctness、相对最强合格组件基线 p95 ≥5% 改善，且无 allocation median 回退。
- **M2**（M1 准入后）：真实 graph-mode serving，预注册 TPOT 或 throughput ≥2% 改善，并通过正确性、内存与尾延迟约束。

禁止将局部 GEMM 加速写成服务加速；禁止将单卡 per-rank 测量写成 TP=4 服务复现；禁止在未经充分硬件测量前切换研究方向。

> 本仓仅整理个人可公开研究材料；代码、旧 profiler 和原始证据以固定上游提交为准，不复制或改写历史结果。
