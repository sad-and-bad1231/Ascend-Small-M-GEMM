# M0 Decision Closure

**推荐：M0 INCONCLUSIVE，进入导师审核收束，不再自动追加诊断。**
不是 GO，也不是机制证伪或正式 NO-GO。`formal_approved=false`。
基于 `e8c7bbb`；本轮零新增硬件进程、零 replay、零 TPOT sweep、零 B2。
权威为上游 `cb095ea9` 的 RESEARCH_CONTRACT、2026-10-14 Handoff，以及本地
已保存的 Issue #20 正文；不声称重新核实了在线审批状态，不向上游发布。

## 1. 正确性：三个标准不能混淆

| 标准 | 现有证据 | 判定 |
|---|---|---|
| 单卡组件容差正确性 | BF16 M=1/8、K5120/N2560；B0/B1 的 QKV、Consumer、完整链路对独立 FP32 reference；finite/shape/dtype/ND stride、V 精确复制、Graph replay/改变输入、M16/bias fallback | **已通过当时冻结的诊断 oracle**，不等于真实 checkpoint/TP4 全覆盖 |
| 逐位重复 | 组件 eager/Graph、B0/B1 配对精确相同；TP4 固定输入/positions 下最终 hidden 和 BF16 logits 非逐位一致 | **TP4 bitwise repeatability 未建立**；本身不是 kernel correctness failure |
| 完整服务 token 等价 | 真实32请求×64token：queued B0/B0 有16/32序列不同；早先 B0/B1 有18/32不同 | **当前自由生成配置不合格**；不改完整 token oracle，不承认服务性能收益 |

组件冻结标准：GEMM `atol=.05, rtol=.01, rel_L2<=.01`；Consumer/chain
`atol=.05, rtol=.005`。这是逐元素组合 atol/rtol，不是要求每个近零元素的
相对误差独立小于 rtol。M1/M2 正式 cosine 等完整验收仍待审核/补证，不能补写 PASS。

TP4 固定前三个 token 输入前缀的8请求、4步中，原始 logits argmax **32/32一致**；
所有观察值 finite，实际 batch/Graph/HCCL 元数据匹配。Prefill M=442/Graph=NONE，
Decode M=8/Graph=FULL。最终 hidden 的跨进程 max-abs=2–6、logits=.34375–.6875，
**不能套用 QKV/Consumer 的 .05 阈值**：它们是不同张量，未冻结对应模型级标准，
也没有独立正确模型参考。前三个强制输出不是 correctness 证据；四步一致不能
证明64-token自由生成等价。原诊断的 raw-logit 精确 gate 只代表该次诊断准入，
不得升级为正式 kernel 错误判据。所有旧失败、stop 和收据保留不改。

因此本轮没有已建立的局部容差违例或可复现非法执行，**不启动64层二分**。
首个已观察差异仍是 Prefill 最终 hidden，具体算子根因 UNKNOWN。B0/B1 服务差异
不能仅归因于 NZ，因为纯 B0 控制也不合格，且早先请求交付轨迹并非完全匹配。

## 2. 真实边界：观察到了什么

本轮 CPU 重算既有4份单卡 trace：M=1/8 × B0/B1，每份5次 Graph Replay，
task/kernel/trace/result 文件逐一匹配此前公布 SHA256。复用原 timestamp join，
仅明确增加 NZ 输入格式参数；默认 ND 检查不变。没有新增设备执行或改变准入阈值。

**OBSERVED，合成 TP4-rank-shaped 组件，不是真实 TP4 服务 dispatch：**

- B0：`aclnnMatmul_MatMulCommon_MatMulV2`，native `MatMulV2_ND_ND_...98513`。
- B1：`aclnnMatmulWeightNz_MatMulCommon_MatMulV2`，native `MatMulV2_NDNZ_ND_...229585`。
- 两者输出 ND；相同 `split_qkv_rmsnorm_rope_kernel`，每 replay 一个 GEMM、一个 Consumer。
  无独立 TransData/copy 任务；NZ cast 在加载时，非 replay。内部 packing/cache 成本 UNKNOWN。
- Consumer 已融合 split、Q/K RMSNorm 和 RoPE，还必须写 Q/K/V；不是待删除的空壳。
  Notify wait 与计算重叠，不能相加成串行损失；Graph 控制不能假设整体消失。

下表是每份 trace 五次 replay 的各项 median，**不是未插桩性能、p95 或跨臂收益比较**。
独立进程没有合格 rollback 控制，不据此选最强性能赢家；各项 median 不要求可加。

| M / arm | GEMM µs | Consumer µs | handoff µs | Graph envelope µs | 跨度5%情景 µs |
|---|---:|---:|---:|---:|---:|
| 1 / B0 | 23.82 | 12.76 | .02 | 40.18 | 2.009 |
| 1 / B1 | 21.06 | 12.34 | .02 | 36.78 | 1.839 |
| 8 / B0 | 25.16 | 18.88 | .02 | 47.44 | 2.372 |
| 8 / B1 | 20.46 | 14.32 | .02 | 38.28 | 1.914 |

既有 B0 三进程时间线同样给出 gap=.02–.04µs，Consumer=14.80–24.76µs；
后者变化不能自动叫作可消除成本。缺失 ACL→NPU flow links、无内部计数器，
profiled 单次 replay 不能替代 unprofiled100-replay critical path。

**当前真实 TP4 的界限：** 权重格式和模型源码路径已确认，但尚无通过服务 oracle
的 matched device trace，真实服务 GEMM/Consumer kernel IDs 与关键路径份额仍 UNKNOWN。
B1 加载时改64个 QKV 权重，不按实际 M 回退；它同时影响 Prefill 和 Decode。
`max_num_seqs=8` 不保证 Prefill M<=8，M442 不在历史 Small-M 组件准入范围内。
不得把此局限误称为已证明数值错误，也不得用单卡 dispatch 冒充服务 dispatch。

## 3. 可消除成本与门槛账

| 成本 | B2 能否消除 | 已有证据边界 |
|---|---|---|
| GEMM 输出 QKV 中间物化写入 + Consumer 读回 | 原生生产消费融合可能避免 | BF16 逻辑字节上限 `2×M×2560×2`：M1=10KiB、M8=80KiB；不是实测 HBM traffic 或延迟 |
| 独立 Consumer task 的调度/启动部分 | 可能，但 Graph 已摊销 Host 工作 | 无独立 overhead control，节省 µs UNKNOWN |
| GEMM→Consumer 可见 handoff gap | 乐观假设可全部移除 | 仅约 .02–.04µs，在已有面板中不足 Graph envelope 的 .1% |
| Q/K RMSNorm、RoPE、最终 Q/K/V 写回 | **必须保留等价工作** | 不能把整个 Consumer duration 当上限内的实际可省成本 |
| 26,214,400B逻辑权重读取、矩阵乘加、HCCL | 不因这一边界融合自动消失 | 不能入账；具体缓存/重叠 UNKNOWN |
| Graph prefix/tail、Notify wait、NZ load cast | 不可整体入账为热链节省 | 控制重叠；加载/编译/额外 workspace/转换须另行收费 |

上述物化字节为中间 buffer 的理想写读机会；若 Cube/Vector 交接仍须 GM spill 或
新增同步，实际节省更小，甚至负收益。不可按某个假定带宽直接换算为实测收益。

**M1：** 以这些 trace 的 median span 仅作数量级情景，5%需要约1.84–2.37µs，
扣除可见 gap 仍需约1.82–2.35µs的**额外、可消除**成本。旧 B0 三进程对应余量
2.10–2.57µs。现有证据尚未分离出这些成本；正式门槛是最强合格基线的 p95，
不是这些 median。B1 组件已具备数值可比性，但稳定性能排序未建立，B3 仍缺失。

PR14/Handoff 保留的 Consumer 增长约2.3%、profiled服务改善约1.3–1.6%未过2%。
若只追回这2.3%增量，其他成本不变，即便 Consumer 占整个候选链路，最大改善也
只有 `2.3/102.3≈2.25%`；实际份额<1时更小。**仅吸收历史增量不足以支持5%的
数量级论证**。这是相同时间统计量下的情景算术，不是 p95反证或 B2 的总收益上限；
历史大CSV仍未重新校验，不能当新测量。B1 的 NZ GEMM 收益也不能算成 B2 Fusion 收益。

**M2：** 串行 latency 情景近似 `服务改善=f×组件改善`，5%/10%/20%组件改善分别
需要目标链路占服务关键路径40%/20%/10%才能贡献2%。真实 f UNKNOWN，存在通信
重叠和调度，不能从 kernel sum 代替，也不能把此式当 throughput 预测。
所以目前没有可辩护的 M1+M2 winning region；但也没有足够上界证伪所有融合机会。

## 4. 推荐判定与审核清单

**M0 INCONCLUSIVE，研究收束等待审核。** 不申请自动进入 B2，不宣称正式 NO-GO，
不继续 seed0/NZ 筛选、TPOT sweep、层级排查或其他课题。当前材料已经能说明：
组件可计算且可比，局部 NZ 不等于融合，显式 gap 很小，真实可省内部成本和合法服务
对照缺失。继续无方向诊断不能修复历史不可恢复 provenance。

需导师裁定的最小事项，草稿仅留个人仓库、**未发送**：

1. 是否接受 clean prospective current-runtime reproduction 作为 M0，尽管历史
   dirty runner/container/checkpoint绑定及完整token收据不可恢复、16份大CSV未验证；
   当前镜像 digest 和跨恢复不可变设备身份也有缺口，不能伪造字节级历史复现。
2. 是否允许重新冻结一个双方相同的确定性服务配置，明确 scheduler/batch/Graph/
   prefill/cache/collective策略和 runtime 支持边界，**仍要求完整有序 token-ID 等价**。
   不预设关闭某开关必能解决问题；teacher forcing/数值相似不能替代正式服务 oracle。
   若允许组件级 M0先行，须明确其能支持的机制结论，不能自行豁免服务要求。
3. 接受何种实际边界归因可完成 M0反例；确认组件容差/cosine、B3资格和 frozen split。
   只有接受前两项且有一项能量化内部可省成本的明确方案，才值得另立有界实验合同。

若上述复现路线不被接受，按历史 provenance/合法服务对照缺失关闭本次 M0准入，
结论仍是证据不足，不是机制证伪。任何后续硬件批次必须有新的冻结 scope、设备
身份、预算、授权与 STOP；所有旧 hardware_stopped 保持 true。

## Reproduction

本轮只做 CPU 重算，原始 tensors/logs/trace 保持本地，历史数据不覆写。
[机制账本及完整任务顺序](../results/m0_decision_closure_20261023/mechanism_accounting.json)，
[组件 dispatch/oracle](M0_B_DISPATCH_20261015.md)，[B0时间线](M0_B_TIMELINE_20261017.md)，
[真实服务结果](M0_TP4_RESULTS_20261022.md)，[TP4数值收据](M0_TP4_NUMERIC_20261023.md)。

```bash
python -S experiments/m0b/decision_closure.py --out results/m0_decision_closure_20261023/mechanism_accounting.json
python -m unittest discover -s experiments/m0b -p 'test_*.py'
python -m unittest discover -s experiments/m0tp4 -p 'test*.py'
```

Reducer 对原始哈希/五次执行/格式/dispatch 不符即停止；不下载新CSV，不改生产或环境。
本轮 CPU 验证：26项组件/归因测试、8项TP4测试通过；无硬件测试冒充 PASS。
