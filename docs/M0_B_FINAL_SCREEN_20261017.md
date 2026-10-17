# Final Small-M quick screen

Base `bda1ce4`; DIAGNOSTIC_ONLY, formal_approved=false. This is a practical
screen, not the formal M1/M2 contract or a new stability investigation.

Frozen before hardware: M=8 seed0, two fresh processes; three independent full
QKV->existing Consumer graphs captured in B0 ND / shadow ND / B1 NZ order.
Identical synthetic CPU tensors/parameters; three device weights temporarily
co-resident, separate output buffers/private Graph allocations. This is not a
production memory result. No profiler, production edits, new kernel or TP4.

Five fixed palindromic cycles: `A S S A | A B B A | A S S A`, A=B0, S=shadow,
B=B1. Each slot10 warmups +100 measured replays; timing Event record/sync/query
API prewarmed three times and events reused. Preserve all60 Event/Wall rows per
process. Full original FP32, eager/Graph exact, five-replay determinism,
changed-input/positions and post-screen oracles apply to all three Graphs.

Conservative2% control gate applies to BOTH Event and Wall: every shadow ABBA
pair, every ABBA B0 bracket, and first/last-cycle B0 median drift. Any failure
invalidates the process, no window removal/retry. Within each central ABBA,
reference=mean(bracketing B0); candidate=mean(two B1); reduction=(reference-
candidate)/reference*100. Primary gain=median of all five Event paired reductions;
Wall gain is reported, never substituted to select a favorable outcome.

Only if both M8 processes VALID and each Event gain>=5% may M1 be enabled with
the same two-process design. Otherwise no M1, integration or fusion development.
INVALID means measurement/allocation-control blocking, not mechanism falsified;
pause Small-M and redirect next research to Ascend MLP Materialization without
starting that new topic's hardware here. Valid but <5% closes NZ as the primary
route; assess existing independent quantified fusion space, not speculative
kernel development.120s/process,2GiB allocated limit, same physical identity.

## Decision: NO-GO next phase; measurement blocked

**Pause current Small-M/NZ primary route. Do not run M1 or integrate/fuse.**
Both fixed processes completed correctly but failed the preregistered controls.
This is not mechanism falsification or proof that fusion cannot help. No retry,
extra window, extra seed or follow-up Stability/Profiler investigation is planned.
Next research focus: **Ascend MLP Materialization**; no new-topic hardware ran here.

| M=8 process | Shadow Event difference abs range % | Max B0 bracket drift Event / Wall % | B0 first/last-cycle drift Event / Wall % | RAW NZ Event reduction % | Decision |
|---|---:|---:|---:|---:|---|
| 1 | 2.293-2.992 | 0.732 / 12.874 | 0.037 / 0.372 | 1.339 | INVALID shadow + Wall |
| 2 | 1.078-1.941 | 0.871 / 2.196 | 0.064 / 0.075 | 7.826 | INVALID Wall bracket |

Process1's shadow is slower than B0 despite identical ND math; this difference
exceeds the control budget even with stable B0 temporal Event medians. Process2
passes Event-only controls, but its retained Wall bracket exceeds2%. The Wall
gate was frozen before hardware; no retrospective threshold relaxation. Even
Event-only screening could not pass **both** processes because process1 fails.
Raw Wall reductions are1.207% and7.709%; accepted gains are null for both.
The favorable second result cannot be selected to trigger conditional M=1.

### Independent fusion space

No independent, quantified >=5% fusion opportunity is established. Prior M=1
device traces measured only0.02-0.04us handoff, not enough by itself for5% of
a42-53us chain. Consumer's full duration is **not** an eliminable fusion cost.
M=8 QKV materialization is40,960bytes each way (81,920bytes write+read); no
measured latency attribution or valid >=5% saving follows from those bytes.
Eliminable memory traffic/cache/internal costs remain UNKNOWN, not zero.
We therefore stop investment now rather than launch a speculative fusion kernel.

### Evidence and limits

Both processes exit0 within120s, all three complete Graphs actually replayed and
passed CPU FP32/eager/Graph/changed-buffer/repeat/post-screen oracles. ND shadow
and cross-process B0 snapshots are exact; NZ satisfies the original full oracle
and paired tolerance. Same input/source/runtime fingerprints and physical card;
final device healthy/idle, production checkouts unchanged. No Profiler or TP4.

Three distinct weights/formats `[ND,ND,NZ]` are deliberately resident with three
private captures and separate QKV outputs. Allocated Torch memory83,258,880bytes,
peak110,184,960bytes in each process, below2GiB. Peak includes load-time format
conversion temporaries; no claim of only three copies during conversion or of
production memory neutrality. Timing Event API prewarm occurred, but retained
Wall excursions remain unlocalized; their cause is UNKNOWN and not investigated.

23 CPU tests pass, including fixed order, exact2% rejection, null accepted gain
on INVALID and M1 denial before the gate. Reducer: `paired_screen.assess(trials)`.
Public [process1](../results/m0b_final_screen_20261017/m8_process1.json) and
[process2](../results/m0b_final_screen_20261017/m8_process2.json) retain all120 raw
Event/Wall rows, all controls/gains and numeric oracle results. RAW reductions
are descriptive and INVALID, not optimization gains or statistical significance.
Raw snapshots/logs/source copies and receipt index stay local under
`m0_receipts/m0b_final_screen_20261017/`; manifest pins source/index hashes.
`formal_approved=false`. The5% quick gate does not amend formal M1/M2 contracts.
