# H100 and two-host validation

The results below use one H100 PCIe 80 GB. The trained serving runs use Qwen3.5-0.8B in bf16 at revision `2fc06364715b967f1860aea9cf38778875588b17`. Local and upstream CUDA workers share this GPU. The separate-host test runs prefill on the Linux H100 and decode on a Mac CPU.

| Measurement | Observed result |
|---|---|
| [Physical two-host trained handoff](results/two-host-trained.json) | All eight generated tokens matched unsplit H100 inference. The cold handoff carried 20,419,660 bytes. A decoder cache hit reproduced the same tokens with zero cross-host KV bytes. |
| [Narrated two-host counting demo](../README.md#demo) | A separate 14-token prompt transferred 20,444,644 KV bytes on the cold request. The Mac reused the full prompt cache on both repeats, including one with KV egress capped at 0 B/s. All three runs matched the nine output tokens from the H100 reference. |
| [Local RoCE and remote TCP](results/followup-service.json) | A 67-token cold request opened four TCP connections. Local prefill used four host-memory RoCE connections on the H100 host. The 33-token partial hit continued correctly. Four concurrent complete hits decoded as a batch of four. |
| [128K end-to-end handoff](results/followup-long.json) | The cold 131,072-token request transferred 1,656,048,772 bytes, with 74.77 s to first token. All four generated tokens matched unsplit inference. |
| [Incremental migration](results/followup-migration.json) | Extending an 8,192-token destination checkpoint to 16,384 tokens transferred 161,472,631 bytes. A cold migration of the same state used 295,754,359 bytes. |
| [Dense and hybrid prefill](dense-profile.md) | At 131,072 tokens the dense 2B model produced 10.00 GiB of state and the hybrid 0.8B model 1.52 GiB. Model sizes and architectures differ. |
| [CUDA graphs and reserved KV](reserved-decode.md) | Batch one reached at least 41.90 tokens/s at every sampled length in the independent 32-step calibration. All batches 1–4 preserved the tested logits, tokens and original checkpoints. |
| [Feed-forward CUDA graphs](decode-graphs.md) | The independent batch-one calibration exceeded 40 tokens/s at 128, 1,024, 8,192 and 32,768 tokens. The 131,072-token median was 32.55 tokens/s. |
| [Eager decode calibration](capacity-study.md) | Across five context lengths and batches 1–4, no batch met 40 tokens/s at every length. At batch 1 the slowest median was 39.03 tokens/s. Batch 4 qualified at 25 tokens/s. |

The [service](results/followup-service-receipt.json), [long-context](results/followup-long-receipt.json) and [migration](results/followup-migration-receipt.json) receipts identify the tested source, commands and raw output by SHA-256. The [capacity study](capacity-study.md) uses single-device measurements to calculate fleet throughput and queue latency. The local RoCE test uses a single server. Geographically separate GPU clusters, production traffic and availability are outside these measurements.

## Eager long-generation serving workload

Three seeded prompts contained 49,377, 37,246 and 15,766 tokens. Every policy saw identical input tokens and scheduled arrivals. Each request produced exactly 1,024 tokens, and outputs matched across policies for all three prompts. The run used a 1,000 MiB/s cross-cluster budget, local RoCE and a batch cap calibrated at 25 tokens/s. The actual decode goodput threshold was 40 tokens/s. [Raw measurements](results/followup-lognormal.json) · [Tested source and invocation](results/followup-lognormal-receipt.json)

| Policy | Completed | Reached 40 tokens/s | Mean / P90 TTFT (s) | Cross-cluster bytes |
|---|---:|---:|---:|---:|
| All local | 3 | 1 | 57.93 / 76.06 | 0 |
| All remote | 3 | 0 | 52.35 / 75.26 | 1,338,625,140 |
| Selective | 3 | 1 | 41.29 / 56.67 | 1,338,625,140 |

Selective routing chose remote prefill on all three long prompts. Timings include shared-GPU contention, service overhead and any compilation for newly encountered shapes. Three observations per policy do not establish a comparative performance advantage.

## Long-generation workload with CUDA graphs and reserved KV

The optimized service used the same arrivals, prompts, output length, link budget and local RoCE transport as the eager workload. Its 40-token/s calibration admitted batch one. All nine requests completed and produced exactly the same 9,216 output tokens as their eager counterparts. None sustained 40 tokens/s in this serving run. Service decode time includes batch queueing and waiting for the compute lock. Multiple worker processes share the GPU. [Raw per-request results](results/followup-combined-lognormal.json) · [Tested source and invocation](results/followup-combined-lognormal-receipt.json)

| Policy | Reached 40 tokens/s | Mean decode tokens/s | Mean / P90 TTFT (s) | Cross-cluster bytes |
|---|---:|---:|---:|---:|
| All local | 0 of 3 | 25.29 | 94.31 / 147.84 | 0 |
| All remote | 0 of 3 | 27.97 | 49.70 / 66.48 | 1,338,625,140 |
| Selective | 0 of 3 | 28.40 | 32.04 / 46.88 | 1,338,625,140 |

Both remote policies again chose the same placement for these three prompts. The eager and reserved-KV configurations used different calibrated batch caps. These measurements do not isolate the effect of the decode kernel.

## One request in flight

Each policy served one 1,024-token prompt and generated 1,024 output tokens with the same calibrated decoder. The three continuations matched exactly. All-local sustained 41.47 decode tokens/s, all-remote 41.18 and selective 41.95. These are measured per-request decode rates after the first token. The different prompt lengths and arrival concurrency in the long-generation benchmark prevent attributing the difference to concurrency alone. [Raw measurements](results/followup-combined-single.json) · [Tested source and invocation](results/followup-combined-single-receipt.json)

## Arrival workloads with the calibrated decoder

All 180 requests completed. Within each scenario, each policy received the same sequence of 12 prompts and generated eight tokens per request. The three policies produced the same output for every matched input. Nineteen requests sustained at least 40 decode tokens/s, including 12 complete-cache-hit requests in the warm scenario. [Per-request measurements](results/followup-combined-suite.json) · [Source and command](results/followup-combined-suite-receipt.json)

| Scenario | Requests at 40 tokens/s, local / remote / selective | P90 TTFT (s), local / remote / selective | Cross-cluster MB, remote / selective |
|---|---:|---:|---:|
| Uniform | 0 / 2 / 1 | 9.87 / 4.29 / 10.20 | 310.33 / 56.51 |
| Burst | 1 / 1 / 0 | 11.22 / 6.06 / 11.00 | 310.33 / 33.05 |
| Skew | 0 / 1 / 0 | 12.57 / 6.36 / 13.11 | 106.08 / 279.33 |
| Bandwidth | 0 / 1 / 0 | 10.23 / 33.15 / 15.43 | 310.33 / 56.51 |
| Warm | 6 / 0 / 6 | 0.16 / 0.18 / 0.19 | 0.00 / 0.00 |

In the bandwidth-drop scenario selective routing reduced egress from 310.33 MB to 56.51 MB relative to all-remote. All-local produced the lowest P90 TTFT in that scenario. For skewed input, selective routing transferred more bytes than all-remote.

## Correctness and packaging

Across the H100 and Mac, all 127 distinct tests ran successfully. The [H100 suite](results/followup-tests.log) passed 124 and skipped the three SSH tests, which passed on the Mac. This covered the trained six-worker federation, cross-site prefix routing, RoCE, eager/graph logit parity and KV ownership. The [H100 receipt](results/followup-tests-receipt.json) identifies the exact source and configuration.

The Mac trained CPU run passed 121 of 127 tests and skipped six device or deployment checks. The dependency-free installation passed 80 and skipped 47. Both cross-host checks subsequently passed with H100 prefill and CPU decode. The `uv build --no-sources` wheel and source archive each contained all 34 package modules, byte-identical to their source files. [CPU test log](results/followup-mac-cpu.log) · [Core test log](results/followup-core.log) · [Two-host trained test](results/followup-physical.log) · [Two-host tiny test](results/followup-physical-tiny.log) · [Archive verification](results/h100-followup-package.json)
