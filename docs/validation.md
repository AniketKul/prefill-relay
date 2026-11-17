# Verified H100 execution

The implementation runs real trained hybrid inference through independent prefill and decode processes, durable prefix caches, layer streaming, adaptive placement and decoder batching. This validation uses one H100 PCIe with 80 GB and three model processes sharing that GPU. All network measurements use loopback and application bandwidth pacing.

## Exact inputs and checks

The trained model is `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17`, in bf16. The environment uses Python 3.10, uv, PyTorch 2.6.0/CUDA 12.4, Transformers 5.18.0, linear-attention kernels 0.4.1 and the optional source-built causal-conv1d 1.5.3.post1. See [CUDA build instructions](cuda-kernels.md).

The [complete run receipt](results/complete-verification.json) binds implementation `916ad39` to file hashes, environment, exact commands and six passing jobs: 90 tests, profile plus correctness checks, service validation, long-context handoff, migration and arrival workloads. The [supplemental receipt](results/supplemental-verification.json) binds `b6edc2d` to another full 90-test GPU pass and tiny-backend CUDA service validation. This small follow-up normalizes the tiny backend's CUDA device identity and replaces a timing-dependent overlap assertion with an actual-transmission check. The trained serving algorithms are unchanged. Documentation and evidence commits retain this tested source state.

The full GPU test suite runs with `PREFILL_RELAY_TRAINED_MODEL=Qwen/Qwen3.5-0.8B` and `PREFILL_RELAY_TEST_DEVICE=cuda`; no tests are skipped. Local CPU and dependency-free test runs are also retained. The package wheel and source archive build with `uv build --no-sources`.

| Check | Observed result | Evidence |
|---|---|---|
| Trained wire handoff | 24 layers, exact transported state and decode logits | [Profile validation](results/h100-profile-final.json) |
| Native decode cache | Every tested decode logit and token matches; stored checkpoint unchanged for distinct-prompt batches 1, 2 and 4 | [Native decode checks](results/native-decode-verification.json), full GPU tests |
| Independent service | Separate prefill/decode PIDs; four simultaneous requests decoded as a batch of 4; tokens match reference | [Service validation](results/h100-service-final.json) |
| Complete prefix hit | Zero transferred bytes, including during a link outage | [Service validation](results/h100-service-final.json) |
| Partial prefix and branches | 33-token prefix reused for a 67-token prompt; shared-prefix branches match reference | [Service validation](results/h100-service-final.json) |
| Cache ownership and failure paths | Restart recovery, corruption, eviction, admission failure, ACK/abort cleanup and rate-limit errors covered | [GPU tests](results/h100-supplemental-tests.log) |
| Controller and role changes | Cache-aware thresholds, bandwidth outage fallback and drained reassignment covered | [GPU tests](results/h100-supplemental-tests.log) |

The final prefix-recomputation check observed a maximum bf16 logit difference of 0.1875, with the same next token. Exact wire transport and native continuation do not imply bitwise equality across different prompt execution shapes.

## Long context and migration

These are three actual service requests, each checked against an unsplit reference. The service starts without a model-shape warmup, so the first request can include compilation. Every row recorded transfer beginning before the last layer was ready.

| Prompt tokens | Service TTFT, s | Transferred MiB | Generated tokens match reference |
|---:|---:|---:|---|
| 8192 | 16.682 | 116.83 | True |
| 32768 | 14.095 | 409.33 | True |
| 131072 | 57.737 | 1579.33 | True |

[Long-context artifact](results/h100-long-final.json). Block size was 256 tokens; the application bandwidth budget was 1000 MiB/s. Backend-only warm prefill at 128K took 8.919 seconds and peaked at 8.57 GiB allocated GPU memory. Service TTFT also includes CPU staging, serialization, SQLite and transfer; the two timings measure different work.

The [migration run](results/h100-migration.json) extended an existing 8192-token prefix to 16384 tokens. The delta transferred 161,472,631 bytes versus 295,754,359 bytes for a cold full copy (45.4% fewer bytes). These counts include the cache HTTP encoding, unlike raw attention-state sizes. All source transfer pins were released after the successful runs. Same-cluster and cross-cluster rebalancing have focused tests; there is no physical multi-cluster experiment.

## Arrival workloads

All-local, all-remote and selective policies receive identical inputs and arrivals within each scenario. Each worker warms the prompt shapes and decode batches 1, 2 and 4 before the measured runs; reusable caches reset between policies. The warm scenario additionally executes one unmeasured cache-warming round. Prompt lengths are 64, 256 and 1024, with 12 requests per policy, 2 offered requests/s and 8 output tokens. The initial aggregate egress budget is 100 MiB/s; the bandwidth scenario reduces it to 5 MiB/s. See [metric definitions](implementation.md#metrics).

All 180 measured requests completed with zero errors. 0 met the 40 decode-token/s SLO, which includes decoder batch/lock waiting. Successful execution is therefore separate from meeting this latency target.

| Scenario | Policy | Completed rps | Mean TTFT, s | P90 TTFT, s | Mean decode tokens/s | SLO passes |
|---|---|---:|---:|---:|---:|---:|
| uniform | all-local | 1.098 | 3.078 | 5.801 | 23.07 | 0/12 |
| uniform | all-remote | 1.080 | 3.055 | 4.399 | 26.85 | 0/12 |
| uniform | selective | 1.017 | 3.406 | 6.749 | 22.60 | 0/12 |
| burst | all-local | 1.035 | 4.010 | 6.737 | 20.98 | 0/12 |
| burst | all-remote | 0.986 | 4.584 | 8.789 | 23.93 | 0/12 |
| burst | selective | 1.043 | 4.626 | 8.404 | 21.68 | 0/12 |
| skew | all-local | 1.192 | 0.762 | 1.078 | 27.58 | 0/12 |
| skew | all-remote | 1.235 | 0.704 | 1.009 | 29.01 | 0/12 |
| skew | selective | 1.480 | 0.637 | 1.036 | 26.89 | 0/12 |
| bandwidth | all-local | 1.084 | 2.980 | 6.508 | 22.01 | 0/12 |
| bandwidth | all-remote | 0.225 | 20.506 | 44.131 | 23.73 | 0/12 |
| bandwidth | selective | 0.784 | 6.773 | 10.837 | 25.02 | 0/12 |
| warm | all-local | 2.022 | 0.145 | 0.194 | 29.24 | 0/12 |
| warm | all-remote | 2.026 | 0.133 | 0.181 | 28.77 | 0/12 |
| warm | selective | 1.014 | 2.672 | 6.383 | 19.56 | 0/12 |

[Final workload artifact](results/h100-workloads-final.json). A warm producer cache can still require local state transfer when role reassignment leaves the decoder without that checkpoint; zero cross-cluster bytes does not imply zero local TCP work. Policy differences are observations on one contended GPU and a short generated workload. They do not establish independent-fleet capacity or a consistent selective-placement speedup. The [earlier baseline](results/h100-workloads-baseline.json) and [its source hashes](results/workloads-baseline-source.json) are retained, but several implementation, kernel, profile and batching settings changed; this is not a single-variable performance comparison.

## Focused optimizations and limits

[Native continuation timings](results/native-decode-benchmark.json) compare the old per-token restoration path with one restoration per 32-token continuation on the same loaded model. The stored throughput counts all 32 output tokens, including the first token available from cached logits; it is a different metric from the workload decode SLO. [Convolution measurements](cuda-kernels.md) show context-dependent prefill gains. The [synthetic SSD benchmark](results/storage-microbenchmark.json) compares identical checkpoint payloads and durability on macOS; its storage improvement is not an inference speedup.

The supplied deployment launches workers on one host. Physical multi-host/WAN/RDMA behavior, heterogeneous GPU fleets, unavailable model weights and production availability are unverified. Directory metadata is in memory; cache data is persistent. Raw tensor transport assumes a trusted private network and is not authenticated or encrypted. These limits are part of the executable deployment's scope.
