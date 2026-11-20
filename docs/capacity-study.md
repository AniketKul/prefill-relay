# Capacity and latency for a long-context workload

Run the batch calibration on an H100 with the trained model, then run the capacity calculation on any CPU. The first input records the original single-request warm prefill timings and cache sizes. Calibration checks the model revision, dtype, GPU type, Torch version and convolution kernels, and preserves the input file's SHA-256.

```bash
uv run --locked --extra trained --extra cuda-kernels python -m prefill_relay.calibration \
  docs/results/h100-profile-final.json --output docs/results/h100-decode-calibration.json
uv run --locked python -m prefill_relay.capacity_study docs/results/h100-decode-calibration.json \
  --output docs/results/h100-capacity-study.json
```

The first command materializes the pretrained hybrid state for each prompt length, copies it to form a contiguous batch, then times native continuation on the GPU. The four batch sizes are 1, 2, 3 and 4. Each shape gets one warmup and three timed continuations of 32 decode steps. Each step produces one new token per sequence. State preparation and batch assembly finish before timing begins. Cache restoration occurs once per continuation. The calibration selects the largest measured batch whose median step time meets the target at all five sampled context lengths. Missing shapes disqualify a batch. The result applies to those sampled lengths and backend settings. The [service benchmarks](validation.md#arrival-workloads) include additional queue and compute-lock waiting.

The second command models 4 remote prefill and 8 local instances using the same H100 profile for every instance. The uncached input length follows a log-normal distribution with log-space mean 9.90 and standard deviation 1.00, conditioned on lengths between 128 and 128,000 tokens. All requests generate 1,024 tokens. The shared link budget is 100 decimal Gbit/s. The throughput search evaluates integer local prefill allocations and thresholds spaced 256 tokens apart, including the two routing endpoints. Stage service times use the mean length on each side of the threshold and linear interpolation between measured profile points. The report includes remote, local and decode capacity, required egress at the chosen request rate, raw compute-saturated egress demand, and the threshold and allocation sweeps through the optimum.

To model different prefill hardware or kernels, pass `--remote-profile FILE` to the capacity calculation. Supply a prefill profile for the same model revision and dtype covering the full requested length range. The remote profile supplies compute timings and cache sizes for all offloaded requests, including the transfer queue and compute-saturated egress estimate. Local prefill, all decode calibration and the two local-only baselines use the local calibration file. The report includes the remote input digest and device. The default uses the local prefill profile for both pools.

The homogeneous baseline assigns the entire instance count to the local cluster and searches its prefill/decode split. The all-remote baseline dedicates all local instances to decode. A second homogeneous baseline spends the combined resource budget on whole local instances. `--remote-unit-cost` and `--local-unit-cost` specify comparable costs per instance-hour. Both default to one, making the count-matched and budget-matched homogeneous baselines coincide. Pricing and independent-instance scaling are explicit user assumptions.

The modeled latency columns use the same 4,096 seeded draws and uniformly spaced arrivals for each policy. The common offered load is 80% of the slowest estimated capacity. Each producer handles one prompt at a time. Completed remote prefills serialize on one shared link, and producers can begin the next request while prior cache transfers complete. Decode admission uses independent slots under ideal continuous batching. TTFT is the time from arrival to decode admission with the first token ready. This queue model uses whole-request transfer, raw KV byte counts and interpolated prefill times. It excludes layer overlap within a request, CPU staging, cache lookup, storage and protocol overhead. Actual TTFT has separate [service measurements](validation.md).

## Results on the H100 PCIe

The slowest measured median across 128, 1,024, 8,192, 32,768 and 131,072 tokens was as follows for each batch.

| Batch | Worst median step, ms | Per-sequence tokens/s at that shape |
|---:|---:|---:|
| 1 | 25.62 | 39.03 |
| 2 | 28.00 | 35.72 |
| 3 | 29.34 | 34.08 |
| 4 | 29.82 | 33.54 |

With eager decode at 40 tokens/s, no tested batch met the target across every sampled context. This eager profile therefore has no calibrated throughput or TTFT estimate at that target. The [reserved-KV CUDA graph profile](reserved-decode.md) qualifies batch one at 40 tokens/s and produces a separate capacity estimate.

At 25 tokens/s, batch 4 satisfied the isolated decode criterion. The capacity calculation evaluated 4,008 threshold/allocation pairs. All policies receive the same modeled 0.838 requests/s in the TTFT simulation.

| Policy | Remote / local prefill / decode | Threshold, tokens | Capacity, requests/s | Egress, Gbit/s | Mean / P90 TTFT, s |
|---|---|---:|---:|---:|---:|
| selective | 4 / 0 / 8 | 0 | 1.048 | 2.980 | 1.104 / 3.318 |
| all remote | 4 / 0 / 8 | 0 | 1.048 | 2.980 | 1.104 / 3.318 |
| all local same count | 0 / 1 / 11 | 128,000 | 1.441 | 0.000 | 10.689 / 26.077 |
| all local same budget | 0 / 1 / 11 | 128,000 | 1.441 | 0.000 | 10.689 / 26.077 |

The [decode measurements](results/h100-decode-calibration.json) contain every timing sample. The [capacity report](results/h100-capacity-study.json) records all assumptions, source and input digests, and search cross-sections. Capacity and TTFT in the table are projections for separate H100 instances. Physical deployment, throughput and latency across those instances remain unmeasured.
