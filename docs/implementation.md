# Implementation and operations

Use `uv sync --locked --extra trained` for the trained backend. The project selects Python 3.10; `uv.lock` records package resolution. GPU measurements here use Linux, PyTorch 2.6.0 with CUDA 12.4, Transformers 5.18.0, and linear-attention kernels 0.4.1.

## Reproduce profiles

Pass an immutable model revision to bind a run to its weights and configuration:

```bash
uv run --locked --extra trained prefill-relay profile \
  --model Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --device cuda --dtype bfloat16 --lengths 1 64 256 1024 8192 32768 131072 \
  --batches 1 --repeats 3 --chunk-size 4096 --validate \
  --output docs/results/local-run/profile.json
```

Each shape receives an untimed prefill warmup. Current profiles restore a native decode cache once per continuation and average up to eight decode steps per timing sample; earlier profiles measured individual restored-cache steps. Output includes all timing samples, medians, raw attention-state bytes, peak device allocation, model revision and environment. `kv_bytes` excludes token IDs, cached logits, serialization headers and transport overhead. The dense-cache estimate in this command uses the model configuration. A [separate dense-model run](dense-profile.md) measures actual prefill time and cache bytes.

The [final single-request profile](results/h100-profile-final.json), using the optional CUDA convolution, contains these observed medians:

| Input tokens | Prefill seconds | Raw attention state, MiB |
|---:|---:|---:|
| 1024 | 0.04227 | 30.84375 |
| 8192 | 0.11067 | 114.84375 |
| 32768 | 0.76389 | 402.84375 |
| 131072 | 8.91948 | 1554.84375 |

Those timings come from one H100 PCIe running the pretrained 0.8B model. To reproduce this configuration, add `--extra cuda-kernels` to the commands above after following [the CUDA build instructions](cuda-kernels.md). They are not service TTFT. [Separate batch measurements](results/pretrained-batches.json) cover batches of 2 and 4 at lengths 1, 64, 1024 and 8192. A batch-only file has no batch-1 interpolation profile and cannot be supplied directly to the runtime controller.

## Run serving workloads

Use the same revision and a profile spanning every possible positive uncached suffix, including length 1:

```bash
uv run --locked --extra trained prefill-relay bench \
  --backend hf --device cuda --model Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --profile docs/results/pretrained-profile.json \
  --scenario uniform --lengths 64 256 1024 --requests 12 \
  --output-tokens 8 --rate 2 --bandwidth-mib 100 \
  --output docs/results/local-run/uniform.json
```

`uniform`, `burst`, `skew` and `bandwidth` run all-local, all-remote and selective policies on identical input sequences. Every worker first warms the requested full prompt shapes and decode batches 1, 2 and 4; prefix caches are then reset. Cache reset and an explicit initial bandwidth give each policy the same starting conditions. Newly encountered suffix shapes can still compile kernels. `--warm-rounds 1` performs an unmeasured warm round after reset. Complete local decoder hits bypass prefill under every policy; the policy determines the route when prefill is needed.

`--scenario suite` runs uniform, burst, skew, changing-bandwidth and warm-cache comparisons in one deployment. Uniform, burst and bandwidth inputs have distinct final tokens to keep real prefill work in the arrival stream.

The `lognormal` scenario samples prompt lengths from the truncated log-normal distribution and runs each policy on the same seeded requests. Use a measured profile spanning the sampled lengths. For example:

```bash
uv run --locked --extra trained --extra cuda-kernels prefill-relay bench \
  --backend hf --device cuda --model Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --scenario lognormal --profile docs/results/h100-decode-calibration.json \
  --calibrate-tps 25 --requests 3 --output-tokens 1024 --block-tokens 256 \
  --rate 0.1 --bandwidth-mib 1000 --output docs/results/local-run/lognormal.json
```

The calibration target chooses an admissible decode batch limit. The workload independently scores per-request goodput against 40 decode tokens/s. The controller counts complete cache hits in decode demand and only charges misses to the prefill and egress stages.

The burst scenario groups arrivals while retaining the specified average interval. Skew repeats a deterministic hot prompt. Bandwidth drops to one twentieth of its initial value halfway through the offered arrival span. These are controlled generated workloads rather than recorded production traffic.

`--scenario long --lengths 1024 8192 32768 131072` runs independent-process remote handoff for each length and checks generated tokens against the reference. `--cache-gib` sets each worker's reusable and staged-transfer cache budget, defaulting to 4 GiB. The receiver has an independent 4 GiB payload bound; Python callers can set `receive_bytes`. `--block-tokens` selects prefix-block granularity, defaulting to 16. Full blocks consume their actual rounded byte units, and distinct numerical variants retain separate content identities. Long runs also need RAM for model state, codec copies and database activity. `--remote-workers` changes the number of logical prefill processes; it does not add GPUs.

The CLI creates temporary cache databases and removes them at exit. For persistent reuse across coordinator lifetimes, instantiate `Coordinator(path=...)` with an application-owned directory and unchanged model/cache configuration. Process spawning requires the usual `if __name__ == '__main__':` guard in executable Python scripts.

### Metrics

The workload runner schedules arrivals independently of completion and bounds the pending queue. Queue overflow and request failures are counted explicitly. It records scheduled arrival, actual submission, worker start, queue wait and completion for every request.

- Throughput is completed requests divided by the interval from the first scheduled arrival to final completion.
- Mean/P90 TTFT includes dispatch delay and executor waiting, then coordinator-observed first-token readiness.
- `transferred_bytes` counts cross-cluster state and migration bytes. Local prefill-to-decode traffic is retained in sender statistics but excluded from this cross-cluster metric.
- Output throughput counts all generated tokens. Per-request decode speed uses `(output_tokens - 1) / decode_s`, excluding the first token produced by prefill.
- The trained decoder restores the native model cache once per continuation, preserving the stored checkpoint. Runtime `decode_s` includes batch coalescing and compute-lock waiting as well as synchronized decode compute. Batch size and batch queue time are retained separately.
- SLO goodput counts completed requests meeting the decode speed threshold and optional TTFT threshold. The runner defaults to 40 decode tokens/s; a one-token output has no decode interval.

The `pipelined` field reports observed overlap between transmission and prefill. Very short prefills may finish before the sender thread is scheduled; this does not prevent layer streaming.

JSON artifacts include configuration, model revision, worker process IDs, per-request placement and transfer data, cache/role statistics, and an input fingerprint. Compare raw artifacts before drawing performance conclusions: logical clusters contend for the same GPU, and loopback application pacing does not reproduce separate machines or network paths.

## Verification map

These are the relevant test suites and validation entry points. Listing a suite identifies coverage, not a claim that every optional environment was executed.

| Behavior | Source | Tests or validation |
|---|---|---|
| Capacity equations, aggregate egress, strict thresholds and allocation search | `analytics.py`, `scheduler.py` | `test_analytics.py`, `test_control.py` |
| Actual cache-hit distributions and complete-hit separation | `empirical.py` | `test_workload.py` |
| Persistent cache ownership, restart recovery, integrity, shared capacity, partial blocks and RAM eviction | `storage.py` | `test_storage.py` |
| Authenticated node access, migration budgets/deltas and rebalancing | `cache_service.py`, `directory.py` | `test_cache_service.py` |
| Framing, chunking, retries, bounded buffers, ACK and abort cleanup | `transport.py` | `test_transport.py`, `test_telemetry.py` |
| Shared cross-process rate reservations and outages | `rate_limit.py` | `test_rate_limit.py` |
| Drained worker role transitions | `telemetry.py` | `test_telemetry.py`, `test_runtime.py` |
| Tensor codec validation, hybrid state reconstruction and pretrained parity | `hf_state.py`, `hf_backend.py`, `hf_decode.py` | `test_hf_state.py`, `test_hf_backend.py`, `test_hf_decode.py`, `profile --validate` |
| Compatible decoder batching, error propagation and drain-on-close | `batch_decode.py` | `test_batch_decode.py` |
| Independent worker handoff, cache-hit bypass and actual routing | `runtime.py`, `worker.py` | `test_runtime.py`, `bench --scenario validate` |
| Arrivals, overload, warm/cold fairness, bandwidth changes and SLO metrics | `workload.py` | `test_workload.py`, `bench` scenarios |

```bash
uv run --locked --extra trained python -m unittest discover -s tests -v
PREFILL_RELAY_TRAINED_MODEL=Qwen/Qwen3.5-0.8B PREFILL_RELAY_TEST_DEVICE=cuda \
  uv run --locked --extra trained python -m unittest discover -s tests -p 'test_hf*.py' -v
```

Opt-in trained tests load the named model from its current default revision; use the pinned `profile --validate` command when reproducing a specific checkpoint. Existing receipts in `docs/results` identify the commands and source hashes for their own runs. Earlier tiny-backend receipts do not certify subsequently added service code.

[Initial pretrained validation](results/pretrained-initial.json) found exact transported tensors, exact decode logits after handoff, exact reconstruction of producer state from a delta, and zero model calls on a complete prompt hit. Prefix recomputation in bf16 had maximum logit difference 0.15625 while preserving the tested next token. That observation is case-specific; it does not guarantee identical tokens for arbitrary inputs or floating-point execution shapes.

[Final validation](validation.md) links the complete source-bound GPU run, subsequent compatibility checks, real migration and arrival-workload results.
