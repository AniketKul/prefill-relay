# prefill-relay

prefill-relay can process an LLM prompt on one machine and continue generating tokens on another. It chooses where to run prefill based on available bandwidth and cached prompt state, transfers hybrid attention state over TCP, and batches compatible requests for local decoding.

## Demo

Watch an H100 start counting and a Mac finish the sequence. On the two repeat requests, the Mac reuses all 14 prompt tokens and generates the same answer with zero cross-host KV bytes. The 52-second demo includes AI-generated narration.

https://github.com/user-attachments/assets/442b9afb-09d0-4f4c-82b9-c7ff93bcd951

The benchmark defaults to CPU workers on one host. In the single-host H100 runs, prefill and decode workers share one GPU. The coordinator can also launch prefill on a separate SSH host. Remote traffic uses TCP with an aggregate bandwidth budget. On a host with a configured RDMA interface, local workers can hand state to one another over RoCE.

[H100 and two-host validation](docs/h100-followup.md) records the measured limits, raw results and tested source revisions.

## Quick start

Python 3.10 is selected by the project. Use `uv`; the lockfile pins dependencies.

```bash
git clone https://github.com/AniketKul/prefill-relay.git
cd prefill-relay
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
uv run --locked prefill-relay search config/example.json \
  --output docs/results/local-run/illustrative
```

The core scheduling, storage, transport, and capacity modules use the standard library. Tests needing optional model dependencies skip when those dependencies are absent. The bundled search configuration is explicitly synthetic; its output is an analytical estimate.

## Trained model on an H100

The trained backend supports Qwen3.5 text checkpoints. The checked-in measurements use `Qwen/Qwen3.5-0.8B`, bf16, and immutable model revision `2fc06364715b967f1860aea9cf38778875588b17`. The trained extra pins PyTorch 2.6.0, Transformers 5.18.0, and Linux linear-attention kernels.

```bash
uv sync --locked --extra trained
uv run --locked --extra trained prefill-relay bench \
  --backend hf --device cuda --model Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --profile docs/results/pretrained-profile.json \
  --scenario validate --lengths 64 --output-tokens 8 \
  --bandwidth-mib 1000 --output docs/results/local-run/validate.json
```

This starts independent model workers, transfers real state, and compares generated tokens against an unsplit reference. Validation exercises cold and complete cache hits, prefix continuation, local and remote prefill, concurrent decoding, and link unavailability.

The optional [CUDA graph decoder with reserved KV](docs/reserved-decode.md) sustained at least 41.90 tokens/s at batch one in the five-length H100 backend calibration. With one request in flight, the three serving policies generated identical 1,024-token continuations at 41.18–41.95 decode tokens/s. Under the long-prompt arrival workload, zero of nine optimized requests sustained 40 tokens/s. [Serving results](docs/h100-followup.md)

An optional [CUDA convolution extra](docs/cuda-kernels.md) provides a tested source build against the locked Torch ABI.

For a smaller CPU or GPU correctness backend:

```bash
uv sync --locked --extra gpu
uv run --locked --extra gpu prefill-relay bench --backend tiny --device cpu \
  --scenario validate --lengths 67 --output docs/results/local-run/tiny.json
```

The tiny model is deterministic and untrained, with three recurrent attention layers and one full-attention layer. It is limited to 8192 tokens. The older `demo` command remains available for its focused tensor-handoff checks.

## What runs

| Capability | Implementation |
|---|---|
| Independent prefill/decode processes and complete-hit bypass | [`runtime.py`](src/prefill_relay/runtime.py), [`worker.py`](src/prefill_relay/worker.py) |
| Trained hybrid inference, bounded prefill chunks, tensor codecs and cache deltas | [`hf_backend.py`](src/prefill_relay/hf_backend.py), [`hf_state.py`](src/prefill_relay/hf_state.py) |
| SQLite persistence, bounded RAM cache and shared transfer allocation | [`storage.py`](src/prefill_relay/storage.py), [`cache_service.py`](src/prefill_relay/cache_service.py) |
| Cache affinity, missing-block migration and hotspot replication | [`directory.py`](src/prefill_relay/directory.py) |
| Layer/chunk streaming, checksums, ACK cleanup and Linux TCP telemetry | [`transport.py`](src/prefill_relay/transport.py) |
| Aggregate bandwidth reservations shared by worker processes | [`rate_limit.py`](src/prefill_relay/rate_limit.py) |
| Cache-aware threshold updates and drained role changes | [`scheduler.py`](src/prefill_relay/scheduler.py), [`telemetry.py`](src/prefill_relay/telemetry.py) |
| Compatible-request decoder batching | [`batch_decode.py`](src/prefill_relay/batch_decode.py) |
| Measured profiling and uniform, burst, skew, changing-bandwidth and log-normal workloads | [`hf_profile.py`](src/prefill_relay/hf_profile.py), [`workload.py`](src/prefill_relay/workload.py), [`bench.py`](src/prefill_relay/bench.py) |

See [the original H100 measurements](docs/validation.md), [multi-site routing](docs/federation.md), [architecture](docs/architecture.md) for invariants and [implementation and operations](docs/implementation.md) for profiling commands, workload metrics, test coverage and evidence boundaries.

## Measurements and limits

[The final pretrained profile](docs/results/h100-profile-final.json) records H100 measurements from 1 through 131072 input tokens with native-cache continuation and the optional CUDA convolution. [The batch profile](docs/results/pretrained-batches.json) records batches of 2 and 4 through 8192 tokens. These are backend compute measurements; service benchmarks also include cache access, CPU staging, serialization, transfer and control overhead.

[The initial trained validation](docs/results/pretrained-initial.json) preserved transported state and decode logits exactly. Recomputing the prompt from a saved bf16 prefix produced a maximum logit difference of 0.15625 in that case, while preserving the next token. Exact wire transport does not imply bitwise equality between different floating-point execution shapes.

The [two-host launcher](docs/ssh-worker.md) supports a remote producer with its own device and prefill profile. The [capacity study](docs/capacity-study.md) measures decode batch sizes and compares projected throughput, egress and latency at equal cost. The [dense and hybrid profiles](docs/dense-profile.md) measure actual cache bytes and prefill time at four lengths through 128K. Independent GPU fleets, a physical WAN and production availability remain outside the measurements.

Control and cache services use bearer authentication on loopback. Raw tensor transport assumes a trusted network and provides neither authentication nor encryption. Keep these services private; no internet-facing serving interface is provided.
