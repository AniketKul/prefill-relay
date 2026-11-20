# CUDA graph decode with reserved KV storage

On the tested H100, batch one met the 40 decode-token/s calibration criterion across all five context lengths. Its slowest median was 23.865 ms per step, or 41.90 tokens/s per sequence. Batches 2, 3 and 4 exceeded 25 ms per step at one or more longer contexts. A separate 25-token/s planning target selected batch 4.

Each full-attention layer allocates room for the prefix plus the requested continuation when decode starts. It copies the prefix once, appends new keys and values in place, and supplies attention with a view limited to the current logical sequence length. Recurrent layers continue to use their existing native state. Stored prefixes stay available to other requests. Reserving the full continuation moves that memory allocation to the start of decode. The mode is opt-in and requires CUDA feed-forward graphs.

```bash
uv run --locked --extra trained --extra cuda-kernels python -m prefill_relay.calibration \
  docs/results/h100-profile-final.json --decode-graphs --reserve-decode-kv \
  --output docs/results/local-run/combined-decode-calibration.json
uv run --locked python -m prefill_relay.capacity_study docs/results/local-run/combined-decode-calibration.json \
  --output docs/results/local-run/combined-capacity-study.json
```

The [device measurements](results/h100-combined-decode-calibration.json) record three warm continuations per shape, with 32 model decode steps per continuation. Each timing includes one restoration of native state and finishes after synchronization. Assembly, network, storage, queueing and graph capture occur outside that timed interval. These are H100 backend measurements. The [capacity study](results/h100-combined-capacity-study.json) uses their worst context-specific median at the selected batch size to estimate capacity under the specified workload.

| Input tokens | Batch 1, ms/step | Batch 2 | Batch 3 | Batch 4 |
|---:|---:|---:|---:|---:|
| 128 | 23.45 | 24.96 | 24.95 | 24.93 |
| 1024 | 23.38 | 24.75 | 24.92 | 24.76 |
| 8192 | 23.40 | 24.69 | 24.75 | 24.68 |
| 32768 | 23.54 | 29.39 | 25.82 | 25.99 |
| 131072 | 23.86 | 25.09 | 25.35 | 25.43 |

The trained CUDA regression compares every decode-step logit and output token to eager inference for four batch sizes with distinct inputs. It also checks that each original prefix tensor remains unchanged. The [GPU tests](../tests/test_mlp_graphs.py) passed for both graph modes. Separate tests cover reserved-layer mask lengths, fixed backing storage and capacity overflow. End-to-end serving rates include service overhead and depend on the workload.
