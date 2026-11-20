# Cache production on one H100

This benchmark runs two pretrained language models on the same GPU and measures how quickly each produces the attention state needed by a decoder. For each prompt, the program divides the resulting state size in bits by the median prefill time. The result is the decimal Gbit/s of link capacity that would be required to export that state in the same time. Actual network throughput is measured by the separate service benchmarks.

The dense model is Granite 3.3 2B Instruct with full grouped-query attention in every layer. The hybrid model is the Qwen3.5 0.8B text decoder with 18 recurrent layers and 6 full-attention layers. Each model runs in bf16 with the same batch size of one, 4096-token chunks, input IDs and PyTorch SDPA setting. Each chunk updates a native Transformers cache. The timer synchronizes the GPU and includes generation of the final-position logits for each chunk. Model loading, tensor byte accounting, serialization and transfer occur outside the timed operation. The implementation counts all populated attention tensors, including the recurrent states in their actual dtypes. Each shape receives one warmup and three timed samples.

| Tokens | Hybrid state (MiB) | Dense state (MiB) | Hybrid prefill (s) | Dense prefill (s) | Hybrid Gbit/s | Dense Gbit/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 30.84 | 80.00 | 0.0420 | 0.0317 | 6.163 | 21.162 |
| 8,192 | 114.84 | 640.00 | 0.1073 | 0.3706 | 8.976 | 14.485 |
| 32,768 | 402.84 | 2560.00 | 0.7492 | 3.7663 | 4.511 | 5.702 |
| 131,072 | 1554.84 | 10240.00 | 8.6828 | 50.0589 | 1.502 | 1.716 |

Run the comparison on a CUDA machine with the locked trained dependencies and optional convolution kernel installed.

```bash
uv run --locked --extra trained --extra cuda-kernels \
  python -m prefill_relay.kv_profile \
  --lengths 1024 8192 32768 131072 --chunk-size 4096 --repeats 3 \
  --output docs/results/local-run/kv-profile.json
```

The [H100 results](results/h100-kv-profile.json) record model revisions, parameter counts, context limits, measured bytes, every timing sample, peak GPU allocations and the SHA-256 digest of the profiling module. The two checkpoints differ in size, layer design and trained weights. This comparison measures their cache production under the specified settings and cannot isolate the effect of attention architecture on latency or model quality.
