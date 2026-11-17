# Optional CUDA convolution

The `trained` extra runs correctly with the reference convolution. The optional `cuda-kernels` extra builds causal-conv1d 1.5.3.post1 against the same locked Torch version and ABI. It requires Linux, a CUDA 12 toolkit and a C++ compiler; the checked configuration uses CUDA 12.8 and g++ 11.4.

```bash
CUDA_HOME=/usr/local/cuda-12.8 uv sync --locked --extra trained --extra cuda-kernels
CUDA_HOME=/usr/local/cuda-12.8 uv run --locked --extra trained --extra cuda-kernels \
  python -c 'import torch, causal_conv1d, causal_conv1d_cuda; print(torch.__version__, causal_conv1d.__version__)'
```

Set `CUDA_HOME` to the installed CUDA 12 toolkit. Keep both extras on subsequent commands. The optional package is excluded on macOS. Use uv 0.11 or newer for the declared matched build dependencies.

The build is forced from source because the tested upstream Torch 2.6 binary wheels failed their import ABI check. The locked build dependencies and variables are in `pyproject.toml`; no custom wheel is distributed here. A clean Linux locked sync and a subsequent locked run were verified, as was resolution on macOS without CUDA. The first source build took about five minutes on the test host.

The [interleaved H100 measurement](results/causal-conv-interleaved.json) compares the two convolution implementations on the same loaded model after warmup:

| Prompt tokens | Reference prefill | CUDA prefill |
|---:|---:|---:|
| 64 | 39.88 ms | 40.86 ms |
| 1024 | 42.86 ms | 43.66 ms |
| 8192 | 127.51 ms | 111.88 ms |
| 32768 | 819.11 ms | 770.21 ms |

Decode timing remained about 27–29 ms. This is an optional optimization with a context-dependent tradeoff. [Correctness checks](results/causal-conv-comparison.json) covered actual trained layer handoff, prefix continuation, batched decoding and complete hits in both modes.

Profiles and service artifacts record the installed convolution version and dispatch module. Regenerate profiles when changing kernels; the original `pretrained-profile.json` used the reference convolution.
