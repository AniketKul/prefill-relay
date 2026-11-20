# CUDA graphs for feed-forward decode blocks

`--decode-graphs` captures the feed-forward block in each of the 24 layers for a given decode batch size. The graph replays fixed-shape matrix operations on the worker. Attention, cache updates and token selection use the ordinary inference path. Prefill and batches above four use eager execution. Captures belong to a model instance and grow lazily as it sees batch sizes 1 through 4. Eager execution remains the default.

The following commands regenerate the graph calibration and the resulting capacity estimates.

```bash
uv run --locked --extra trained --extra cuda-kernels python -m prefill_relay.calibration \
  docs/results/h100-profile-final.json --decode-graphs --output docs/results/local-run/graph-decode-calibration.json
uv run --locked python -m prefill_relay.capacity_study docs/results/local-run/graph-decode-calibration.json \
  --output docs/results/local-run/graph-capacity-study.json
```

The [paired experiment](results/h100-mlp-graph-trial.json) alternates eager and graph generation four times per shape on one loaded H100 model. At all three context lengths, every one of the 32 decode logits and every output token matched bitwise. These are medians in milliseconds per generated decode step.

| Prefix | Decode steps | Eager | Graph |
|---:|---:|---:|---:|
| 128 | 7 | 25.14 | 24.21 |
| 128 | 32 | 24.89 | 24.12 |
| 1024 | 7 | 25.29 | 24.47 |
| 1024 | 32 | 24.88 | 24.56 |
| 131072 | 7 | 27.57 | 28.84 |
| 131072 | 32 | 25.14 | 24.31 |

The first eight-token graph request for batch 1 took 185 ms including captures. It ran after an eager request on the same loaded model and reserved an additional 82 MiB in PyTorch's allocator. Restarting a worker discards its graphs. The [regression tests](../tests/test_mlp_graphs.py) exercise changing inputs on different CUDA streams, saved outputs, immutability of the original prefix and every greedy decode logit at batches 1, 2, 3 and 4.

The independent [20-shape graph calibration](results/h100-graph-decode-calibration.json) found batch 1 above 40 tokens/s at 128 through 32,768 sampled tokens. At 131,072 its three samples were 27.50, 31.08 and 30.72 ms per step. The five-context median-based criterion therefore found no qualifying batch at 40 tokens/s. Batch 4 qualified at 25 tokens/s. The [capacity estimates](results/h100-graph-capacity-study.json) preserve those outcomes. A compute result at a particular shape does not establish an end-to-end service SLO.

To repeat the paired experiment with the published script, run the following from the repository root.

```bash
mkdir -p docs/results/local-run
(cd docs/results/local-run && uv run --project ../../.. --locked --extra trained --extra cuda-kernels python ../../experiments/mlp_trial.py)
```
