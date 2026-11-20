# Local RDMA handoff

The coordinator can use Linux rdma-core rsockets for the local prefill-to-decode handoff. Select a configured RDMA interface with an IPv4 address and an active port. The receiver accepts peers on the same host address. The control and cache services continue to use loopback. The remote prefill path uses TCP.

With the locked trained environment installed and the interface address exported as `PREFILL_RELAY_RDMA_ADDRESS`, run the validation on the H100 host.

```bash
uv run --locked --extra trained --extra cuda-kernels prefill-relay bench \
  --backend hf --device cuda --model Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --local-rdma --scenario validate --lengths 67 --output-tokens 8 \
  --output docs/results/local-run/rdma.json
```

The result records `handoff_transport`, the sender's acknowledged frame count and the returned tokens. It omits the interface address. The transfer stages model state in host memory, applies the common frame checksums and waits for acknowledgements. The option must successfully open RDMA connections to complete a local handoff. These checks exercise same-host RoCE. GPU-direct DMA and switch throughput require separate measurements.
