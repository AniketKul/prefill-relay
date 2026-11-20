# Remote prefill over SSH

A coordinator can launch a prefill worker on a second host with OpenSSH. Install the same revision on both hosts and run `uv sync --locked --extra gpu` in both checkouts for the tiny model. For the trained backend, install its dependency extras instead. Configure noninteractive access to the SSH host and permit local and remote port forwarding. The account on the remote host needs `uv` on PATH or in `~/.local/bin`.

The coordinator opens local forwards to the remote worker's control and cache endpoints. Reverse forwards give the producer access to the coordinator's rate service and selected decoder's tensor receiver. All forwards listen on loopback. SSH carries the model state between machines. Multiple data streams through one SSH session share its outer TCP connection. Kernel telemetry on the inner sockets measures the loopback legs. The protocol therefore supports real remote handoff while physical link saturation and WAN loss require separate measurement.

By default the remote worker owns a temporary cache directory. Supply a dedicated remote cache directory to retain checkpoints across sessions. The launcher uses an existing authenticated OpenSSH master when available, otherwise it starts and owns a private master. Closing the worker's SSH input stops the remote process and releases its cache. Normal exit also cancels all forwards created by that worker.

To check a remote checkout directly, set `PREFILL_RELAY_SSH_HOST` to an SSH alias and `PREFILL_RELAY_SSH_PROJECT` to the checkout path on that host. The default path is `prefill-relay` relative to the SSH login directory. Set `PREFILL_RELAY_REMOTE_DEVICE=cuda` to use the remote GPU. The test decodes locally on CPU and compares with a local reference.

```bash
uv run --locked --extra gpu python -m unittest discover -s tests -p test_ssh_worker.py -v
```

Through the CLI, set the local decode device separately from the remote prefill device. The host alias below is a placeholder.

```bash
uv run --locked --extra gpu prefill-relay bench --backend tiny --device cpu \
  --remote-host gpu-lab --remote-project prefill-relay --remote-device cuda \
  --scenario validate --lengths 67 --output docs/results/local-run/ssh.json
```

The trained physical handoff check uses a fixed 12-token prompt and compares an eight-token cold and warm continuation against the unsplit remote producer. Set `PREFILL_RELAY_SSH_HOST` to the configured SSH alias and run the following on the CPU decoder host.

```bash
PREFILL_RELAY_TRAINED_MODEL=Qwen/Qwen3.5-0.8B PREFILL_RELAY_REMOTE_DEVICE=cuda \
  uv run --locked --extra trained python -m unittest discover -s tests -p test_cross_host_trained.py -v
```
