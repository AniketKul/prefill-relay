# Multiple prefill and PD clusters

Each site contributes a local prefill/decode (PD) pool and an upstream producer pool. The federation registers every worker's cache client in one directory. A producer in any site can send KV state directly to a decoder in any PD site. Local prefill always uses the chosen PD site's local workers.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from prefill_relay.federation import Federation
from prefill_relay.runtime import Coordinator


def main():
    with TemporaryDirectory() as path:
        sites = {
            name: Coordinator(Path(path) / name, bandwidth_bytes_s=rate, threshold=16)
            for name, rate in [('east', 200 * 2**20), ('west', 100 * 2**20)]
        }
        with Federation(sites) as serving:
            result = serving.run(list(range(64)), output_tokens=4, pd_site='west')
            print(result['producer_site'], result['pd_site'], result['generated_tokens'])


if __name__ == '__main__':
    main()
```

This example starts six local CPU processes with the small backend. Each site's coordinator also accepts the trained backend, its own worker counts, measured profiles, and its own SSH producer configuration. PD receivers run on the coordinator machine. All sites must use the same model, precision and cache block geometry. Create new coordinators with distinct persistent cache directories and let the federation manage their lifetimes.

Placement first chooses a PD site. A complete match on a decoder takes priority, followed by longest local cached prefix and load. The optional `pd_site` pins this choice. For upstream selection the federation compares availability, observed egress congestion, matching producer prefix length, fraction of busy workers, egress queue time and configured rate. Within the selected pair the existing threshold controller chooses local or remote prefill and retains its cache migration, role changes and per-node admission controls. An existing eligible decoder checkpoint can complete locally with zero cross-cluster bytes, including during an egress outage. Reported TTFT includes global placement.

`producer_site` and `pd_site` identify the site for each side of the returned handoff. A complete decoder cache hit returns `producer_site=None`. The same producer workers, locks and source egress budget are shared by requests to all PD destinations. Bandwidth configurations represent explicit per-site outbound budgets. When circuits share a physical link, allocate per-site caps whose sum fits that link. A site's local and remote workers share its configured reservation clock for outbound transfers. Update a site's budget with `sites[name].set_bandwidth(bytes_per_second)`. The global directory names its two worker pools `name/local` and `name/remote`.

The six-worker CPU regression exercises both cross-site directions, global cache affinity, cross-site prefix migration, concurrent PD destinations sharing a producer, and zero-bandwidth fallback. It compares generated tokens with local reference inference and checks the source rate counters. This is a logical multi-cluster deployment on one host.


Run the regression on CUDA with the optional pinned trained-model handoff:

```bash
PREFILL_RELAY_FEDERATION_DEVICE=cuda PREFILL_RELAY_FEDERATION_TRAINED=1 \
  uv run --locked --extra trained --extra cuda-kernels \
  python -m unittest discover -s tests -p test_federation.py -v
```
