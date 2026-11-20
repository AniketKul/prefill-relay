"""Pair native generation and decode-only feed-forward graphs on one model."""
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from prefill_relay import hf_backend, hf_decode, mlp_graphs


def timed(backend, state, n, graphed):
    torch.cuda.synchronize()
    start = perf_counter()
    if graphed:
        output = backend.generate(state, n)
    else:
        output = hf_decode.generate_continuation(backend, state, n)
    torch.cuda.synchronize()
    return output, perf_counter() - start


def run():
    model = 'Qwen/Qwen3.5-0.8B'
    revision = '2fc06364715b967f1860aea9cf38778875588b17'
    backend = hf_backend.HFHybrid(model, revision)
    backend.enable_decode_graphs()
    seed = backend.tokenizer('A reusable prefix reduces repeated computation. ', return_tensors='pt').input_ids[0]
    report = dict(model=model, revision=revision, device=torch.cuda.get_device_name(), torch=torch.__version__,
                  kernels=backend.kernel_provenance, rows=[], source_sha256={})
    for source in [Path(__file__), *[Path(m.__file__) for m in (hf_backend, hf_decode, mlp_graphs)]]:
        report['source_sha256'][source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    for length in (128, 1024, 131072):
        tokens = seed.repeat((length + seed.numel() - 1) // seed.numel())[:length].to(backend.device)
        _, state = backend.prefill(tokens)
        if length == 128:
            # First request includes graph creation for all 24 MLPs. Both paths see a real 8-token request.
            reference, eager_s = timed(backend, state, 8, False)
            memory_before = torch.cuda.memory_reserved()
            actual, graph_s = timed(backend, state, 8, True)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            report['first_eight_tokens'] = dict(eager_s=eager_s, graph_s=graph_s, count=len(backend._decode_graphs.graphs),
                tokens_equal=True, additional_reserved_bytes=torch.cuda.memory_reserved() - memory_before)
            assert report['first_eight_tokens']['count'] == 24
        traces = []
        for use_graph in (False, True):
            logits = []
            h = backend.model.register_forward_hook(lambda mod, args, result: logits.append(result.logits.clone()))
            try:
                actual, _ = timed(backend, state, 33, use_graph)
            finally:
                h.remove()
            traces.append((logits, actual))
        assert len(traces[0][0]) == len(traces[1][0]) == 32
        for a, b in zip(traces[0][0], traces[1][0]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(traces[0][1], traces[1][1], rtol=0, atol=0)
        del traces
        for n in (8, 33):
            samples = {'eager': [], 'graphs': []}
            for first in (False, True, False, True):
                for use_graph in (first, not first):
                    _, elapsed = timed(backend, state, n, use_graph)
                    samples['graphs' if use_graph else 'eager'].append(elapsed / (n - 1))
            row = dict(length=length, batch=1, steps=n-1, logits_bitwise_equal=True, samples_s=samples,
                       median_step_s={k:median(v) for k,v in samples.items()})
            report['rows'].append(row)
            print(json.dumps(row), flush=True)
    Path('mlp-graph-trial.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    run()
