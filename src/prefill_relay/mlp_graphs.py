"""Opt-in CUDA graph replay for fixed-shape feed-forward blocks during decode."""
from contextlib import contextmanager
from functools import wraps
import threading

import torch


class DecodeMLPGraphs:
    """Each worker owns its buffers. Attention, cache updates and validation stay eager."""

    def __init__(self, backend, max_batch=4):
        if backend.device.type != 'cuda' or backend.model.training:
            raise ValueError('decode graphs require a CUDA model in eval mode')
        self.device = backend.device
        self.max_batch = max_batch
        self.graphs = {}
        self.lock = threading.Lock()
        self.local = threading.local()
        self.stream = torch.cuda.Stream(device=self.device)
        self.last_decode = None
        for index, layer in enumerate(backend.model.model.layers):
            layer.mlp.forward = self._wrap(index, layer.mlp.forward)

    def _wrap(self, index, eager):
        @wraps(eager)
        def run(x):
            if (not getattr(self.local, 'active', False) or x.ndim != 3 or x.shape[1] != 1
                    or x.shape[0] > self.max_batch or x.device != self.device):
                return eager(x)
            key = index, x.shape, x.dtype
            if key not in self.graphs:
                caller = torch.cuda.current_stream(self.device)
                self.stream.wait_stream(caller)
                with torch.cuda.stream(self.stream):
                    arg = torch.empty_like(x)
                    arg.copy_(x)
                    eager(arg)  # Initialize stream-local cuBLAS workspace before capture.
                    graph = torch.cuda.CUDAGraph()
                    graph.capture_begin(capture_error_mode='thread_local')
                    try:
                        result = eager(arg)
                    finally:
                        graph.capture_end()
                caller.wait_stream(self.stream)
                self.graphs[key] = graph, arg, result
            graph, arg, result = self.graphs[key]
            arg.copy_(x)
            graph.replay()
            # A caller can retain module outputs across requests via a hook.
            return result.clone()
        return run

    @contextmanager
    def enabled(self):
        with self.lock:
            stream = torch.cuda.current_stream(self.device)
            if self.last_decode is not None:
                stream.wait_event(self.last_decode)
            self.local.active = True
            try:
                yield
            finally:
                self.local.active = False
                self.last_decode = stream.record_event()
