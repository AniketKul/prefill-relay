"""Bounded equal-length decoder batching with drain-on-close semantics."""
from collections import deque
from concurrent.futures import Future
import threading
import time

import torch

from .analytics import count, positive


class BatchDecoder:
    """Coalesce compatible requests; a batch holds the shared compute lock.

    States are exclusively owned by the caller until decode returns. Only HF
    batch-shaped caches support merging; the tiny correctness backend runs singly.
    """
    def __init__(self, backend, compute_lock, max_batch=4, wait_s=.005, max_pending=32):
        count(max_batch, 'maximum batch', zero=False)
        count(max_pending, 'pending capacity', zero=False)
        positive(wait_s, 'batch wait', zero=True)
        self.backend = backend
        self.compute_lock = compute_lock
        self.batched = getattr(backend, 'sequence_axis', getattr(backend, 'kv_sequence_dim', 0)) == 2
        self.max_batch = max_batch if self.batched else 1
        self.wait_s = wait_s
        self.max_pending = max_pending
        self.queue = deque()
        self.condition = threading.Condition()
        self.closed = False
        self.thread = threading.Thread(target=self._run, name='decode-batcher', daemon=True)
        self.thread.start()

    def _key(self, state, output_tokens):
        count(output_tokens, 'output tokens', zero=False)
        if not isinstance(state, dict): raise ValueError('expected a cache dictionary')
        required = ('length', 'tokens', 'model_config', 'last_logits')
        if any(not isinstance(state.get(k), torch.Tensor) for k in required):
            raise ValueError('incomplete cache metadata')
        length = state['length']
        if length.numel() != 1 or length.dtype != torch.int64:
            raise ValueError('invalid cache length')
        length = int(length)
        if length < 1 or state['tokens'].shape != (length,) or state['tokens'].dtype != torch.int64:
            raise ValueError('expected a single-request token cache')
        if state['last_logits'].shape != (1, self.backend.vocab_size):
            raise ValueError('invalid cached logits')
        identity = state['model_config']
        expected = getattr(self.backend, 'identity', getattr(getattr(self.backend, 'model', None), 'config', None))
        if isinstance(expected, torch.Tensor) and (expected.dtype != identity.dtype or not torch.equal(expected.cpu(), identity.cpu())):
            raise ValueError('different model cache')
        if identity.ndim != 1 or identity.dtype not in (torch.uint8, torch.int64):
            raise ValueError('invalid model identity')
        schema = []
        for key, value in state.items():
            if not isinstance(value, torch.Tensor) or value.device != torch.device(self.backend.device):
                raise ValueError('cache tensors must be on the model device')
            if key in required: continue
            if not key.startswith('layer.') or value.ndim == 0 or (self.batched and value.shape[0] != 1):
                raise ValueError('invalid layer cache')
            schema.append((key, tuple(value.shape), str(value.dtype)))
        if not schema: raise ValueError('missing layer cache')
        limit = getattr(self.backend, 'max_context', None)
        if limit is not None and length + output_tokens - 1 > limit:
            raise ValueError('decode exceeds model context limit')
        return length, output_tokens, str(identity.dtype), tuple(identity.cpu().tolist()), tuple(sorted(schema))

    def decode(self, state: dict[str, torch.Tensor], output_tokens: int) -> dict:
        key = self._key(state, output_tokens)
        future = Future()
        with self.condition:
            if self.closed: raise RuntimeError('decoder is closed')
            if len(self.queue) >= self.max_pending: raise RuntimeError('decode queue is full')
            self.queue.append((key, state, future, time.perf_counter()))
            self.condition.notify()
        return future.result()

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join()

    def _next_batch(self):
        with self.condition:
            while not self.queue and not self.closed: self.condition.wait()
            if not self.queue: return []
            first = self.queue.popleft()
            group = [first]
            deadline = first[3] + self.wait_s
            while len(group) < self.max_batch:
                for job in list(self.queue):
                    if job[0] == first[0]:
                        self.queue.remove(job)
                        group.append(job)
                        if len(group) == self.max_batch: break
                remaining = deadline - time.perf_counter()
                if len(group) == self.max_batch or self.closed or remaining <= 0: break
                self.condition.wait(remaining)
            return group

    @torch.inference_mode()
    def _decode_batch(self, jobs):
        backend = self.backend
        batch = len(jobs)
        states = [job[1] for job in jobs]
        output_tokens = jobs[0][0][1]
        with self.compute_lock:
            if batch == 1:
                state = states[0]
            else:
                state = {key: states[0][key] for key in ('length', 'model_config')}
                state['tokens'] = torch.stack([s['tokens'] for s in states])
                state['last_logits'] = torch.stack([s['last_logits'] for s in states])
                for key in states[0]:
                    if key.startswith('layer.'):
                        state[key] = torch.cat([s[key] for s in states], dim=0)
            if str(backend.device).startswith('cuda'): torch.cuda.synchronize(backend.device)
            start = time.perf_counter()
            native = getattr(backend, 'generate', None)
            if native is not None:
                generated = native(state, output_tokens)
            else:
                token = state['last_logits'].reshape(batch, backend.vocab_size).argmax(dim=-1)
                pieces = [token]
                for _ in range(output_tokens-1):
                    inputs = token.reshape(batch, 1) if batch > 1 else token.reshape(1)
                    logits, state = backend.forward(inputs, state)
                    token = logits.reshape(batch, backend.vocab_size).argmax(dim=-1)
                    pieces.append(token)
                generated = torch.stack(pieces, dim=1)
            if str(backend.device).startswith('cuda'): torch.cuda.synchronize(backend.device)
            elapsed = time.perf_counter()-start
            tokens = generated.cpu().tolist()
        return [dict(generated_tokens=sequence, decode_s=elapsed, batch_size=batch,
                     batch_queue_s=start-job[3]) for sequence, job in zip(tokens, jobs)]

    def _run(self):
        try:
            while True:
                jobs = self._next_batch()
                if not jobs: return
                try:
                    results = self._decode_batch(jobs)
                except BaseException as error:
                    for _, _, future, _ in jobs: future.set_exception(error)
                else:
                    for job, result in zip(jobs, results): job[2].set_result(result)
        finally:
            with self.condition:
                self.closed = True
                while self.queue:
                    self.queue.popleft()[2].set_exception(RuntimeError('decoder stopped'))
                self.condition.notify_all()
