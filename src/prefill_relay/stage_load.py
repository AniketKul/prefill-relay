"""Time-weighted service occupancy and placement queues by serving stage."""
from collections import Counter
from contextlib import contextmanager
import threading
import time


class StageLoad:
    stages = ('local_prefill', 'remote_prefill', 'decode')

    def __init__(self, roles, clock=time.monotonic):
        self.roles = dict(roles)
        self.clock = clock
        self.lock = threading.Lock()
        self.started = self.last = clock()
        self.active = Counter()
        self.blocked = {}
        self.busy_seconds = Counter()
        self.capacity_seconds = Counter()
        self.queue_seconds = Counter()

    def _advance(self):
        now = self.clock()
        elapsed = now - self.last
        self.last = now
        for node, stage in self.roles.items():
            self.capacity_seconds[stage] += elapsed
            if self.active[node]:
                self.busy_seconds[stage] += elapsed
        for stages in self.blocked.values():
            for stage in stages:
                self.queue_seconds[stage] += elapsed

    def role(self, node, stage):
        with self.lock:
            self._advance()
            self.roles[node] = stage

    def waiting(self, request, stages=()):
        with self.lock:
            self._advance()
            if stages:
                self.blocked[request] = stages
            else:
                self.blocked.pop(request, None)

    @contextmanager
    def working(self, node):
        with self.lock:
            self._advance()
            self.active[node] += 1
        try:
            yield
        finally:
            with self.lock:
                self._advance()
                self.active[node] -= 1

    def snapshot(self, reset=False):
        with self.lock:
            self._advance()
            elapsed = self.last - self.started
            rows = {}
            for stage in self.stages:
                capacity = self.capacity_seconds[stage]
                rows[stage] = {
                    'utilization': self.busy_seconds[stage] / capacity if capacity else 0.,
                    'mean_queue': self.queue_seconds[stage] / elapsed if elapsed else 0.,
                    'queue_depth': sum(stage in stages for stages in self.blocked.values()),
                    'instances': sum(role == stage for role in self.roles.values()),
                }
            if reset:
                self.busy_seconds.clear()
                self.capacity_seconds.clear()
                self.queue_seconds.clear()
                self.started = self.last
            return rows
