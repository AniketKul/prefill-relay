"""Route requests across multiple local PD and upstream prefill clusters."""
from contextlib import ExitStack
from copy import deepcopy
import threading
import time

from .directory import GlobalDirectory


class Federation:
    """A site contributes local PD workers and one independent producer egress budget."""

    def __init__(self, sites):
        self.sites = dict(sites)
        if (not self.sites or len(set(self.sites.values())) != len(self.sites) or
                any(not isinstance(name, str) or not name or len(name) > 100 for name in self.sites)):
            raise ValueError('expected distinct coordinators with nonempty site names')
        self.lock = threading.RLock()

    def __enter__(self):
        self.routes = {}
        self.directory = GlobalDirectory()
        self.stack = ExitStack()
        try:
            for name, site in self.sites.items():
                if site.workers:
                    raise ValueError('provide new coordinators owned by the federation')
                self.stack.callback(setattr, site, 'control_lock', site.control_lock)
                site.control_lock = self.lock
                self.stack.enter_context(site)
                for cluster, node, client, _ in site.directory._candidates(site.namespace):
                    self.directory.register(f'{name}/{cluster}', node, client)
            if len({site.namespace for site in self.sites.values()}) != 1:
                raise ValueError('sites must serve the same model and precision')
            if len({site.config['block_tokens'] for site in self.sites.values()}) != 1:
                raise ValueError('sites must use the same cache block geometry')
            for local_name, local in self.sites.items():
                for source_name, source in self.sites.items():
                    key = local_name, source_name
                    if local is source:
                        self.routes[key] = local.directory, local.controller
                        continue
                    pair = GlobalDirectory()
                    for role, owner in [('local', local), ('remote', source)]:
                        for _, node, client, _ in owner.directory._candidates(owner.namespace, role):
                            pair.register(role, node, client)
                    controller = deepcopy(local.controller)
                    if controller is not None:
                        if source.controller is None:
                            raise ValueError('all producers need profiles when a PD site uses a profile')
                        workload, _, local_profile, _, nlocal = controller.args
                        remote_profile = source.controller.args[1]
                        if (remote_profile.lengths[0] > local_profile.lengths[0] or
                                remote_profile.lengths[-1] < local_profile.lengths[-1]):
                            raise ValueError('upstream profile must cover PD routing lengths')
                        controller.args = workload, remote_profile, local_profile, source.nremote, nlocal
                        controller.last_long = None
                    self.routes[key] = pair, controller
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *exc):
        self.stack.close()

    def run(self, tokens, output_tokens=8, policy='selective', *, pd_site=None):
        """Choose decoder locality first. Rank available upstreams by prefix and load."""
        if pd_site is not None and pd_site not in self.sites:
            raise ValueError('unknown PD site')
        if policy not in ('selective', 'all-local', 'all-remote'):
            raise ValueError('invalid policy')
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError('invalid tokens')
        start = time.perf_counter()
        with self.lock:
            local_candidates, remote_candidates = [], []
            for name, site in self.sites.items():
                for node, load in site.roles.active.items():
                    self.directory.update_load(f'{name}/local', node, load)
                busy = sum(site.locks[n].locked() for n in site.workers if n.startswith('remote'))
                for node in site.workers:
                    if node.startswith('remote'):
                        self.directory.update_load(f'{name}/remote', node, int(site.locks[node].locked()))
                if pd_site is None or pd_site == name:
                    node, prefix = self.directory.best(f'{name}/local', site.namespace, tokens)
                    if node:
                        # Prefer complete checkpoints already on a decoder.
                        _, decoder_prefix = self.directory.best(f'{name}/local', site.namespace, tokens,
                                                                nodes=site.roles.available('decode'))
                        decoder_hit = decoder_prefix == len(tokens)
                        local_candidates.append((not decoder_hit, -prefix, site.queue + sum(site.roles.active.values()), name))
                if policy == 'all-local':
                    continue
                node, prefix = self.directory.best(f'{name}/remote', site.namespace, tokens)
                if node:
                    rate = site.rate.stats()
                    pending = rate['pending_reservations']
                    link = site.monitor.snapshot(pending)
                    congested = pending >= 8 or link['utilization'] >= .85
                    remote_candidates.append((site.bandwidth == 0, congested, -prefix, busy / site.nremote,
                                              rate['queued_seconds'], -site.bandwidth, name))
            if not local_candidates:
                raise RuntimeError('no available local PD cluster')
            local_name = min(local_candidates)[-1]
            source_name = min(remote_candidates)[-1] if remote_candidates and policy != 'all-local' else local_name
            local, source = self.sites[local_name], self.sites[source_name]
            directory, controller = self.routes[local_name, source_name]
        dispatch = time.perf_counter() - start
        row = local._run(tokens, output_tokens, policy, source, directory, controller)
        row.update(pd_site=local_name, producer_site=source_name if row['target'] == 'remote' else
                   local_name if row['producer'] else None, placement_s=dispatch)
        row['ttft_s'] += dispatch
        row['total_s'] = time.perf_counter() - start
        return row
