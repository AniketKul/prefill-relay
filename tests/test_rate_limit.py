import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from prefill_relay.rate_limit import RateService, rate_client


def reserve(service, size, token='test-token'):
    request = Request(service.url, data=json.dumps({'bytes': size}).encode(),
                      headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
    with urlopen(request, timeout=2) as response:
        return json.load(response)['delay_s']


class RateLimitTests(unittest.TestCase):
    def test_independent_producers_share_aggregate_clock(self):
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
        code = ('import sys; from prefill_relay.rate_limit import rate_client; '
                'rate_client(sys.argv[1], sys.argv[2], int(sys.argv[3]))')
        with RateService(600, 'test-token') as service:
            start = time.monotonic()
            children = [subprocess.Popen([sys.executable, '-c', code, service.url, 'test-token', '100'], env=env,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(6)]
            for child in children:
                out, err = child.communicate(timeout=5)
                self.assertEqual(child.returncode, 0, err.decode())
            elapsed = time.monotonic() - start
            self.assertGreaterEqual(elapsed, 1.0)
            self.assertLess(elapsed, 5)
            self.assertEqual(service.stats()['bytes'], 600)
            self.assertEqual(service.stats()['reservations'], 6)
            self.assertEqual(service.stats()['pending_reservations'], 0)

    def test_zero_bandwidth_fails_quickly_and_recovers(self):
        with RateService(0, 'test-token') as service:
            start = time.monotonic()
            with self.assertRaisesRegex(ConnectionError, 'unavailable'):
                rate_client(service.url, 'test-token', 50)
            self.assertLess(time.monotonic() - start, .5)
            service.set_bandwidth(10000)
            rate_client(service.url, 'test-token', 50)
            self.assertEqual(service.stats()['bytes'], 50)
            self.assertEqual(service.stats()['rejected'], 1)

    def test_auth_and_input_bounds(self):
        with RateService(1000, 'test-token') as service:
            with self.assertRaisesRegex(ValueError, 'token'):
                rate_client(service.url, 'wrong-token', 1)
            for value in [-1, 1.5, True, 4 * 1024**3 + 1]:
                with self.assertRaises(ValueError):
                    rate_client(service.url, 'test-token', value)
                with self.assertRaises(HTTPError) as error:
                    reserve(service, value)
                self.assertEqual(error.exception.code, 400)
            self.assertEqual(service.stats()['bytes'], 0)
        for value in [-1, float('nan'), float('inf'), True, 10**16, 10**1000]:
            with self.assertRaises(ValueError):
                RateService(value, 'test-token')
        with self.assertRaises(ValueError):
            rate_client('https://example.com/reserve', 'test-token', 1)

    def test_bandwidth_change_keeps_existing_reservations(self):
        with RateService(1000, 'test-token') as service:
            first = reserve(service, 5000)
            self.assertGreaterEqual(first, 4.9)
            service.set_bandwidth(100000)
            second = reserve(service, 1000)
            self.assertGreaterEqual(second, 4)
            self.assertEqual(service.stats()['pending_reservations'], 2)
            service.set_bandwidth(0)
            with self.assertRaises(ConnectionError):
                rate_client(service.url, 'test-token', 1)
            service.set_bandwidth(100000)
            third = reserve(service, 1000)
            self.assertGreaterEqual(third, 4)

    def test_waiting_clients_do_not_hold_server_lock(self):
        with RateService(100, 'test-token') as service:
            errors = []
            def run():
                try:
                    rate_client(service.url, 'test-token', 50)
                except Exception as error:
                    errors.append(error)
            client = threading.Thread(target=run)
            client.start()
            deadline = time.monotonic() + 2
            while not service.stats()['reservations'] and time.monotonic() < deadline:
                time.sleep(.005)
            start = time.monotonic()
            service.set_bandwidth(10000)
            self.assertEqual(service.stats()['pending_reservations'], 1)
            self.assertLess(time.monotonic() - start, .1)
            client.join(timeout=2)
            self.assertFalse(client.is_alive())
            self.assertFalse(errors)

    def test_pending_queue_is_bounded_and_drains(self):
        with RateService(1000, 'test-token', max_pending=2) as service:
            reserve(service, 100)
            delay = reserve(service, 100)
            with self.assertRaisesRegex(ConnectionError, 'queue is full'):
                rate_client(service.url, 'test-token', 1)
            self.assertEqual(service.stats()['pending_reservations'], 2)
            time.sleep(delay + .01)
            self.assertEqual(service.stats()['pending_reservations'], 0)
            rate_client(service.url, 'test-token', 1)
            self.assertEqual(service.stats()['reservations'], 3)


if __name__ == '__main__':
    unittest.main()
