"""Optional IPv4 host-memory RDMA streams using Linux rdma-core rsockets."""
import ctypes as C
import errno
import fcntl
from functools import lru_cache
import ipaddress
import math
import os
import select
import socket
import socketserver
import struct
import threading
import time


@lru_cache(maxsize=1)
def _library():
    lib = C.CDLL('librdmacm.so.1', use_errno=True)
    for name in ('rsend', 'rrecv'):
        fn = getattr(lib, name)
        fn.argtypes = [C.c_int, C.c_void_p, C.c_size_t, C.c_int]
        fn.restype = C.c_ssize_t
    return lib


class _PollFD(C.Structure):
    _fields_ = [('fd', C.c_int), ('events', C.c_short), ('revents', C.c_short)]


def _check(result):
    if result < 0:
        number = C.get_errno()
        raise OSError(number, os.strerror(number))
    return result


def _address(host, port):
    address = ipaddress.IPv4Address(host)
    if address.is_unspecified or address.is_multicast:
        raise ValueError('specify the IPv4 address of a local RDMA interface')
    return struct.pack('=H', socket.AF_INET) + struct.pack('!H4s8x', port, address.packed)


class RDMAStream:
    def __init__(self, fd=None):
        self.api = _library()
        self.fd = _check(self.api.rsocket(socket.AF_INET, socket.SOCK_STREAM, 0) if fd is None else fd)
        self.timeout = 10
        try:
            _check(self.api.rfcntl(self.fd, fcntl.F_SETFL, os.O_NONBLOCK))
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.fd >= 0:
            self.api.rclose(self.fd)
            self.fd = -1

    def settimeout(self, timeout):
        self.timeout = timeout

    def poll(self, event, deadline):
        descriptor = _PollFD(self.fd, event, 0)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            result = self.api.rpoll(C.byref(descriptor), 1, math.ceil(remaining * 1000))
            if result < 0 and C.get_errno() == errno.EINTR:
                continue
            return bool(_check(result))

    def _wait(self, event, deadline):
        if not self.poll(event, deadline):
            raise TimeoutError('RDMA stream timed out')

    def sendall(self, data):
        if not data:
            return
        pointer = C.cast(C.c_char_p(data), C.c_void_p).value
        offset = 0
        deadline = time.monotonic() + self.timeout
        while offset < len(data):
            sent = self.api.rsend(self.fd, pointer + offset, len(data) - offset, socket.MSG_NOSIGNAL)
            if sent < 0 and C.get_errno() in (errno.EINTR, errno.EAGAIN):
                self._wait(select.POLLOUT, deadline)
                continue
            if not _check(sent):
                raise ConnectionError('RDMA peer closed during send')
            offset += sent

    def recv(self, size):
        data = C.create_string_buffer(size)
        deadline = time.monotonic() + self.timeout
        while True:
            result = self.api.rrecv(self.fd, data, size, 0)
            if result < 0 and C.get_errno() in (errno.EINTR, errno.EAGAIN):
                self._wait(select.POLLIN, deadline)
                continue
            return data.raw[:_check(result)]


def connect(address, timeout=10):
    raw = _address(*address)
    stream = RDMAStream()
    try:
        stream.settimeout(timeout)
        result = stream.api.rconnect(stream.fd, raw, len(raw))
        if result < 0 and C.get_errno() == errno.EINPROGRESS:
            stream._wait(select.POLLOUT, time.monotonic() + timeout)
            status, size = C.c_int(), C.c_uint(4)
            _check(stream.api.rgetsockopt(stream.fd, socket.SOL_SOCKET, socket.SO_ERROR,
                                          C.byref(status), C.byref(size)))
            if status.value:
                raise OSError(status.value, os.strerror(status.value))
        else:
            _check(result)
        return stream
    except BaseException:
        stream.close()
        raise


class RDMAServer(socketserver.ThreadingMixIn, socketserver.BaseServer):
    """Bind a single interface and accept only connections from this host address."""
    def __init__(self, address, handler):
        raw = _address(*address)
        super().__init__(address, handler)
        self.listener = RDMAStream()
        self.closing = threading.Event()
        try:
            api, fd = self.listener.api, self.listener.fd
            _check(api.rbind(fd, raw, len(raw)))
            _check(api.rlisten(fd, 16))
            actual, size = C.create_string_buffer(16), C.c_uint(16)
            _check(api.rgetsockname(fd, actual, C.byref(size)))
            self.server_address = (address[0], struct.unpack('!H', actual.raw[2:4])[0])
        except BaseException:
            self.listener.close()
            raise

    def serve_forever(self, poll_interval=.05):
        while not self.closing.is_set():
            if not self.listener.poll(select.POLLIN, time.monotonic() + poll_interval):
                continue
            address, size = C.create_string_buffer(16), C.c_uint(16)
            fd = self.listener.api.raccept(self.listener.fd, address, C.byref(size))
            if fd < 0 and C.get_errno() in (errno.EAGAIN, errno.EINTR, errno.ECONNABORTED):
                continue
            peer = socket.inet_ntoa(address.raw[4:8])
            stream = RDMAStream(_check(fd))
            if peer != self.server_address[0]:
                stream.close()
                continue
            self.process_request(stream, (peer, struct.unpack('!H', address.raw[2:4])[0]))

    def shutdown_request(self, request):
        request.close()

    def shutdown(self):
        self.closing.set()

    def server_close(self):
        self.closing.set()
        super().server_close()
        self.listener.close()
