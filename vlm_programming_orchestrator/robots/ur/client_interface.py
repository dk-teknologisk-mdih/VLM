"""
Background listener for the UR primary/secondary client interface.

Collects controller-reported runtime exceptions and error popups so program
aborts can be detected even when the dashboard only reports STOPPED.
"""

import logging
import socket
import struct
import threading
from typing import Optional

logger = logging.getLogger(__name__)

MESSAGE_TYPE_ROBOT_MESSAGE = 20

ROBOT_MESSAGE_TYPE_POPUP = 2
ROBOT_MESSAGE_TYPE_RUNTIME_EXCEPTION = 10


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Socket closed by peer")
        data += chunk
    return data


def _read_message(sock: socket.socket) -> bytes:
    header = _recv_exact(sock, 4)
    size = struct.unpack(">i", header)[0]
    return header + _recv_exact(sock, size - 4)


def parse_error(msg: bytes) -> Optional[str]:
    """Returns a readable error string for runtime exceptions / error popups, else None."""
    if len(msg) < 15 or msg[4] != MESSAGE_TYPE_ROBOT_MESSAGE:
        return None
    # Layout: size(4) type(1) timestamp(8) source(1) robot_message_type(1)
    robot_message_type = msg[14]
    offset = 15

    if robot_message_type == ROBOT_MESSAGE_TYPE_RUNTIME_EXCEPTION:
        line, column = struct.unpack_from(">ii", msg, offset)
        text = msg[offset + 8:].decode("utf-8", errors="replace")
        return f"Runtime exception at line {line}, column {column}: {text}"

    if robot_message_type == ROBOT_MESSAGE_TYPE_POPUP:
        # request_id(4) requested_type(4) warning(1) error(1) blocking(1) title_size(1)
        is_error = bool(msg[offset + 9])
        if not is_error:
            return None
        title_size = msg[offset + 11]
        title_start = offset + 12
        title = msg[title_start:title_start + title_size].decode("utf-8", errors="replace")
        text = msg[title_start + title_size:].decode("utf-8", errors="replace")
        return f"Error popup '{title}': {text}"

    return None


class RobotMessageListener:
    """Reads robot messages on a background thread and records reported errors."""

    def __init__(self, host: str, port: int, connect_timeout: float = 5.0):
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self._errors: list[str] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None

    def start(self) -> None:
        """Clears previously recorded errors and (re)starts listening."""
        self.stop()
        with self._lock:
            self._errors.clear()
        if not self.host:
            logger.warning("RobotMessageListener has no host configured; not listening.")
            return
        try:
            self._sock = socket.create_connection(
                (self.host, self.port), timeout=self.connect_timeout)
            # Blocking reads; stop() closes the socket to unblock recv.
            self._sock.settimeout(None)
        except OSError as e:
            logger.warning(f"Could not connect to UR client interface {self.host}:{self.port}: {e}")
            self._sock = None
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def errors(self) -> list[str]:
        with self._lock:
            return list(self._errors)

    def _run(self) -> None:
        sock = self._sock
        while sock is not None and not self._stop_event.is_set():
            try:
                msg = _read_message(sock)
            except (OSError, ConnectionError, struct.error) as e:
                if not self._stop_event.is_set():
                    logger.warning(f"UR client interface connection lost: {e}")
                return
            try:
                error = parse_error(msg)
            except (IndexError, struct.error):
                continue
            if error:
                logger.error(f"UR controller reported: {error}")
                with self._lock:
                    self._errors.append(error)
