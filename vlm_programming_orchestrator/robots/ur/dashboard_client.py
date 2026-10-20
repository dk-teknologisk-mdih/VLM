"""Minimal client for the UR Dashboard Server (default TCP port 29999).

Shared by `simulator.py` (URSim) and `robot_connection.py` (real controller) —
both talk to the same line-based text protocol, just at different IPs.
"""
# pylint: disable=W1203, W0718

import logging
import socket
from typing import Optional

logger = logging.getLogger("orchestrator")


class DashboardClient:
    """Thin wrapper around the UR Dashboard Server's newline-terminated text protocol."""

    def __init__(self, host: str, port: int = 29999, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None
        self.sock_file = None

    def is_connected(self) -> bool:
        return self.sock is not None

    def connect(self) -> bool:
        """Open the socket and consume the server's welcome banner."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect((self.host, self.port))
            self.sock_file = self.sock.makefile("r")
            banner = self._receive()  # e.g. "Connected: Universal Robots Dashboard Server"
            logger.info(f"Dashboard connected to {self.host}:{self.port}: {banner}")
            return True
        except (socket.error, socket.timeout) as e:
            logger.error(f"Could not connect to Dashboard Server {self.host}:{self.port}: {e}")
            self._close()
            return False

    def disconnect(self) -> None:
        try:
            if self.sock is not None:
                self.send("quit")
        except Exception:
            pass
        finally:
            self._close()

    def _close(self) -> None:
        if self.sock_file is not None:
            try:
                self.sock_file.close()
            except Exception:
                pass
            self.sock_file = None
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def send(self, command: str) -> str:
        """Send one command and return the single-line response."""
        assert self.sock is not None, "Dashboard socket is not connected."
        logger.debug(f"TX -> Dashboard: {command}")
        self.sock.sendall((command + "\n").encode("utf-8"))
        return self._receive()

    def _receive(self) -> str:
        assert self.sock_file is not None, "Dashboard socket is not connected."
        line = self.sock_file.readline().strip()
        logger.debug(f"RX <- Dashboard: {line}")
        return line

    # ----- Convenience wrappers for the commands this backend needs -----

    def load(self, program: str) -> str:
        """`load <program>.urp` — returns e.g. 'Loading program: ...' or 'File not found: ...'."""
        return self.send(f"load {program}")

    def play(self) -> str:
        return self.send("play")

    def stop(self) -> str:
        return self.send("stop")

    def pause(self) -> str:
        return self.send("pause")

    def close_popup(self) -> str:
        return self.send("close popup")

    def unlock_protective_stop(self) -> str:
        return self.send("unlock protective stop")

    def is_running(self) -> bool:
        """Parses the 'Program running: true|false' response."""
        response = self.send("running")
        if ":" not in response:
            # Anything else ('could not understand', empty line on a half-open socket)
            # must not be silently read as "finished".
            raise RuntimeError(f"Unexpected 'running' response from dashboard: {response!r}")
        return response.split(":", 1)[1].strip().lower() == "true"

    def program_state(self) -> str:
        """Returns 'PLAYING', 'PAUSED' or 'STOPPED' from the 'STOPPED <program>' response."""
        response = self.send("programState")
        parts = response.split()
        if not parts:
            raise RuntimeError("Empty 'programState' response from dashboard.")
        return parts[0].strip().upper()

    def robot_mode(self) -> str:
        """Parses the 'Robotmode: RUNNING' style response, returning just the mode word."""
        response = self.send("robotmode")
        return (response.split(":", 1)[1] if ":" in response else response).strip().upper()

    def safety_status(self) -> str:
        """Parses the 'Safetystatus: NORMAL' style response, returning just the status word."""
        response = self.send("safetystatus")
        return (response.split(":", 1)[1] if ":" in response else response).strip().upper()
