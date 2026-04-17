# pylint: disable=W1203, W0718

import socket
import logging
from typing import Optional

from .config import Config

logger = logging.getLogger("orchestrator")


# ========================================================
# Robot Connection (TCP Socket Client)
# ========================================================

class RobotConnection:
    """
    Manages the TCP socket connection to the ABB robot's LLM_Host.
    """

    def __init__(self, config: Config):
        self.config = config
        self.sock: Optional[socket.socket] = None
        self.sock_file = None

    def connect(self) -> bool:
        """Connect to the robot's TCP server and wait for READY."""
        logger.info(
            f"Connecting to robot at {self.config.robot_ip}:{self.config.robot_port}...")
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.config.socket_timeout)
            self.sock.connect((self.config.robot_ip, self.config.robot_port))
            self.sock_file = self.sock.makefile("r")

            # Wait for READY handshake
            response = self._receive()
            if response == "READY":
                logger.info("Robot connection established. Received READY.")
                return True
            else:
                logger.error(f"Unexpected handshake response: {response}")
                return False
        except (socket.error, socket.timeout) as e:
            logger.error(f"Failed to connect to robot: {e}")
            return False

    def disconnect(self):
        """Gracefully disconnect from the robot."""
        try:
            if self.sock:
                self._send("QUIT")
                response = self._receive()
                logger.info(f"Disconnect response: {response}")
        except Exception:
            pass
        finally:
            self._close()

    def ping(self) -> bool:
        """Health check."""
        try:
            self._send("PING")
            response = self._receive()
            return response == "PONG"
        except Exception:
            return False

    def load_module(self, filename: str) -> tuple[bool, str]:
        """
        Send LOAD command and wait for acknowledgement.
        Returns (success, message).
        """
        self._send(f"LOAD:{filename}")
        response = self._receive()

        if response == "LOAD_OK":
            logger.info(f"Module '{filename}' loaded successfully on robot.")
            return True, response
        elif response.startswith("LOAD_ERR:"):
            reason = response.split(":", 1)[1]
            logger.error(f"Module load failed: {reason}")
            return False, reason
        else:
            logger.error(f"Unexpected load response: {response}")
            return False, response

    def start_execution(self) -> tuple[bool, str]:
        """
        Send START command, wait for START_OK, then wait for DONE.
        Returns (success, message).
        """
        self._send("START")
        response = self._receive()

        if response == "START_OK":
            logger.info(
                "Execution started on robot. Waiting for completion...")
        elif response.startswith("EXEC_ERR:"):
            reason = response.split(":", 1)[1]
            logger.error(f"Start failed: {reason}")
            return False, reason
        else:
            logger.error(f"Unexpected start response: {response}")
            return False, response

        # Now wait for DONE or EXEC_ERR
        # Use a longer timeout for execution since the robot may take time
        assert self.sock is not None, "Socket is not connected."
        old_timeout = self.sock.gettimeout()
        self.sock.settimeout(600)  # 10 minutes max execution time
        try:
            response = self._receive()
        finally:
            self.sock.settimeout(old_timeout)

        if response == "DONE":
            logger.info("Robot execution completed successfully.")
            return True, response
        elif response.startswith("EXEC_ERR:"):
            reason = response.split(":", 1)[1]
            logger.error(f"Execution error: {reason}")
            return False, reason
        else:
            logger.error(f"Unexpected execution response: {response}")
            return False, response

    def _send(self, message: str):
        """Send a message to the robot (newline-terminated)."""
        assert self.sock is not None, "Socket is not connected."
        logger.debug(f"TX -> Robot: {message}")
        self.sock.sendall((message + "\n").encode("utf-8"))

    def _receive(self) -> str:
        """Receive a newline-terminated message from the robot."""
        assert self.sock_file is not None, "Socket file is not initialized."
        line = self.sock_file.readline().strip()
        logger.debug(f"RX <- Robot: {line}")
        return line

    def _close(self):
        """Close socket resources."""
        try:
            if self.sock_file:
                self.sock_file.close()
            if self.sock:
                self.sock.close()
        except Exception:
            pass
        self.sock = None
        self.sock_file = None
