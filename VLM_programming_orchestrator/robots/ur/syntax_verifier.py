"""Docker-based URScript compile check.

Implements the SyntaxVerifier interface (see robots/base.py) by shelling out
to `validate_script()` (validate_urscript.py), which spins up a throwaway
URSim container running URControl in script-replay mode.
"""
# pylint: disable=W1203, W0718

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from ...config import Config
from .validate_urscript import validate_script

logger = logging.getLogger("orchestrator")


class UrscriptSyntaxVerifier:
    """Checks URScript compiles by running it through URControl in a disposable container."""

    def __init__(self, config: Config):
        self.config = config
        self._docker_available: bool | None = None

    def is_connected(self) -> bool:
        return bool(self._docker_available)

    def connect(self) -> bool:
        """Check that the `docker` CLI is available."""
        if shutil.which("docker") is None:
            logger.error("Docker not found on PATH; cannot run URScript syntax checks.")
            self._docker_available = False
            return False
        try:
            subprocess.run(["docker", "version"], capture_output=True, timeout=10, check=True)
            self._docker_available = True
            return True
        except (subprocess.SubprocessError, OSError) as e:
            logger.error(f"Docker is installed but not usable: {e}")
            self._docker_available = False
            return False

    def verify(self, code: str) -> tuple[bool, str]:
        """Write `code` to a temp .script file and validate it via URControl."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".script", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(code)
            tmp_path = Path(tmp.name)

        try:
            exit_code, message = validate_script(
                tmp_path,
                image=self.config.ur_validate_image,
                robot=self.config.ur_validate_robot_model,
                timeout=self.config.ur_validate_timeout,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        if exit_code == 0:
            return True, ""
        logger.warning(f"URScript validation failed: {message}")
        return False, message
