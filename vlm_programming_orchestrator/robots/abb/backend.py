"""ABB backend registration: wires RobotStudio + LLM_Host into a RobotBackend."""

from ...config import Config
from ..base import RobotBackend
from .module_preparer import RapidModulePreparer
from .robot_connection import RobotConnection
from .robotstudio import RobotStudioAutomation

ROBOT_TYPE = "ABB"


def create_backend(config: Config) -> RobotBackend:
    """Build the ABB `RobotBackend` for the given config."""
    robotstudio = RobotStudioAutomation(config)
    return RobotBackend(
        syntax_verifier=robotstudio,
        simulator=robotstudio,
        error_describer=robotstudio,
        executor=RobotConnection(config),
        module_preparer=RapidModulePreparer(),
    )
