"""UR backend registration: wires URSim + real controller into a RobotBackend."""

from ...config import Config
from ..base import RobotBackend
from .module_preparer import URScriptModulePreparer
from .robot_connection import URRobotExecutor
from .simulator import URSimSimulator
from .syntax_verifier import UrscriptSyntaxVerifier

ROBOT_TYPE = "UR"


def create_backend(config: Config) -> RobotBackend:
    """Build the UR `RobotBackend` for the given config."""
    simulator = URSimSimulator(config)
    return RobotBackend(
        syntax_verifier=UrscriptSyntaxVerifier(config),
        simulator=simulator,
        error_describer=simulator,
        executor=URRobotExecutor(config),
        module_preparer=URScriptModulePreparer(),
    )
