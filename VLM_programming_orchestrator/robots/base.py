"""Structural interfaces for the robot-specific nodes of the pipeline.

These map 1:1 onto the "Robot-specific operations" subgraph in
system_block_diagram.mmd: Syntax Verification, Simulation Verification,
Describe error as prompt, Execute Robot. A robot vendor plugin (see
`robots/abb/`) implements these as plain classes — no inheritance required,
since `Protocol` matching is structural.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class SyntaxVerifier(Protocol):
    """Checks generated code compiles/parses for the target robot."""

    def connect(self) -> bool: ...
    def is_connected(self) -> bool: ...
    def verify(self, code: str) -> tuple[bool, str]: ...


@runtime_checkable
class Simulator(Protocol):
    """Runs a simulated execution of the code and reports success/failure."""

    def start_simulation(self) -> bool: ...
    def stop_simulation(self) -> bool: ...
    def wait_for_simulation_complete(self, timeout: float = 300) -> tuple[bool, str]: ...


@runtime_checkable
class ErrorDescriber(Protocol):
    """Turns a raw verifier/simulator error into text suitable for an LLM fix prompt."""

    def describe_error(self, raw_error: str) -> str: ...


@runtime_checkable
class RobotExecutor(Protocol):
    """Transfers and runs the validated code on the physical robot."""

    def connect(self) -> bool: ...
    def is_connected(self) -> bool: ...
    def disconnect(self) -> None: ...
    def load_module(self, filename: str) -> tuple[bool, str]: ...
    def start_execution(self) -> tuple[bool, str]: ...


@runtime_checkable
class ModulePreparer(Protocol):
    """Rewrites LLM-generated code to match the robot's entry-point convention."""

    def prepare(self, code: str) -> str: ...


@dataclass
class RobotBackend:
    """Bundle of the concrete implementations for one robot vendor."""

    syntax_verifier: SyntaxVerifier
    simulator: Simulator
    error_describer: ErrorDescriber
    executor: RobotExecutor
    module_preparer: ModulePreparer
