"""
Test script for RobotStudio automation.
"""

from vlm_programming_orchestrator import Config
from vlm_programming_orchestrator.robots.abb.robotstudio import RobotStudioAutomation


def main():
    """Test RobotStudio automation by connecting, pasting code, validating syntax, and simulating."""
    config = Config()
    rs = RobotStudioAutomation(config)
    connected = rs.connect()
    if not connected:
        print("RobotStudio not available. Skipping test.")
        return

    # Test pasting code and applying
    test_code = ""
    with open("test_code.modx", "r", encoding="utf-8") as f:
        test_code = f.read()

    syntax_ok, error_msg = rs.verify(test_code)
    if syntax_ok:
        print("Syntax validation passed.")
    else:
        print(f"Syntax validation failed: {error_msg}")
        return

    # Test starting simulation
    started = rs.start_simulation()
    if started:
        print("Simulation started successfully.")
        success, message = rs.wait_for_simulation_complete()
        if success:
            print("Simulation completed successfully.")
        else:
            print(f"Simulation failed: {message}")

    else:
        print("Failed to start simulation.")

    rs.stop_simulation()
    print("Simulation stopped.")


if __name__ == "__main__":
    main()
