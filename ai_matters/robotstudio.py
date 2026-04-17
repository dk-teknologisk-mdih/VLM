# pylint: disable=W1203, W0718, C0301

import time
import logging
from enum import Enum
import pyperclip
from pywinauto.application import Application
from pywinauto.controls.uia_controls import ListItemWrapper


from .config import Config

logger = logging.getLogger("orchestrator")

class RobotStudioTabs(Enum):
    """Enum for RobotStudio tab names. Adjust as needed for your RS version."""
    FILE = "&File"
    HOME = "&Home"
    MODELING = "&Modeling"
    SIMULATION = "&Simulation"
    CONTROLLER = "&Controller"
    RAPID = "&RAPID"
    ADDINS = "&Add-Ins"


# ========================================================
# RobotStudio UI Automation
# ========================================================
class RobotStudioAutomation:
    """ 
    Automates RobotStudio via pywinauto for syntax validation and simulation. 
    """

    def __init__(self, config: Config):
        self.config = config
        self.app = None
        self.main_window = None

    def connect_to_robotstudio(self) -> bool:
        """Attach to a running RobotStudio instance."""

        try:
            self.app = Application(backend="uia").connect(
                title_re=f".*{self.config.robotstudio_window_title}.*",
                timeout=10,
            )
            logger.info("Connected to RobotStudio.")
            self.main_window = self.app.top_window()
            logger.info(f"RobotStudio main window: {self.main_window}")
            return True
        except Exception as e:
            logger.error(f"Could not connect to RobotStudio: {e}")
            return False

    def change_tab(self, tab_name: RobotStudioTabs) -> bool:
        """Change to a specific tab in RobotStudio. Adjust selectors as needed."""
        if not self.app or not self.main_window:
            logger.warning("RobotStudio not connected. Cannot change tab.")
            return False

        try:
            tab = self.main_window.Ribbon.RibbonTabBar.child_window(title=tab_name.value, control_type="TabItem")
            self.main_window.set_focus()
            tab.click_input()
            logger.info(f"Switched to {tab_name} tab.")
            return True
        except Exception as e:
            logger.error(f"Could not switch to {tab_name} tab: {e}")
            return False

    def paste_code_and_apply(self, code: str) -> tuple[bool, str]:
        """
        Paste code into the RAPID editor and click Apply.
        Returns (syntax_ok, error_message).
        """
        if not self.app or not self.main_window:
            logger.warning("RobotStudio not connected. Skipping validation.")
            return True, ""

        self.change_tab(RobotStudioTabs.RAPID)

        try:
            # --- Step 1: Copy code to clipboard ---
            pyperclip.copy(code)

            # --- Step 2: Open the editor ---
            # 
            
            self.clear_output_pane()
            
            # Ctrl+Shift+M to reset program pointers    
            self.main_window.type_keys("^+m")
            
            # If error occurs, it means there were syntax errors in the existing code
            errors = []
            output_texts = self.get_output_pane_text()
            for item in output_texts:
                if "Error" in item[0]:
                    errors.append(item[1])  # error message is in the second column
                    
            if errors:
                error_message = "\n".join(errors)
                logger.error("Syntax errors found in existing code, falling back to slow navigation method.")
                
                # Slow method: Click through the tree RAPID -> T_ROB1 -> Module
                cid = self.config.controller_id
                mod = self.config.module_name
                
                rapid_node = self.main_window.child_window(title=f"/{cid}/RAPID", control_type="ListItem")
                t_rob_node = self.main_window.child_window(title_re=f"/{cid}/RAPID/T_ROB1 \\(.*\\)$", control_type="ListItem")
                module_node = self.main_window.child_window(title_re=f"/{cid}/RAPID/T_ROB1.*/{mod}$", control_type="ListItem")

                # Only expand a parent if its child isn't already visible
                if not t_rob_node.exists(timeout=1):
                    rapid_node.child_window(control_type="Text").double_click_input()
                    time.sleep(0.5)

                if not module_node.exists(timeout=1):
                    t_rob_node.child_window(control_type="Text").double_click_input()
                    time.sleep(0.5)

                module_node.child_window(control_type="Text").double_click_input()
                logger.info("Navigated to program pointer location using tree view.")
            else:
                logger.info("Program pointers reset successfully. No syntax errors in existing code.")
                go_to_pp = self.main_window.child_window(title="CmdBarCtl_ProgramShowPP", control_type="Button")
                go_to_pp.click()
            
                logger.info("Navigated to program pointer location in RAPID editor.")

            # --- Step 3: Focus the RAPID editor pane ---
            #
            editor = self.main_window.child_window(auto_id="EditorViewLower")
            editor.set_focus()
            editor.type_keys("^a")  # Select all
            editor.type_keys("^v")  # Paste

            logger.info("Code pasted into RobotStudio editor.")

            # --- Step 4: Clear output pane ---
            self.clear_output_pane()

            # --- Step 5: Click the Apply button ---
            apply_btn = self.main_window.child_window(title="CmdBarCtl_RapidApplyAll", control_type="Button")
            apply_btn.set_focus()
            apply_btn.click()

            logger.info("Apply button clicked. Waiting for compilation...")
            time.sleep(3)

            # --- Step 6: Check for errors in the Output pane ---
            #
            output_texts = self.get_output_pane_text()
            errors = []
            for item in output_texts:
                if "Error" in item[0]:
                    errors.append(item[1])  # error message is in the second column

            if errors:
                errors.pop()  # Remove the last item which is a summary like "1 error(s)"
                error_message = "\n".join(errors)
                logger.error(f"Syntax errors found: {error_message}")
                return False, error_message

            # No errors found
            return True, ""

        except Exception as e:
            logger.error(f"RobotStudio automation error: {e}")
            return False, str(e)

    def clear_output_pane(self) -> bool:
        """Clear the Output pane in RobotStudio."""
        if not self.app or not self.main_window:
            logger.warning("RobotStudio not connected. Cannot clear output pane.")
            return False

        try:
            output_window = self.main_window.child_window(title="Output", control_type="Pane")
            output_window.set_focus()
            output_window.right_click_input()
            clear_option = self.main_window.child_window(title="MenuItem_OutputWindowClear", control_type="MenuItem")
            clear_option.click_input()
            logger.info("Output pane cleared.")
            return True
        except Exception as e:
            logger.error(f"Could not clear output pane: {e}")
            return False

    def get_output_pane_text(self) -> list[tuple[str, str]]:
        """Get the text content of the Output pane."""
        if not self.app or not self.main_window:
            logger.warning("RobotStudio not connected. Cannot read output pane.")
            return []

        try:
            output_window = self.main_window.child_window(title="Output", control_type="Pane")
            data_items: list[ListItemWrapper] = output_window.child_window(control_type='DataGrid').children(control_type='DataItem')
            output_texts = []
            for item in data_items:
                texts = item.children()[0].children_texts()
                output_texts.append(tuple(texts))
            return output_texts

        except Exception as e:
            logger.error(f"Could not read output pane: {e}")
            return []

    def start_simulation(self) -> bool:
        """
        Start the simulation in RobotStudio.
        Returns True if simulation started successfully.
        """
        if not self.app or not self.main_window:
            return True

        try:
            self.change_tab(RobotStudioTabs.SIMULATION)
            self.clear_output_pane()
            start_btn = self.main_window.child_window(title="CmdBarCtl_SimulationPlay", control_type="Button")
            start_btn.set_focus()
            start_btn.click_input()

            logger.info("Simulation started.")
            return True
        except Exception as e:
            logger.error(f"Could not start simulation: {e}")
            return False

    def stop_simulation(self) -> bool:
        """
        Stop the simulation in RobotStudio.
        Returns True if simulation stopped successfully.
        """
        if not self.app or not self.main_window:
            return True

        try:
            self.change_tab(RobotStudioTabs.SIMULATION)
            stop_btn = self.main_window.child_window(title="CmdBarCtl_SimulationStop", control_type="Button")
            stop_btn.set_focus()
            stop_btn.click_input()

            logger.info("Simulation stopped.")
            return True
        except Exception as e:
            logger.error(f"Could not stop simulation: {e}")
            return False

    def wait_for_simulation_complete(self, timeout: float = 60*5) -> tuple[bool, str]:
        """
        Wait for simulation to finish.
        """
        logger.info(f"Waiting for simulation to complete (timeout={timeout}s)...")

        output_texts = self.get_output_pane_text()
        start_time = time.time()
        dot_count = 0
        while time.time() - start_time < timeout:
            output_texts = self.get_output_pane_text()
            for item in output_texts:
                if "Program stopped" in item[1]:
                    elapsed = time.time() - start_time
                    print(f"\r{' ' * dot_count}\r", end="", flush=True)
                    logger.info(f"Simulation completed successfully in {elapsed:.1f}s.")
                    return True, "Simulation completed successfully."
                if "Error" in item[0]:
                    elapsed = time.time() - start_time
                    print(f"\r{' ' * dot_count}\r", end="", flush=True)
                    # TODO: Extract more specific error message if needed
                    logger.error(f"Simulation error detected after {elapsed:.1f}s.")
                    return False, "Simulation error detected."
            time.sleep(2)
            print(".", end="", flush=True)
            dot_count += 1

        elapsed = time.time() - start_time
        print(f"\r{' ' * dot_count}\r", end="", flush=True)
        logger.warning(f"Simulation did not complete within {elapsed:.1f}s (timeout).")
        return False, "Simulation did not complete within the timeout period."
