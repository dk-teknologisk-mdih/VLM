"""Main entry point for the VLM -> LLM -> RobotStudio -> Robot pipeline.

Implements the full 12-step flow via the VLMOrchestrator state machine:
  1. Run the GUI
  2. VLM trajectory planning
  3. LLM code generation (save code; auto-fix on errors)
  4. RobotStudio paste / apply / validate
  5. Start simulation; wait for completion
  6. Human review
  7. Copy module to FTP share
  8. LOAD over socket (wait LOAD_OK)
  9. START over socket (wait DONE)
 10. Loop back to step 1 (configurable)

Usage:
  python main_pipeline.py                  # continuous loop (default)
  python main_pipeline.py --once           # single cycle
  python main_pipeline.py --from-file      # reuse saved GUI config
  python main_pipeline.py --no-robotstudio --no-human-review --once
"""

import argparse
import sys
from pathlib import Path

import urllib3
import yaml
from dotenv import load_dotenv

from vlm_programming_orchestrator import (Config, VLMOrchestrator,
                                          run_continuous)
from vlm_programming_orchestrator.gui.gui import run_app

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def load_gui_config_from_file(path: Path) -> dict:
    """Load GUI config from a YAML file. This allows skipping the GUI for the first cycle."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "target_position" in cfg and isinstance(cfg["target_position"], list):
        cfg["target_position"] = tuple(cfg["target_position"])
    return cfg


def main():
    """Parse command-line arguments, build config, and run the orchestrator."""
    load_dotenv()

    parser = argparse.ArgumentParser(description="VLM -> Robot pipeline")
    parser.add_argument("--from-file", action="store_true",
                        help="Reuse saved GUI config instead of opening the GUI (first cycle only).")
    parser.add_argument("--once", action="store_true",
                        help="Run a single cycle instead of looping.")
    parser.add_argument("--no-robotstudio", action="store_true",
                        help="Skip validation and simulation for the selected robot backend.")
    parser.add_argument("--no-human-review", action="store_true",
                        help="Skip the human review step.")
    parser.add_argument("--robot-ip", type=str,
                        default=None, help="Override robot IP.")
    parser.add_argument("--robot-port", type=int,
                        default=None, help="Override robot port.")
    parser.add_argument("--ur-dashboard-port", type=int, default=None,
                        help="Override UR Dashboard Server port (UR backend).")
    parser.add_argument("--ftp-dir", type=str, default=None,
                        help="Override FTP shared directory.")
    parser.add_argument("--robot-type", type=str, default=None,
                        help="Robot backend to use, e.g. ABB or UR.")
    parser.add_argument("--code-language", type=str, default=None,
                        help="Code language to generate, e.g. Rapid or URScript.")
    parser.add_argument("--ursim-ip", type=str, default=None,
                        help="Override URSim host IP (UR backend).")
    parser.add_argument("--ur-robot-ip", type=str, default=None,
                        help="Override real UR controller IP (UR backend).")
    args = parser.parse_args()

    # Build the single unified config
    config = Config()
    if args.no_robotstudio:
        config.use_robotstudio_validation = False
    if args.no_human_review:
        config.require_human_review = False
    if args.robot_ip:
        config.robot_ip = args.robot_ip
    if args.robot_port:
        config.robot_port = args.robot_port
    if args.ftp_dir:
        config.ftp_shared_dir = Path(args.ftp_dir)

    if not config.api_key or not config.base_url:
        print("ERROR: API_KEY and BASE_URL must be set (e.g. via .env).",
              file=sys.stderr)
        sys.exit(1)

    # Optionally preload a saved GUI config (skips the GUI for the first cycle)
    initial_gui = None
    if args.from_file:
        full_path = Path("vlm_programming_orchestrator/gui") / config.gui_config_path
        if not full_path.exists():
            print(f"ERROR: saved GUI config not found at {full_path}",
                  file=sys.stderr)
            sys.exit(1)
        initial_gui = load_gui_config_from_file(full_path)
        print(f"Loaded GUI config from {full_path}")

    if args.once:
        run_app(config, initial_gui=initial_gui, run_once=True)
    else:
        run_app(config, initial_gui=initial_gui, run_once=False)


if __name__ == "__main__":
    main()
