#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_IMAGE = "universalrobots/ursim_e-series"

ROBOT_MODELS = [
    "UR3", "UR5", "UR10", "UR15", "UR16", "UR18", "UR20", "UR30", "UR8LONG",
    "UR3_DC", "UR5_DC", "UR10_DC", "UR15_DC", "UR16_DC", "UR18_DC",
    "UR20_DC", "UR30_DC", "UR8LONG_DC",
]

# "Error in the script: Syntax error on line 3: end"
_ERROR_LINE_RE = re.compile(r"Error in the script: (.+) on line (\d+): (.*)")
_ERROR_ANY_RE  = re.compile(r"Error in the script: (.*)")
# Presence of this line confirms URControl reached the compilation stage
_LOADED_RE = re.compile(r"URControl: loading script:")


def _build_docker_cmd(script: Path, image: str, robot: str, timeout: int) -> list[str]:
    # ${VERSION%.*} strips the build suffix: 5.25.1.130388 -> 5.25.1
    shell = (
        "source /ursim/version.sh && "
        'MVER="${VERSION%.*}" && '
        f"ln -sf /ursim/.urcontrol/urcontrol.conf.{robot} /ursim/.urcontrol/urcontrol.conf 2>/dev/null && "
        f"timeout {timeout} /ursim/URControl -s /tmp/validate.script -r -m \"$MVER\" 2>&1; "
        "exit 0"
    )
    return [
        "docker", "run", "--rm",
        "--entrypoint", "",
        "-v", f"{script}:/tmp/validate.script:ro",
        image,
        "/bin/bash", "-c", shell,
    ]


def validate_script(
    script: Path, image: str = DEFAULT_IMAGE, robot: str = "UR5",
    timeout: int = 10, verbose: bool = False,
) -> tuple[int, str]:
    """Run URControl against `script` in a throwaway container.

    Returns (exit_code, message): 0/"" = valid, 1/error text = syntax
    error(s), 2/error text = tool or setup error (e.g. Docker missing).
    """
    cmd = _build_docker_cmd(script, image, robot, timeout)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return 2, "'docker' not found — is Docker installed and on PATH?"

    output = result.stdout + result.stderr
    if verbose:
        print(output, file=sys.stderr, end="")

    if not _LOADED_RE.search(output):
        message = (
            f"URControl did not reach the compilation stage.\n"
            f"  Check image ({image}) and robot ({robot}) are correct."
        )
        return 2, message

    name = script.name
    structured = _ERROR_LINE_RE.findall(output)
    if structured:
        lines = [f"{name}:{line_no}: error: {kind}: {detail}"
                  for kind, line_no, detail in structured]
        return 1, "\n".join(lines)

    # Fallback for error lines that don't include a line number
    unstructured = _ERROR_ANY_RE.findall(output)
    if unstructured:
        lines = [f"{name}: error: {msg}" for msg in unstructured]
        return 1, "\n".join(lines)

    return 0, ""


def _validate(script: Path, image: str, robot: str, timeout: int, verbose: bool) -> int:
    code, message = validate_script(script, image, robot, timeout, verbose)
    if code != 0 and message:
        print(message, file=sys.stderr)
    return code


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate URScript .script file syntax using URControl from URSim.",
        epilog=(
            "Exit codes: 0 = valid, 1 = syntax error(s), 2 = tool/setup error.\n"
            "Requires Docker with the universalrobots/ursim_e-series image."
        ),
    )
    ap.add_argument("script", help=".script file to validate")
    ap.add_argument(
        "--image", default=DEFAULT_IMAGE, metavar="IMAGE",
        help=f"URSim Docker image to use (default: {DEFAULT_IMAGE})",
    )
    ap.add_argument(
        "--robot", default="UR5", choices=ROBOT_MODELS, metavar="MODEL",
        help="Robot model config (default: UR5). Choices: " + ", ".join(ROBOT_MODELS),
    )
    ap.add_argument(
        "--timeout", type=int, default=10, metavar="N",
        help="Seconds before killing URControl (default: 10). "
             "Must be >5 to allow controller initialization.",
    )
    ap.add_argument(
        "--verbose", action="store_true",
        help="Print raw URControl output to stderr",
    )
    args = ap.parse_args()

    script = Path(args.script)
    if not script.exists():
        print(f"error: file not found: {args.script}", file=sys.stderr)
        sys.exit(2)

    sys.exit(_validate(script.resolve(), args.image, args.robot, args.timeout, args.verbose))


if __name__ == "__main__":
    main()
