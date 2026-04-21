"""Multi-monitor utilities for the VLM GUI and dialogs."""

import logging
import os

logger = logging.getLogger(__name__)

DISPLAY_ENV_VAR = "VLM_GUI_DISPLAY"


class _FallbackMonitor:
    """Tk-based fallback when screeninfo is unavailable."""
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


def get_target_monitor(index=None):
    """Return the monitor object for the requested display index.

    Resolution order: explicit ``index`` arg → ``VLM_GUI_DISPLAY`` env var → 0.
    Falls back to a virtual monitor sized from Tk's primary screen if
    ``screeninfo`` is unavailable or enumeration fails.
    """
    if index is None:
        env_val = os.environ.get(DISPLAY_ENV_VAR)
        if env_val is not None:
            try:
                index = int(env_val)
            except ValueError:
                logger.warning(
                    "Invalid %s=%r; using display 0", DISPLAY_ENV_VAR, env_val)
                index = 0
        else:
            index = 0
    
    print(f"Getting target monitor for index {index} (env {DISPLAY_ENV_VAR}={os.environ.get(DISPLAY_ENV_VAR)})")

    try:
        from screeninfo import get_monitors  # pylint: disable=C0415
        monitors = get_monitors()
        if not monitors:
            raise RuntimeError("screeninfo returned no monitors")
        if index < 0 or index >= len(monitors):
            logger.warning(
                "Display index %d out of range (0..%d); using 0",
                index, len(monitors) - 1)
            index = 0
        return monitors[index]
    except Exception as exc:  # pylint: disable=W0718
        logger.warning("screeninfo unavailable (%s); using primary screen", exc)
        import tkinter as tk  # pylint: disable=C0415
        tmp = tk.Tk()
        tmp.withdraw()
        w = tmp.winfo_screenwidth()
        h = tmp.winfo_screenheight()
        tmp.destroy()
        return _FallbackMonitor(0, 0, w, h)


def center_on_monitor(window, width, height, monitor=None):
    """Position a Tk window of size ``width x height`` centered on a monitor."""
    if monitor is None:
        monitor = get_target_monitor()
    x = monitor.x + (monitor.width - width) // 2
    y = monitor.y + (monitor.height - height) // 2
    window.geometry(f"{width}x{height}+{x}+{y}")
    return monitor
