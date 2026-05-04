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
        self.name = "fallback"


def _match_monitor_by_name(monitors, name):
    """Return the first monitor whose name contains ``name`` (case-insensitive).

    On Windows, ``screeninfo`` reports names like ``\\\\.\\DISPLAY1`` which
    correspond to the display numbers shown in Windows Settings. So a config
    value of ``"DISPLAY1"`` or just ``"1"`` will both match.
    """
    needle = str(name).strip().lower()
    if not needle:
        return None
    for m in monitors:
        if needle in str(getattr(m, "name", "") or "").lower():
            return m
    return None


def get_target_monitor(name=None):
    """Return the monitor object for the requested display name.

    Resolution order: explicit ``name`` arg → ``VLM_GUI_DISPLAY`` env var →
    primary monitor. ``name`` is matched as a case-insensitive substring of
    the screeninfo monitor name (e.g. ``"DISPLAY3"`` or ``"3"`` matches
    ``\\\\.\\DISPLAY3`` on Windows).

    Falls back to a virtual monitor sized from Tk's primary screen if
    ``screeninfo`` is unavailable or enumeration fails.
    """
    if name is None:
        name = os.environ.get(DISPLAY_ENV_VAR)

    try:
        from screeninfo import get_monitors  # pylint: disable=C0415
        monitors = get_monitors()
        if not monitors:
            raise RuntimeError("screeninfo returned no monitors")

        if name is not None and str(name).strip() != "":
            match = _match_monitor_by_name(monitors, name)
            if match is not None:
                return match
            available = [getattr(m, "name", "?") for m in monitors]
            logger.warning(
                "Display name %r not found in monitors %s; using primary",
                name, available)

        for m in monitors:
            if getattr(m, "is_primary", False):
                return m
        return monitors[0]
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
