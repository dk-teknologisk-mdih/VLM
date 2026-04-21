"""
Human review utilities for the VLM programming orchestrator.

Dialogs are rendered as custom Tk Toplevel windows centered on the monitor
selected via the ``VLM_GUI_DISPLAY`` environment variable (same convention as
the wizard GUI). On Windows the feedback prompt also opens TabTip (the
on-screen touch keyboard) and closes it when the dialog is dismissed.
"""
# pylint: disable=W1203, W0718

import logging
import os
import subprocess
import sys
import tkinter as tk

from .gui.display_utils import center_on_monitor, get_target_monitor

logger = logging.getLogger("orchestrator")

# Visual styling for dialogs (matches wizard palette loosely)
_BG = "#0d0221"
_BG_MID = "#1a0a30"
_FG = "#f1f5ff"
_ACCENT = "#00f5d4"
_ACCENT_OK = "#58D68D"


# ---------------------------------------------------------------------------
# Windows TabTip on-screen keyboard
# ---------------------------------------------------------------------------

_TABTIP_PATHS = [
    r"C:\Program Files\Common Files\microsoft shared\ink\TabTip.exe",
    r"C:\Program Files (x86)\Common Files\microsoft shared\ink\TabTip.exe",
]

# COM IDs used by ITipInvocation to actually toggle the touch keyboard panel.
# Just launching TabTip.exe on Win10/11 only starts the background service;
# the keyboard window is shown by calling ITipInvocation::Toggle.
_CLSID_UIHostNoLaunch = "{4CE576FA-83DC-4F88-951C-9D0782B4E376}"
_IID_ITipInvocation = "{37C994E7-432B-4834-A2F7-DCE1F13B834B}"


def _toggle_tabtip_via_com() -> bool:
    """Force the TabTip panel to appear via its COM ITipInvocation interface.

    Returns True on success. Windows-only.
    """
    if sys.platform != "win32":
        return False
    try:
        # pylint: disable=C0415
        import ctypes
        from ctypes import HRESULT, POINTER
        from ctypes.wintypes import HWND
        from comtypes import COMMETHOD, GUID, CoCreateInstance, IUnknown
        from comtypes import CLSCTX_INPROC_HANDLER, CLSCTX_LOCAL_SERVER

        class ITipInvocation(IUnknown):
            _iid_ = GUID(_IID_ITipInvocation)
            _methods_ = [
                COMMETHOD([], HRESULT, "Toggle",
                          (["in"], HWND, "hwnd")),
            ]

        tip = CoCreateInstance(
            GUID(_CLSID_UIHostNoLaunch),
            interface=ITipInvocation,
            clsctx=CLSCTX_INPROC_HANDLER | CLSCTX_LOCAL_SERVER,
        )
        # Pass the desktop window as owner; required by the API.
        desktop = ctypes.windll.user32.GetDesktopWindow()
        tip.Toggle(desktop)
        return True
    except Exception as exc:
        logger.warning(f"ITipInvocation toggle failed: {exc}")
        return False


def _open_touch_keyboard():
    """Launch + show the Windows touch keyboard (TabTip). No-op on non-Windows."""
    if sys.platform != "win32":
        return
    # Step 1: make sure the TabTip background process is running.
    for path in _TABTIP_PATHS:
        if os.path.exists(path):
            try:
                # ShellExecute (os.startfile) honours TabTip's UAC manifest;
                # subprocess.Popen would raise WinError 740.
                os.startfile(path)  # type: ignore[attr-defined]
                break
            except Exception as exc:
                logger.warning(f"Failed to launch TabTip at {path}: {exc}")
    else:
        logger.warning("TabTip.exe not found; skipping on-screen keyboard")
        return
    # Step 2: actually show the keyboard panel via COM.
    _toggle_tabtip_via_com()


def _close_touch_keyboard():
    """Hide the Windows touch keyboard. No-op on non-Windows."""
    if sys.platform != "win32":
        return
    # Toggle hides the panel if it's currently shown. Fall back to taskkill
    # if the COM call fails for any reason.
    if _toggle_tabtip_via_com():
        return
    try:
        subprocess.run(
            ["taskkill", "/IM", "TabTip.exe", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception as exc:
        logger.debug(f"Failed to close TabTip: {exc}")


# ---------------------------------------------------------------------------
# Dialog helpers
# ---------------------------------------------------------------------------

def _surface(window):
    """Force a Toplevel/Tk window to the front."""
    try:
        window.attributes("-topmost", True)
        window.lift()
        window.focus_force()
        # Drop topmost so it doesn't compete with other apps later
        window.after(500, lambda: window.attributes("-topmost", False))
    except Exception:
        pass


def _make_dialog(title, width, height, parent=None):
    """Create a styled Toplevel centered on the target monitor.

    If `parent` is provided (an existing Tk root / Toplevel), the dialog is
    parented to it and no temporary root is created. Otherwise a hidden
    temporary root is created and returned alongside the dialog.

    Returns (owner, dialog). `owner` is the temporary root when one was
    created, else None.
    """
    if parent is None:
        root = tk.Tk()
        root.withdraw()
        owner = root
        dialog = tk.Toplevel(root)
    else:
        owner = None
        dialog = tk.Toplevel(parent)
    monitor = get_target_monitor()
    dialog.title(title)
    dialog.configure(bg=_BG)
    center_on_monitor(dialog, width, height, monitor)
    # Force the Toplevel to actually map and draw before the caller blocks
    # on wait_window(). NOTE: do NOT call dialog.transient(root) here -- the
    # root is withdrawn, and on Windows a transient owner that is unmapped
    # can prevent the Toplevel from being shown at all.
    dialog.deiconify()
    dialog.update_idletasks()
    return owner, dialog


def _styled_button(parent, text, command, bg=_ACCENT, fg=_BG):
    return tk.Button(
        parent, text=text, command=command,
        font=("Arial", 11, "bold"),
        fg=fg, bg=bg,
        activebackground=_ACCENT_OK, activeforeground=_BG,
        relief=tk.FLAT, padx=20, pady=10, cursor="hand2",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _yes_no_dialog(title, message, yes_text="Yes", no_text="No", parent=None) -> bool:
    """Custom Yes/No dialog rendered on the configured monitor."""
    try:
        owner, dialog = _make_dialog(title, 560, 280, parent=parent)
    except Exception as exc:
        logger.error(f"Dialog init failed: {exc}")
        response = input(f"{title}\n{message}\n(y/n): ").strip().lower()
        return response in ("y", "yes")

    state = {"result": False}

    header = tk.Label(
        dialog, text=title, font=("Arial", 14, "bold"),
        fg=_ACCENT, bg=_BG, pady=12,
    )
    header.pack(fill=tk.X)

    body = tk.Label(
        dialog, text=message, font=("Arial", 11),
        fg=_FG, bg=_BG, wraplength=520, justify=tk.CENTER,
    )
    body.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))

    btn_frame = tk.Frame(dialog, bg=_BG)
    btn_frame.pack(fill=tk.X, pady=(0, 18))

    def _set(val):
        state["result"] = val
        dialog.destroy()

    yes_btn = _styled_button(btn_frame, yes_text, lambda: _set(True))
    yes_btn.pack(side=tk.RIGHT, padx=20)
    no_btn = _styled_button(
        btn_frame, no_text, lambda: _set(False), bg=_BG_MID, fg=_FG)
    no_btn.pack(side=tk.RIGHT)

    dialog.protocol("WM_DELETE_WINDOW", lambda: _set(False))
    dialog.bind("<Return>", lambda e: _set(True))
    dialog.bind("<Escape>", lambda e: _set(False))

    _surface(dialog)
    try:
        dialog.grab_set()
    except Exception:
        pass

    if owner is not None:
        owner.wait_window(dialog)
        try:
            owner.destroy()
        except Exception:
            pass
    else:
        parent.wait_window(dialog)
    return state["result"]


def ask_human_review(parent=None) -> bool:
    """Pop-up asking if the simulation looked correct. Returns True if approved."""
    return _yes_no_dialog(
        "Human Review",
        "The simulation has completed.\n\n"
        "Did the robot behavior look correct?\n\n"
        "Click 'Yes' to send the code to the physical robot.\n"
        "Click 'No' to regenerate the code.",
        parent=parent,
    )


def ask_skip_validation(parent=None) -> bool:
    """Ask the user whether to skip RobotStudio validation. Returns True to skip."""
    return _yes_no_dialog(
        "RobotStudio Not Available",
        "Could not connect to RobotStudio for validation.\n\n"
        "Do you want to skip validation and proceed anyway?",
        parent=parent,
    )


def _feedback_dialog(parent=None) -> str:
    """Custom feedback dialog with on-screen keyboard support."""
    try:
        width, height = 760, 380
        owner, dialog = _make_dialog(
            "What went wrong?", width, height, parent=parent)
        # Shift the dialog upward so the touch keyboard (which docks to the
        # bottom of the screen) doesn't cover the OK/Cancel buttons.
        monitor = get_target_monitor()
        x = int(monitor.x) + (int(monitor.width) - width) // 2
        y = int(monitor.y) + max(40, int(monitor.height) // 8)
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        dialog.update_idletasks()
    except Exception as exc:
        logger.error(f"Feedback dialog init failed: {exc}")
        return input("Describe what was wrong: ").strip()

    state = {"text": ""}

    header = tk.Label(
        dialog, text="What went wrong?",
        font=("Arial", 14, "bold"), fg=_ACCENT, bg=_BG, pady=10,
    )
    header.pack(fill=tk.X)

    prompt = tk.Label(
        dialog,
        text="Describe what was wrong with the simulation so the LLM can fix it:",
        font=("Arial", 11), fg=_FG, bg=_BG, wraplength=720, justify=tk.LEFT,
    )
    prompt.pack(fill=tk.X, padx=20, pady=(0, 10))

    text_widget = tk.Text(
        dialog, height=8, font=("Arial", 12),
        bg=_BG_MID, fg=_FG, insertbackground=_ACCENT,
        relief=tk.FLAT, wrap=tk.WORD,
    )
    text_widget.pack(fill=tk.BOTH, expand=True, padx=20)

    btn_frame = tk.Frame(dialog, bg=_BG)
    btn_frame.pack(fill=tk.X, pady=14)

    def _submit():
        state["text"] = text_widget.get("1.0", tk.END).strip()
        _close_touch_keyboard()
        dialog.destroy()

    def _cancel():
        state["text"] = ""
        _close_touch_keyboard()
        dialog.destroy()

    ok_btn = _styled_button(btn_frame, "OK", _submit)
    ok_btn.pack(side=tk.RIGHT, padx=20)
    cancel_btn = _styled_button(
        btn_frame, "Cancel", _cancel, bg=_BG_MID, fg=_FG)
    cancel_btn.pack(side=tk.RIGHT)

    dialog.protocol("WM_DELETE_WINDOW", _cancel)
    dialog.bind("<Escape>", lambda e: _cancel())
    dialog.bind("<Control-Return>", lambda e: _submit())

    _surface(dialog)
    try:
        dialog.grab_set()
    except Exception:
        pass
    text_widget.focus_set()

    # Open touch keyboard once the dialog is on-screen
    dialog.after(150, _open_touch_keyboard)

    if owner is not None:
        owner.wait_window(dialog)
        try:
            owner.destroy()
        except Exception:
            pass
    else:
        parent.wait_window(dialog)
    return state["text"]


def ask_human_review_with_feedback(parent=None) -> tuple[bool, str]:
    """
    Pop-up asking if the simulation looked correct.
    On rejection, prompts for free-text feedback to pass to the LLM.
    Returns (approved, feedback_text).
    """
    approved = ask_human_review(parent=parent)
    if approved:
        return True, ""
    try:
        feedback = _feedback_dialog(parent=parent)
    except Exception as exc:
        logger.error(f"Feedback dialog failed: {exc}")
        feedback = input("Describe what was wrong: ").strip()
    return False, feedback


if __name__ == "__main__":
    ok, fb = ask_human_review_with_feedback()
    print(f"Approved: {ok}, Feedback: {fb!r}")
