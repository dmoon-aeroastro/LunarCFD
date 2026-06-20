"""LunarCFD application entry point.

Launches the main Tkinter GUI.  If the GUI crashes with an unhandled
exception, the error is appended to crash_log.txt and a minimal recovery
window is shown with options to view the log or restart the app.

Dependencies:
    - sys, os, traceback, datetime, tkinter
    - gui.main_window (primary UI layer)
"""

# GNU General Public License v3.0 Header
# Copyright (C) 2026 LunarCFD Development Team
# This program is free software: you can redistribute it and/or modify it under the terms
# of the GNU General Public License as published by the Free Software Foundation, version 3.

import os
import sys
import datetime
import traceback
import tkinter as tk

# Top-level import so PyInstaller can trace the dependency.
from gui.main_window import launch_main_gui

# Keep the crash log next to main.py regardless of the working directory
# the app was launched from.
CRASH_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crash_log.txt")


def launch_fallback_window(error_msg):
    """Show a minimal crash-report window (no dependency on the main GUI)."""
    root = tk.Tk()
    root.title("LunarCFD - Crash Report")
    root.geometry("600x400")
    root.configure(bg="#2b2b2b")

    lbl_title = tk.Label(root, text="LunarCFD crashed - details below", fg="#ff4444",
                         bg="#2b2b2b", font=("Arial", 14, "bold"))
    lbl_title.pack(pady=10)

    txt_area = tk.Text(root, bg="#1e1e1e", fg="#ffffff", font=("Consolas", 10))
    txt_area.insert(tk.END, error_msg)
    txt_area.config(state=tk.DISABLED)
    txt_area.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

    frame_btns = tk.Frame(root, bg="#2b2b2b")
    frame_btns.pack(pady=15)

    def open_log():
        # startfile is non-blocking (os.system would freeze this window
        # until notepad closed).
        try:
            os.startfile(CRASH_LOG)
        except OSError:
            pass

    def trigger_restart():
        root.destroy()
        os.execv(sys.executable, [sys.executable] + sys.argv)

    tk.Button(frame_btns, text="Open crash_log.txt", command=open_log, width=20,
              bg="#444444", fg="white").grid(row=0, column=0, padx=10)
    tk.Button(frame_btns, text="Restart LunarCFD", command=trigger_restart, width=20,
              bg="#008800", fg="white").grid(row=0, column=1, padx=10)

    root.mainloop()


def main():
    """Run the GUI; on unhandled exception, log and show the recovery window."""
    try:
        launch_main_gui()
    except Exception as ex:
        # Exception (not BaseException): a deliberate Ctrl+C or sys.exit()
        # should quit quietly, not open the crash window.
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_report = f"--- CRASH LOG REPORT [{timestamp}] ---\n"
        error_report += f"Exception Type: {type(ex).__name__}\n"
        error_report += f"Message: {str(ex)}\n\n"
        error_report += "Traceback Details:\n"
        error_report += traceback.format_exc()

        try:
            with open(CRASH_LOG, "a") as f:
                f.write(error_report + "\n" + "=" * 60 + "\n")
        except Exception:
            pass

        launch_fallback_window(error_report)


if __name__ == "__main__":
    main()
