"""Application Entry Point and System Crash-Recovery Backdoor Management.

This script manages the initialization sequence of the LunarCFD framework, wrap-protecting
the entire execution stack within a structural top-level BaseException processing lifecycle.
If any system-level corruption, memory degradation, or unhandled traceback occurs, the script
safely isolates resources, logs states, and stands up a zero-dependency recovery engine.

Dependencies:
    - sys, os, traceback, datetime, tkinter
    - gui.main_window (Primary UI Layer)
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
from tkinter import messagebox

# FIX: Move import to the top level so PyInstaller can trace the dependency!
from gui.main_window import launch_main_gui

def launch_fallback_window(error_msg):
    """Generates a zero-dependency Tkinter crash recovery window."""
    root = tk.Tk()
    root.title("LunarCFD - Emergency Recovery Console")
    root.geometry("600x400")
    root.configure(bg="#2b2b2b")
    
    lbl_title = tk.Label(root, text="CRITICAL APPLICATION CRASH DETECTED", fg="#ff4444", bg="#2b2b2b", font=("Arial", 14, "bold"))
    lbl_title.pack(pady=10)
    
    txt_area = tk.Text(root, bg="#1e1e1e", fg="#ffffff", font=("Consolas", 10))
    txt_area.insert(tk.END, error_msg)
    txt_area.config(state=tk.DISABLED)
    txt_area.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
    
    frame_btns = tk.Frame(root, bg="#2b2b2b")
    frame_btns.pack(pady=15)
    
    def open_log():
        os.system("notepad.exe crash_log.txt")
        
    def trigger_restart():
        root.destroy()
        os.execv(sys.executable, [sys.executable] + sys.argv)

    tk.Button(frame_btns, text="Open crash_log.txt", command=open_log, width=20, bg="#444444", fg="white").grid(row=0, column=0, padx=10)
    tk.Button(frame_btns, text="Restart LunarCFD", command=trigger_restart, width=20, bg="#008800", fg="white").grid(row=0, column=1, padx=10)
    
    root.mainloop()

def main():
    """Execution entry wrapper parsing configuration flags and spawning the primary UI."""
    try:
        # The top-level import now guarantees launch_main_gui is fully available here
        launch_main_gui()
    except BaseException as ex:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_report = f"--- CRASH LOG REPORT [{timestamp}] ---\n"
        error_report += f"Exception Type: {type(ex).__name__}\n"
        error_report += f"Message: {str(ex)}\n\n"
        error_report += "Traceback Details:\n"
        error_report += traceback.format_exc()
        
        try:
            with open("crash_log.txt", "a") as f:
                f.write(error_report + "\n" + "="*60 + "\n")
        except Exception:
            pass
            
        launch_fallback_window(error_report)

if __name__ == "__main__":
    main()