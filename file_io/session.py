"""Automated Application State Autosave/Recovery Modules.

Saves configurations to disk every 60 seconds to prevent telemetry loss 
during physical system reboots or terminal crashes.

Dependencies:
    - json, os
"""

# GNU General Public License v3.0 Header
# Copyright (C) 2026 LunarCFD Development Team

import json
import os
from tkinter import messagebox

class SessionManager:
    """Controls serialization updates for background parameters cache workflows."""
    
    def __init__(self, main_window_instance):
        self.app = main_window_instance
        self.store_path = "autosave.json"

    def execute_autosave_state(self):
        """Serializes configuration entries directly into a local storage document."""
        state_payload = {
            "reynolds_number": self.app.ent_re.get(),
            "flow_velocity": self.app.ent_vel.get() if hasattr(self.app, 'ent_vel') else "25.0",
            "angle_of_attack": self.app.ent_aoa.get() if hasattr(self.app, 'ent_aoa') else "0.0",
            "configured_ram_limit": self.app.config_ram_limit,
            "timestamp_epoch": os.time.time() if hasattr(os, 'time') else 0
        }
        try:
            with open(self.store_path, "w") as dest:
                json.dump(state_payload, dest, indent=4)
        except Exception:
            pass # Suppress writing blocks from introducing trace exceptions

    def check_and_restore_autosave(self):
        """Detects unmanaged failures on startup and offers state recovery paths."""
        if os.path.exists(self.store_path):
            opt = messagebox.askyesno("Session Recovery", "LunarCFD identified an autosave.json configuration. Restore parameters?")
            if opt:
                try:
                    with open(self.store_path, "r") as src:
                        payload = json.load(src)
                    
                    # Restore Reynolds
                    self.app.ent_re.delete(0, "end")
                    self.app.ent_re.insert(0, payload.get("reynolds_number", "100"))
                    
                    # Restore Velocity
                    if hasattr(self.app, 'ent_vel'):
                        self.app.ent_vel.delete(0, "end")
                        self.app.ent_vel.insert(0, payload.get("flow_velocity", "25.0"))
                        
                    # Restore Angle of Attack
                    if hasattr(self.app, 'ent_aoa'):
                        self.app.ent_aoa.delete(0, "end")
                        self.app.ent_aoa.insert(0, payload.get("angle_of_attack", "0.0"))
                        
                    self.app.config_ram_limit = payload.get("configured_ram_limit", self.app.config_ram_limit)
                except Exception as err:
                    messagebox.showwarning("Restore Failed", f"Could not parse state file safely: {str(err)}")