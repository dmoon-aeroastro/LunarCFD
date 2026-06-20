"""Session parameter autosave/restore.

Saves the solver input parameters to ``autosave.json`` next to the
application each time a run starts, and offers to restore them on the
next launch.  Only GUI entry values are stored — no flow-field data.

Dependencies:
    - json, os, time
"""

# GNU General Public License v3.0 Header
# Copyright (C) 2026 LunarCFD Development Team

import json
import os
import time
from tkinter import messagebox


class SessionManager:
    """Serializes GUI parameter entries to/from autosave.json."""

    def __init__(self, main_window_instance):
        self.app = main_window_instance
        self.store_path = "autosave.json"

    # Entry-widget attributes saved/restored by name.
    _ENTRY_FIELDS = (
        "ent_re", "ent_vel", "ent_aoa", "ent_pressure", "ent_iters",
        "ent_dt", "ent_omega", "ent_conv_res", "ent_theta",
        "ent_t_wall", "ent_t_inf", "ent_chord_m", "ent_hist_interval",
    )

    def execute_autosave_state(self):
        """Write current GUI parameters to autosave.json (called on solver start)."""
        state_payload = {"timestamp_epoch": time.time()}
        for name in self._ENTRY_FIELDS:
            widget = getattr(self.app, name, None)
            if widget is not None:
                state_payload[name] = widget.get()
        for var_name in ("var_grid", "var_airfoil", "var_solver"):
            var = getattr(self.app, var_name, None)
            if var is not None:
                state_payload[var_name] = var.get()
        try:
            with open(self.store_path, "w") as dest:
                json.dump(state_payload, dest, indent=4)
        except Exception:
            pass  # autosave is best-effort; never interrupt a run to report it

    def check_and_restore_autosave(self):
        """Offer to restore parameters from a previous session on startup."""
        if not os.path.exists(self.store_path):
            return
        opt = messagebox.askyesno(
            "Session Recovery",
            "Restore solver parameters from the previous session?")
        if not opt:
            return
        try:
            with open(self.store_path, "r") as src:
                payload = json.load(src)

            for name in self._ENTRY_FIELDS:
                widget = getattr(self.app, name, None)
                if widget is not None and name in payload:
                    widget.delete(0, "end")
                    widget.insert(0, payload[name])

            for var_name in ("var_grid", "var_airfoil", "var_solver"):
                var = getattr(self.app, var_name, None)
                if var is not None and var_name in payload:
                    var.set(payload[var_name])
        except Exception as err:
            messagebox.showwarning(
                "Restore Failed",
                f"Could not parse state file safely: {str(err)}")
