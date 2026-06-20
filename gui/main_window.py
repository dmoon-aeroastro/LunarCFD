import tkinter as tk
import threading
import psutil
import os
import sys
import time
import colorsys
import numpy as np
from file_io.session import SessionManager
from solver.core_bfm import run_bfm_simulation

class MainWindowApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LunarCFD v0.1.2.0")
        self.root.geometry("1400x768")
        self.kill_event = threading.Event()
        self.pause_event = threading.Event()
        self.solver_running = False
        self._help_win = None
        self._re_calc_win = None
        self._dt_calc_win = None
        self._mi_calc_win = None
        self._last_matrices = None
        self._body_mask = None
        self.config_ram_limit = (psutil.virtual_memory().available / (1024 * 1024)) * 0.75
        self.session_io = SessionManager(self)
        self.build_ui_layout()
        self.start_watchdog()
        # Offer to restore parameters from the previous session once the
        # window has painted (autosave.json is written on each solver start).
        self.root.after(300, self.session_io.check_and_restore_autosave)

    def build_ui_layout(self):
        # ── Toolbar ──────────────────────────────────────────────────────────
        toolbar = tk.Frame(self.root, bg="#e1e1e1", bd=1, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn_run = tk.Button(toolbar, text="Run Solver", command=self.start_solver)
        self.btn_run.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_pause = tk.Button(toolbar, text="Pause", command=self.trigger_pause)
        self.btn_pause.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_finish = tk.Button(toolbar, text="Finish Simulation",
                                    command=self.trigger_finish, bg="#d1e7dd")
        self.btn_finish.pack(side=tk.LEFT, padx=5, pady=5)

        tk.Button(toolbar, text="Restart Process",
                  command=self.process_level_restart).pack(side=tk.LEFT, padx=5, pady=5)

        tk.Button(toolbar, text="Help", command=self.toggle_help,
                  bg="#ddeeff").pack(side=tk.RIGHT, padx=5, pady=5)

        tk.Button(toolbar, text="Re Calculator", command=self._open_re_calculator,
                  bg="#ddeeff").pack(side=tk.RIGHT, padx=5, pady=5)

        tk.Button(toolbar, text="dt Calculator", command=self._open_dt_calculator,
                  bg="#ddeeff").pack(side=tk.RIGHT, padx=5, pady=5)

        tk.Button(toolbar, text="Max-Iter Calculator", command=self._open_max_iters_calculator,
                  bg="#ddeeff").pack(side=tk.RIGHT, padx=5, pady=5)

        # ── Status bar (bottom) — RAM monitor only ────────────────────────────
        self.status_bar = tk.Frame(self.root, bd=1, relief=tk.SUNKEN, bg="#eaeaea")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.ram_canvas = tk.Canvas(self.status_bar, width=200, height=15,
                                    bg="#dddddd", highlightthickness=0)
        self.ram_canvas.pack(side=tk.RIGHT, padx=10)
        self.lbl_ram = tk.Label(self.status_bar, text="RAM: 0MB", bg="#eaeaea")
        self.lbl_ram.pack(side=tk.RIGHT)

        # ── History panel (right sidebar) ─────────────────────────────────────
        history_frame = tk.Frame(self.root, width=460, bg="#f0f0f0", bd=1, relief=tk.SUNKEN)
        history_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 5), pady=5)
        history_frame.pack_propagate(False)

        tk.Label(history_frame, text="Iteration History",
                 font=("Arial", 10, "bold"), bg="#f0f0f0").pack(anchor="w", padx=5, pady=(5, 0))
        header_txt = f"{'Iter':>7}  {'Residual':>11}  {'Cl':>9}  {'Cd':>9}  {'L/D':>9}"
        tk.Label(history_frame, text=header_txt, font=("Courier", 8),
                 bg="#e0e0e0", anchor="w", relief=tk.FLAT).pack(fill=tk.X, padx=5)

        hist_container = tk.Frame(history_frame, bg="#f0f0f0")
        hist_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        hist_scroll = tk.Scrollbar(hist_container)
        hist_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_text = tk.Text(hist_container, font=("Courier", 8), bg="#f9f9f9",
                                    yscrollcommand=hist_scroll.set,
                                    wrap=tk.NONE, state=tk.DISABLED)
        self.history_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        hist_scroll.config(command=self.history_text.yview)

        # ── Left sidebar (parameters) ─────────────────────────────────────────
        panel_frame = tk.Frame(self.root, width=200, bg="#f5f5f5")
        panel_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        panel_frame.pack_propagate(False)

        def lbl_entry(text, default):
            tk.Label(panel_frame, text=text, bg="#f5f5f5", anchor="w").pack(fill=tk.X, padx=5)
            e = tk.Entry(panel_frame)
            e.insert(0, default)
            e.pack(fill=tk.X, padx=5)
            return e

        self.ent_re       = lbl_entry("Re:", "500000")
        self.ent_vel      = lbl_entry("Velocity (m/s):", "100.0")
        self.ent_aoa      = lbl_entry("AoA (deg):", "0")
        self.ent_pressure = lbl_entry("Air Pressure (Pa):", "101325")
        self.ent_iters    = lbl_entry("Max Iterations:", "150000")
        self.ent_dt       = lbl_entry("dt (time step):", "2e-4")
        self.ent_omega    = lbl_entry("Omega (Relaxation):", "0.6")
        self.ent_conv_res = lbl_entry("Convergence Residual:", "1e-8")
        self.ent_t_wall   = lbl_entry("Wall Temp (K):", "320")
        self.ent_t_inf    = lbl_entry("Air Temp (K):", "300")
        self.ent_chord_m  = lbl_entry("Chord (m, blank=auto):", "1")

        # Wire Re and Vel fields so the chord entry auto-updates when either changes.
        # The chord display is purely cosmetic — the solver recomputes it the same way.
        self._chord_last_auto = ""
        for _w in (self.ent_re, self.ent_vel):
            _w.bind("<FocusOut>", lambda e: self._recompute_auto_chord())
            _w.bind("<Return>",   lambda e: self._recompute_auto_chord())

        # ── Airfoil selector ──────────────────────────────────────────────────
        tk.Label(panel_frame, text="Airfoil:", bg="#f5f5f5", anchor="w").pack(fill=tk.X, padx=5)
        self.var_airfoil = tk.StringVar(value="NACA 0012")
        airfoil_menu = tk.OptionMenu(panel_frame, self.var_airfoil,
                                     "NACA 0012", "NACA 2412", "NACA 4412", "NACA 0006",
                                     "Rectangle 1×8")
        airfoil_menu.config(anchor="w", bg="#f5f5f5")
        airfoil_menu.pack(fill=tk.X, padx=5)

        # ── Grid selector ─────────────────────────────────────────────────────
        tk.Label(panel_frame, text="Grid Size:", bg="#f5f5f5", anchor="w").pack(fill=tk.X, padx=5)
        self.var_grid = tk.StringVar(value="BFM 96×48")
        grid_menu = tk.OptionMenu(panel_frame, self.var_grid,
                                  "BFM 64×32",
                                  "BFM 96×48",
                                  "BFM 128×64",
                                  "BFM 256×128",
                                  "BFM 320×160")
        grid_menu.config(anchor="w", bg="#f5f5f5")
        grid_menu.pack(fill=tk.X, padx=5)
        self.var_grid.trace_add("write", lambda *_: self._on_grid_change())

        self.ent_hist_interval = lbl_entry("History Row Interval:", "50")

        self.var_show_matrix = tk.BooleanVar(value=True)
        self.var_show_matrix.trace_add("write", lambda *_: self._refresh_matrix_overlay())
        tk.Checkbutton(panel_frame, text="Show/Update flow field",
                       variable=self.var_show_matrix,
                       bg="#f5f5f5", anchor="w").pack(fill=tk.X, padx=5)

        self.var_show_geom = tk.BooleanVar(value=True)
        self.var_show_geom.trace_add("write", lambda *_: self._refresh_matrix_overlay())
        tk.Checkbutton(panel_frame, text="Show geometry outline",
                       variable=self.var_show_geom,
                       bg="#f5f5f5", anchor="w").pack(fill=tk.X, padx=5)

        # Higher-order faces, Rhie–Chow, and local time-stepping are now STANDARD
        # (always on) — see the run payload below.  The implicit SIMPLE solver is
        # temporarily disabled pending a future update.  These checkboxes were
        # removed in v0.1.2.0.

        # ── Status button (bottom of left panel) ──────────────────────────────
        self.btn_status = tk.Label(panel_frame, text="Ready",
                                   bg="#cccccc", fg="#333333",
                                   font=("Arial", 11, "bold"),
                                   relief=tk.RAISED, bd=2, pady=8)
        self.btn_status.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=(5, 5))

        # ── Central dashboard ─────────────────────────────────────────────────
        self.display_dashboard = tk.Frame(self.root, bg="#ffffff", bd=2, relief=tk.SUNKEN)
        self.display_dashboard.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.lbl_dashboard = tk.Label(self.display_dashboard,
                 text="Performance Monitors [Target: NACA 0012 | Body-fitted (O-grid)]",
                 font=("Arial", 12, "bold"), bg="#ffffff")
        self.lbl_dashboard.pack(anchor="w", padx=5, pady=(5, 2))

        # ── Two-column monitor layout ─────────────────────────────────────────
        monitors_frame = tk.Frame(self.display_dashboard, bg="#ffffff")
        monitors_frame.pack(fill=tk.X, padx=5, pady=(0, 4))

        left_col  = tk.Frame(monitors_frame, bg="#ffffff")
        right_col = tk.Frame(monitors_frame, bg="#ffffff")
        left_col.pack(side=tk.LEFT, fill=tk.X, expand=True)
        right_col.pack(side=tk.LEFT, fill=tk.X, expand=True)

        def metric_entry(parent, initial):
            """Read-only entry that users can select and copy."""
            var = tk.StringVar(value=initial)
            e = tk.Entry(parent, textvariable=var, state="readonly",
                         readonlybackground="#ffffff", relief=tk.FLAT,
                         font=("Arial", 10), fg="#000000")
            e.pack(anchor="w", fill=tk.X, padx=4, pady=1)
            return var

        # Left column — core flow metrics
        self.var_iter = metric_entry(left_col,  "Iteration: 0")
        self.var_time = metric_entry(left_col,  "Elapsed: 0.00s")
        self.var_res  = metric_entry(left_col,  "Residual: —")
        self.var_cl   = metric_entry(left_col,  "Cl: —")
        self.var_cd   = metric_entry(left_col,  "Cd: —")
        self.var_ld   = metric_entry(left_col,  "L/D: —")

        # Right column — derived/extended metrics
        self.var_cm      = metric_entry(right_col, "Cm: —")
        self.var_lift_n  = metric_entry(right_col, "Lift: — N/m")
        self.var_drag_n  = metric_entry(right_col, "Drag: — N/m")
        self.var_cl_std  = metric_entry(right_col, "Cl Amp (σ): —")
        self.var_st      = metric_entry(right_col, "Strouhal: —")
        self.var_nu      = metric_entry(right_col, "Nusselt: —")

        # ── Matrix field selector ─────────────────────────────────────────────
        field_sel_frame = tk.Frame(self.display_dashboard, bg="#ffffff")
        field_sel_frame.pack(anchor="w", padx=5, pady=(2, 0))
        tk.Label(field_sel_frame, text="Field:", bg="#ffffff",
                 font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.var_field = tk.StringVar(value="Pressure")
        _field_menu = tk.OptionMenu(field_sel_frame, self.var_field,
                                    "Velocity (u)", "Speed |V|", "Pressure",
                                    "Vorticity", "Temperature")
        _field_menu.config(bg="#f5f5f5", font=("Arial", 9))
        _field_menu.pack(side=tk.LEFT, padx=4)
        self.var_field.trace_add("write", lambda *_: self._refresh_matrix_overlay())

        tk.Label(self.display_dashboard, text="Flowfield Visualisation:",
                 font=("Arial", 10, "bold"), bg="#ffffff").pack(anchor="w", padx=5, pady=(6, 0))

        # Colour key – pack BOTTOM first so the canvas can expand to fill the middle
        self.key_canvas = tk.Canvas(self.display_dashboard, height=52, bg="#111111",
                                    highlightthickness=0)
        self.key_canvas.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 5))

        # Canvas fills all remaining space; re-renders automatically on resize
        self.matrix_canvas = tk.Canvas(self.display_dashboard, bg="#111111",
                                       highlightthickness=0)
        self.matrix_canvas.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        self.matrix_canvas.bind("<Configure>", lambda e: self._refresh_matrix_overlay())

        # Populate chord with auto-computed value using startup defaults (Re=100, V=25)
        self._recompute_auto_chord()

    # ── Watchdog ───────────────────────────────────────────────────────────────
    def start_watchdog(self):
        def loop():
            while True:
                rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                pct = min(1.0, rss_mb / self.config_ram_limit)
                self.root.after(0, lambda p=pct, r=rss_mb: (
                    self.ram_canvas.delete("all"),
                    self.ram_canvas.create_rectangle(0, 0, int(200 * p), 15, fill="green"),
                    self.lbl_ram.config(text=f"RAM: {int(r)}MB")
                ))
                threading.Event().wait(2.0)
        threading.Thread(target=loop, daemon=True).start()

    # ── Solver control ────────────────────────────────────────────────────────
    def start_solver(self):
        if self.solver_running:
            return
        self.solver_running = True
        self.kill_event.clear()
        self.pause_event.clear()
        self.btn_pause.config(text="Pause")
        self._set_status("running")

        # Reset metric displays
        self.var_iter.set("Iteration: 0")
        self.var_time.set("Elapsed: 0.00s")
        self.var_res.set("Residual: —")
        self.var_cl.set("Cl: —")
        self.var_cd.set("Cd: —")
        self.var_ld.set("L/D: —")
        self.var_cm.set("Cm: —")
        self.var_lift_n.set("Lift: — N/m")
        self.var_drag_n.set("Drag: — N/m")
        self.var_cl_std.set("Cl Amp (σ): —")
        self.var_st.set("Strouhal: —")
        self.var_nu.set("Nusselt: —")
        self._last_matrices = None

        # Clear history table
        self.history_text.config(state=tk.NORMAL)
        self.history_text.delete("1.0", tk.END)
        self.history_text.config(state=tk.DISABLED)

        try:
            conv_res = float(self.ent_conv_res.get())
        except ValueError:
            conv_res = 1e-8
        try:
            dt_val = float(self.ent_dt.get())
            dt_val = max(1e-7, min(0.5, dt_val))
        except ValueError:
            dt_val = 0.2

        try:
            hist_interval = max(1, int(self.ent_hist_interval.get()))
        except ValueError:
            hist_interval = 50

        try:
            t_wall = float(self.ent_t_wall.get())
        except ValueError:
            t_wall = 320.0
        try:
            t_inf = float(self.ent_t_inf.get())
        except ValueError:
            t_inf = 300.0

        _chord_txt = self.ent_chord_m.get().strip()
        try:
            chord_m_val = float(_chord_txt) if _chord_txt else None
        except ValueError:
            chord_m_val = None

        # Map airfoil name to NACA series code
        _airfoil_map = {
            "NACA 0012": "0012", "NACA 2412": "2412",
            "NACA 4412": "4412", "NACA 0006": "0006",
            "Rectangle 1×8": "rect_1x8",
        }
        airfoil_code = _airfoil_map.get(self.var_airfoil.get(), "0012")

        # Update dashboard title
        self.lbl_dashboard.config(
            text=f"Performance Monitors [{self.var_airfoil.get()} | Body-fitted (O-grid)]")

        payload = {
            "re":            float(self.ent_re.get()),
            "vel":           float(self.ent_vel.get()),
            "aoa":           float(self.ent_aoa.get()),
            "pressure":      float(self.ent_pressure.get()),
            "omega":         float(self.ent_omega.get()),
            "max_iters":     int(self.ent_iters.get()),
            "conv_res":      conv_res,
            "dt":            dt_val,
            "grid":          self.var_grid.get().split("x")[0],
            "hist_interval": hist_interval,
            # v0.1.2.0: these three are now standard (always on) — checkboxes removed.
            "ho_faces":      True,
            "rhie_chow":     True,
            "local_dt":      True,
            "t_wall":        t_wall,
            "t_inf":         t_inf,
            "airfoil":       airfoil_code,
            "chord_m":       chord_m_val,   # None → auto-derive from Re/vel
        }

        # Autosave the parameters that produced this run (restored on next launch)
        self.session_io.execute_autosave_state()

        # Parse the BFM n_xi × n_eta from the grid label (e.g. "BFM 96×48").
        import re as _re
        _grid_lbl = self.var_grid.get()
        _bfm_m = _re.search(r'BFM\s+(\d+)[×x](\d+)', _grid_lbl)
        if _bfm_m:
            payload["n_xi"]  = int(_bfm_m.group(1))
            payload["n_eta"] = int(_bfm_m.group(2))
        else:
            payload["n_xi"], payload["n_eta"] = 96, 48   # safe fallback

        # v0.1.2.0: always the explicit time-marching solver.  The implicit SIMPLE
        # steady solver is temporarily disabled (to be re-enabled in a future
        # update); the run_bfm_simple module is kept in the tree for that.
        threading.Thread(target=run_bfm_simulation, args=(self, payload), daemon=True).start()

    def trigger_pause(self):
        if not self.solver_running:
            return
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.btn_pause.config(text="Pause")
            self._set_status("running")
        else:
            self.pause_event.set()
            self.btn_pause.config(text="Continue")
            self._set_status("paused")

    def trigger_finish(self):
        self.kill_event.set()
        self.pause_event.clear()      # Unblock the solver if it is paused
        self._set_status("finalizing")

    def process_level_restart(self):
        os.execv(sys.executable, [sys.executable] + sys.argv)

    # ── Matrix display slice — depends on selected grid ───────────────────────
    def _matrix_slice(self):
        g = self.var_grid.get().split("x")[0]
        if g == "80":  return slice(20, 60),  slice(0,   60)   # chord=32 cx=20 TE@52 ✓
        if g == "320": return slice(120, 200), slice(60, 220)  # chord=128 cx=80 TE@208 (+12 wake)
        return slice(60, 100), slice(30, 115)                  # chord=64 cx=40 TE@104 (+11 wake)

    def _recompute_auto_chord(self):
        """Fill chord entry with Re × ν_air / V when not manually overridden.

        Only overwrites the field if it is blank or still contains the last
        value we wrote here (i.e. the user hasn't typed something different).
        """
        try:
            re_val  = float(self.ent_re.get())
            vel_val = float(self.ent_vel.get())
        except ValueError:
            return
        auto = re_val * 1.5e-5 / max(vel_val, 0.01)
        auto_str = f"{auto:.4g}"   # e.g. "6e-05", "0.0006", "1.5"
        current = self.ent_chord_m.get().strip()
        if current == "" or current == self._chord_last_auto:
            self.ent_chord_m.delete(0, tk.END)
            self.ent_chord_m.insert(0, auto_str)
            self._chord_last_auto = auto_str

    def _on_grid_change(self):
        """Placeholder for future per-grid auto-adjustments."""
        pass

    def _render_matrix_colored(self, data, field_name="Velocity (u)"):
        """Render a 2-D numpy array onto matrix_canvas, scaled to fill the space.

        The image is built at native grid resolution using vectorised numpy
        colour mapping, then zoomed up with tk's built-in integer zoom so the
        whole grid always fits.

        Blue (HSV hue 0.667) = low value, Red (HSV hue 0.000) = high value.
        """
        cv = self.matrix_canvas
        cv.update_idletasks()
        W = cv.winfo_width()
        H = cv.winfo_height()
        if W < 2 or H < 2:
            return

        nrows, ncols = data.shape
        if nrows == 0 or ncols == 0:
            return

        # Largest integer pixel multiplier so the full grid fits inside the canvas
        pix = max(1, min(W // ncols, H // nrows))

        # ── Vectorised HSV (S=1, V=1) → RGB colour mapping ───────────────────
        # Signed normalisation: map [d_min, d_max] → [0, 1].
        # This is correct for ALL fields:
        #   • Pressure   : negative on suction side → blue, positive at stagnation → red
        #   • Vorticity  : signed (CW/CCW)
        #   • Velocity u : can be negative in wake/separation
        #   • Speed |V|  : always ≥ 0, so d_min ≈ 0 and behaviour is unchanged
        #   • Temperature: stored as ΔT ≥ 0, also unchanged
        # Previously np.abs(data) was used, which made suction (large negative p)
        # appear red — the same as stagnation — producing an inverted pressure map.
        d_min = float(data.min())
        d_max = float(data.max())
        d_range = d_max - d_min
        if d_range < 1e-10:
            # Flat field: centre the colour scale on the mean value
            d_mid   = (d_min + d_max) / 2.0
            d_min   = d_mid - 0.5
            d_max   = d_mid + 0.5
            d_range = 1.0

        norm = np.clip((data - d_min) / d_range, 0.0, 1.0)
        hue  = 0.667 * (1.0 - norm)       # blue=low, red=high

        h6   = hue * 6.0
        hi   = np.floor(h6).astype(np.uint8) % 6
        f    = h6 - np.floor(h6)
        q    = 1.0 - f
        ones  = np.ones_like(f)
        zeros = np.zeros_like(f)

        # Fully-saturated HSV sector breakdown
        r_f = np.select([hi==0, hi==1, hi==2, hi==3, hi==4, hi==5],
                        [ones,  q,     zeros, zeros, f,     ones ])
        g_f = np.select([hi==0, hi==1, hi==2, hi==3, hi==4, hi==5],
                        [f,     ones,  ones,  q,     zeros, zeros])
        b_f = np.select([hi==0, hi==1, hi==2, hi==3, hi==4, hi==5],
                        [zeros, zeros, f,     ones,  ones,  q    ])

        r_i = (r_f * 255).astype(np.uint8)
        g_i = (g_f * 255).astype(np.uint8)
        b_i = (b_f * 255).astype(np.uint8)

        # ── Geometry: solid interior + white outline ──────────────────────────
        show_geom = self.var_show_geom.get()
        if show_geom and hasattr(self, '_body_mask') and self._body_mask is not None:
            _mask = self._body_mask
            if _mask.shape == data.shape:
                # Fill interior with dark grey so bogus interpolated values
                # (O-grid wall cells mapped inside the polygon) don't bleed through
                r_i[_mask] = 50;  g_i[_mask] = 50;  b_i[_mask] = 50
                # White outline: body cells adjacent to at least one fluid cell
                fluid = ~_mask
                adj   = (np.roll(fluid,  1, 0) | np.roll(fluid, -1, 0) |
                         np.roll(fluid,  1, 1) | np.roll(fluid, -1, 1))
                bnd   = _mask & adj
                r_i[bnd] = 255;  g_i[bnd] = 255;  b_i[bnd] = 255

        # ── Build PhotoImage at native resolution, then integer-zoom ─────────
        packed   = ((r_i.astype(np.uint32) << 16) |
                    (g_i.astype(np.uint32) <<  8) |
                     b_i.astype(np.uint32))
        row_strs = ["{" + " ".join(f"#{v:06x}" for v in row) + "}" for row in packed]

        img = tk.PhotoImage(width=ncols, height=nrows)
        img.put(" ".join(row_strs))
        if pix > 1:
            img = img.zoom(pix)

        img_w = ncols * pix
        img_h = nrows * pix
        x0    = max(0, (W - img_w) // 2)
        y0    = max(0, (H - img_h) // 2)

        cv.delete("all")
        cv.create_image(x0, y0, image=img, anchor=tk.NW)
        cv.image = img          # keep reference — prevents garbage collection

        # ── Scale bar ─────────────────────────────────────────────────────────
        _meta = (self._last_matrices or {}).get("_meta")
        if _meta:
            _ncc  = float(_meta.get("n_cells_per_chord", 0))   # data cells per chord
            _cm   = float(_meta.get("chord_m", 0))             # chord in metres
            if _ncc > 0 and _cm > 0:
                bar_px    = int(_ncc * pix)
                bar_label = "1 chord"
                bar_cm    = _cm
                # Rescale so bar occupies 10–40 % of canvas width
                if bar_px > W * 0.40:
                    bar_px //= 2;  bar_cm /= 2;  bar_label = "½ chord"
                elif bar_px < W * 0.10:
                    bar_px = min(bar_px * 2, int(W * 0.40))
                    bar_cm *= 2;   bar_label = "2 chords"

                # Format the physical length
                if bar_cm >= 1.0:
                    cm_str = f"{bar_cm:.2f} m"
                elif bar_cm >= 0.01:
                    cm_str = f"{bar_cm * 100:.1f} cm"
                else:
                    cm_str = f"{bar_cm * 1000:.2f} mm"

                sx = x0 + 14
                sy = y0 + img_h - 14
                # Dark backing rectangle for readability
                cv.create_rectangle(sx - 4, sy - 20, sx + bar_px + 4, sy + 8,
                                    fill="#000000", outline="#444444")
                # Bar line and end ticks
                cv.create_line(sx, sy, sx + bar_px, sy, fill="#ffffff", width=2)
                cv.create_line(sx, sy - 5, sx, sy + 5, fill="#ffffff", width=2)
                cv.create_line(sx + bar_px, sy - 5, sx + bar_px, sy + 5,
                               fill="#ffffff", width=2)
                # Label
                cv.create_text(sx + bar_px // 2, sy - 7,
                               text=f"{bar_label}  ({cm_str})",
                               fill="#ffffff", font=("Arial", 8, "bold"), anchor="s")

        self._update_color_key(d_min, d_max, field_name)

    def _update_color_key(self, d_min, d_max, field_name="Velocity (u)"):
        """Redraw the colour-scale bar beneath the flowfield matrix.

        The bar runs blue (low=d_min) -> red (high=d_max), with five tick labels
        showing the actual physical values at 0 %, 25 %, 50 %, 75 %, 100 % of
        the colour range.  For pressure this means the left tick is the suction
        minimum (negative) and the right tick is the stagnation maximum (positive).
        Called automatically by _render_matrix_colored.
        """
        cv = self.key_canvas
        cv.update_idletasks()
        W = cv.winfo_width()
        if W < 10:
            W = 300
        cv.delete("all")

        try:
            real_vel = float(self.ent_vel.get())
        except ValueError:
            real_vel = 25.0

        # Determine scale factor and unit label per field
        if field_name in ("Velocity (u)", "Speed |V|"):
            scale  = real_vel        # dimensionless → m/s
            unit   = "m/s"
        elif field_name == "Temperature":
            # stored as ΔT above freestream (T - T_inf), in Kelvin
            scale  = 1.0
            unit   = "ΔT K"
        elif field_name == "Pressure":
            scale  = 1.0
            unit   = "(rel)"
        elif field_name == "Vorticity":
            scale  = 1.0
            unit   = "1/s"
        else:
            scale  = 1.0
            unit   = ""

        N = 200
        bar_y0, bar_y1 = 2, 22

        for lvl in range(N):
            norm = lvl / (N - 1)
            hue  = 0.667 * (1.0 - norm)
            ri, gi, bi = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            color = f"#{int(ri*255):02x}{int(gi*255):02x}{int(bi*255):02x}"
            x0 = int(lvl / N * W)
            x1 = max(x0 + 1, int((lvl + 1) / N * W))
            cv.create_rectangle(x0, bar_y0, x1, bar_y1, fill=color, outline="")

        d_range = d_max - d_min
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            val    = (d_min + frac * d_range) * scale
            x      = int(frac * (W - 1))
            cv.create_line(x, bar_y1, x, bar_y1 + 5, fill="#888888")
            anchor = "nw" if frac == 0.0 else ("ne" if frac == 1.0 else "n")
            cv.create_text(x, bar_y1 + 6, text=f"{val:.4g}",
                           fill="#aaaaaa", font=("Arial", 14), anchor=anchor)

        cv.create_text(W - 2, bar_y0, text=unit,
                       fill="#000000", font=("Arial", 14), anchor="ne")

    # ── GUI update callbacks (called from solver thread via root.after) ────────
    def update_live_metrics(self, i, res, elapsed):
        self.var_iter.set(f"Iteration: {i}")
        self.var_time.set(f"Elapsed: {elapsed:.2f}s")
        self.var_res.set(f"Residual: {res:.6f}")

    def set_body_mask(self, mask):
        """Receive the exact boolean body mask from the solver (once per run)."""
        self._body_mask = mask

    def _get_active_matrix(self):
        """Return the 2D array for the currently selected display field, or None."""
        if not self._last_matrices:
            return None
        field_map = {
            "Velocity (u)": "u",
            "Speed |V|":    "speed",
            "Pressure":     "pressure",
            "Vorticity":    "vorticity",
            "Temperature":  "temperature",
        }
        key = field_map.get(self.var_field.get(), "u")
        return self._last_matrices.get(key, None)

    def _refresh_matrix_overlay(self):
        """Re-render the matrix when the field, geometry toggle, or canvas size changes."""
        if not self.var_show_matrix.get():
            self.matrix_canvas.delete("all")
            return
        data = self._get_active_matrix()
        if data is None:
            return
        self._render_matrix_colored(data, self.var_field.get())

    @staticmethod
    def _fmt_force(v):
        """Adaptive formatter for physical force values (N/m).

        Uses scientific notation when |v| < 0.001 to avoid showing "0.000"
        for small but non-zero forces (e.g. at Re=100 where chord is tiny).
        """
        av = abs(v)
        if av == 0.0:
            return "0.000"
        if av < 0.001:
            return f"{v:.3e}"
        if av < 1000.0:
            return f"{v:.3f}"
        return f"{v:.4g}"

    def update_live_results(self, matrices, cl, cd, ld_ratio,
                            cm=0.0, lift_n=0.0, drag_n=0.0,
                            cl_std=0.0, st=float('nan'), nu_val=0.0):
        """Refresh all metrics and flowfield preview during the run (every N iters)."""
        import math
        self._last_matrices = matrices
        self.var_cl.set(f"Cl: {cl:.4f}")
        self.var_cd.set(f"Cd: {cd:.4f}")
        # L/D is meaningless when there is effectively no lift (e.g. a
        # symmetric airfoil at AoA=0): it becomes the ratio of two near-zero
        # numbers and reads as noise.  Show "—" below a small |Cl| threshold.
        self.var_ld.set("L/D: —" if abs(cl) < 0.02 else f"L/D: {ld_ratio:.4f}")
        self.var_cm.set(f"Cm: {cm:.4f}")
        self.var_lift_n.set(f"Lift: {self._fmt_force(lift_n)} N/m")
        self.var_drag_n.set(f"Drag: {self._fmt_force(drag_n)} N/m")
        self.var_cl_std.set(f"Cl Amp (σ): {cl_std:.4f}")
        self.var_st.set(f"Strouhal: {'—' if math.isnan(st) else f'{st:.4f}'}")
        self.var_nu.set(f"Nusselt: {'—' if nu_val == 0.0 else f'{nu_val:.2f}'}")
        if self.var_show_matrix.get():
            data = self._get_active_matrix()
            if data is not None:
                self._render_matrix_colored(data, self.var_field.get())

    def append_history_row(self, i, res, cl, cd, ld):
        """Add one row to the right-side iteration history table."""
        self.history_text.config(state=tk.NORMAL)
        row = f"{i:>7}  {res:>11.3e}  {cl:>9.4f}  {cd:>9.4f}  {ld:>9.3f}\n"
        self.history_text.insert(tk.END, row)
        self.history_text.see(tk.END)
        self.history_text.config(state=tk.DISABLED)

    def display_final_results(self, matrices, cl, cd, ld_ratio,
                              cm=0.0, lift_n=0.0, drag_n=0.0,
                              cl_std=0.0, st=float('nan'), nu_val=0.0,
                              state="Complete"):
        import math
        self._last_matrices = matrices
        self.solver_running = False
        self.pause_event.clear()
        self.btn_pause.config(text="Pause")
        _state_labels = {
            "Converged (steady)":     ("success", "Converged: Steady"),
            "Converged (periodic)":   ("success", "Converged: Periodic"),
            "Max iterations reached": ("success", "Max Iterations Reached"),
            "Finalized":              ("success", "Finalized by User"),
            "NaN error":              ("failure", "Failed: NaN Error"),
        }
        btn_state, btn_text = _state_labels.get(state, ("success", "Simulation Complete"))
        self._set_status(btn_state, btn_text)
        self.var_cl.set(f"Cl: {cl:.4f}")
        self.var_cd.set(f"Cd: {cd:.4f}")
        # L/D is meaningless when there is effectively no lift (e.g. a
        # symmetric airfoil at AoA=0): it becomes the ratio of two near-zero
        # numbers and reads as noise.  Show "—" below a small |Cl| threshold.
        self.var_ld.set("L/D: —" if abs(cl) < 0.02 else f"L/D: {ld_ratio:.4f}")
        self.var_cm.set(f"Cm: {cm:.4f}")
        self.var_lift_n.set(f"Lift: {self._fmt_force(lift_n)} N/m")
        self.var_drag_n.set(f"Drag: {self._fmt_force(drag_n)} N/m")
        self.var_cl_std.set(f"Cl Amp (σ): {cl_std:.4f}")
        self.var_st.set(f"Strouhal: {'—' if math.isnan(st) else f'{st:.4f}'}")
        self.var_nu.set(f"Nusselt: {'—' if nu_val == 0.0 else f'{nu_val:.2f}'}")
        if self.var_show_matrix.get():
            data = self._get_active_matrix()
            if data is not None:
                self._render_matrix_colored(data, self.var_field.get())

    # ── Status button helper ──────────────────────────────────────────────────
    def _set_status(self, state, text=None):
        """Update the colour-coded status label at the bottom of the left panel.

        state values:
          'ready'      — grey   (initial / after restart)
          'running'    — yellow (solver active)
          'paused'     — yellow (solver paused)
          'finalizing' — yellow (stop requested, draining)
          'success'    — green  (any clean completion)
          'failure'    — red    (NaN / solver error)

        Pass text= to override the default label (e.g. convergence reason).
        """
        cfg = {
            "ready":      ("Ready",               "#cccccc", "#333333"),
            "running":    ("Simulation Running",  "#f0c040", "#222222"),
            "paused":     ("Paused",              "#f0c040", "#222222"),
            "finalizing": ("Finalizing...",       "#f0c040", "#222222"),
            "success":    ("Simulation Complete", "#4caf50", "#ffffff"),
            "failure":    ("Simulation Failed",   "#e53935", "#ffffff"),
        }
        default_text, bg, fg = cfg.get(state, cfg["ready"])
        self.btn_status.config(text=text or default_text, bg=bg, fg=fg)

    # ── Re Calculator window ──────────────────────────────────────────────────
    def _open_re_calculator(self):
        """Open the Reynolds number calculator window (singleton)."""
        if self._re_calc_win and self._re_calc_win.winfo_exists():
            self._re_calc_win.destroy()
            self._re_calc_win = None
            return

        win = tk.Toplevel(self.root)
        win.title("Reynolds Number Calculator")
        win.geometry("420x360")
        win.configure(bg="#f5f5f5")
        win.resizable(False, False)
        self._re_calc_win = win

        tk.Label(win, text="Reynolds Number Calculator",
                 font=("Arial", 13, "bold"), bg="#f5f5f5").pack(padx=10, pady=(12, 2))
        tk.Label(win, text="Re = V × L / ν     (ν from Sutherland's law)",
                 font=("Arial", 9, "italic"), bg="#f5f5f5", fg="#555555").pack()

        # ── Input frame ───────────────────────────────────────────────────────
        frm = tk.Frame(win, bg="#f5f5f5")
        frm.pack(fill=tk.X, padx=10, pady=(10, 2))

        def row(label, default, row_n):
            tk.Label(frm, text=label, bg="#f5f5f5", width=22,
                     anchor="e").grid(row=row_n, column=0, sticky="e", pady=3)
            sv = tk.StringVar(value=default)
            tk.Entry(frm, textvariable=sv, width=14).grid(
                row=row_n, column=1, sticky="w", padx=(6, 0))
            return sv

        sv_vel   = row("Velocity V (m/s):",      self.ent_vel.get().strip(),     0)
        sv_chord = row("Chord / Length L (m):",  self.ent_chord_m.get().strip(), 1)
        sv_temp  = row("Air temperature (°C):",  "15",                           2)

        # ── Output labels ─────────────────────────────────────────────────────
        sep = tk.Frame(win, height=1, bg="#cccccc")
        sep.pack(fill=tk.X, padx=10, pady=(8, 2))

        # Anchor StringVars to the window so Python's GC cannot collect them
        # before the labels update (local vars would be eligible for GC).
        win._sv_vel   = sv_vel
        win._sv_chord = sv_chord
        win._sv_temp  = sv_temp

        var_re     = win._var_re     = tk.StringVar(value="—")
        var_nu     = win._var_nu     = tk.StringVar(value="—")
        var_regime = win._var_regime = tk.StringVar(value="—")
        var_equiv  = win._var_equiv  = tk.StringVar(value="—")

        out_frm = tk.Frame(win, bg="#f5f5f5")
        out_frm.pack(fill=tk.X, padx=10, pady=2)

        tk.Label(out_frm, textvariable=var_re,
                 font=("Arial", 15, "bold"), bg="#f5f5f5", fg="#1a1a8c"
                 ).pack(anchor="center", pady=(6, 0))
        tk.Label(out_frm, textvariable=var_nu,
                 font=("Arial", 9), bg="#f5f5f5", fg="#555555"
                 ).pack(anchor="center")

        tk.Label(out_frm, textvariable=var_regime,
                 font=("Arial", 10, "bold"), bg="#f5f5f5", wraplength=380,
                 justify="center").pack(anchor="center", pady=(6, 0))
        tk.Label(out_frm, textvariable=var_equiv,
                 font=("Arial", 9, "italic"), bg="#f5f5f5", fg="#555555",
                 wraplength=380, justify="center").pack(anchor="center")

        # ── Apply button ──────────────────────────────────────────────────────
        btn_apply = tk.Button(win, text="Apply Re to Solver", state=tk.DISABLED,
                              bg="#4caf50", fg="white", font=("Arial", 10, "bold"),
                              padx=10)
        btn_apply.pack(pady=(10, 8))

        # ── Computation logic ─────────────────────────────────────────────────
        _last_re = [None]

        def _regime(re):
            if re < 100:
                return ("Creeping / Stokes flow",
                        "Microscopic organism swimming, dust settling in still air.")
            if re < 1_000:
                return ("Laminar — solver accurate",
                        "Smoke from a candle, flow around a pin. LunarCFD works well here.")
            if re < 5_000:
                return ("Laminar / transitional — works well",
                        "Small model aircraft, blood flow in large arteries.")
            if re < 50_000:
                return ("Transitional — results approximate",
                        "R/C model plane wing, golf ball in flight.")
            if re < 500_000:
                return ("Low turbulence — turbulence model advised",
                        "Bicycle rider, small drone propeller, competitive swimmer.")
            if re < 3_000_000:
                return ("Turbulent — turbulence model required",
                        "Full-size car at highway speed, light aircraft wing.")
            if re < 15_000_000:
                return ("Fully turbulent — beyond current solver",
                        "Airliner wing, commercial wind turbine blade.")
            return ("Hypersonic / extreme turbulence",
                    "Rocket ascent, space-shuttle re-entry (compressibility matters too).")

        def _compute(*_):
            try:
                V   = float(sv_vel.get())
                L   = float(sv_chord.get())
                T_c = float(sv_temp.get())
            except ValueError:
                var_re.set("—")
                var_nu.set("—")
                var_regime.set("—")
                var_equiv.set("—")
                btn_apply.config(state=tk.DISABLED)
                _last_re[0] = None
                return

            if V <= 0 or L <= 0:
                var_re.set("Invalid (V and L must be > 0)")
                var_nu.set("—")
                var_regime.set("—")
                var_equiv.set("—")
                btn_apply.config(state=tk.DISABLED)
                _last_re[0] = None
                return

            T_k = T_c + 273.15
            nu  = 4.131e-9 * T_k**2.5 / (T_k + 110.4)
            Re  = V * L / nu

            _last_re[0] = Re
            var_re.set(f"Re = {Re:,.0f}")
            var_nu.set(f"ν = {nu:.4e} m²/s  (T = {T_c:.1f} °C)")
            regime_str, equiv_str = _regime(Re)
            var_regime.set(regime_str)
            var_equiv.set(equiv_str)
            btn_apply.config(state=tk.NORMAL)

        def _apply():
            if _last_re[0] is not None:
                self.ent_re.delete(0, tk.END)
                self.ent_re.insert(0, f"{int(round(_last_re[0]))}")
            win.destroy()                 # close the calculator after applying
            self._re_calc_win = None

        btn_apply.config(command=_apply)

        for sv in (sv_vel, sv_chord, sv_temp):
            sv.trace_add("write", _compute)

        _compute()  # initial result

    # ── Time-step (dt) Calculator window ──────────────────────────────────────
    def _open_dt_calculator(self):
        """Open the dt calculator (singleton): estimates the largest CFL-stable
        time step for the selected BFM grid."""
        import math, re as _re
        if self._dt_calc_win and self._dt_calc_win.winfo_exists():
            self._dt_calc_win.destroy()
            self._dt_calc_win = None
            return

        win = tk.Toplevel(self.root)
        win.title("Time Step (dt) Calculator")
        win.geometry("460x380")
        win.configure(bg="#f5f5f5")
        win.resizable(False, False)
        self._dt_calc_win = win

        tk.Label(win, text="Time Step (dt) Calculator",
                 font=("Arial", 13, "bold"), bg="#f5f5f5").pack(padx=10, pady=(12, 2))
        tk.Label(win, text="dt = CFL × (smallest cell) / velocity",
                 font=("Arial", 9, "italic"), bg="#f5f5f5", fg="#555555").pack()
        tk.Label(win,
                 text="The Body-fitted solver auto-limits dt to this CFL-stable value, so this\n"
                      "is the largest efficient dt for your grid. A smaller dt is always safe\n"
                      "(just more iterations). Only the Body-fitted (O-grid) solver is modelled.",
                 font=("Arial", 8), bg="#f5f5f5", fg="#777777", justify="center").pack(pady=(2, 6))

        frm = tk.Frame(win, bg="#f5f5f5")
        frm.pack(fill=tk.X, padx=10, pady=(4, 2))

        tk.Label(frm, text="Grid:", bg="#f5f5f5", width=14,
                 anchor="e").grid(row=0, column=0, sticky="e", pady=3)
        _grids = ["BFM 64×32", "BFM 96×48", "BFM 128×64",
                  "BFM 256×128", "BFM 320×160"]
        _cur = self.var_grid.get()
        sv_grid = tk.StringVar(value=_cur if _cur in _grids else _grids[1])
        tk.OptionMenu(frm, sv_grid, *_grids).grid(row=0, column=1, sticky="w", padx=(6, 0))

        tk.Label(frm, text="CFL target:", bg="#f5f5f5", width=14,
                 anchor="e").grid(row=1, column=0, sticky="e", pady=3)
        # 0.05 = the solver's stable cap for the default central scheme.  Even
        # if a larger value is entered the solver re-clamps to its own limit,
        # so this can only ever suggest a dt at or below the stable one.
        sv_cfl = tk.StringVar(value="0.05")
        tk.Entry(frm, textvariable=sv_cfl, width=14).grid(row=1, column=1, sticky="w", padx=(6, 0))

        sep = tk.Frame(win, height=1, bg="#cccccc")
        sep.pack(fill=tk.X, padx=10, pady=(8, 2))

        # Anchor StringVars to the window so the GC cannot collect them.
        win._sv_grid = sv_grid;  win._sv_cfl = sv_cfl
        var_dt   = win._var_dt   = tk.StringVar(value="—")
        var_info = win._var_info = tk.StringVar(value="—")

        out = tk.Frame(win, bg="#f5f5f5")
        out.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(out, textvariable=var_dt, font=("Arial", 15, "bold"),
                 bg="#f5f5f5", fg="#1a1a8c").pack(anchor="center", pady=(6, 0))
        tk.Label(out, textvariable=var_info, font=("Arial", 9), bg="#f5f5f5",
                 fg="#555555", justify="center").pack(anchor="center")

        btn_apply = tk.Button(win, text="Apply dt to Solver", state=tk.DISABLED,
                              bg="#4caf50", fg="white", font=("Arial", 10, "bold"), padx=10)
        btn_apply.pack(pady=(10, 4))
        tk.Button(win, text="Close", command=win.destroy, bg="#ddeeff",
                  font=("Arial", 10), padx=12, pady=2).pack(pady=(0, 8))

        _last_dt = [None]
        _R_ff = 15.0   # default far-field radius (sim chord units), matches solver

        def _compute(*_):
            try:
                cfl = float(sv_cfl.get())
            except ValueError:
                cfl = -1.0
            m = _re.search(r'BFM\s+(\d+)[×x](\d+)', sv_grid.get())
            if not m or cfl <= 0:
                var_dt.set("—")
                var_info.set("Enter a positive CFL target (0.1–0.5 typical).")
                btn_apply.config(state=tk.DISABLED);  _last_dt[0] = None
                return
            n_xi = int(m.group(1));  n_eta = int(m.group(2))
            alpha = 6.0 + max(0.0, math.log2(n_eta / 32.0))   # matches solver default
            # Smallest cell: first radial cell (sinh stretch over radius R) vs
            # circumferential spacing (equi-arc, airfoil perimeter ≈ 2.04 chord).
            first_radial = (math.sinh(alpha / n_eta) / math.sinh(alpha)) * _R_ff
            circ         = 2.04 / n_xi
            min_face     = min(first_radial, circ)
            which        = "radial (near-wall)" if first_radial < circ else "circumferential"
            dt = cfl * min_face                                # velocity = 1 in sim units
            _last_dt[0] = dt
            var_dt.set(f"dt ≈ {dt:.2e}")
            var_info.set(f"BFM {n_xi}×{n_eta},  α ≈ {alpha:.1f}\n"
                         f"smallest cell ≈ {min_face:.4f} chord  ({which})\n"
                         f"≈ {1.0/dt:,.0f} steps per chord-crossing")
            btn_apply.config(state=tk.NORMAL)

        def _apply():
            if _last_dt[0] is not None:
                self.ent_dt.delete(0, tk.END)
                self.ent_dt.insert(0, f"{_last_dt[0]:.2e}")
            win.destroy()                 # close the calculator after applying
            self._dt_calc_win = None

        btn_apply.config(command=_apply)
        for sv in (sv_grid, sv_cfl):
            sv.trace_add("write", _compute)
        _compute()   # initial result

    def _open_max_iters_calculator(self):
        """Open the max-iterations calculator (singleton): how many iterations a
        run needs to develop a target number of chord-crossings of flow time.
        Circulation (lift) develops over ~10–20 crossings (Wagner): ~10 → ~90%
        of steady Cl, ~15–20 → ~95%+.  iters ≈ crossings / dt."""
        import math, re as _re
        if self._mi_calc_win and self._mi_calc_win.winfo_exists():
            self._mi_calc_win.destroy()
            self._mi_calc_win = None
            return

        win = tk.Toplevel(self.root)
        win.title("Max Iterations Calculator")
        win.geometry("480x420")
        win.configure(bg="#f5f5f5")
        win.resizable(False, False)
        self._mi_calc_win = win

        tk.Label(win, text="Max Iterations Calculator",
                 font=("Arial", 13, "bold"), bg="#f5f5f5").pack(padx=10, pady=(12, 2))
        tk.Label(win, text="iters ≈ chord-crossings ÷ dt   (dt = the CFL-limited step)",
                 font=("Arial", 9, "italic"), bg="#f5f5f5", fg="#555555").pack()
        tk.Label(win,
                 text="Circulation (lift) develops over ~10–20 chord-crossings of flow\n"
                      "time — the Wagner effect: ~10 crossings → ~90% of steady Cl,\n"
                      "~15–20 → ~95%+.  This sizes Max Iterations to develop that far.",
                 font=("Arial", 8), bg="#f5f5f5", fg="#777777", justify="center").pack(pady=(2, 6))

        frm = tk.Frame(win, bg="#f5f5f5")
        frm.pack(fill=tk.X, padx=10, pady=(4, 2))

        tk.Label(frm, text="Grid:", bg="#f5f5f5", width=18,
                 anchor="e").grid(row=0, column=0, sticky="e", pady=3)
        _grids = ["BFM 64×32", "BFM 96×48", "BFM 128×64",
                  "BFM 256×128", "BFM 320×160"]
        _cur = self.var_grid.get()
        sv_grid = tk.StringVar(value=_cur if _cur in _grids else _grids[1])
        tk.OptionMenu(frm, sv_grid, *_grids).grid(row=0, column=1, sticky="w", padx=(6, 0))

        tk.Label(frm, text="Target chord-crossings:", bg="#f5f5f5", width=18,
                 anchor="e").grid(row=1, column=0, sticky="e", pady=3)
        sv_cross = tk.StringVar(value="15")
        tk.Entry(frm, textvariable=sv_cross, width=14).grid(row=1, column=1, sticky="w", padx=(6, 0))

        tk.Label(frm, text="dt (blank = CFL max):", bg="#f5f5f5", width=18,
                 anchor="e").grid(row=2, column=0, sticky="e", pady=3)
        sv_dt = tk.StringVar(value=self.ent_dt.get().strip())  # default to current dt field
        tk.Entry(frm, textvariable=sv_dt, width=14).grid(row=2, column=1, sticky="w", padx=(6, 0))

        sep = tk.Frame(win, height=1, bg="#cccccc")
        sep.pack(fill=tk.X, padx=10, pady=(8, 2))

        win._sv_grid = sv_grid;  win._sv_cross = sv_cross;  win._sv_dt = sv_dt
        var_mi   = win._var_mi   = tk.StringVar(value="—")
        var_info = win._var_info = tk.StringVar(value="—")

        out = tk.Frame(win, bg="#f5f5f5")
        out.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(out, textvariable=var_mi, font=("Arial", 15, "bold"),
                 bg="#f5f5f5", fg="#1a1a8c").pack(anchor="center", pady=(6, 0))
        tk.Label(out, textvariable=var_info, font=("Arial", 9), bg="#f5f5f5",
                 fg="#555555", justify="center").pack(anchor="center")

        btn_apply = tk.Button(win, text="Apply to Max Iterations", state=tk.DISABLED,
                              bg="#4caf50", fg="white", font=("Arial", 10, "bold"), padx=10)
        btn_apply.pack(pady=(10, 4))
        tk.Button(win, text="Close", command=win.destroy, bg="#ddeeff",
                  font=("Arial", 10), padx=12, pady=2).pack(pady=(0, 8))

        _last_mi = [None]
        _R_ff = 15.0          # far-field radius (matches solver)
        _CFL  = 0.05          # central scheme's stable cap (matches solver)

        def _compute(*_):
            m = _re.search(r'BFM\s+(\d+)[×x](\d+)', sv_grid.get())
            try:
                ncross = float(sv_cross.get())
            except ValueError:
                ncross = -1.0
            if not m or ncross <= 0:
                var_mi.set("—")
                var_info.set("Enter a positive number of chord-crossings (10–20 typical).")
                btn_apply.config(state=tk.DISABLED);  _last_mi[0] = None
                return
            n_xi = int(m.group(1));  n_eta = int(m.group(2))
            alpha = 6.0 + max(0.0, math.log2(n_eta / 32.0))      # matches solver default
            first_radial = (math.sinh(alpha / n_eta) / math.sinh(alpha)) * _R_ff
            circ         = 2.04 / n_xi
            min_face     = min(first_radial, circ)
            dt_cfl       = _CFL * min_face                        # velocity = 1 in sim units
            _txt = sv_dt.get().strip()
            try:
                dt_user = float(_txt) if _txt else None
            except ValueError:
                dt_user = None
            # The solver clamps dt to the CFL value, so the effective dt is the
            # smaller of the user's dt and the CFL limit.
            dt_eff = min(dt_user, dt_cfl) if dt_user else dt_cfl
            spc    = 1.0 / dt_eff                                 # steps per chord-crossing
            mi     = int(math.ceil(ncross * spc))
            _last_mi[0] = mi
            var_mi.set(f"Max Iterations ≈ {mi:,}")
            var_info.set(f"BFM {n_xi}×{n_eta},  dt ≈ {dt_eff:.2e}\n"
                         f"≈ {spc:,.0f} steps / chord-crossing × {ncross:g} crossings\n"
                         f"(~90% Cl at 10 crossings, ~95%+ at 15–20)")
            btn_apply.config(state=tk.NORMAL)

        def _apply():
            if _last_mi[0] is not None:
                self.ent_iters.delete(0, tk.END)
                self.ent_iters.insert(0, str(_last_mi[0]))
            win.destroy()                 # close after applying
            self._mi_calc_win = None

        btn_apply.config(command=_apply)
        for sv in (sv_grid, sv_cross, sv_dt):
            sv.trace_add("write", _compute)
        _compute()   # initial result

    # ── Help window ───────────────────────────────────────────────────────────
    def toggle_help(self):
        """Open the help window, or close it if already open."""
        if self._help_win and self._help_win.winfo_exists():
            self._help_win.destroy()
            self._help_win = None
        else:
            self._open_help_window()

    def _open_help_window(self):
        win = tk.Toplevel(self.root)
        win.title("LunarCFD — Help")
        win.geometry("620x720")
        win.configure(bg="#f5f5f5")
        win.resizable(True, True)
        self._help_win = win

        frame = tk.Frame(win, bg="#f5f5f5")
        frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(12, 4))

        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        txt = tk.Text(frame, wrap=tk.WORD, bg="#f5f5f5", relief=tk.FLAT,
                      font=("Arial", 10), padx=10, pady=6,
                      yscrollcommand=scrollbar.set)
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=txt.yview)

        txt.tag_configure("h1",    font=("Arial", 15, "bold"), spacing3=6)
        txt.tag_configure("h2",    font=("Arial", 11, "bold"), spacing1=10, spacing3=2)
        txt.tag_configure("param", font=("Arial", 10, "bold"))
        txt.tag_configure("body",  font=("Arial", 10), spacing3=3)
        txt.tag_configure("mono",  font=("Courier", 9), spacing3=3)

        def h1(t):   txt.insert(tk.END, t + "\n", "h1")
        def h2(t):   txt.insert(tk.END, t + "\n", "h2")
        def param(t): txt.insert(tk.END, t + "\n", "param")
        def body(t): txt.insert(tk.END, t + "\n", "body")

        h1("LunarCFD  v0.1.2.0")
        body(
            "LunarCFD is a 2D computational fluid dynamics simulator for airfoil "
            "analysis. It solves the incompressible Navier-Stokes equations with an "
            "explicit fractional-step projection method on a Body-Fitted O-grid "
            "(BFM) that wraps a curvilinear mesh directly around the airfoil surface "
            "for clean boundary-layer resolution. Above Re = 500 it adds the Menter "
            "k-ω SST turbulence model."
        )
        body(
            "It computes lift (Cl), drag (Cd), pitching moment about the quarter "
            "chord (Cm), lift-to-drag ratio, physical forces (N/m), and Nusselt "
            "number, and shows a live colour-mapped flowfield with a geometry "
            "outline and a physical scale bar. Angle of attack is applied by "
            "rotating the inflow velocity vector; the airfoil stays axis-aligned. "
            "Defaults: chord = 1 m, AoA = 0°, V = 100 m/s, Re = 500 000."
        )
        h2("What is standard in v0.1.2.0")
        body(
            "Three numerical features are now always on (their checkboxes were "
            "removed): higher-order convective faces, Rhie–Chow momentum "
            "interpolation, and local time-stepping. Local time-stepping drives the "
            "run to a STEADY state several times faster, but it is not time-"
            "accurate, so unsteady quantities — vortex shedding, Strouhal number, "
            "and the Cl oscillation amplitude — are no longer physically meaningful "
            "and the Strouhal readout will normally show '—'. The implicit SIMPLE "
            "steady solver is temporarily disabled and will return in a later "
            "update. The auto-stop convergence sensor is always active (the "
            "'run to max iters' override was removed)."
        )

        h2("Simulation Parameters")

        param("Re — Reynolds Number")
        body(
            "Ratio of inertial to viscous forces. Low Re (e.g. 100) gives smooth "
            "laminar flow; higher Re (500+) produces vortex shedding and unsteady "
            "effects. The physical velocity and chord length are combined with air "
            "viscosity to reach the target Re inside the solver. Default: 500 000."
        )

        h2("Choosing Reynolds Number")
        body(
            "Reynolds number links your real-world conditions to the solver. "
            "The formula is:\n"
            "    Re = V × L / ν\n"
            "where V is freestream velocity (m/s), L is chord length (m), and "
            "ν is the kinematic viscosity of air (≈ 1.5 × 10⁻⁵ m²/s at 15 °C).\n\n"
            "Use the Re Calculator button in the toolbar to enter your actual "
            "velocity, chord, and air temperature — it computes Re and ν "
            "automatically using Sutherland's law, and lets you apply the result "
            "directly to the solver with one click.\n\n"
            "Regime guide (what LunarCFD can do at each Re):\n"
            "  Re < 100             Creeping / Stokes flow. Accurate.\n"
            "  Re 100 – 1 000       Laminar. Accurate.\n"
            "  Re 1 000 – 5 000     Laminar / transitional. Works well.\n"
            "  Re 5 000 – 50 000    Transitional. Cl good; Cd approximate.\n"
            "  Re 50 000 – 500 000  Turbulent (k-ω SST active). Validated band —\n"
            "                       Cl within a few % of published data.\n"
            "  Re ~ 500 000 – 1e6   Turbulent. Spot-checked OK (NACA0012 Cl 95–97%\n"
            "                       of reference); treat as the upper tested edge.\n"
            "  Re > ~1e6            Extrapolation — physics still 2D/steady-RANS,\n"
            "                       so use with caution.\n\n"
            "k-ω SST turns on automatically at Re ≥ 500. The solver has been tested "
            "up to Re ≈ 500 000 (relatively low for aerospace) with spot checks near "
            "Re = 1 000 000. Cd is over-predicted at all turbulent Re (see Accuracy "
            "& Margins of Error). Try Re = 200 (candle flame), 1 000 (pin in "
            "airflow), 3 000 (small model wing), or 500 000 (drone / small wing)."
        )

        param("Velocity (m/s)")
        body(
            "Freestream speed in real-world units. Scales the colour-key axis "
            "(dimensionless → m/s) and sets the physical dynamic pressure used to "
            "compute lift and drag in N/m. Does not affect the dimensionless flow "
            "physics, which are governed by Re alone. Default: 100.0."
        )

        param("AoA (deg) — Angle of Attack")
        body(
            "Angle between the freestream direction and the airfoil chord line. "
            "Positive AoA tilts the nose upward, producing positive lift (Cl > 0). "
            "Typical range: −20° to +20°; stall occurs around 10–15° depending on "
            "Re and airfoil shape. Default: 0°."
        )

        param("Air Pressure (Pa)")
        body(
            "Atmospheric pressure used to compute real-world air density "
            "(ρ = P / (R·T_std)) and hence physical lift and drag forces in N/m. "
            "Default: 101325 (sea-level standard)."
        )

        param("Max Iterations")
        body(
            "Maximum time-stepping iterations before the solver stops. The solver "
            "may stop earlier if convergence is detected. Increase for finer grids "
            "or flows that take longer to settle.\n\n"
            "Suggested minimums (use the Max Iterations Calculator to size for a "
            "target number of chord-crossings):\n"
            "  BFM  64× 32 →   3 000\n"
            "  BFM  96× 48 →   5 000\n"
            "  BFM 128× 64 →  10 000\n"
            "  BFM 256×128 →  20 000\n"
            "  BFM 320×160 →  30 000"
        )

        param("dt — Time Step")
        body(
            "Dimensionless time increment per iteration. Smaller dt is more stable "
            "but needs more iterations to simulate the same physical time. "
            "The BFM solver automatically clips dt to a CFL-stable value, so any "
            "input is safe. Suggested BFM values by grid: 96×48 → 5e-3, "
            "128×64 → 2e-3, 256×128 → 2e-4, 320×160 → 5e-4. Default: 2e-4."
        )

        param("Omega — SOR Relaxation Factor")
        body(
            "Relaxation factor for the BFM pressure Poisson (Jacobi) solver. "
            "Values below 1.0 are under-relaxed (stable, slower convergence per "
            "sweep); the value is clamped to [0.05, 1.0]. 0.6 is "
            "stable across all grid sizes. Default: 0.6."
        )

        param("Convergence Residual")
        body(
            "Velocity residual threshold for declaring steady-state convergence. "
            "If the mean velocity change per iteration drops below this value the "
            "solver stops early. For periodic/vortex-shedding flows this threshold "
            "is rarely met. Default: 1e-8."
        )

        param("Chord (m, blank = auto)")
        body(
            "Physical chord length in metres, used to scale lift/drag forces and "
            "the flowfield scale bar. Default: 1.0 m (a realistic full-scale chord). "
            "Leave blank to let the solver derive it from Re × ν_air / V "
            "(e.g. Re=100, V=100 m/s → chord ≈ 0.015 mm — physically tiny but "
            "dimensionlessly correct). The field auto-fills with the computed value "
            "whenever Re or Velocity changes; typing your own value overrides it."
        )

        param("Solver — Body-fitted (O-grid)")
        body(
            "A curvilinear O-grid wraps directly around the airfoil surface, giving "
            "good boundary-layer resolution and clean pressure and vorticity fields "
            "for all symmetric and cambered airfoils.\n\n"
            "At Re ≥ 500 the solver automatically activates the Menter k-ω SST "
            "turbulence model. This adds two transport equations (turbulent kinetic "
            "energy k and specific dissipation rate ω) solved every timestep, "
            "plus a spatially-varying eddy viscosity ν_t that augments the molecular "
            "viscosity in both momentum and heat-transfer equations. The model "
            "blends k-ω behaviour near the wall with k-ε behaviour in the freestream "
            "— the same approach used in OpenFOAM, Fluent, and SU2."
        )

        param("Airfoil")
        body(
            "NACA 4-digit series airfoil used in the simulation.\n"
            "  0012 — symmetric, no camber. Baseline for AoA studies.\n"
            "  2412 — 2 % camber at 40 % chord. Slight positive Cl at AoA=0.\n"
            "  4412 — 4 % camber. Stronger camber lift, used in many light aircraft.\n"
            "  0006 — thin symmetric section (6 % thickness), lower drag."
        )

        param("Wall Temp (K)")
        body(
            "Surface temperature of the airfoil in Kelvin. Used by the passive "
            "scalar energy equation to set the wall thermal boundary condition. "
            "Higher values increase the surface-to-air temperature difference, "
            "which raises the Nusselt number. Default: 320 K."
        )

        param("Air Temp (K)")
        body(
            "Freestream air temperature in Kelvin. Sets the inflow and far-field "
            "thermal boundary condition for the energy equation. "
            "Default: 300 K (≈ 27 °C)."
        )

        param("Grid Size")
        body(
            "BFM O-grid resolution as circumferential × radial cells. The flowfield "
            "always renders at 320×320 pixels regardless of solve resolution.\n\n"
            "  BFM  64× 32 — coarsest, fastest. Quick exploration.\n"
            "  BFM  96× 48 — recommended starting point.\n"
            "  BFM 128× 64 — good standard accuracy.\n"
            "  BFM 256×128 — fine near-wall resolution.\n"
            "  BFM 320×160 — highest available; best for high-Re boundary layers.\n\n"
            "Finer grids resolve the boundary layer better and cost more time and "
            "memory. With local time-stepping standard (v0.1.2.0) the finer grids "
            "are far cheaper than before because each cell advances at its own "
            "stable step. All grids use an R = 15-chord far-field radius, sinh "
            "near-wall stretching, and (at Re ≥ 500) k-ω SST with the automatic "
            "wall treatment that blends the viscous-sublayer and log-law ω limits."
        )

        param("History Row Interval")
        body(
            "Iterations between rows added to the history table on the right. "
            "Lower values give more detail; higher values reduce clutter. "
            "Default: 50."
        )

        param("Show/Update flow field")
        body(
            "Toggles the live colour-mapped flowfield display. Unchecking saves "
            "rendering time on slow machines. The field selector above the canvas "
            "chooses which quantity is mapped: u-velocity, speed |V|, pressure, "
            "vorticity, or temperature. The colour scale runs blue (low) → red "
            "(high); tick labels on the colour bar are in physical units."
        )

        param("Show geometry outline")
        body(
            "Draws a white outline around the airfoil boundary and fills the "
            "interior with dark grey, so interpolated values inside the solid body "
            "do not mislead. The scale bar in the bottom-left corner shows 1 chord "
            "length in physical units (e.g. '1 chord  (1.00 m)')."
        )

        h2("Toolbar Calculators")
        body(
            "Three helper windows in the toolbar size your inputs correctly:"
        )
        param("Re Calculator")
        body(
            "Enter real-world velocity (m/s), chord length (m), and air temperature "
            "(°C). It computes the kinematic viscosity of air via Sutherland's law "
            "and then Re = V × L / ν, shows the flow regime and a real-world "
            "equivalent, and the 'Apply Re to Solver' button drops the value into "
            "the Re field."
        )
        param("dt Calculator")
        body(
            "Pick a BFM grid and a CFL target (0.05 is the solver's stable cap for "
            "the default scheme). It finds the smallest cell on that grid (the "
            "sinh-stretched first radial cell or the circumferential spacing) and "
            "returns dt = CFL × min_face, plus the number of steps per chord-"
            "crossing. The solver re-clamps any dt to its own stable limit, so this "
            "only ever suggests a safe value. 'Apply' writes it to the dt field."
        )
        param("Max Iterations Calculator")
        body(
            "Pick a BFM grid, a target number of chord-crossings (15 is a good "
            "default — lift is ~95%+ developed by then), and a dt (blank uses the "
            "CFL-max). It returns the required Max Iterations as crossings ÷ dt. "
            "Use it so a run develops far enough for Cl to be accurate rather than "
            "stopping early and under-reading lift."
        )

        h2("Accuracy & Margins of Error")
        body(
            "LunarCFD is a research and learning tool. The figures below are from "
            "this build's validation runs against published NACA polars (Abbott & "
            "von Doenhoff / NASA TMR), fully developed (~15 chord-crossings) with "
            "the standard settings. Tested up to Re ≈ 500 000 — relatively low for "
            "aerospace — with spot checks near Re = 1 000 000."
        )

        param("Lift (Cl) — within a few percent")
        body(
            "For attached flow (AoA ≲ 8°) Cl lands within roughly ±5% of published "
            "data. Measured: NACA 0012 at Re = 1e6 — AoA 0° → Cl ≈ 0; 4° → 0.416 "
            "(ref 0.43, 97%); 8° → 0.800 (ref 0.84, 95%). NACA 2412 at Re = 5e5, "
            "AoA 0° → Cl ≈ 0.21 (ref ~0.22, 95%). The lift curve has the right "
            "shape — zero for symmetric sections at AoA 0, linear with angle and "
            "camber, equal-and-opposite for ±AoA, correct zero-lift angle. "
            "IMPORTANT: lift develops slowly (the Wagner effect); if a run is "
            "stopped too early Cl reads low. Use the Max Iterations Calculator to "
            "size the run to ~15 chord-crossings, or let the auto-stop settle it."
        )

        param("Drag (Cd) — over-predicted; use as a guide")
        body(
            "Cd is the least reliable output. It is over-predicted at turbulent Re, "
            "typically by a factor of ~1.3–2× the reference (e.g. 0012 AoA 4° gives "
            "Cd ≈ 0.019 vs a reference ~0.009). Three causes, all inherent to the "
            "method: (1) the flow is treated as fully turbulent from the leading "
            "edge — no laminar run / transition — which inflates skin friction; "
            "(2) the solver is 2D, while turbulent drag is partly a 3D phenomenon; "
            "(3) 2D steady RANS over-predicts drag even in professional tools. Use "
            "Cd comparatively (which airfoil/AoA has more drag) or as an order-of-"
            "magnitude estimate, not for a precision drag budget."
        )

        param("Pitching moment (Cm) — matches published")
        body(
            "Quarter-chord Cm comes out right: ≈ 0 for symmetric sections at all "
            "attached angles, and for cambered sections NACA 2412 ≈ −0.05 and "
            "4412 ≈ −0.10, matching published values (measured 2412 AoA 0° → −0.051)."
        )

        param("Steady-only — no unsteady quantities")
        body(
            "With local time-stepping standard (v0.1.2.0) the solver targets the "
            "STEADY state and is not time-accurate. Vortex shedding, Strouhal "
            "number, and the Cl oscillation amplitude (σ) are therefore not "
            "physically meaningful; Strouhal normally shows '—'. If you need "
            "time-accurate shedding, that requires a build with local time-stepping "
            "turned off (a future option)."
        )

        param("Stall, high AoA, and very high Re")
        body(
            "Steady 2D RANS cannot reliably predict separated flow. Past stall "
            "(typically AoA > 12–15° for NACA 0012) the solver still returns a "
            "number, but it is not trustworthy at any Re or grid size — Cl plateaus/"
            "drops and Cd spikes; the trends are qualitatively right, the values "
            "are not. Above the tested band (Re > ~1e6) the same 2D/steady-RANS "
            "and fully-turbulent assumptions apply, so treat results as "
            "extrapolation."
        )

        param("Summary")
        body(
            "Trust:      Cl and Cm for attached flow (AoA ≲ 8°), pressure-field\n"
            "            shape, Nusselt trends.\n"
            "Use a guide: Cd (over-predicted ~1.3–2×).\n"
            "Don't trust: anything post-stall; unsteady quantities (Strouhal, σ);\n"
            "            absolute Cd for a drag budget."
        )

        h2("Output Values")

        param("Cl — Lift Coefficient")
        body(
            "Dimensionless lift force normalised by dynamic pressure and chord "
            "length. Positive means upward lift. For a symmetric airfoil at AoA=0, "
            "Cl ≈ 0."
        )

        param("Cd — Drag Coefficient")
        body(
            "Dimensionless drag in the freestream direction. Always positive. "
            "Includes pressure drag and viscous drag as resolved by the grid.\n\n"
            "WARNING — Cd is significantly less reliable than Cl at high Re. "
            "2D RANS (which this solver uses) typically over-predicts drag by "
            "20–50% at Re > 500 000 because: (1) skin-friction drag in a "
            "turbulent boundary layer is inherently a 3D phenomenon; (2) the "
            "solver assumes the flow is fully turbulent from the leading edge "
            "(no laminar run, no transition) which inflates friction drag; "
            "(3) any wake unsteadiness or separation is suppressed by the "
            "2D steady RANS assumption. Use Cd as an order-of-magnitude guide "
            "only above Re ≈ 500 000. Cl is much more trustworthy."
        )

        param("L/D — Lift-to-Drag Ratio")
        body("Cl ÷ Cd. Higher values indicate better aerodynamic efficiency.")

        param("Cm — Pitching Moment Coefficient")
        body(
            "Moment about the quarter-chord point (x = LE + 0.25 × chord). "
            "Positive Cm is nose-up; cambered sections are nose-down (negative) — "
            "NACA 2412 ≈ −0.05, 4412 ≈ −0.10, matching published values. For a "
            "symmetric airfoil (NACA 0012) Cm ≈ 0 at all attached angles of attack."
        )

        param("Lift / Drag — Physical Forces (N/m)")
        body(
            "Real-world lift and drag per unit span in Newtons per metre. "
            "Computed as Cl/Cd × ½ρV² × chord, where ρ is derived from "
            "Air Pressure via the ideal gas law. With chord=1 m and V=100 m/s, "
            "forces are directly readable (e.g. Cl=0.23 → Lift ≈ 17 N/m). "
            "The display switches to scientific notation automatically when "
            "values fall below 0.001 N/m (e.g. at very small auto-derived chords)."
        )

        param("Cl Amp (σ) — Lift Oscillation Amplitude")
        body(
            "Standard deviation of Cl over the last averaging window. "
            "NOTE (v0.1.2.0): with local time-stepping standard the solver is "
            "steady, so this stays near zero and is not a physical shedding "
            "amplitude — it only reflects the residual settling."
        )

        param("Strouhal — Strouhal Number")
        body(
            "NOTE (v0.1.2.0): the standard settings target the STEADY state and "
            "are not time-accurate, so a meaningful shedding frequency cannot be "
            "measured — Strouhal will normally show '—'. The description below "
            "applies only to a time-accurate (non-local-time-stepping) build.\n\n"
            "St = f × chord / V  — a dimensionless vortex-shedding frequency.\n\n"
            "f is the rate at which vortices are shed alternately from the upper "
            "and lower surfaces into the wake (von Kármán vortex street). "
            "Dividing by chord/V normalises by geometry and speed, making St "
            "comparable across cases: St ≈ 0.2 is typical for bluff bodies "
            "(cylinders, flat plates); lifting airfoils at moderate AoA give "
            "St ≈ 0.1–0.3.\n\n"
            "At low Re or AoA=0 the flow is steady — no shedding, no oscillating "
            "Cl — so the display shows '—'. It only appears once enough periodic "
            "Cl samples have been collected to identify a reliable frequency. "
            "A large Cl Amp (σ) alongside a finite St confirms genuine periodic "
            "shedding rather than numerical noise."
        )

        param("Nusselt — Nusselt Number")
        body(
            "Nu = h × chord / k  — ratio of convective to conductive heat transfer "
            "over one chord length.\n\n"
            "h is the surface heat-transfer coefficient; k is the thermal "
            "conductivity of air. Nu = 1 means the moving fluid adds no advantage "
            "over a completely stagnant layer; higher Nu means flow is carrying "
            "heat away much more efficiently. Typical values: Nu ≈ 2–10 for "
            "laminar airfoil flow; much higher with vortex shedding because shed "
            "vortices continuously sweep fresh cool fluid against the hot surface.\n\n"
            "Requires Wall Temp ≠ Air Temp (shows '—' when the difference is "
            "less than 0.1 K). The Temperature field display shows "
            "ΔT = T − T_air (blue=0, red=T_wall−T_air) to maximise contrast."
        )

        txt.config(state=tk.DISABLED)

        tk.Button(win, text="Close Help", command=win.destroy,
                  bg="#ddeeff", font=("Arial", 10), padx=12, pady=4
                  ).pack(pady=(4, 10))


def launch_main_gui():
    root = tk.Tk()
    app = MainWindowApp(root)
    root.mainloop()
