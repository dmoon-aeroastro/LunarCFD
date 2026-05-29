import tkinter as tk
from tkinter import messagebox
import threading
import psutil
import os
import sys
import time
import colorsys
import numpy as np
from file_io.session import SessionManager
from solver.core import run_fluid_simulation

class MainWindowApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LunarCFD v0.4.0.0")
        self.root.geometry("1400x768")
        self.kill_event = threading.Event()
        self.pause_event = threading.Event()
        self.solver_running = False
        self._help_win = None
        self.config_ram_limit = (psutil.virtual_memory().available / (1024 * 1024)) * 0.75
        self.session_io = SessionManager(self)
        self.build_ui_layout()
        self.start_watchdog()

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

        self.ent_re       = lbl_entry("Re:", "100")
        self.ent_vel      = lbl_entry("Velocity (m/s):", "25.0")
        self.ent_aoa      = lbl_entry("AoA (deg):", "5.0")
        self.ent_pressure = lbl_entry("Air Pressure (Pa):", "101325")
        self.ent_iters    = lbl_entry("Max Iterations:", "3000")
        self.ent_dt       = lbl_entry("dt (time step):", "0.2")
        self.ent_omega    = lbl_entry("Omega (Relaxation):", "0.6")
        self.ent_conv_res = lbl_entry("Convergence Residual:", "1e-6")
        self.ent_theta    = lbl_entry("THETA (0=central, 1=upwind):", "0.5")

        # ── Grid selector ─────────────────────────────────────────────────────
        tk.Label(panel_frame, text="Grid Size:", bg="#f5f5f5", anchor="w").pack(fill=tk.X, padx=5)
        self.var_grid = tk.StringVar(value="160x160")
        grid_menu = tk.OptionMenu(panel_frame, self.var_grid, "80x80", "160x160", "320x320")
        grid_menu.config(anchor="w", bg="#f5f5f5")
        grid_menu.pack(fill=tk.X, padx=5)

        self.ent_hist_interval = lbl_entry("History Row Interval:", "50")

        self.var_show_geom = tk.BooleanVar(value=True)
        self.var_show_geom.trace_add("write", lambda *_: self._refresh_matrix_overlay())
        tk.Checkbutton(panel_frame, text="Show geometry outline",
                       variable=self.var_show_geom,
                       bg="#f5f5f5", anchor="w").pack(fill=tk.X, padx=5)

        # ── Status button (bottom of left panel) ──────────────────────────────
        self.btn_status = tk.Label(panel_frame, text="Ready",
                                   bg="#cccccc", fg="#333333",
                                   font=("Arial", 11, "bold"),
                                   relief=tk.RAISED, bd=2, pady=8)
        self.btn_status.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=(5, 5))

        # ── Central dashboard ─────────────────────────────────────────────────
        self.display_dashboard = tk.Frame(self.root, bg="#ffffff", bd=2, relief=tk.SUNKEN)
        self.display_dashboard.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        tk.Label(self.display_dashboard,
                 text="Performance Monitors [Target: NACA 0012]",
                 font=("Arial", 12, "bold"), bg="#ffffff").pack(anchor="w", padx=5, pady=(5, 2))

        def metric_entry(initial):
            """Read-only entry that users can select and copy."""
            var = tk.StringVar(value=initial)
            e = tk.Entry(self.display_dashboard, textvariable=var, state="readonly",
                         readonlybackground="#ffffff", relief=tk.FLAT,
                         font=("Arial", 10), fg="#000000")
            e.pack(anchor="w", fill=tk.X, padx=8, pady=1)
            return var

        self.var_iter = metric_entry("Current Iteration: 0")
        self.var_time = metric_entry("Elapsed Time: 0.00s")
        self.var_res  = metric_entry("Residual: Waiting...")
        self.var_cl   = metric_entry("Cl: Waiting...")
        self.var_cd   = metric_entry("Cd: Waiting...")
        self.var_ld   = metric_entry("L/D Ratio: Waiting...")

        tk.Label(self.display_dashboard, text="Velocity Flowfield Matrix Preview:",
                 font=("Arial", 10, "bold"), bg="#ffffff").pack(anchor="w", padx=5, pady=(10, 0))

        # Colour key – pack BOTTOM first so matrix_frame can expand to fill the middle
        self.key_canvas = tk.Canvas(self.display_dashboard, height=52, bg="#111111",
                                    highlightthickness=0)
        self.key_canvas.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 5))

        # Smaller font so the full 40×40 slice fits without wrapping; scrollbars for overflow
        matrix_frame = tk.Frame(self.display_dashboard, bg="#111111")
        matrix_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        _msy = tk.Scrollbar(matrix_frame)
        _msy.pack(side=tk.RIGHT, fill=tk.Y)
        _msx = tk.Scrollbar(matrix_frame, orient=tk.HORIZONTAL)
        _msx.pack(side=tk.BOTTOM, fill=tk.X)
        self.matrix_text_box = tk.Text(matrix_frame, font=("Courier", 7), bg="#111111",
                                       fg="#888888",
                                       state=tk.DISABLED, wrap=tk.NONE,
                                       yscrollcommand=_msy.set, xscrollcommand=_msx.set)
        self.matrix_text_box.pack(fill=tk.BOTH, expand=True)
        _msy.config(command=self.matrix_text_box.yview)
        _msx.config(command=self.matrix_text_box.xview)

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
        self.var_iter.set("Current Iteration: 0")
        self.var_time.set("Elapsed Time: 0.00s")
        self.var_res.set("Residual: Waiting...")
        self.var_cl.set("Cl: Waiting...")
        self.var_cd.set("Cd: Waiting...")
        self.var_ld.set("L/D Ratio: Waiting...")

        # Clear history table
        self.history_text.config(state=tk.NORMAL)
        self.history_text.delete("1.0", tk.END)
        self.history_text.config(state=tk.DISABLED)

        try:
            conv_res = float(self.ent_conv_res.get())
        except ValueError:
            conv_res = 1e-6
        try:
            theta = float(self.ent_theta.get())
            theta = max(0.0, min(1.0, theta))
        except ValueError:
            theta = 0.5
        try:
            dt_val = float(self.ent_dt.get())
            dt_val = max(0.001, min(0.5, dt_val))   # clamp to sane range
        except ValueError:
            dt_val = 0.2

        try:
            hist_interval = max(1, int(self.ent_hist_interval.get()))
        except ValueError:
            hist_interval = 50

        payload = {
            "re":            float(self.ent_re.get()),
            "vel":           float(self.ent_vel.get()),
            "aoa":           float(self.ent_aoa.get()),
            "pressure":      float(self.ent_pressure.get()),
            "omega":         float(self.ent_omega.get()),
            "max_iters":     int(self.ent_iters.get()),
            "conv_res":      conv_res,
            "theta":         theta,
            "dt":            dt_val,
            "grid":          self.var_grid.get().split("x")[0],
            "hist_interval": hist_interval,
        }
        threading.Thread(target=run_fluid_simulation, args=(self, payload), daemon=True).start()

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
        g = self.var_grid.get()
        if g == "80x80":   return slice(20, 60),  slice(0,   60)  # chord=32 cx=20 TE@52 ✓
        if g == "320x320": return slice(120, 200), slice(60, 220)  # chord=128 cx=80 TE@208 (+12 wake)
        return slice(60, 100), slice(30, 115)                      # chord=64 cx=40 TE@104 (+11 wake)

    def _render_matrix_colored(self, data):
        """Render a 2-D numpy array into matrix_text_box with a rainbow gradient.

        Blue  (HSV hue 0.667) = 0 m/s
        Red   (HSV hue 0.000) = max speed in the current slice

        Colours are quantized to 200 levels and reused across cells in the same
        frame so only O(200) tag_configure calls are made per refresh.
        """
        tb = self.matrix_text_box
        tb.config(state=tk.NORMAL)
        tb.delete("1.0", tk.END)

        speeds = np.abs(data)
        max_sp = float(speeds.max())
        if max_sp < 1e-10:
            max_sp = 1.0          # prevent division by zero in still flow

        N = 200                   # colour quantization levels
        show_geom = self.var_show_geom.get()
        seen = set()
        nrows, ncols = data.shape

        # Prefer the exact boolean body mask passed from the solver.
        # Falls back to velocity-threshold detection before the first mask arrives.
        _mask = None
        if show_geom and hasattr(self, '_body_mask') and self._body_mask is not None:
            sr, sc = self._matrix_slice()
            _mask = self._body_mask[sr, sc]

        def _is_boundary(r, c):
            """True if this cell sits on the fluid/solid interface."""
            if _mask is not None:
                if not _mask[r, c]:
                    return False                  # fluid cell — not boundary
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < nrows and 0 <= nc < ncols:
                        if not _mask[nr, nc]:
                            return True           # body cell touching fluid
                return False
            # Fallback: velocity threshold
            if abs(float(data[r, c])) >= 0.05:
                return False
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < nrows and 0 <= nc < ncols:
                    if abs(float(data[nr, nc])) >= 0.05:
                        return True
            return False

        for r in range(nrows):
            for c in range(ncols):
                if c:
                    tb.insert(tk.END, " ")
                if show_geom and _is_boundary(r, c):
                    tag = "geom"
                    if tag not in seen:
                        tb.tag_configure(tag, foreground="#ffffff")
                        seen.add(tag)
                else:
                    lvl = int(abs(float(data[r, c])) / max_sp * (N - 1))
                    tag = f"c{lvl:03d}"
                    if tag not in seen:
                        norm       = lvl / (N - 1)
                        hue        = 0.667 * (1.0 - norm)
                        ri, gi, bi = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                        tb.tag_configure(
                            tag,
                            foreground=f"#{int(ri*255):02x}{int(gi*255):02x}{int(bi*255):02x}"
                        )
                        seen.add(tag)
                tb.insert(tk.END, f"{data[r, c]:4.1f}", tag)
            tb.insert(tk.END, "\n")

        tb.config(state=tk.DISABLED)
        self._update_color_key(max_sp)

    def _update_color_key(self, max_sp):
        """Redraw the colour-scale bar beneath the flowfield matrix.

        The bar runs blue (0 m/s) -> red (max_sp * real_vel m/s), with five
        tick labels evenly spaced.  Called automatically by _render_matrix_colored.
        """
        cv = self.key_canvas
        cv.update_idletasks()
        W = cv.winfo_width()
        if W < 10:
            W = 300          # fallback before widget is fully realized
        cv.delete("all")

        try:
            real_vel = float(self.ent_vel.get())
        except ValueError:
            real_vel = 25.0

        N = 200
        bar_y0, bar_y1 = 2, 22

        # Gradient bar — N coloured rectangles spanning the full width
        for lvl in range(N):
            norm = lvl / (N - 1)
            hue  = 0.667 * (1.0 - norm)
            ri, gi, bi = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            color = f"#{int(ri*255):02x}{int(gi*255):02x}{int(bi*255):02x}"
            x0 = int(lvl / N * W)
            x1 = max(x0 + 1, int((lvl + 1) / N * W))
            cv.create_rectangle(x0, bar_y0, x1, bar_y1, fill=color, outline="")

        # Tick marks and velocity labels
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            speed_ms = frac * max_sp * real_vel
            x        = int(frac * (W - 1))
            cv.create_line(x, bar_y1, x, bar_y1 + 5, fill="#888888")
            anchor = "nw" if frac == 0.0 else ("ne" if frac == 1.0 else "n")
            cv.create_text(x, bar_y1 + 6, text=f"{speed_ms:.1f}",
                           fill="#aaaaaa", font=("Arial", 14), anchor=anchor)

        # Unit label top-right
        cv.create_text(W - 2, bar_y0, text="m/s",
                       fill="#000000", font=("Arial", 14), anchor="ne")

    # ── GUI update callbacks (called from solver thread via root.after) ────────
    def update_live_metrics(self, i, res, elapsed):
        self.var_iter.set(f"Current Iteration: {i}")
        self.var_time.set(f"Elapsed Time: {elapsed:.2f}s")
        self.var_res.set(f"Residual: {res:.6f}")

    def set_body_mask(self, mask):
        """Receive the exact boolean body mask from the solver (once per run)."""
        self._body_mask = mask

    def _refresh_matrix_overlay(self):
        """Re-render the matrix immediately when the geometry toggle changes."""
        if not hasattr(self, '_last_matrix') or self._last_matrix is None:
            return
        sr, sc = self._matrix_slice()
        self._render_matrix_colored(self._last_matrix[sr, sc])

    def update_live_results(self, matrix, cl, cd, ld_ratio):
        """Refresh Cl/Cd/LD and flowfield preview during the run (every N iters)."""
        self._last_matrix = matrix
        self.var_cl.set(f"Cl: {cl:.4f}")
        self.var_cd.set(f"Cd: {cd:.4f}")
        self.var_ld.set(f"L/D Ratio: {ld_ratio:.4f}")
        sr, sc = self._matrix_slice()
        self._render_matrix_colored(matrix[sr, sc])

    def append_history_row(self, i, res, cl, cd, ld):
        """Add one row to the right-side iteration history table."""
        self.history_text.config(state=tk.NORMAL)
        row = f"{i:>7}  {res:>11.3e}  {cl:>9.4f}  {cd:>9.4f}  {ld:>9.3f}\n"
        self.history_text.insert(tk.END, row)
        self.history_text.see(tk.END)
        self.history_text.config(state=tk.DISABLED)

    def display_final_results(self, matrix, cl, cd, ld_ratio, state="Complete"):
        self._last_matrix = matrix
        self.solver_running = False
        self.pause_event.clear()
        self.btn_pause.config(text="Pause")
        self._set_status("failure" if state == "NaN error" else "success")
        self.var_cl.set(f"Cl: {cl:.4f}")
        self.var_cd.set(f"Cd: {cd:.4f}")
        self.var_ld.set(f"L/D Ratio: {ld_ratio:.4f}")
        sr, sc = self._matrix_slice()
        self._render_matrix_colored(matrix[sr, sc])


    # ── Status button helper ──────────────────────────────────────────────────
    def _set_status(self, state):
        """Update the colour-coded status label at the bottom of the left panel.

        state values:
          'ready'      — grey  (initial / after restart)
          'running'    — yellow (solver active)
          'paused'     — yellow (solver paused)
          'finalizing' — yellow (stop requested, draining)
          'success'    — green  (any clean completion)
          'failure'    — red    (NaN / solver error)
        """
        cfg = {
            "ready":      ("Ready",               "#cccccc", "#333333"),
            "running":    ("Simulation Running",  "#f0c040", "#222222"),
            "paused":     ("Paused",              "#f0c040", "#222222"),
            "finalizing": ("Finalizing...",       "#f0c040", "#222222"),
            "success":    ("Simulation Complete", "#4caf50", "#ffffff"),
            "failure":    ("Simulation Failed",   "#e53935", "#ffffff"),
        }
        text, bg, fg = cfg.get(state, cfg["ready"])
        self.btn_status.config(text=text, bg=bg, fg=fg)

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

        h1("LunarCFD")
        body(
            "LunarCFD is a 2D computational fluid dynamics simulator for airfoil "
            "analysis. It solves the incompressible Navier-Stokes equations using an "
            "explicit projection method on a structured Cartesian grid, modelling a "
            "NACA 0012 airfoil at user-defined flow conditions."
        )
        body(
            "The solver computes lift (Cl) and drag (Cd) coefficients and displays a "
            "live velocity flowfield. Angle of attack is applied by rotating the inflow "
            "velocity vector, keeping the airfoil geometry axis-aligned. Grid refinement "
            "from 80×80 to 320×320 improves accuracy at the cost of computation time."
        )

        h2("Simulation Parameters")

        param("Re — Reynolds Number")
        body(
            "Ratio of inertial to viscous forces. Low Re (e.g. 100) gives smooth "
            "laminar flow; higher Re (500+) produces vortex shedding and unsteady "
            "effects. The physical velocity and chord length are combined with air "
            "viscosity to reach the target Re inside the solver. Default: 100."
        )

        param("Velocity (m/s)")
        body(
            "Freestream speed in real-world units. Used only to convert the "
            "dimensionless simulation output into physical m/s values shown in the "
            "flowfield colour key. Does not affect the flow physics, which are "
            "governed by Re. Default: 25.0."
        )

        param("AoA (deg) — Angle of Attack")
        body(
            "Angle between freestream flow and the airfoil chord. Positive AoA "
            "produces positive lift (Cl > 0). Range: typically −20° to +20°. "
            "Default: 5.0."
        )

        param("Air Pressure (Pa)")
        body(
            "Reference atmospheric pressure for display purposes. Not used in "
            "the flow solver equations. Default: 101325 (sea-level standard)."
        )

        param("Max Iterations")
        body(
            "Maximum time-stepping iterations before the solver stops. The solver "
            "may stop earlier if convergence is detected. Increase for finer grids "
            "or flows that take longer to settle.\n"
            "  Suggested:  80×80 → 3000,  160×160 → 5000,  320×320 → 10000+."
        )

        param("dt — Time Step")
        body(
            "Dimensionless time increment per iteration. Smaller dt is more stable "
            "but needs more iterations to simulate the same physical time. Values "
            "of 0.1–0.3 are appropriate for most cases. Default: 0.2."
        )

        param("Omega — SOR Relaxation Factor")
        body(
            "Controls relaxation in the pressure Poisson solver (Successive "
            "Over-Relaxation). Values below 1.0 are under-relaxed (stable, slower); "
            "above 1.0 are over-relaxed (faster, less stable). 0.6 is stable across "
            "all grid sizes. Default: 0.6."
        )

        param("Convergence Residual")
        body(
            "Velocity residual threshold for declaring steady-state convergence. "
            "If the mean velocity change per iteration drops below this value the "
            "solver stops early. For periodic/vortex-shedding flows this threshold "
            "is rarely met. Default: 1e-6."
        )

        param("THETA — Advection Blending")
        body(
            "Blends central-difference (THETA=0) and upwind (THETA=1) advection. "
            "Central is more accurate; upwind is more diffusive but stable. A Van "
            "Leer flux limiter switches automatically to upwind at steep gradients "
            "regardless of THETA. Default: 0.5."
        )

        param("Grid Size")
        body(
            "Simulation grid resolution. Larger grids capture finer flow features "
            "and give more accurate Cl/Cd but are much slower.\n"
            "   80×80   — ~1–2 min.   Good for exploring parameters.\n"
            "  160×160  — ~10–15 min. Recommended for analysis.\n"
            "  320×320  — ~1–2 hr.   Best accuracy."
        )

        param("History Row Interval")
        body(
            "Iterations between rows added to the history table on the right. "
            "Lower values give more detail; higher values reduce clutter. "
            "Default: 50."
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
            "Includes pressure drag and viscous drag as resolved by the grid."
        )

        param("L/D — Lift-to-Drag Ratio")
        body("Cl ÷ Cd. Higher values indicate better aerodynamic efficiency.")

        txt.config(state=tk.DISABLED)

        tk.Button(win, text="Close Help", command=win.destroy,
                  bg="#ddeeff", font=("Arial", 10), padx=12, pady=4
                  ).pack(pady=(4, 10))


def launch_main_gui():
    root = tk.Tk()
    app = MainWindowApp(root)
    root.mainloop()
