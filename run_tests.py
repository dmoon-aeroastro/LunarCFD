"""LunarCFD Integration Test Suite (Body-fitted O-grid solver).

Runs the BFM solver against a mock GUI and validates physical correctness:

  - NACA 0012 at AoA=0   -> Cl ~= 0  (symmetric airfoil, zero angle of attack)
  - NACA 0012 at AoA=+5  -> Cl > 0   (positive lift), Cd > 0, L/D > 0
  - NACA 2412 (cambered) -> Cl > 0 at AoA=0 (camber lift)
  - All 5 display matrix fields present (u, speed, pressure, vorticity, temperature)
  - Nusselt number positive and finite when T_wall != T_inf

The legacy Cartesian immersed-boundary solver has been removed; this suite
covers the O-grid (BFM) solver only.

Run from C:\\LunarCFD_Local with:
    py -3.11 run_tests.py
"""

import sys
import math
import threading
import time
import numpy as np
from collections import namedtuple

# Force UTF-8 output so Unicode characters print cleanly on Windows
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, "C:/LunarCFD_Local")
from solver.core_bfm import run_bfm_simulation

# ── Mock GUI ──────────────────────────────────────────────────────────────────

class _MockRoot:
    """Executes root.after() callbacks synchronously (no Tk event loop needed)."""
    def after(self, delay, func, *args):
        func(*args)

class MockGUI:
    def __init__(self):
        self.kill_event     = threading.Event()
        self.pause_event    = threading.Event()
        self.solver_running = True
        self.root           = _MockRoot()
        self.result         = {}
        self.last_live      = {}   # last update_live_results call

    def update_live_metrics(self, i, res, elapsed):
        pass

    def update_live_results(self, matrices, cl, cd, ld,
                            cm=0.0, lift_n=0.0, drag_n=0.0,
                            cl_std=0.0, st=float('nan'), nu_val=0.0):
        self.last_live = dict(matrices=matrices, cl=cl, cd=cd, ld=ld,
                              cm=cm, lift_n=lift_n, drag_n=drag_n,
                              cl_std=cl_std, st=st, nu_val=nu_val)

    def append_history_row(self, i, res, cl, cd, ld):
        pass

    def set_body_mask(self, mask):
        self.body_mask = mask

    def display_final_results(self, matrices, cl, cd, ld,
                              cm=0.0, lift_n=0.0, drag_n=0.0,
                              cl_std=0.0, st=float('nan'), nu_val=0.0,
                              state="Complete"):
        self.result = dict(matrices=matrices, cl=cl, cd=cd, ld=ld,
                           cm=cm, lift_n=lift_n, drag_n=drag_n,
                           cl_std=cl_std, st=st, nu_val=nu_val,
                           state=state)


# ── Helpers ───────────────────────────────────────────────────────────────────

Result = namedtuple("Result", [
    "cl", "cd", "ld", "cm", "lift_n", "drag_n", "cl_std", "st", "nu_val",
    "state", "matrices", "gui"
])

failures = []

def check(name, condition, detail=""):
    tag    = "PASS" if condition else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"    {tag}  {name}{suffix}")
    if not condition:
        failures.append(name)


def run_bfm_case(label, **params):
    """Run one BFM solver case and return a Result namedtuple."""
    defaults = dict(vel=25.0, pressure=101325, max_iters=4000,
                    conv_res=1e-6, dt=0.2, hist_interval=50,
                    t_wall=320.0, t_inf=300.0,
                    n_xi=64, n_eta=32, ogrid_R=8.0)
    defaults.update(params)
    gui = MockGUI()
    t0  = time.perf_counter()
    run_bfm_simulation(gui, defaults)
    elapsed = time.perf_counter() - t0
    r  = gui.result
    nan = float("nan")
    cl     = r.get("cl",     nan)
    cd     = r.get("cd",     nan)
    ld     = r.get("ld",     nan)
    cm     = r.get("cm",     nan)
    lift_n = r.get("lift_n", nan)
    drag_n = r.get("drag_n", nan)
    cl_std = r.get("cl_std", nan)
    st     = r.get("st",     nan)
    nu_val = r.get("nu_val", nan)
    state  = r.get("state",  "unknown")
    mats   = r.get("matrices", {})
    st_str = "—" if (not np.isfinite(st)) else f"{st:.3f}"
    print(f"  [BFM {label:<32s}]  Cl={cl:+.4f}  Cd={cd:.4f}  L/D={ld:+.4f}"
          f"  Nu={nu_val:.2f}  St={st_str}  [{state}]  {elapsed:.1f}s")
    return Result(cl, cd, ld, cm, lift_n, drag_n, cl_std, st, nu_val,
                  state, mats, gui)


# ── Test cases ────────────────────────────────────────────────────────────────

print("=" * 65)
print("LunarCFD Integration Test Suite (BFM)")
print("=" * 65)

print("\n[1] BFM: NACA 0012 at AoA=0 should produce Cl ~= 0")
bfm0 = run_bfm_case("NACA0012 Re=100 AoA=0", re=100, aoa=0,  airfoil="0012")
check("BFM Cl ~= 0  (|Cl| < 0.08)",  abs(bfm0.cl) < 0.08,  f"Cl={bfm0.cl:.4f}")
check("BFM Cd > 0",                   bfm0.cd > 0,           f"Cd={bfm0.cd:.4f}")
check("BFM result finite",            np.isfinite(bfm0.cl) and np.isfinite(bfm0.cd))

print("\n[2] BFM: NACA 0012 at AoA=+5 should produce Cl > 0")
bfm5 = run_bfm_case("NACA0012 Re=100 AoA=+5", re=100, aoa=5, airfoil="0012")
check("BFM Cl > 0",     bfm5.cl > 0,   f"Cl={bfm5.cl:.4f}")
check("BFM Cd > 0",     bfm5.cd > 0,   f"Cd={bfm5.cd:.4f}")
check("BFM L/D > 0",    bfm5.ld > 0,   f"L/D={bfm5.ld:.4f}")
check("BFM Cl in [0.04, 0.50]",
      0.04 <= bfm5.cl <= 0.50,         f"Cl={bfm5.cl:.4f}")

print("\n[3] BFM: NACA 2412 (cambered) at AoA=0 should produce Cl > 0")
bfm24 = run_bfm_case("NACA2412 Re=100 AoA=0", re=100, aoa=0, airfoil="2412")
check("BFM NACA2412 Cl > 0 at AoA=0",  bfm24.cl > 0,  f"Cl={bfm24.cl:.4f}")
check("BFM NACA2412 Cd > 0",           bfm24.cd > 0,  f"Cd={bfm24.cd:.4f}")

print("\n[4] BFM: All 5 matrix fields present in BFM output")
bfm_mats = bfm5.matrices
for field in sorted({"u", "speed", "pressure", "vorticity", "temperature"}):
    check(f"BFM field '{field}' present",
          field in bfm_mats and isinstance(bfm_mats[field], np.ndarray),
          f"{'OK' if field in bfm_mats else 'MISSING'}")

print("\n[5] BFM: Nusselt number positive and finite")
check("BFM Nu(AoA=0) > 0",   bfm0.nu_val > 0,   f"Nu={bfm0.nu_val:.2f}")
check("BFM Nu(AoA=+5) > 0",  bfm5.nu_val > 0,   f"Nu={bfm5.nu_val:.2f}")
check("BFM Nu finite",
      np.isfinite(bfm0.nu_val) and np.isfinite(bfm5.nu_val),
      f"Nu0={bfm0.nu_val:.2f} Nu5={bfm5.nu_val:.2f}")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
if failures:
    print(f"RESULT: {len(failures)} FAILURE(S): {', '.join(failures)}")
    sys.exit(1)
else:
    print("RESULT: All tests passed.")
    sys.exit(0)
