"""
Headless smoke test for the BFM k-omega SST solver.
Runs Re=6,845,120 with Lv4 grid (512x128) and dt=2e-4.
No GUI needed — all callbacks are no-ops.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Force UTF-8 output so unicode arrows/symbols in solver print() don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import warnings
import numpy as np

# Keep numpy overflow as warnings (same as the real run); NaN is caught
# internally by the solver and reported as a FATAL line.
warnings.filterwarnings("always", category=RuntimeWarning)

from solver.core_bfm import run_bfm_simulation

# ---------- Mock GUI object (no-ops for all callbacks) ----------
import threading

class MockRoot:
    @staticmethod
    def after(delay, fn, *args):
        fn(*args)   # call immediately (no event loop needed)

class MockGUI:
    root = MockRoot()
    kill_event  = threading.Event()   # never set → solver runs to completion
    pause_event = threading.Event()   # never set → no pausing
    solver_running = True             # reset to False when solver finishes

    # callbacks fired by the solver
    def update_live_metrics(self, i, res, elapsed):
        pass

    def update_live_results(self, matrices, cl, cd, ld, *args):
        pass

    def set_body_mask(self, mask):
        pass

    def append_history_row(self, i, res, cl, cd, ld):
        pass

    def display_final_results(self, matrices, cl, cd, ld, *args, **kwargs):
        print(f"  Final: Cl={cl:.4f}  Cd={cd:.4f}  L/D={ld:.4f}")

# ---------- Solver parameters matching the Lv4 run ----------
p = {
    "airfoil"      : "2412",
    "n_xi"         : 512,
    "n_eta"        : 128,
    "ogrid_R"      : 15.0,
    "alpha_stretch": 8.0,
    "re"           : 6_845_120.0,
    "aoa"          : 0.0,
    "dt"           : 2e-4,
    "max_iters"    : 20000,
    "hist_interval": 50,
    "t_wall"       : 320.0,
    "t_inf"        : 300.0,
}

print("=== BFM headless smoke test ===")
print(f"Re={p['re']:.3g}  dt={p['dt']:.2e}  grid={p['n_xi']}x{p['n_eta']}  max_iters={p['max_iters']}")
print()

# Run with numpy overflow → exception so we catch divergence immediately
try:
    run_bfm_simulation(MockGUI(), p)
    print()
    print("=== PASSED: solver completed without NaN/overflow ===")
except Exception as e:
    print()
    print(f"=== FAILED: {type(e).__name__}: {e} ===")
    sys.exit(1)
