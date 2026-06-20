"""LunarCFD Verification & Validation harness.

Runs an angle-of-attack sweep for NACA 0012 at a fixed Reynolds number and
grid, and compares the solver's Cl/Cd against published reference values, so
that scheme/grid/setting changes can be judged by AGREEMENT WITH DATA rather
than by eye.  This is the objective basis a publication needs.

Usage:
    python val_benchmark.py [scheme] [grid] [re]
      scheme : "baseline" (central_jst) | "rc" (central_jst + Rhie-Chow)
      grid   : "96" | "128" | "256"  (BFM n_xi; n_eta = n_xi/2)
      re     : Reynolds number (default 1e6)

Reference NACA 0012 lift/drag, ~Re 1e6, attached regime (alpha < ~10 deg).
These are representative experimental / panel-with-BL (XFOIL-class) values
collated from Abbott & von Doenhoff (1959) "Theory of Wing Sections" and the
NASA Turbulence Modeling Resource NACA 0012 case; they are reference targets,
not solver output.  Lift-curve slope ~0.105-0.11 /deg; Cd_min ~0.008.
"""
import sys, os, threading
sys.path.insert(0, os.path.dirname(__file__))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
from solver.core_bfm import run_bfm_simulation

# Reference NACA 0012 (Re ~ 1e6).  alpha[deg] : (Cl_ref, Cd_ref)
REF_0012 = {
    0: (0.000, 0.0080),
    2: (0.210, 0.0083),
    4: (0.430, 0.0090),
    6: (0.640, 0.0102),
    8: (0.840, 0.0123),
}

class _Root:
    @staticmethod
    def after(d, f, *a): f(*a)

class _GUI:
    def __init__(s):
        s.root=_Root(); s.kill_event=threading.Event(); s.pause_event=threading.Event()
        s.solver_running=True; s.last_res=9.9; s.res={}
    def update_live_metrics(s, i, res, e): s.last_res=res
    def update_live_results(s, *a): pass
    def set_body_mask(s, m): pass
    def append_history_row(s, *a): pass
    def display_final_results(s, m, cl, cd, ld, *a, **k):
        fin = all(np.all(np.isfinite(m[k2])) for k2 in ("pressure","u","speed"))
        s.res = dict(cl=cl, cd=cd, fin=fin)

def run_case(airfoil, aoa, re, nxi, neta, rc, iters):
    p = {"airfoil":airfoil, "aoa":aoa, "re":re, "n_xi":nxi, "n_eta":neta,
         "dt":2e-4, "max_iters":iters, "hist_interval":iters, "t_wall":320.0,
         "t_inf":300.0, "vel":30.0, "pressure":101325.0, "chord_m":0.5,
         "rhie_chow":rc}
    g=_GUI(); run_bfm_simulation(g, p); return g.res, g.last_res

def main():
    scheme = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    gridn  = sys.argv[2] if len(sys.argv) > 2 else "128"
    re     = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0e6
    nxi    = int(gridn);  neta = nxi // 2
    rc     = (scheme == "rc")
    alphas = [0, 4, 8]            # enough to fit a lift slope + check a mid point
    iters  = 9000

    print(f"=== V&V: NACA0012  Re={re:.0e}  BFM {nxi}x{neta}  scheme={scheme} ===", flush=True)
    print(f"{'a':>3} | {'Cl_solver':>9} {'Cl_ref':>7} {'Cl_err%':>8} | "
          f"{'Cd_solver':>9} {'Cd_ref':>7} {'Cd_x':>5} | {'res':>8}", flush=True)
    cls=[]
    for a in alphas:
        res, lr = run_case("0012", float(a), re, nxi, neta, rc, iters)
        clr, cdr = REF_0012[a]
        cls.append(res['cl'])
        clerr = 100.0*(res['cl']-clr)/clr if clr != 0 else float('nan')
        cdx   = res['cd']/cdr if cdr != 0 else float('nan')
        flag  = "" if res['fin'] else "  <NaN!>"
        print(f"{a:>3} | {res['cl']:>9.4f} {clr:>7.3f} {clerr:>7.1f}% | "
              f"{res['cd']:>9.5f} {cdr:>7.4f} {cdx:>4.2f}x | {lr:>8.1e}{flag}", flush=True)
    # lift-curve slope (deg) from a=0..8
    slope = (cls[-1]-cls[0]) / (alphas[-1]-alphas[0])
    print(f"--- lift-slope = {slope:.4f} /deg  (reference ~0.105-0.110 /deg, "
          f"i.e. {100*slope/0.105:.0f}% of reference)", flush=True)
    print("=== V&V DONE ===", flush=True)

if __name__ == "__main__":
    main()
