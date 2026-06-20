"""Cambered-airfoil V&V: NACA 2412 & 4412 vs published polars.

Symmetric 0012 only exercises the symmetry-preserving fix.  Cambered sections
test the real circulation physics: nonzero Cl at alpha=0, the zero-lift angle,
and the (nose-down, negative) quarter-chord pitching moment.

Reference Cl/Cd/Cm in the attached regime, high-Re (~3e6) wind-tunnel /
XFOIL-class values from Abbott & von Doenhoff (1959):
  2412: a_L0 ~ -2.1 deg, slope ~0.105/deg, Cm_c/4 ~ -0.05, Cd_min ~0.006
  4412: a_L0 ~ -4.0 deg, slope ~0.105/deg, Cm_c/4 ~ -0.10, Cd_min ~0.007
"""
import sys, os, threading
sys.path.insert(0, os.path.dirname(__file__))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
from solver.core_bfm import run_bfm_simulation

# alpha[deg] : (Cl_ref, Cd_ref, Cm_c4_ref)
REF = {
    "2412": {0: (0.22, 0.0062, -0.047), 4: (0.64, 0.0072, -0.049), 8: (1.05, 0.0110, -0.050)},
    "4412": {0: (0.42, 0.0070, -0.092), 4: (0.85, 0.0083, -0.095), 8: (1.25, 0.0125, -0.098)},
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
    def display_final_results(s, m, cl, cd, ld, cm, *a, **k):
        fin = all(np.all(np.isfinite(m[k2])) for k2 in ("pressure","u","speed"))
        s.res = dict(cl=cl, cd=cd, cm=cm, fin=fin)

def run_case(airfoil, aoa, re, nxi, neta, iters):
    p = {"airfoil":airfoil, "aoa":aoa, "re":re, "n_xi":nxi, "n_eta":neta,
         "dt":2e-4, "max_iters":iters, "hist_interval":iters, "t_wall":320.0,
         "t_inf":300.0, "vel":30.0, "pressure":101325.0, "chord_m":0.5,
         "rhie_chow":False}
    g=_GUI(); run_bfm_simulation(g, p); return g.res, g.last_res

def main():
    foil  = sys.argv[1] if len(sys.argv) > 1 else "2412"
    gridn = sys.argv[2] if len(sys.argv) > 2 else "128"
    re    = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0e6
    nxi   = int(gridn); neta = nxi // 2
    iters = 9000
    alphas = [0, 4, 8]

    print(f"=== V&V: NACA{foil}  Re={re:.0e}  BFM {nxi}x{neta} ===", flush=True)
    print(f"{'a':>3} | {'Cl':>8} {'Clref':>6} {'err%':>7} | "
          f"{'Cd':>8} {'Cdref':>6} {'x':>4} | {'Cm':>7} {'Cmref':>6} | {'res':>8}", flush=True)
    cls=[]
    for a in alphas:
        res, lr = run_case(foil, float(a), re, nxi, neta, iters)
        clr, cdr, cmr = REF[foil][a]
        cls.append(res['cl'])
        clerr = 100.0*(res['cl']-clr)/clr if clr else float('nan')
        cdx   = res['cd']/cdr if cdr else float('nan')
        flag  = "" if res['fin'] else "  <NaN!>"
        print(f"{a:>3} | {res['cl']:>8.4f} {clr:>6.2f} {clerr:>6.0f}% | "
              f"{res['cd']:>8.5f} {cdr:>6.4f} {cdx:>3.1f}x | "
              f"{res['cm']:>+7.4f} {cmr:>+6.3f} | {lr:>8.1e}{flag}", flush=True)
    slope = (cls[-1]-cls[0]) / (alphas[-1]-alphas[0])
    a_l0  = alphas[0] - cls[0]/slope if slope else float('nan')
    print(f"--- slope={slope:.4f}/deg (ref ~0.105) ; "
          f"Cl(a=0)={cls[0]:+.3f} (camber lift) ; a_L0~{a_l0:+.2f}deg", flush=True)
    print("=== V&V DONE ===", flush=True)

if __name__ == "__main__":
    main()
