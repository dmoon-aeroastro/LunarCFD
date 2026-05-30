import numpy as np
import time
import os
import psutil
import numba
from numba import njit, prange
from solver.geometry import get_naca_mask

@njit(parallel=True)
def _sor_step(press, pn, b, mask, omega, ny, nx):
    """
    Parallel Jacobi SOR step for the pressure Poisson equation.

    Reads exclusively from pn (previous-iteration snapshot) and writes interior
    fluid cells to press.  Because every cell reads only from pn — which is
    never written inside this function — all row-level updates are independent
    and safe to parallelise with prange.

    This is mathematically identical to the original numpy SOR (same Jacobi
    update order, same clipping), so Cl/Cd values are unchanged.  Speed
    improvement comes from multi-core parallelism and elimination of numpy
    temporary arrays.
    """
    for j in prange(1, ny - 1):
        for i in range(1, nx - 1):
            if not mask[j, i]:
                p_new = (pn[j, i+1] + pn[j, i-1] +
                         pn[j+1, i] + pn[j-1, i] - b[j, i]) * 0.25
                if   p_new >  5.0: p_new =  5.0
                elif p_new < -5.0: p_new = -5.0
                p_sor = (1.0 - omega) * pn[j, i] + omega * p_new
                if   p_sor >  5.0: p_sor =  5.0
                elif p_sor < -5.0: p_sor = -5.0
                press[j, i] = p_sor

def run_fluid_simulation(gui, p):
    start_time = time.perf_counter()
    _g = str(p.get("grid", "160"))
    print(f"--- Solver Core Started (Re: {p.get('re')}, AoA: {p.get('aoa')}, Grid: {_g}x{_g}, dt: {p.get('dt', 0.2)}) ---")

    # Change 1 — High process priority: gives the solver preferential CPU scheduling
    try:
        psutil.Process(os.getpid()).nice(psutil.HIGH_PRIORITY_CLASS)
    except Exception:
        pass

    # Change 2+3 — Set Numba thread count for parallel SOR (clamped 1–6)
    n_cores = max(1, min(6, int(p.get("cores", 1))))
    numba.set_num_threads(n_cores)

    # Grid configs: (nx, ny, chord, center_x, center_y)
    _gcfg = {"80":  (80,  80,  32,  20,  40),
             "160": (160, 160, 64,  40,  80),
             "320": (320, 320, 128, 80,  160)}
    nx, ny, _chord0, _cx0, _cy0 = _gcfg.get(_g, _gcfg["160"])
    u, v, press = np.zeros((ny, nx)), np.zeros((ny, nx)), np.zeros((ny, nx))
    mask = get_naca_mask(ny, nx, chord_length=_chord0, center_x=_cx0, center_y=_cy0)
    if hasattr(gui, 'set_body_mask'):
        gui.root.after(0, gui.set_body_mask, mask)

    y_indices, x_indices = np.where(mask)
    chord_length = float(x_indices.max() - x_indices.min() + 1) if len(x_indices) > 0 else 32.0

    sim_vel = 1.0
    nu      = (sim_vel * chord_length) / max(p.get("re", 100), 1)
    dt      = float(p.get("dt", 0.2))
    rho     = 1.0
    omega   = float(p.get("omega", 0.6))
    conv_threshold = float(p.get("conv_res", 1e-6))

    # ── Advection blending ────────────────────────────────────────────────────
    # THETA=1 pure upwind (stable, high ν_num); THETA=0 pure central (accurate, low ν_num).
    # Van Leer flux limiter adapts spatially: approaches (1-THETA) central in smooth
    # regions, falls back to full upwind at steep gradients regardless of THETA.
    # Lower THETA → lower effective ν_num → higher meaningful Re ceiling.
    THETA = float(p.get("theta", 0.5))

    # ── Smagorinsky SGS turbulence model ─────────────────────────────────────
    # nu_t = (Cs*Delta)^2 * |S|, Cs=0.032 (CS_SQ=0.001).
    # Conservative constant: negligible (<1%) at Re<=500 (laminar regime),
    # adds ~3-15% of physical nu at Re>=1000 to represent unresolved turbulent
    # mixing on this 64-cell chord.  Cs=0.1 was too large and made
    # Re=500 behave identically to Re=100 by swamping physical viscosity.
    CS_SQ = 0.001

    fluid_above = ~np.roll(mask,  1, 0)
    fluid_below = ~np.roll(mask, -1, 0)
    fluid_left  = ~np.roll(mask,  1, 1)
    fluid_right = ~np.roll(mask, -1, 1)
    n_fluid_nbrs = (fluid_above.astype(float) + fluid_below.astype(float)
                  + fluid_left.astype(float)  + fluid_right.astype(float))
    n_fluid_nbrs = np.where(mask, np.maximum(n_fluid_nbrs, 1.0), 1.0)
    fa = fluid_above.astype(float)
    fb = fluid_below.astype(float)
    fl = fluid_left.astype(float)
    fr = fluid_right.astype(float)

    aoa_rad  =  np.radians(p.get("aoa", 0))
    u_inflow =  sim_vel * np.cos(aoa_rad)
    v_inflow = -sim_vel * np.sin(aoa_rad)   # flip: array-y downward, physical-y upward

    u[:, :] = u_inflow;  v[:, :] = v_inflow
    u[mask] = 0.0;       v[mask] = 0.0

    q_ref = 0.5 * rho * sim_vel**2 * chord_length

    def compute_cl_cd():
        p_inf = float(np.mean(press[:, 0]))
        pa = np.roll(press,  1, 0);  pb = np.roll(press, -1, 0)
        pl = np.roll(press,  1, 1);  pr = np.roll(press, -1, 1)
        fy = float(np.sum(-fa[mask]*(pa[mask]-p_inf) + fb[mask]*(pb[mask]-p_inf)))
        fx = float(np.sum( fl[mask]*(pl[mask]-p_inf) - fr[mask]*(pr[mask]-p_inf)))
        lift = fy * np.cos(aoa_rad) - fx * np.sin(aoa_rad)
        drag = fx * np.cos(aoa_rad) + fy * np.sin(aoa_rad)
        cl = lift / (q_ref + 1e-10)
        cd = drag / (q_ref + 1e-10)
        return float(cl), float(cd), float(cl / (abs(cd) + 1e-10))

    max_iters     = int(p["max_iters"])
    hist_interval = max(1, int(p.get("hist_interval", 50)))

    # ── Warmup: wait 4 chord-crossing times before time-averaging starts ──────
    # One crossing = chord / (sim_vel * dt) iters.  At 80×80 that's 160 iters;
    # 4× gives 640 — past the declining transient seen in data (Cl settles ~650).
    # Capped at max_iters//2 so short deliberate runs still produce an average.
    _flow_through = int(chord_length / (sim_vel * dt))
    warmup = max(200, min(4 * _flow_through, max_iters // 2))

    res_history = []
    # Trailing window: keep the last _AVG_WIN post-warmup samples.
    # Using the tail rather than a full cumulative sum prevents early transient
    # drift from biasing the reported Cl/Cd (the root cause of the 0.21 vs 0.17
    # discrepancy on 80×80).  20 samples × 50-iter interval = last ~1 000 iters.
    _AVG_WIN = 20
    cl_samples, cd_samples = [], []
    conv_state = "Max iterations reached"
    res = 0.0
    cl, cd, ld_ratio = 0.0, 0.0, 0.0

    # ── Pre-allocated work buffers (eliminates per-step heap allocation) ──────
    # u_old: residual tracking  |  pn: SOR workspace  |  _pdiff: convergence norm
    u_old          = np.zeros_like(u)
    pn             = np.zeros_like(press)
    _pdiff         = np.zeros_like(press)
    _sor_threshold = 1e-4 * nx * ny / 6400   # pre-computed once; scales with grid

    # Shift buffers — replace np.roll() (which always allocates a full copy).
    # Naming: _X_l/r = left/right shift (axis=1); _X_u/d = up/down shift (axis=0).
    # Filled in-place each step; reused wherever that shifted array is needed.
    # Adding a new airfoil only changes the mask — these buffers are geometry-agnostic.
    _u_l = np.empty_like(u);  _u_r = np.empty_like(u)
    _u_u = np.empty_like(u);  _u_d = np.empty_like(u)
    _v_l = np.empty_like(v);  _v_r = np.empty_like(v)
    _v_u = np.empty_like(v);  _v_d = np.empty_like(v)
    _vel_l = np.empty_like(u); _vel_r = np.empty_like(u)
    _vel_u = np.empty_like(u); _vel_d = np.empty_like(u)
    _pa = np.empty_like(press); _pb = np.empty_like(press)   # press shifts for Neumann BC
    _pl = np.empty_like(press); _pr = np.empty_like(press)

    try:
        for i in range(max_iters):

            if gui.kill_event.is_set():
                conv_state = "Finalized";  break

            while gui.pause_event.is_set() and not gui.kill_event.is_set():
                time.sleep(0.05)
            if gui.kill_event.is_set():
                conv_state = "Finalized";  break

            # ── BCs ───────────────────────────────────────────────────────────
            u[:, 0] = u_inflow;   v[:, 0] = v_inflow
            u[0,  :] = u[1,  :]; v[0,  :] = v[1,  :]
            u[-1, :] = u[-2, :]; v[-1, :] = v[-2, :]

            np.copyto(u_old, u)

            # Fill shift buffers from current u, v (axis-1: l=i+1, r=i-1; axis-0: u=j+1, d=j-1)
            _u_l[:, :-1] = u[:, 1:];   _u_l[:, -1]  = u[:, 0]
            _u_r[:, 1:]  = u[:, :-1];  _u_r[:, 0]   = u[:, -1]
            _u_u[:-1, :] = u[1:, :];   _u_u[-1, :]  = u[0, :]
            _u_d[1:, :]  = u[:-1, :];  _u_d[0, :]   = u[-1, :]
            _v_l[:, :-1] = v[:, 1:];   _v_l[:, -1]  = v[:, 0]
            _v_r[:, 1:]  = v[:, :-1];  _v_r[:, 0]   = v[:, -1]
            _v_u[:-1, :] = v[1:, :];   _v_u[-1, :]  = v[0, :]
            _v_d[1:, :]  = v[:-1, :];  _v_d[0, :]   = v[-1, :]

            # ── Central-difference gradients (shared by Smagorinsky + blending) ─
            dudx_c = (_u_l - _u_r) / 2
            dvdy_c = (_v_u - _v_d) / 2
            dudy_c = (_u_u - _u_d) / 2
            dvdx_c = (_v_l - _v_r) / 2

            # ── Smagorinsky turbulence model ──────────────────────────────────
            S_mag  = np.sqrt(2*dudx_c**2 + 2*dvdy_c**2 + (dudy_c + dvdx_c)**2)
            nu_eff = nu + CS_SQ * S_mag    # effective (laminar + SGS) viscosity

            # ── Van Leer flux limiter + THETA floor ───────────────────────────
            # Smoothness indicator per direction (minmod ratio of consecutive
            # differences in velocity magnitude).  theta_eff = THETA in smooth
            # flow; → 1 (full upwind) at steep gradients regardless of THETA.
            # Setting THETA=0 in the GUI gives near-central in smooth regions
            # (lowest ν_num, highest meaningful Re) while the limiter preserves
            # stability where the solution is rough.
            _eps  = 1e-10
            _vel  = np.abs(u) + np.abs(v)
            _vel_l[:, :-1] = _vel[:, 1:];   _vel_l[:, -1] = _vel[:, 0]
            _vel_r[:, 1:]  = _vel[:, :-1];  _vel_r[:, 0]  = _vel[:, -1]
            _vel_u[:-1, :] = _vel[1:, :];   _vel_u[-1, :] = _vel[0, :]
            _vel_d[1:, :]  = _vel[:-1, :];  _vel_d[0, :]  = _vel[-1, :]
            _dfx = _vel_l - _vel;  _dbx = _vel - _vel_r
            _dfy = _vel_u - _vel;  _dby = _vel - _vel_d
            _mm_x = np.where(_dfx*_dbx > 0,
                             np.minimum(np.abs(_dfx),np.abs(_dbx))
                             / (np.maximum(np.abs(_dfx),np.abs(_dbx)) + _eps), 0.0)
            _mm_y = np.where(_dfy*_dby > 0,
                             np.minimum(np.abs(_dfy),np.abs(_dby))
                             / (np.maximum(np.abs(_dfy),np.abs(_dby)) + _eps), 0.0)
            # theta_eff ∈ [THETA, 1]: THETA when smooth, 1 at discontinuities
            _tx = THETA + (1.0 - THETA) * (1.0 - _mm_x)
            _ty = THETA + (1.0 - THETA) * (1.0 - _mm_y)

            dudx_uw = np.where(u > 0, u - _u_r, _u_l - u)
            dvdx_uw = np.where(u > 0, v - _v_r, _v_l - v)
            dudy_uw = np.where(v > 0, u - _u_d, _u_u - u)
            dvdy_uw = np.where(v > 0, v - _v_d, _v_u - v)

            dudx = _tx * dudx_uw + (1-_tx) * dudx_c
            dvdx = _tx * dvdx_uw + (1-_tx) * dvdx_c
            dudy = _ty * dudy_uw + (1-_ty) * dudy_c
            dvdy = _ty * dvdy_uw + (1-_ty) * dvdy_c

            # ── Momentum update — point-implicit diffusion ────────────────────
            # Treats diffusion implicitly (one Jacobi step) making the scheme
            # unconditionally stable for diffusion at any nu_eff.
            # u_new*(1+4α) = u - dt*adv_u + α*Σneighbours,  α = dt*nu_eff
            _nbrs_u = _u_d + _u_u + _u_r + _u_l
            _nbrs_v = _v_d + _v_u + _v_r + _v_l
            _adv_u  = u*dudx + v*dudy
            _adv_v  = u*dvdx + v*dvdy
            _denom  = 1.0 + 4.0 * dt * nu_eff
            u[1:-1,1:-1] = (u[1:-1,1:-1] - dt*_adv_u[1:-1,1:-1]
                            + dt*nu_eff[1:-1,1:-1]*_nbrs_u[1:-1,1:-1]) / _denom[1:-1,1:-1]
            v[1:-1,1:-1] = (v[1:-1,1:-1] - dt*_adv_v[1:-1,1:-1]
                            + dt*nu_eff[1:-1,1:-1]*_nbrs_v[1:-1,1:-1]) / _denom[1:-1,1:-1]
            u[mask] = 0.0;  v[mask] = 0.0

            # ── Pressure Poisson ──────────────────────────────────────────────
            # Re-fill x/y shifts — u and v changed during momentum update
            _u_l[:, :-1] = u[:, 1:];   _u_l[:, -1]  = u[:, 0]
            _u_r[:, 1:]  = u[:, :-1];  _u_r[:, 0]   = u[:, -1]
            _v_u[:-1, :] = v[1:, :];   _v_u[-1, :]  = v[0, :]
            _v_d[1:, :]  = v[:-1, :];  _v_d[0, :]   = v[-1, :]
            div = ((_u_l - _u_r) + (_v_u - _v_d)) / 2
            b   = (rho/dt) * div
            for _ in range(300):
                np.copyto(pn, press)
                _sor_step(press, pn, b, mask, omega, ny, nx)   # parallel Jacobi SOR
                press[1:-1,1:-1] = np.clip(press[1:-1,1:-1], -5, 5)  # safety clamp
                press[mask] = pn[mask]
                press[:,0]=press[:,1];  press[:,-1]=press[:,-2]
                press[0,:]=press[1,:];  press[-1,:]=press[-2,:]
                press -= np.mean(press)
                np.subtract(press, pn, out=_pdiff)
                np.absolute(_pdiff, out=_pdiff)
                if _pdiff.sum() < _sor_threshold:  break

            # ── Neumann BC on solid cells ─────────────────────────────────────
            _pa[1:, :]  = press[:-1, :];  _pa[0, :]  = press[-1, :]
            _pb[:-1, :] = press[1:, :];   _pb[-1, :] = press[0, :]
            _pl[:, 1:]  = press[:, :-1];  _pl[:, 0]  = press[:, -1]
            _pr[:, :-1] = press[:, 1:];   _pr[:, -1] = press[:, 0]
            press[mask] = ((fluid_above*_pa+fluid_below*_pb
                           +fluid_left*_pl+fluid_right*_pr)/n_fluid_nbrs)[mask]

            # ── Velocity correction ───────────────────────────────────────────
            u[1:-1,1:-1] -= (dt/rho)*(press[1:-1,2:]-press[1:-1,0:-2])/2
            v[1:-1,1:-1] -= (dt/rho)*(press[2:,1:-1]-press[0:-2,1:-1])/2
            u[mask] = 0.0;  v[mask] = 0.0

            # Safety clamp: prevents blow-up when omega>1 or at sharp gradients
            u = np.clip(u, -5.0, 5.0)
            v = np.clip(v, -5.0, 5.0)

            # ── Convective outflow BC (lets vortices exit cleanly) ────────────
            u[:,-1] -= dt * u_inflow * (u[:,-1] - u[:,-2])
            v[:,-1] -= dt * u_inflow * (v[:,-1] - v[:,-2])

            res = float(np.mean(np.abs(u - u_old)))
            res_history.append(res)

            if np.isnan(u).any() or np.isnan(press).any():
                print(f"FATAL ERROR: NaN at iteration {i}.")
                conv_state = "NaN error";  break

            # ── Steady convergence ────────────────────────────────────────────
            if i > 200 and res < conv_threshold:
                conv_state = "Converged (steady)"
                print(f"Converged at {i} (res={res:.2e}).");  break

            # ── Periodic / plateau convergence ────────────────────────────────
            # Triggers when residual has stopped improving AND is either:
            #   (a) oscillating  — any single step jumped >2% (vortex shedding)
            #   (b) trending up  — second half of recent window 0.5%+ worse than
            #                      first half (slow numerical drift / Cd creep)
            # The trending-up clause catches the slow-drift case that never shows
            # a sharp 2% single-step jump but steadily inflates Cd over 10 k+ iters.
            if i > warmup + 400 and len(res_history) >= 400:
                recent   = res_history[-200:]
                sharp_osc   = any(recent[j] > recent[j-1] * 1.02 for j in range(1, len(recent)))
                trending_up = float(np.mean(recent[100:])) > float(np.mean(recent[:100])) * 1.005
                oscillating = sharp_osc or trending_up
                plateau     = min(recent) >= min(res_history[-400:-200]) * 0.95
                if oscillating and plateau:
                    conv_state = "Converged (periodic)"
                    print(f"Plateau at {i} (res={res:.2e}). Periodic regime.");  break

            # ── Live GUI updates ──────────────────────────────────────────────
            elapsed = time.perf_counter() - start_time
            gui.root.after(0, gui.update_live_metrics, i, res, elapsed)

            if i % 50 == 0 and i > 0:
                cl, cd, ld_ratio = compute_cl_cd()
                gui.root.after(0, gui.update_live_results, u.copy(), cl, cd, ld_ratio)
                if i >= warmup:
                    cl_samples.append(cl)
                    cd_samples.append(cd)

            if i % hist_interval == 0 and i > 0:
                gui.root.after(0, gui.append_history_row, i, res, cl, cd, ld_ratio)

        # ── Final result ──────────────────────────────────────────────────────
        if len(cl_samples) > 0:
            # Use only the last _AVG_WIN samples (settled flow, not the transient)
            _used_cl = cl_samples[-_AVG_WIN:]
            _used_cd = cd_samples[-_AVG_WIN:]
            cl       = float(np.mean(_used_cl))
            cd       = float(np.mean(_used_cd))
            ld_ratio = cl / (abs(cd) + 1e-10)
            print(f"Time-averaged (last {len(_used_cl)} of {len(cl_samples)} samples): "
                  f"Cl={cl:.4f} Cd={cd:.4f} L/D={ld_ratio:.4f}")
        else:
            cl, cd, ld_ratio = compute_cl_cd()

        print(f"Finished [{conv_state}]. Cl={cl:.4f} Cd={cd:.4f} L/D={ld_ratio:.4f}")
        gui.root.after(0, gui.append_history_row, i, res, cl, cd, ld_ratio)
        gui.root.after(0, gui.display_final_results, u, cl, cd, ld_ratio, conv_state)

    except Exception as e:
        print(f"SOLVER ERROR: {e}")
        import traceback; traceback.print_exc()
    finally:
        gui.solver_running = False
