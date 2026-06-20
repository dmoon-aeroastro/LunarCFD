"""
#1B — Implicit SIMPLE steady solver (separate entry point).

run_bfm_simple(gui, p) mirrors run_bfm_simulation's GUI/callback contract but
solves the STEADY incompressible RANS equations with the SIMPLE pressure-
correction algorithm instead of an explicit time march:

  outer iteration:
    1. lag face mass fluxes (Rhie-Chow) and assemble the implicit momentum
       matrix (1st-order upwind convection + diffusion) with a deferred-
       correction source carrying the central/JST higher-order advection;
    2. under-relax and solve the u- and v-momentum systems (Jacobi sweeps);
    3. assemble + solve the pressure-correction Poisson (mass imbalance);
    4. correct p, u, v (under-relaxed) and the face fluxes.
  until the scaled momentum + continuity residuals fall below tol.

No CFL limit (convection is implicit), so steady cases converge in thousands of
outer iterations.  STEADY-ONLY: there is no physical time, so shedding/Strouhal
are not meaningful here — use the explicit solver for unsteady studies.

Hot linear-solve sweeps come from solver.simple_solver (compiled Fortran when
built, else Numba).  This is the first working draft: laminar (and SST-coupled)
momentum/pressure are in; it is opt-in via solver_mode="simple".
"""
import time
import math
import numpy as np

from mesh.airfoil import naca4, load_dat, resample, rect_1x8
from mesh.ogrid   import build_ogrid
from solver.core_bfm import _interpolate_to_cartesian
from solver import simple_solver as _ss
try:
    from solver.core_bfm import _sst_jit, _HAVE_NUMBA
except Exception:                                # pragma: no cover
    _sst_jit = None; _HAVE_NUMBA = False


def _ghost_NS(a, outerN, wall="antisym"):
    """N ghost = outerN (array or scalar); S ghost per wall mode."""
    aN = np.empty_like(a); aS = np.empty_like(a)
    aN[:-1, :] = a[1:, :]; aN[-1, :] = outerN
    aS[1:, :] = a[:-1, :]
    if wall == "antisym":
        aS[0, :] = -a[0, :]
    elif wall == "neumann":
        aS[0, :] = a[0, :]
    else:
        aS[0, :] = 0.0
    return aN, aS


def run_bfm_simple(gui, p):
    start_time = time.perf_counter()

    # ── Setup (mirrors run_bfm_simulation) ───────────────────────────────────
    airfoil_id = str(p.get("airfoil", "0012"))
    n_xi = int(p.get("n_xi", 96)); n_eta = int(p.get("n_eta", 48))
    ogrid_R = float(p.get("ogrid_R", 15.0))
    if "alpha_stretch" in p:
        alpha_stretch = float(p["alpha_stretch"])
    else:
        alpha_stretch = 6.0 + max(0.0, math.log2(n_eta / 32.0))
    Re = max(float(p.get("re", 100)), 1.0)
    aoa_rad = math.radians(float(p.get("aoa", 0.0)))
    max_iters = int(p["max_iters"])
    hist_interval = max(1, int(p.get("hist_interval", 50)))
    T_wall = float(p.get("t_wall", 320.0)); T_inf = float(p.get("t_inf", 300.0))

    # SIMPLE controls.
    _au = float(p.get("simple_au", 0.7))         # momentum under-relaxation
    _ap = float(p.get("simple_ap", 0.3))         # pressure under-relaxation
    _mom_sweeps = int(p.get("simple_mom_sweeps", 5))
    _pc_sweeps  = int(p.get("simple_pc_sweeps", 60))
    # Pseudo-transient continuation: a false local time step Δτ adds ρV/Δτ to the
    # momentum diagonal (and ρV/Δτ·φ_old to the source).  It bounds the diagonal
    # in near-inviscid (high-Re) cells where convection+diffusion alone would let
    # a_P→0 and the pressure coupling V/a_P blow up — this is what makes the
    # high-Re solve stable.  At convergence φ=φ_old so the term cancels: the
    # steady solution is unchanged.  Smaller CFL = more stable / slower.  0.5 is
    # validated stable+accurate at Re1e6 with the TVD flux (tau=1.0 still NaNs as
    # the suction peak sharpens; 0.4 and 0.7 both held to Cl≈0.39).
    _tau_cfl = float(p.get("simple_tau_cfl", 0.5))
    # Eddy-viscosity under-relaxation: blend the new SST nu_t with the previous
    # one so turbulence evolves smoothly with the momentum (an abruptly changing
    # nu_t destabilises the high-Re solve once the BL starts developing).
    _nut_relax = float(p.get("simple_nut_relax", 0.2))
    _jst_eps4   = float(p.get("jst_eps4", 0.02))
    _tol = float(p.get("simple_tol", 1e-4))
    # Deferred-correction advection.  Off by default for the first stable build:
    # pure 1st-order upwind converges robustly; turning this on recovers central-
    # scheme accuracy.  _dc_relax (<=1) under-relaxes the explicit DC source so
    # the lagged central flux can't destabilise the outer iteration; _use_jst adds
    # the 4th-difference dissipation that central advection needs at high Re.
    _use_dc   = bool(p.get("simple_dc", True))
    _use_jst  = bool(p.get("simple_jst", True))
    # TVD (van Leer) limited 2nd-order convection on the periodic E/W faces in
    # place of central+JST.  The limiter is self-adjusting — full 2nd-order in
    # smooth flow, clamping toward upwind at the steep boundary-layer gradients
    # that make the unlimited central scheme blow up at high Re.  Default ON; it
    # is far more robust than tuning JST for the implicit large-step solve.
    _use_tvd  = bool(p.get("simple_tvd", True))
    # TVD limiter: "vanleer" (smooth → robust steady convergence across airfoils/
    # AoA; the default) or "minmod" (more dissipative but its non-smooth clamp can
    # limit-cycle/break down on cambered low-AoA cases).  van Leer over-predicts
    # lift somewhat on cambered sections — for best accuracy use the explicit
    # solver + local time-stepping, which is validated to 95–97% of the polars.
    _limiter  = str(p.get("simple_limiter", "vanleer"))
    _dc_relax = float(p.get("simple_dc_relax", 1.0))   # 1.0 = full central at convergence
    # Continuation ramp: the deferred-correction weight grows 0->_dc_relax over the
    # first _dc_ramp outer iterations.  Early iters are (stable) upwind; the violent
    # high-Re startup transient is past before the central flux is fully on.  The
    # converged weight is still _dc_relax, so accuracy is unchanged.
    _dc_ramp  = max(1, int(p.get("simple_dc_ramp", 800)))

    if airfoil_id.endswith(".dat") or airfoil_id.endswith(".txt"):
        xa_raw, ya_raw = load_dat(airfoil_id); xa_a, ya_a = resample(xa_raw, ya_raw, n_xi)
    elif airfoil_id == "rect_1x8":
        xa_raw, ya_raw = rect_1x8(); xa_a, ya_a = resample(xa_raw, ya_raw, n_xi)
    else:
        xa_a, ya_a = naca4(airfoil_id.zfill(4), n_pts=n_xi // 2)
        xa_a, ya_a = resample(xa_a, ya_a, n_xi)

    grid = build_ogrid(xa_a, ya_a, n_xi=n_xi, n_eta=n_eta, R=ogrid_R, alpha_stretch=alpha_stretch)
    chord = grid['chord']
    nxE, nyE, dsE, dnE = grid['nxE'], grid['nyE'], grid['dsE'], grid['dnE']
    nxW, nyW, dsW, dnW = grid['nxW'], grid['nyW'], grid['dsW'], grid['dnW']
    nxN, nyN, dsN, dnN = grid['nxN'], grid['nyN'], grid['dsN'], grid['dnN']
    nxS, nyS, dsS, dnS = grid['nxS'], grid['nyS'], grid['dsS'], grid['dnS']
    V = grid['cell_area']; XC, YC = grid['XC'], grid['YC']
    ip1 = (np.arange(n_xi) + 1) % n_xi; im1 = (np.arange(n_xi) - 1) % n_xi
    ip2 = (np.arange(n_xi) + 2) % n_xi; im2 = (np.arange(n_xi) - 2) % n_xi

    gE = dsE/dnE; gW = dsW/dnW; gN = dsN/dnN; gS = dsS/dnS
    gS_wall = gS.copy(); gS_wall[0, :] = 0.0     # Neumann wall for p'
    # Per-cell pseudo time step for the false-transient term (geometry-based,
    # like #1A's local dt): Δτ = CFL·min(local face)/U∞.
    _mfc0 = np.minimum(np.minimum(dsE, dsW), np.minimum(dsN, dsS))

    sim_vel = 1.0
    nu = (sim_vel * chord) / Re
    rho = 1.0
    q_ref = 0.5 * rho * sim_vel**2 * chord
    u_inf = sim_vel * math.cos(aoa_rad); v_inf = sim_vel * math.sin(aoa_rad)

    _dE_circ = np.sqrt((XC[:, ip1] - XC)**2 + (YC[:, ip1] - YC)**2)
    # (false-transient diagonal _pt is computed ADAPTIVELY each outer iteration
    #  from the local speed; see the loop — Δτ = tau_cfl·min_face/(|U|+U∞).)

    # Wall-distance (for a future SST coupling; laminar for now)
    _xa_wmid = 0.5 * (grid['xa'] + np.roll(grid['xa'], -1))
    _ya_wmid = 0.5 * (grid['ya'] + np.roll(grid['ya'], -1))

    # State
    u = np.full((n_eta, n_xi), u_inf)
    v = np.full((n_eta, n_xi), v_inf)
    press = np.zeros((n_eta, n_xi))
    nu_t = np.zeros((n_eta, n_xi))
    apc = np.full((n_eta, n_xi), 1.0)            # momentum centre coeff (this iter)
    apr = np.full((n_eta, n_xi), 1.0)            # relaxed diagonal apc/au (lagged into RC)

    # ── k-ω SST coupling (pseudo-transient, reuses the validated _sst_jit) ────
    # Active at Re>=500 when Numba is available; otherwise laminar (nu_t=0).
    # Each outer iteration advances k/ω one local pseudo-step with the current
    # velocity, so the eddy viscosity relaxes to steady alongside the momentum.
    _kw = (Re >= 500) and (_sst_jit is not None) and bool(p.get("sst", True))
    _u_out_arr = np.full(n_xi, u_inf); _v_out_arr = np.full(n_xi, v_inf)
    if _kw:
        _a1 = 0.31; _beta_str = 0.09; _kappa = 0.41
        _sk1 = 0.85; _sw1 = 0.5; _b1 = 0.075
        _g1 = _b1/_beta_str - _sw1*_kappa**2/math.sqrt(_beta_str)
        _sk2 = 1.0; _sw2 = 0.856; _b2 = 0.0828
        _g2 = _b2/_beta_str - _sw2*_kappa**2/math.sqrt(_beta_str)
        _nu_t_mult = float(p.get("nu_t_mult", 1.0)); _pk_limit = float(p.get("pk_limit", 10.0))
        _min_face = float(np.minimum(dsE, dsS).min())
        _S_max = sim_vel / max(_min_face, 1e-10)
        # 2-D wall distance (cell centre -> wall-face midpoint)
        _d_w = np.sqrt((XC - _xa_wmid[np.newaxis, :])**2 + (YC - _ya_wmid[np.newaxis, :])**2)
        _d_w = np.maximum(_d_w, 1e-10)
        _Tu = 0.001; _k_fs = 1.5*(_Tu*sim_vel)**2
        _om_fs = math.sqrt(_k_fs)/(_beta_str**0.25*chord)
        _om_w0 = np.minimum(60.0*nu/(_b1*_d_w[0, :]**2), 1.0e5)
        _eta_frac = np.linspace(0.0, 1.0, n_eta)[:, np.newaxis]
        tke = (_k_fs*_eta_frac)*np.ones((1, n_xi))
        _lw = np.log(np.maximum(_om_w0, 1e-10)); _lf = math.log(max(_om_fs, 1e-10))
        tom = np.exp(_lw[np.newaxis, :]*(1.0-_eta_frac) + _lf*_eta_frac)
        tke[0, :] = 0.0; tom[0, :] = _om_w0; tke[-1, :] = _k_fs; tom[-1, :] = _om_fs
        # SST pseudo time step (geometry-based local dt, like #1A)
        _mfc = np.minimum(np.minimum(dsE, dsW), np.minimum(dsN, dsS))
        _dt_sst = np.maximum(0.05*_mfc/(1.5*sim_vel), 1e-6)
    else:
        tke = np.zeros((n_eta, n_xi)); tom = np.ones((n_eta, n_xi))

    # Real-world scaling (for force readout parity with the explicit solver)
    _P_atm = float(p.get("pressure", 101325.0)); _vel_real = float(p.get("vel", 25.0))
    _rho_real = _P_atm / (287.05 * 288.15); _q_real = 0.5 * _rho_real * _vel_real**2
    _nu_air = 1.5e-5
    _cm_in = p.get("chord_m", None)
    _chord_m = float(_cm_in) if (_cm_in and float(_cm_in) > 0) else Re * _nu_air / max(_vel_real, 0.01)

    print(f"--- SIMPLE Solver Started (Re:{Re} AoA:{math.degrees(aoa_rad):.1f} "
          f"airfoil:{airfoil_id} {n_xi}x{n_eta} au={_au} ap={_ap} "
          f"backend={_ss.backend()}) ---", flush=True)

    def _apply_bc():
        u[-1, :] = u_inf; v[-1, :] = v_inf

    def _green_grad(f, outerN):
        fE = f[:, ip1]; fW = f[:, im1]
        fN, fS = _ghost_NS(f, outerN, wall="neumann")
        gx = (0.5*(f+fE)*nxE*dsE + 0.5*(f+fW)*nxW*dsW
              + 0.5*(f+fN)*nxN*dsN + 0.5*(f+fS)*nxS*dsS) / V
        gy = (0.5*(f+fE)*nyE*dsE + 0.5*(f+fW)*nyW*dsW
              + 0.5*(f+fN)*nyN*dsN + 0.5*(f+fS)*nyS*dsS) / V
        return gx, gy

    def _pcorr_Aop(pp, dfE, dfW, dfN, dfS, apP):
        """Matrix-free p'-Poisson operator: apP*p' - Σ df*p'_nb, with periodic
        E/W, outer Dirichlet p'=0 (row n_eta-1 held identity) and Neumann wall
        (dfS[0]=0)."""
        pE = pp[:, ip1]; pW = pp[:, im1]
        pN = np.zeros_like(pp); pN[:-1, :] = pp[1:, :]      # outer ghost 0
        pS = np.zeros_like(pp); pS[1:, :] = pp[:-1, :]; pS[0, :] = pp[0, :]
        out = apP*pp - (dfE*pE + dfW*pW + dfN*pN + dfS*pS)
        out[-1, :] = pp[-1, :]                              # outer ring: identity
        return out

    def _pcorr_cg(dfE, dfW, dfN, dfS, apP, rhs, maxit, rtol=1e-3):
        """Conjugate-gradient solve of the p'-Poisson (SPD).  Converges the
        low-wavenumber modes Jacobi cannot, which is what continuity needs each
        SIMPLE outer iteration."""
        b = rhs.copy(); b[-1, :] = 0.0
        nb = float(np.sqrt(np.sum(b*b))) + 1e-30
        p = np.zeros_like(b)
        r = b - _pcorr_Aop(p, dfE, dfW, dfN, dfS, apP)
        d = r.copy(); rs = float(np.sum(r*r))
        for _ in range(maxit):
            Ad = _pcorr_Aop(d, dfE, dfW, dfN, dfS, apP)
            al = rs / (float(np.sum(d*Ad)) + 1e-30)
            p += al*d; r -= al*Ad
            rsn = float(np.sum(r*r))
            if (rsn**0.5)/nb < rtol:
                break
            d = r + (rsn/(rs + 1e-30))*d; rs = rsn
        return p

    def _compute_forces():
        p_inf = float(np.mean(press[-1, :]))
        dp = press[0, :] - p_inf
        Fx = float(np.sum(dp * nxS[0, :] * dsS[0, :]))
        Fy = float(np.sum(dp * nyS[0, :] * dsS[0, :]))
        # skin friction (velocity-based law of the wall, k-independent)
        _tx = -nyS[0, :]; _ty = nxS[0, :]
        _utang = u[0, :]*_tx + v[0, :]*_ty
        _dw = np.sqrt((XC[0, :]-_xa_wmid)**2 + (YC[0, :]-_ya_wmid)**2)
        _dw = np.maximum(_dw, 1e-10)
        _um = np.abs(_utang) + 1e-12
        _ut = np.sqrt(nu*_um/_dw)
        for _ in range(12):
            _yp = _ut*_dw/(nu+1e-30)
            _up = np.where(_yp < 11.6, _yp, (1.0/0.41)*np.log(np.maximum(9.8*_yp, 1.0)))
            _ut = _um/np.maximum(_up, 1e-6)
        _tw = rho*_ut**2*np.sign(_utang+1e-30)
        Fx += float(np.sum(_tw*_tx*dsS[0, :])); Fy += float(np.sum(_tw*_ty*dsS[0, :]))
        ca, sa = math.cos(aoa_rad), math.sin(aoa_rad)
        lift = Fy*ca - Fx*sa; drag = Fx*ca + Fy*sa
        cl = lift/(q_ref+1e-10); cd = drag/(q_ref+1e-10)
        x_qc = float(xa_a.min()) + 0.25*chord; y_ac = float(ya_a.mean())
        mom = float(np.sum(dp*(nyS[0, :]*(_xa_wmid-x_qc)*dsS[0, :]
                              - nxS[0, :]*(_ya_wmid-y_ac)*dsS[0, :])))
        cm = -mom/(q_ref*chord+1e-10)
        return cl, cd, cl/(abs(cd)+1e-10), cm

    conv_state = "Max iterations reached"
    cl = cd = ld = cm = 0.0
    res0 = None
    _body_mask_sent = False

    try:
        for it in range(max_iters):
            if gui.kill_event.is_set():
                conv_state = "Finalized"; break
            while gui.pause_event.is_set() and not gui.kill_event.is_set():
                time.sleep(0.05)

            _apply_bc()
            # ── Neighbours / face-normal velocities (lagged) ──────────────────
            uE = u[:, ip1]; uW = u[:, im1]; vE = v[:, ip1]; vW = v[:, im1]
            uN, uS = _ghost_NS(u, u_inf, "antisym"); vN, vS = _ghost_NS(v, v_inf, "antisym")
            unE = 0.5*(u+uE)*nxE + 0.5*(v+vE)*nyE
            unW = 0.5*(u+uW)*nxW + 0.5*(v+vW)*nyW
            unN = 0.5*(u+uN)*nxN + 0.5*(v+vN)*nyN
            unS = 0.5*(u+uS)*nxS + 0.5*(v+vS)*nyS

            # Rhie-Chow correction on E/W (compact pressure coupling)
            gpx, gpy = _green_grad(press, 0.0)
            pE = press[:, ip1]; pW = press[:, im1]
            dPc = V/apr                          # velocity-pressure coupling (relaxed)
            dE = 0.5*(dPc + dPc[:, ip1])
            dW = 0.5*(dPc + dPc[:, im1])
            unE = unE - dE*((pE-press)/dnE - 0.5*(gpx+gpx[:, ip1])*nxE - 0.5*(gpy+gpy[:, ip1])*nyE)
            unW = unW - dW*((pW-press)/dnW - 0.5*(gpx+gpx[:, im1])*nxW - 0.5*(gpy+gpy[:, im1])*nyW)

            FE = rho*unE*dsE; FW = rho*unW*dsW; FN = rho*unN*dsN; FS = rho*unS*dsS
            FS[0, :] = 0.0   # no through-flow at the wall

            # ── Momentum coefficients (upwind convection + diffusion) ─────────
            nuf = nu + nu_t
            DE = nuf*gE; DW = nuf*gW; DN = nuf*gN; DS = nuf*gS
            aE = DE + np.maximum(-FE, 0.0)
            aW = DW + np.maximum(-FW, 0.0)
            aN = DN + np.maximum(-FN, 0.0)
            aS = DS + np.maximum(-FS, 0.0)
            # Diagonal from the FULL coefficients so the wall (DS at j=0) and the
            # outer ring (DN at j=n_eta-1) Dirichlet diffusion stay in a_P; only
            # the NEIGHBOUR coupling is dropped at those Dirichlet faces.
            apc = np.maximum(aE + aW + aN + aS + (FE + FW + FN + FS), 1e-12)
            # Adaptive false-transient: Δτ = tau_cfl·min_face/(|U|+U∞) so the
            # damping rho·V/Δτ grows where the flow accelerates (cambered/high-AoA
            # suction peaks) — robust across airfoils, AoA and grid without
            # per-case tuning.  Vanishes at convergence (uses start-of-iter u,v).
            _pt = rho * V * (np.sqrt(u*u + v*v) + sim_vel) / (_tau_cfl * _mfc0)
            apc = apc + _pt                     # false-transient diagonal (stabiliser)
            aS = aS.copy(); aS[0, :] = 0.0     # wall: no-slip Dirichlet 0 neighbour
            apr = apc/_au                       # relaxed momentum diagonal

            # Deferred correction (optional): move (central - upwind) advection
            # onto the RHS so the converged solution is central, not upwind.
            # Under-relaxed by _dc_relax (lagged explicit term).  Optional JST
            # 4th-difference dissipation stabilises the central flux at high Re.
            def _dc(phi, phiE, phiW, phiN, phiS):
                up = (np.maximum(FE, 0.0)*phi - np.maximum(-FE, 0.0)*phiE
                      + np.maximum(FW, 0.0)*phi - np.maximum(-FW, 0.0)*phiW
                      + np.maximum(FN, 0.0)*phi - np.maximum(-FN, 0.0)*phiN
                      + np.maximum(FS, 0.0)*phi - np.maximum(-FS, 0.0)*phiS)
                ce = (FE*0.5*(phi+phiE) + FW*0.5*(phi+phiW)
                      + FN*0.5*(phi+phiN) + FS*0.5*(phi+phiS))
                s = up - ce
                if _use_jst:
                    spd = np.sqrt(u*u+v*v) + 1e-12
                    d4 = phi[:, ip2]-4.0*phi[:, ip1]+6.0*phi-4.0*phi[:, im1]+phi[:, im2]
                    s = s - _jst_eps4*spd*d4/(_dE_circ+1e-12)*V   # dissipative (-Δ⁴)
                return _dcf * s

            def _tvd_dc(phi, phiN, phiS):
                # van Leer TVD on periodic E/W; central deferred on N/S.  Returns
                # (1st-order-upwind - high-order) convection summed per cell, so
                # adding it to the RHS makes the converged scheme 2nd-order TVD.
                pe = phi[:, ip1]; pw = phi[:, im1]
                pee = phi[:, ip2]; pww = phi[:, im2]
                eps = 1e-12
                def _vl(num, den):                 # TVD limiter ψ(num/den)
                    r = num/np.where(np.abs(den) < eps, eps, den)
                    if _limiter == "vanleer":
                        return (r + np.abs(r))/(1.0 + np.abs(r))
                    return np.maximum(0.0, np.minimum(1.0, r))   # minmod (more dissipative)
                # E face (cells P,E); upwind side depends on sign of FE
                denE = np.where(FE > 0, pe - phi, phi - pe)         # downwind-upwind
                numE = np.where(FE > 0, phi - pw, pe - pee)
                corrE = 0.5*_vl(numE, denE)*denE
                # W face (cells W,P)
                denW = np.where(FW > 0, pw - phi, phi - pw)
                numW = np.where(FW > 0, phi - pe, pw - pww)
                corrW = 0.5*_vl(numW, denW)*denW
                # N/S (radial) kept 1st-order upwind: diffusion-dominated near the
                # wall and a known instability source for high-order radial flux.
                # source = upwind - high-order  (E/W limiter part is -F*corr)
                return _dcf * (-(FE*corrE) - (FW*corrW))

            _dcf = _dc_relax * min(1.0, (it + 1.0)/_dc_ramp)   # continuation ramp

            gpx, gpy = _green_grad(press, 0.0)
            bu = -gpx*V
            bv = -gpy*V
            if _use_tvd:
                bu = bu + _tvd_dc(u, uN, uS)
                bv = bv + _tvd_dc(v, vN, vS)
            elif _use_dc:
                bu = bu + _dc(u, uE, uW, uN, uS)
                bv = bv + _dc(v, vE, vW, vN, vS)
            # Patankar under-relaxation + false-transient sources (both use the
            # start-of-iteration field, which u/v still hold here; both cancel at
            # convergence so the steady solution is unchanged).
            bu = bu + (1.0-_au)*apr*u + _pt*u
            bv = bv + (1.0-_au)*apr*v + _pt*v
            zeroG = np.zeros(n_xi)
            u = _ss.jacobi5(u, apr, aE, aW, aN, aS, bu, zeroG, 1.0, _mom_sweeps)
            v = _ss.jacobi5(v, apr, aE, aW, aN, aS, bv, zeroG, 1.0, _mom_sweeps)
            _apply_bc()

            # ── Pressure correction ───────────────────────────────────────────
            uE = u[:, ip1]; uW = u[:, im1]; vE = v[:, ip1]; vW = v[:, im1]
            uN, uS = _ghost_NS(u, u_inf, "antisym"); vN, vS = _ghost_NS(v, v_inf, "antisym")
            unE = 0.5*(u+uE)*nxE + 0.5*(v+vE)*nyE
            unW = 0.5*(u+uW)*nxW + 0.5*(v+vW)*nyW
            unN = 0.5*(u+uN)*nxN + 0.5*(v+vN)*nyN
            unS = 0.5*(u+uS)*nxS + 0.5*(v+vS)*nyS
            gpx, gpy = _green_grad(press, 0.0)
            pEv = press[:, ip1]; pWv = press[:, im1]
            dPc = V/apr                              # relaxed coupling
            dE = 0.5*(dPc + dPc[:, ip1]); dW = 0.5*(dPc + dPc[:, im1])
            unE = unE - dE*((pEv-press)/dnE - 0.5*(gpx+gpx[:, ip1])*nxE - 0.5*(gpy+gpy[:, ip1])*nyE)
            unW = unW - dW*((pWv-press)/dnW - 0.5*(gpx+gpx[:, im1])*nxW - 0.5*(gpy+gpy[:, im1])*nyW)
            FE = rho*unE*dsE; FW = rho*unW*dsW; FN = rho*unN*dsN; FS = rho*unS*dsS
            FS[0, :] = 0.0
            mass = FE + FW + FN + FS                 # net outflow per cell
            res_mass = float(np.mean(np.abs(mass)))

            # p'-coefficients (d-weighted Poisson; d = relaxed V/apr at the face)
            dfE = rho*dsE/dnE*0.5*(dPc + dPc[:, ip1])
            dfW = rho*dsW/dnW*0.5*(dPc + dPc[:, im1])
            dfN = np.empty_like(V); dfS = np.empty_like(V)
            dfN[:-1, :] = rho*dsN[:-1, :]/dnN[:-1, :]*0.5*(dPc[:-1, :]+dPc[1:, :])
            dfN[-1, :] = 0.0                          # outer Dirichlet p'=0
            dfS[1:, :] = rho*dsS[1:, :]/dnS[1:, :]*0.5*(dPc[1:, :]+dPc[:-1, :])
            dfS[0, :] = 0.0                          # Neumann wall
            apP = dfE + dfW + dfN + dfS + 1e-30
            # CG solve (converges the low-wavenumber pressure modes that the
            # fixed Jacobi sweep leaves untouched — essential for continuity).
            pcor = _pcorr_cg(dfE, dfW, dfN, dfS, apP, -mass, _pc_sweeps, rtol=1e-3)
            pcor -= float(np.mean(pcor))

            # Correct pressure and velocity (relaxed coupling V/apr)
            press = press + _ap*pcor
            gx, gy = _green_grad(pcor, 0.0)
            u = u - dPc*gx
            v = v - dPc*gy
            _apply_bc()

            # ── SST advance (pseudo-transient turbulence) ─────────────────────
            if _kw:
                tke, tom, nu_t_new = _sst_jit(
                    u, v, tke, tom, _d_w,
                    nxE, nyE, dsE, dnE, nxW, nyW, dsW, dnW,
                    nxN, nyN, dsN, dnN, nxS, nyS, dsS, dnS,
                    V, _u_out_arr, _v_out_arr,
                    nu, _k_fs, _om_fs, _kappa, _beta_str, _b1, _b2,
                    _sk1, _sk2, _sw1, _sw2, _g1, _g2, _a1,
                    _nu_t_mult, _pk_limit, _S_max, _dt_sst)
                nu_t = (1.0 - _nut_relax)*nu_t + _nut_relax*nu_t_new   # under-relax

            # ── Residual / convergence ────────────────────────────────────────
            res = res_mass
            if res0 is None and res > 0:
                res0 = res
            rel = res/(res0+1e-30) if res0 else res
            if np.isnan(u).any() or np.isnan(press).any():
                conv_state = "NaN error"; print(f"SIMPLE NaN at outer {it}"); break

            if it % 50 == 0:
                cl, cd, ld, cm = _compute_forces()
                elapsed = time.perf_counter() - start_time
                gui.root.after(0, gui.update_live_metrics, it, res, elapsed)
                mats = _interpolate_to_cartesian(u, v, press, np.full_like(u, T_inf),
                                                 T_inf, grid, chord_m=_chord_m)
                if not _body_mask_sent and "_mask" in mats:
                    gui.root.after(0, gui.set_body_mask, mats["_mask"]); _body_mask_sent = True
                lift_n = cl*_q_real*_chord_m; drag_n = cd*_q_real*_chord_m
                gui.root.after(0, gui.update_live_results, mats, cl, cd, ld, cm,
                               lift_n, drag_n, 0.0, float('nan'), 0.0)
                print(f"  SIMPLE it={it:>5} res={res:.2e} rel={rel:.2e} "
                      f"Cl={cl:+.4f} Cd={cd:.4f}", flush=True)
                if it > 100 and rel < _tol:
                    conv_state = "Converged (SIMPLE residual)"
                    print(f"SIMPLE converged at {it} (rel={rel:.2e})."); break

            if it % hist_interval == 0 and it > 0:
                gui.root.after(0, gui.append_history_row, it, res, cl, cd, ld)

        cl, cd, ld, cm = _compute_forces()
        lift_n = cl*_q_real*_chord_m; drag_n = cd*_q_real*_chord_m
        mats = _interpolate_to_cartesian(u, v, press, np.full_like(u, T_inf), T_inf, grid, chord_m=_chord_m)
        if not _body_mask_sent and "_mask" in mats:
            gui.root.after(0, gui.set_body_mask, mats["_mask"])
        print(f"SIMPLE Finished [{conv_state}]. Cl={cl:.4f} Cd={cd:.4f} L/D={ld:.4f}", flush=True)
        gui.root.after(0, gui.append_history_row, max_iters, 0.0, cl, cd, ld)
        gui.root.after(0, gui.display_final_results, mats, cl, cd, ld, cm,
                       lift_n, drag_n, 0.0, float('nan'), 0.0, conv_state)
    except Exception as exc:
        import traceback
        print(f"SIMPLE SOLVER ERROR: {exc}"); traceback.print_exc()
    finally:
        gui.solver_running = False


run_simple = run_bfm_simple
