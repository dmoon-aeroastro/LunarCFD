"""
Body-Fitted Mesh (BFM) CFD solver for LunarCFD.

Uses an O-grid generated around an arbitrary airfoil with Finite Volume
discretisation on curvilinear coordinates.  Output format is identical to
solver/core.py so the GUI and test suite use the same callbacks.

Physics
-------
  - 2-D incompressible Navier-Stokes, projection method (fractional step)
  - Cell-centred Cartesian velocity (u, v) stored at O-grid cell centres
  - Face-normal fluxes from grid metrics (nxE/nyE/dsE/dnE etc.)
  - Upwind advection  +  central diffusion  +  point-implicit diffusion correction
  - SOR pressure Poisson with Dirichlet BC at outer boundary (p=0) and
    Neumann (dp/dn=0) at the wall
  - Temperature passive scalar (same treatment as Cartesian solver)

Coordinate convention
---------------------
  - j=0        : wall-adjacent cells (j increases radially outward)
  - j=n_eta-1  : outer-boundary cells
  - i periodic : circumferential direction (CCW around airfoil)
"""

import time
import math
import numpy as np

from mesh.airfoil import naca4, load_dat, resample, rect_1x8
from mesh.ogrid   import build_ogrid


# ── Optional Numba acceleration of the pressure-Poisson sweep ─────────────────
# The fixed-sweep Jacobi smoother is the dominant per-timestep cost (it runs
# ~200×/step, and each NumPy sweep allocates ~6 temporary arrays).  Fusing it
# into one compiled, multi-core loop with NO temporaries is a large speedup with
# IDENTICAL arithmetic (no fastmath, so results match the NumPy path to round-
# off).  Falls back to NumPy automatically if Numba is unavailable.
try:
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _jacobi_sweeps_jit(press, cE, cW, cN, cS, coeff, rhs, omega, n_sweeps):
        n_eta, n_xi = press.shape
        new = np.empty_like(press)
        for _ in range(n_sweeps):
            for j in prange(n_eta):
                for i in range(n_xi):
                    ip = i + 1 if i + 1 < n_xi else 0          # periodic E
                    im = i - 1 if i >= 1 else n_xi - 1         # periodic W
                    pN = press[j + 1, i] if (j + 1) < n_eta else 0.0   # Dirichlet outer
                    pS = press[j - 1, i] if j >= 1 else press[j, i]    # Neumann wall
                    pn = (cE[j, i] * press[j, ip] + cW[j, i] * press[j, im]
                          + cN[j, i] * pN + cS[j, i] * pS - rhs[j, i]) / coeff[j, i]
                    if pn > 5.0:
                        pn = 5.0
                    elif pn < -5.0:
                        pn = -5.0
                    new[j, i] = (1.0 - omega) * press[j, i] + omega * pn
            s = 0.0
            for j in range(n_eta):
                for i in range(n_xi):
                    s += new[j, i]
            m = s / (n_eta * n_xi)
            for j in prange(n_eta):
                for i in range(n_xi):
                    press[j, i] = new[j, i] - m
        return press

    @njit(parallel=True, cache=True)
    def _mom_predict_jit(u, v, nu_t,
                         nxE, nyE, dsE, dnE, nxW, nyW, dsW, dnW,
                         nxN, nyN, dsN, dnN, nxS, nyS, dsS, dnS,
                         cell_area, dE_circ, u_out, v_out,
                         dt_arr, nu, jst_eps4, jst_on):
        """Compiled momentum predictor: face-normal velocities + central
        (optionally central_jst) advection + point-implicit diffusion → u*,v*.

        Bit-faithful transcription of the NumPy `central`/`central_jst` branch
        (same operation order, no fastmath) so Cl/Cd match the NumPy path to
        round-off.  The legacy `upwind` scheme and `ho_faces` stay on the NumPy
        path; this kernel covers only the default central family.  The outer ring
        (j=n_eta-1) is overwritten by _apply_vel_bc after the call, exactly as in
        the NumPy path, so its predictor value here is harmless."""
        n_eta, n_xi = u.shape
        u_star = np.empty_like(u)
        v_star = np.empty_like(v)
        for j in prange(n_eta):
            for i in range(n_xi):
                ip  = i + 1 if i + 1 < n_xi else 0
                im  = i - 1 if i >= 1 else n_xi - 1
                ip2 = i + 2 if i + 2 < n_xi else i + 2 - n_xi
                im2 = i - 2 if i - 2 >= 0 else i - 2 + n_xi

                uc = u[j, i];  vc = v[j, i]
                uE = u[j, ip]; vE = v[j, ip]
                uW = u[j, im]; vW = v[j, im]
                if j + 1 < n_eta:
                    uN = u[j + 1, i]; vN = v[j + 1, i]
                else:
                    uN = u_out[i];    vN = v_out[i]          # outer ghost = freestream
                if j >= 1:
                    uS = u[j - 1, i]; vS = v[j - 1, i]
                else:
                    uS = -uc;         vS = -vc               # wall ghost (u_face=0)

                # Face-normal velocities
                unE = 0.5*(uc + uE)*nxE[j, i] + 0.5*(vc + vE)*nyE[j, i]
                unW = 0.5*(uc + uW)*nxW[j, i] + 0.5*(vc + vW)*nyW[j, i]
                unN = 0.5*(uc + uN)*nxN[j, i] + 0.5*(vc + vN)*nyN[j, i]
                unS = 0.5*(uc + uS)*nxS[j, i] + 0.5*(vc + vS)*nyS[j, i]

                # Central face values
                uEf = 0.5*(uc + uE); vEf = 0.5*(vc + vE)
                uWf = 0.5*(uc + uW); vWf = 0.5*(vc + vW)
                uNf = 0.5*(uc + uN); vNf = 0.5*(vc + vN)
                uSf = 0.5*(uc + uS); vSf = 0.5*(vc + vS)

                ca = cell_area[j, i]
                adv_u = (unE*uEf*dsE[j, i] + unW*uWf*dsW[j, i]
                       + unN*uNf*dsN[j, i] + unS*uSf*dsS[j, i]) / ca
                adv_v = (unE*vEf*dsE[j, i] + unW*vWf*dsW[j, i]
                       + unN*vNf*dsN[j, i] + unS*vSf*dsS[j, i]) / ca

                if jst_on:
                    spd = (uc*uc + vc*vc)**0.5 + 1e-12
                    d4u = u[j, ip2] - 4.0*u[j, ip] + 6.0*uc - 4.0*u[j, im] + u[j, im2]
                    d4v = v[j, ip2] - 4.0*v[j, ip] + 6.0*vc - 4.0*v[j, im] + v[j, im2]
                    adv_u = adv_u + jst_eps4*spd*d4u/(dE_circ[j, i] + 1e-12)
                    adv_v = adv_v + jst_eps4*spd*d4v/(dE_circ[j, i] + 1e-12)

                # Point-implicit diffusion (unconditionally stable)
                ddE = dsE[j, i]/dnE[j, i]; ddW = dsW[j, i]/dnW[j, i]
                ddN = dsN[j, i]/dnN[j, i]; ddS = dsS[j, i]/dnS[j, i]
                nbrs_u = uE*ddE + uW*ddW + uN*ddN + uS*ddS
                nbrs_v = vE*ddE + vW*ddW + vN*ddN + vS*ddS
                eff = (nu + nu_t[j, i]) / ca
                sum_ds_dn = ddE + ddW + ddN + ddS
                dtc = dt_arr[j, i]
                denom = 1.0 + dtc*eff*sum_ds_dn + 1e-30
                u_star[j, i] = (uc - dtc*adv_u + dtc*eff*nbrs_u) / denom
                v_star[j, i] = (vc - dtc*adv_v + dtc*eff*nbrs_v) / denom
        return u_star, v_star

    @njit(parallel=True, cache=True)
    def _sst_jit(u, v, tke, tom, d_w,
                 nxE, nyE, dsE, dnE, nxW, nyW, dsW, dnW,
                 nxN, nyN, dsN, dnN, nxS, nyS, dsS, dnS,
                 cell_area, u_out, v_out,
                 nu, k_fs, om_fs, kappa, beta_str, b1, b2,
                 sk1, sk2, sw1, sw2, g1, g2, a1,
                 nu_t_mult, pk_limit, S_max, dt_arr):
        """Compiled k-ω SST update: per-cell wall treatment, Green-Gauss
        gradients, F1/F2 blending, ν_t with Bradshaw limiter, semi-implicit k/ω
        transport.  Bit-faithful with the NumPy block (same op order, same
        clips/NaN guards), so Cl/Cd and the ν_t field match to round-off.

        Two passes: pass A computes the wall ω BC and the in-log wall-k value
        IN PLACE on tke[0,:] (cell j=1 reads it as its south neighbour); pass B
        is the gradient + transport sweep writing fresh output arrays.  The two
        loops form an implicit barrier so there is no read/write race on row 0."""
        n_eta, n_xi = u.shape
        bstr4 = beta_str**0.25
        om_w  = np.empty(n_xi)

        # ── Pass A: wall ω BC and viscous/log wall-k value (mutates tke[0,:]) ──
        for i in prange(n_xi):
            d1      = d_w[0, i]
            k1      = tke[1, i] if tke[1, i] > 1e-10 else 1e-10
            om_log  = k1**0.5 / (kappa * d1 * bstr4)
            om_vis2 = (60.0 * nu / (b1 * d1 * d1))**2
            ow      = (om_vis2 + om_log*om_log)**0.5
            om_w[i] = ow if ow < 1.0e5 else 1.0e5
            # k wall BC: Neumann (k[j=1]) where log layer dominates, else 0.
            tke[0, i] = tke[1, i] if om_log > om_vis2**0.5 else 0.0

        tke_new = np.empty_like(tke)
        tom_new = np.empty_like(tom)
        nu_t    = np.empty_like(tke)

        # ── Pass B: gradients, blending, eddy viscosity, k/ω transport ────────
        for j in prange(n_eta):
            for i in range(n_xi):
                ip = i + 1 if i + 1 < n_xi else 0
                im = i - 1 if i >= 1 else n_xi - 1

                uc = u[j, i];   vc = v[j, i]
                uE = u[j, ip];  vE = v[j, ip]
                uW = u[j, im];  vW = v[j, im]
                if j + 1 < n_eta:
                    uN = u[j + 1, i]; vN = v[j + 1, i]
                else:
                    uN = u_out[i];    vN = v_out[i]
                if j >= 1:
                    uS = u[j - 1, i]; vS = v[j - 1, i]
                else:
                    uS = -uc;         vS = -vc

                kc  = tke[j, i];  omc = tom[j, i]
                kE  = tke[j, ip]; kW  = tke[j, im]
                omE = tom[j, ip]; omW = tom[j, im]
                if j + 1 < n_eta:
                    kN = tke[j + 1, i]; omN = tom[j + 1, i]
                else:
                    kN = k_fs;          omN = om_fs
                if j >= 1:
                    kS = tke[j - 1, i]; omS = tom[j - 1, i]
                else:
                    kS = 0.0;           omS = om_w[i]

                # Corrected face-normal velocities (upwind switch for k/ω adv)
                unE = 0.5*(uc + uE)*nxE[j, i] + 0.5*(vc + vE)*nyE[j, i]
                unW = 0.5*(uc + uW)*nxW[j, i] + 0.5*(vc + vW)*nyW[j, i]
                unN = 0.5*(uc + uN)*nxN[j, i] + 0.5*(vc + vN)*nyN[j, i]
                unS = 0.5*(uc + uS)*nxS[j, i] + 0.5*(vc + vS)*nyS[j, i]

                ca_sst = cell_area[j, i] if cell_area[j, i] > 1e-6 else 1e-6

                # Velocity gradients (Green-Gauss); wall S-face → uSf = 0
                uEf = 0.5*(uc + uE); vEf = 0.5*(vc + vE)
                uWf = 0.5*(uc + uW); vWf = 0.5*(vc + vW)
                uNf = 0.5*(uc + uN); vNf = 0.5*(vc + vN)
                uSf = 0.5*(uc + uS); vSf = 0.5*(vc + vS)
                dudx = (uEf*nxE[j, i]*dsE[j, i] + uWf*nxW[j, i]*dsW[j, i]
                      + uNf*nxN[j, i]*dsN[j, i] + uSf*nxS[j, i]*dsS[j, i]) / ca_sst
                dudy = (uEf*nyE[j, i]*dsE[j, i] + uWf*nyW[j, i]*dsW[j, i]
                      + uNf*nyN[j, i]*dsN[j, i] + uSf*nyS[j, i]*dsS[j, i]) / ca_sst
                dvdx = (vEf*nxE[j, i]*dsE[j, i] + vWf*nxW[j, i]*dsW[j, i]
                      + vNf*nxN[j, i]*dsN[j, i] + vSf*nxS[j, i]*dsS[j, i]) / ca_sst
                dvdy = (vEf*nyE[j, i]*dsE[j, i] + vWf*nyW[j, i]*dsW[j, i]
                      + vNf*nyN[j, i]*dsN[j, i] + vSf*nyS[j, i]*dsS[j, i]) / ca_sst

                S2 = 2.0*dudx*dudx + 2.0*dvdy*dvdy + (dudy + dvdx)**2
                if S2 < 0.0:
                    S2 = 0.0
                S = S2**0.5
                if S2 > S_max*S_max:
                    S2 = S_max*S_max
                if S > S_max:
                    S = S_max

                k_s  = kc if kc > 0.0 else 0.0
                om_s = omc if omc > 1e-10 else 1e-10

                # ∇k, ∇ω (Green-Gauss)
                dkdx = (0.5*(kc+kE)*nxE[j, i]*dsE[j, i] + 0.5*(kc+kW)*nxW[j, i]*dsW[j, i]
                      + 0.5*(kc+kN)*nxN[j, i]*dsN[j, i] + 0.5*(kc+kS)*nxS[j, i]*dsS[j, i]) / ca_sst
                dkdy = (0.5*(kc+kE)*nyE[j, i]*dsE[j, i] + 0.5*(kc+kW)*nyW[j, i]*dsW[j, i]
                      + 0.5*(kc+kN)*nyN[j, i]*dsN[j, i] + 0.5*(kc+kS)*nyS[j, i]*dsS[j, i]) / ca_sst
                domdx = (0.5*(omc+omE)*nxE[j, i]*dsE[j, i] + 0.5*(omc+omW)*nxW[j, i]*dsW[j, i]
                       + 0.5*(omc+omN)*nxN[j, i]*dsN[j, i] + 0.5*(omc+omS)*nxS[j, i]*dsS[j, i]) / ca_sst
                domdy = (0.5*(omc+omE)*nyE[j, i]*dsE[j, i] + 0.5*(omc+omW)*nyW[j, i]*dsW[j, i]
                       + 0.5*(omc+omN)*nyN[j, i]*dsN[j, i] + 0.5*(omc+omS)*nyS[j, i]*dsS[j, i]) / ca_sst

                gk_gom = dkdx*domdx + dkdy*domdy
                if gk_gom != gk_gom:          # NaN
                    gk_gom = 0.0
                if gk_gom > 1e20:
                    gk_gom = 1e20
                elif gk_gom < -1e20:
                    gk_gom = -1e20
                cd_t = 2.0*sw2 / om_s * gk_gom
                CD_kw = cd_t if cd_t > 1e-10 else 1e-10

                dw  = d_w[j, i]
                r1a = k_s**0.5 / (beta_str*om_s*dw + 1e-30)
                r1b = 500.0*nu / (om_s*dw*dw + 1e-30)
                r1c = 4.0*sw2*k_s / (CD_kw*dw*dw + 1e-30)
                _mx = r1a if r1a > r1b else r1b
                arg1 = _mx if _mx < r1c else r1c
                if arg1 > 50.0:
                    arg1 = 50.0
                F1 = math.tanh(arg1**4)
                r2a = 2.0*k_s**0.5 / (beta_str*om_s*dw + 1e-30)
                r2b = 500.0*nu / (om_s*dw*dw + 1e-30)
                r2 = r2a if r2a > r2b else r2b
                if r2 > 50.0:
                    r2 = 50.0
                F2 = math.tanh(r2**2)

                sk  = F1*sk1 + (1.0 - F1)*sk2
                sw  = F1*sw1 + (1.0 - F1)*sw2
                bbl = F1*b1  + (1.0 - F1)*b2
                gbl = F1*g1  + (1.0 - F1)*g2

                _den_nt = a1*om_s if a1*om_s > S*F2 else S*F2
                nt = a1*k_s / _den_nt
                if nt > 1.0e4*nu:
                    nt = 1.0e4*nu
                nt = nt * nu_t_mult
                if j == 0:
                    nt = 0.0
                nu_t[j, i] = nt

                Pk_cap = pk_limit*beta_str*k_s*om_s
                Pk = nt*S2 if nt*S2 < Pk_cap else Pk_cap

                ddE = dsE[j, i]/dnE[j, i]; ddW = dsW[j, i]/dnW[j, i]
                ddN = dsN[j, i]/dnN[j, i]; ddS = dsS[j, i]/dnS[j, i]
                ca = cell_area[j, i]
                dtc = dt_arr[j, i]

                # k transport (upwind advection)
                kE_uw = kc if unE > 0 else kE
                kW_uw = kc if unW > 0 else kW
                kN_uw = kc if unN > 0 else kN
                kS_uw = kc if unS > 0 else kS
                adv_k = (unE*kE_uw*dsE[j, i] + unW*kW_uw*dsW[j, i]
                       + unN*kN_uw*dsN[j, i] + unS*kS_uw*dsS[j, i]) / ca
                Dk = (nu + sk*nt) / ca
                nk = kE*ddE + kW*ddW + kN*ddN + kS*ddS
                dk = ddE + ddW + ddN + ddS
                kn = (kc - dtc*adv_k + dtc*Dk*nk + dtc*Pk) / (1.0 + dtc*Dk*dk + dtc*beta_str*om_s + 1e-30)
                if kn < 0.0:
                    kn = 0.0

                # ω transport (semi-implicit cross-diffusion neutralisation)
                xd = 2.0*(1.0 - F1)*sw2 / om_s * gk_gom
                if xd < 0.0:
                    xd = 0.0
                omE_uw = omc if unE > 0 else omE
                omW_uw = omc if unW > 0 else omW
                omN_uw = omc if unN > 0 else omN
                omS_uw = omc if unS > 0 else omS
                adv_om = (unE*omE_uw*dsE[j, i] + unW*omW_uw*dsW[j, i]
                        + unN*omN_uw*dsN[j, i] + unS*omS_uw*dsS[j, i]) / ca
                Dom = (nu + sw*nt) / ca
                nom = omE*ddE + omW*ddW + omN*ddN + omS*ddS
                Pom_cap = pk_limit*bbl*om_s*om_s
                Pom = gbl*S2 if gbl*S2 < Pom_cap else Pom_cap
                on = (omc - dtc*adv_om + dtc*Dom*nom + dtc*Pom + dtc*xd) / (
                     1.0 + dtc*Dom*dk + dtc*bbl*om_s + dtc*xd/om_s + 1e-30)
                if on < 1e-10:
                    on = 1e-10

                # Runaway guards (no-op for well-behaved cells)
                if kn > 1.0:
                    kn = 1.0
                if on > 1e6:
                    on = 1e6

                # Boundary conditions overwrite transport result
                if j == 0:
                    kn = 0.0;    on = om_w[i]
                elif j == n_eta - 1:
                    kn = k_fs;   on = om_fs
                tke_new[j, i] = kn
                tom_new[j, i] = on
        return tke_new, tom_new, nu_t

    _HAVE_NUMBA = True
except Exception:                                              # pragma: no cover
    _HAVE_NUMBA = False


# ── Optional compiled Fortran kernels (f2py) ─────────────────────────────────
# Built from solver/bfm_explicit_f.f90; bit-faithful with the Numba kernels.
# Preferred over Numba when present; falls back to Numba, then NumPy.
try:
    from . import bfm_explicit_f as _FF
    _HAVE_FORTRAN = True
except Exception:
    try:
        import bfm_explicit_f as _FF              # flat-import fallback
        _HAVE_FORTRAN = True
    except Exception:
        _FF = None
        _HAVE_FORTRAN = False


# ── Solver entry point ────────────────────────────────────────────────────────

def run_bfm_simulation(gui, p):
    """
    Run one BFM simulation.  Called in a background thread; callbacks delivered
    via gui.root.after(0, ...) exactly as in run_fluid_simulation().

    Parameters recognised in *p* beyond the Cartesian solver:
        airfoil  : str, e.g. "0012" (NACA) or an absolute path to a .dat file
        n_xi     : int, circumferential cells (default 96)
        n_eta    : int, radial cells (default 48)
        ogrid_R  : float, far-field radius in chord lengths (default 8.0)
    """
    start_time = time.perf_counter()

    # ── Grid / flow parameters ────────────────────────────────────────────────
    airfoil_id  = str(p.get("airfoil", "0012"))
    n_xi        = int(p.get("n_xi",   96))
    n_eta       = int(p.get("n_eta",  48))

    # Far-field radius: 15 chord lengths by default (was 8).
    # Larger domain reduces far-field BC contamination at higher Re.
    ogrid_R     = float(p.get("ogrid_R", 15.0))

    # Near-wall sinh stretching: auto-scale alpha with radial resolution so
    # the first cell stays adequately thin as n_eta grows.
    #   n_eta  32 → alpha 6.0
    #   n_eta  48 → alpha 6.6
    #   n_eta  64 → alpha 7.0
    #   n_eta 128 → alpha 8.0
    #   n_eta 160 → alpha 8.3
    # Base raised from 3.0 to 6.0 (2026-06): the old thick first cell
    # over-resolved the outer field but UNDER-resolved the boundary layer,
    # which thickened the numerical BL and decambered the airfoil — lift was
    # ~40% low.  A thinner first cell recovers it: an alpha sweep at
    # NACA 0012, AoA=4°, Re=5e5, 128×64 peaked at alpha≈7 (Cl 0.149→0.247,
    # matching a 4× finer uniform grid) and stayed symmetric (Cl≈0 at AoA=0).
    # Above ~9 the outer field starves and lift drops again.  At typical
    # settings (dt=2e-4) the CFL limiter does not bite until alpha≳9, so this
    # accuracy gain is essentially free; the alpha>6 branch already tightens
    # the CFL target for the thinner cells.
    if "alpha_stretch" in p:
        alpha_stretch = float(p["alpha_stretch"])
    else:
        alpha_stretch = 6.0 + max(0.0, math.log2(n_eta / 32.0))

    # MUSCL slope-correction factor for E/W faces. 0.5 = full 2nd-order
    # (unstable at high Re on fine grids), 0.25 = blended (default), 0.0 = pure
    # 1st-order upwind. Exposed as a parameter for diagnostics/ablation.
    _muscl_f    = float(p.get("muscl_factor", 0.25))

    # Pressure-solver under-relaxation from the GUI "Omega" field (was
    # hardcoded 0.6, silently ignoring the user's setting). Clamped to a
    # stable range for the Jacobi-style sweep used here.
    _sor_omega  = min(max(float(p.get("omega", 0.6)), 0.05), 1.0)

    # Pressure-Poisson solver.  "jacobi" = the legacy fixed-sweep smoother;
    # "cg" = matrix-free conjugate gradient to a tolerance.  The 200 Jacobi
    # sweeps barely move the LOW-wavenumber pressure modes on a 128-wide grid
    # (measured: ~0% residual reduction), leaving the global stagnation /
    # circulation pressure unconverged → suppressed, refinement-worsening lift.
    # CG converges those modes, so it is the accuracy fix.
    _p_solver   = str(p.get("p_solver", "jacobi"))
    _p_sweeps   = int(p.get("p_sweeps", 200))
    # Use the compiled Numba Jacobi kernel when available (identical arithmetic,
    # just faster).  Falls back to the NumPy sweep automatically.
    _jit        = bool(p.get("jit", True)) and _HAVE_NUMBA
    _cg_rtol    = float(p.get("cg_rtol", 1e-3))
    _cg_maxit   = int(p.get("cg_maxit", 600))

    # Diagnostic global multiplier on the SST eddy viscosity ν_t.  1.0 = normal
    # SST; 0.0 = effectively laminar (effective Re = molecular Re everywhere).
    # Used to test whether outer/freestream over-diffusion (ν_t capped at ~100ν,
    # i.e. effective Re ~1e4) is decambering the airfoil and suppressing lift.
    _nu_t_mult  = float(p.get("nu_t_mult", 1.0))

    # Rhie–Chow momentum interpolation (Rhie & Chow, AIAA J. 1983) — opt-in.
    # Adds a 3rd-derivative pressure damping to the circumferential continuity
    # flux to suppress the collocated-grid checkerboard the central scheme
    # admits.  Off by default while it is validated against the symmetry gate.
    _rhie_chow  = bool(p.get("rhie_chow", False))

    # Advection scheme: "central_jst" (default), "central", or "upwind".
    # The legacy "upwind" MUSCL scheme's |un|-weighted dissipation, although
    # mirror-symmetric in form, destabilises the antisymmetric mode and gives
    # symmetric bodies a large spurious steady Cl (NACA 0012 at AoA=0 gave
    # Cl≈0.2–0.35 instead of 0).  "central" advection keeps the symmetric
    # solution stable (Cl→0 to machine precision); "central_jst" adds 4th-order
    # symmetric dissipation for high-Re / fine-grid robustness without
    # reintroducing the instability.  See project memory for the full diagnosis.
    _adv_scheme = str(p.get("adv_scheme", "central_jst"))
    _jst_eps4   = float(p.get("jst_eps4", 0.02))   # 4th-order dissipation coeff
    # Higher-order (4th-order central) interpolation of the convective face
    # values on the periodic E/W faces.  O(h⁴) vs the default O(h²), so it
    # sharpens the leading-edge suction peak WITHOUT adding dissipation —
    # recovers a little more lift once the flow is fully developed.  Opt-in.
    _ho_faces   = bool(p.get("ho_faces", False))
    # Turbulence production-limiter multiple of the destruction rate.  Menter
    # (2003) uses ~10.  An earlier attempt at 10× inflated Cd 4–6× — but that was
    # because the wall-shear closure was τ_w ∝ √k, which over-reads when k grows.
    # The wall shear is now a VELOCITY-based law-of-the-wall (see _compute_forces),
    # insensitive to k magnitude, so production can be raised to develop a real
    # turbulent boundary layer.  At 1.0 production is clamped to destruction
    # (k stays ≈ freestream → ν_t ≈ 0 → laminar-like BL that separates and sheds);
    # 10.0 lets k develop and hold the BL attached.  Validated with the new
    # wall function + raised ν_t cap.
    _pk_limit   = float(p.get("pk_limit", 10.0))

    # Use the compiled momentum predictor (advection + point-implicit diffusion)
    # for the default central / central_jst schemes.  The legacy "upwind" MUSCL
    # path and the opt-in 4th-order "ho_faces" interpolation stay on NumPy (their
    # wider/limited stencils aren't transcribed).  Bit-faithful with the NumPy
    # branch, so it only changes speed, not the answer.
    _use_mom_jit = (_jit and _adv_scheme in ("central", "central_jst")
                    and not _ho_faces)
    # Compiled-kernel backend: prefer Fortran when built, else Numba, else NumPy.
    # `fortran` param (default True) lets the user force the Numba path off it.
    # _fortran_pref gates the pressure-Poisson and SST kernels (independent of the
    # advection scheme); _use_fortran additionally requires the compiled momentum
    # predictor to apply (central/central_jst, no higher-order faces).  Keeping
    # them separate means higher-order faces (standard in v0.1.2.0, momentum stays
    # on NumPy) does NOT disable the Fortran pressure/SST kernels.
    _fortran_pref = (_HAVE_FORTRAN and _jit and bool(p.get("fortran", True)))
    _use_fortran  = (_fortran_pref and _use_mom_jit)

    # ── Local time-stepping (steady accelerator, #1A) ─────────────────────────
    # Opt-in.  Replaces the single global dt (set by the SMALLEST cell's CFL)
    # with a per-cell pseudo time step dt_local[j,i] = CFL·min_face_local /
    # (v_scale·U∞), so large far-field cells march far faster than the tiny
    # wall/TE cells.  Collapses the ~500k-iter development cost toward thousands
    # of iters — but ONLY for STEADY cases: per-cell dt destroys time-accuracy,
    # so vortex shedding / Strouhal are meaningless with it on.  The projection
    # is recast in dt-weighted form (Poisson coeff ·dt_face, RHS = ρ∮u*·n, no
    # /dt); with a CONSTANT dt field this is algebraically identical to the
    # global-dt scheme (dt cancels in the Jacobi/CG iteration), so local_dt=False
    # is bit-for-bit the old solver.
    _local_dt   = bool(p.get("local_dt", False))
    # Velocity scale for the local CFL.  >1 keeps the suction-peak cells (where
    # |U| exceeds U∞ by ~2×) stable; 1.5 is a safe default for airfoils.
    _lts_vscale = max(float(p.get("lts_vscale", 1.5)), 1e-6)

    # Point-vortex far-field correction (default on).  A lifting airfoil's bound
    # circulation induces a velocity ∝ Γ/r that is still finite at the outer
    # boundary (R≈15c); clamping the outer ring to pure u_∞ (Dirichlet) fights
    # that circulation and suppresses lift.  Instead superpose the velocity of a
    # point vortex Γ = −½·Cl·U·c (Kutta–Joukowski; Cl>0 → clockwise → Γ<0),
    # lagged from the force calc and under-relaxed by ff_relax to damp the
    # Cl→BC→Cl feedback.  The induced speed at R=15 is only ~Γ/(2πR) (a few % of
    # U∞), so this is a boundary-consistency correction, not a large forcing.
    # Default OFF: a 2026-06-14 ablation (4412 α=4) showed the correction moves
    # Cl by only −0.1% at R=15c — the induced speed there (~Γ/2πR) is ~0.3% of
    # U∞, so it cannot explain the ~65% lift deficit (that is the central scheme
    # + grid resolution smearing the suction peak; the zero-lift angles are
    # already correct).  Kept as opt-in, like rhie_chow, for use with a smaller
    # outer domain where the Dirichlet error — and thus this correction — matters.
    _farfield_vortex = bool(p.get("farfield_vortex", False))
    _ff_relax        = float(p.get("ff_relax", 0.2))

    # Symmetry-probe interval (0 = off): every N steps, print the mirror
    # asymmetry of u/v/p under the cell mirror i↔n_xi-1-i (u/p even, v odd).
    # For a symmetric body at AoA=0 these should stay near round-off; used to
    # regression-test the trailing-edge / advection symmetry.
    _mirror_every = int(p.get("mirror_check_every", 0))

    Re          = max(float(p.get("re", 100)),  1.0)
    aoa_deg     = float(p.get("aoa",  0.0))
    aoa_rad     = math.radians(aoa_deg)
    dt          = float(p.get("dt",   0.2))
    max_iters   = int(p["max_iters"])
    hist_interval = max(1, int(p.get("hist_interval", 50)))
    T_wall      = float(p.get("t_wall", 320.0))
    T_inf       = float(p.get("t_inf",  300.0))

    print(f"--- BFM Solver Started (Re:{Re} AoA:{aoa_deg} airfoil:{airfoil_id} "
          f"n_xi:{n_xi} n_eta:{n_eta} R:{ogrid_R} alpha:{alpha_stretch:.2f} "
          f"SST:{'ON' if Re >= 500 else 'OFF'}) ---")

    # ── Airfoil / body coordinates ────────────────────────────────────────────
    if airfoil_id.endswith(".dat") or airfoil_id.endswith(".txt"):
        xa_raw, ya_raw = load_dat(airfoil_id)
        xa_a,   ya_a   = resample(xa_raw, ya_raw, n_xi)
    elif airfoil_id == "rect_1x8":
        xa_raw, ya_raw = rect_1x8()
        xa_a,   ya_a   = resample(xa_raw, ya_raw, n_xi)
    else:
        xa_a, ya_a = naca4(airfoil_id.zfill(4), n_pts=n_xi // 2)
        xa_a, ya_a = resample(xa_a, ya_a, n_xi)

    # ── O-grid ────────────────────────────────────────────────────────────────
    grid = build_ogrid(xa_a, ya_a, n_xi=n_xi, n_eta=n_eta, R=ogrid_R,
                       alpha_stretch=alpha_stretch)
    chord = grid['chord']   # ≈ 1.0 (normalised)

    nxE, nyE, dsE, dnE = grid['nxE'], grid['nyE'], grid['dsE'], grid['dnE']
    nxW, nyW, dsW, dnW = grid['nxW'], grid['nyW'], grid['dsW'], grid['dnW']
    nxN, nyN, dsN, dnN = grid['nxN'], grid['nyN'], grid['dsN'], grid['dnN']
    nxS, nyS, dsS, dnS = grid['nxS'], grid['nyS'], grid['dsS'], grid['dnS']
    cell_area           = grid['cell_area']
    XC, YC              = grid['XC'], grid['YC']

    ip1 = (np.arange(n_xi) + 1) % n_xi
    im1 = (np.arange(n_xi) - 1) % n_xi
    ip2 = (np.arange(n_xi) + 2) % n_xi   # two cells east  (for MUSCL)
    im2 = (np.arange(n_xi) - 2) % n_xi   # two cells west


    # ── Poisson coefficients (precomputed, time-invariant) ───────────────────
    cE = dsE / dnE;  cW = dsW / dnW
    cN = dsN / dnN;  cS_raw = dsS / dnS

    # Wall BC: south face of j=0 → Neumann → zero coefficient
    cS = cS_raw.copy()
    cS[0, :] = 0.0

    coeff_denom = cE + cW + cN + cS + 1e-30

    # ── Physical parameters ───────────────────────────────────────────────────
    sim_vel = 1.0
    nu      = (sim_vel * chord) / Re
    rho     = 1.0
    q_ref   = 0.5 * rho * sim_vel**2 * chord

    u_inf   =  sim_vel * math.cos(aoa_rad)
    v_inf   =  sim_vel * math.sin(aoa_rad)

    # Far-field point-vortex geometry: outer-ring cell centres relative to the
    # airfoil centroid.  _u_out/_v_out are the outer-ring Dirichlet values; they
    # equal (u_inf, v_inf) until the lagged circulation update fills in the
    # induced velocity, so with _farfield_vortex off they are a no-op.
    _xv_c = float(np.mean(xa_a));  _yv_c = float(np.mean(ya_a))
    _dx_o = XC[-1, :] - _xv_c
    _dy_o = YC[-1, :] - _yv_c
    _r2_o = np.maximum(_dx_o**2 + _dy_o**2, 1e-12)
    _gamma_smooth = 0.0
    _u_out = np.full(n_xi, u_inf)
    _v_out = np.full(n_xi, v_inf)

    # ── CFL-limited time step ─────────────────────────────────────────────────
    # The BFM grid has chord=1.0 in physical units.  The Cartesian solver uses
    # chord≈32 pixels with dx=1 → effective dt_phys ≈ dt/32.  To get the same
    # CFL number on the BFM grid, scale dt to the minimum cell face length.
    # CFL_target = 0.35 gives a stable margin for upwind + point-implicit.
    _min_face = float(np.minimum(grid['dsE'], grid['dsS']).min())
    # CFL target depends on the advection scheme.  The central schemes
    # (central / central_jst — the default) carry almost no numerical
    # dissipation, so they go CHAOTICALLY UNSTABLE above a fairly low CFL —
    # and the limit falls as Re rises (less physical viscosity to damp the
    # scheme).  Validated on NACA 2412, Re=1e6, 96×48: clean at CFL≈0.05,
    # diverged (residual ~0.2, Cl thrashing) at CFL≈0.25.  Clamp them hard at
    # 0.05 so ANY dt the user (or the dt Calculator) enters is made safe — the
    # solver always reduces dt to a stable value.  The legacy upwind scheme is
    # dissipative and keeps the old, looser 0.25–0.35 target.
    if _adv_scheme in ("central", "central_jst"):
        _cfl_target = 0.05
    else:
        _cfl_target = 0.25 if alpha_stretch > 6.0 else 0.35
    dt_bfm    = min(float(dt), _cfl_target * _min_face / sim_vel)
    dt_bfm    = max(dt_bfm, 1e-6)
    print(f"    dt_input={dt:.4f} → dt_bfm={dt_bfm:.5f}  "
          f"min_face={_min_face:.4f}  CFL≈{sim_vel*dt_bfm/_min_face:.2f}")
    dt = dt_bfm   # replace dt with the stable value throughout

    # Physical strain-rate ceiling (constant) — clips only degenerate
    # near-singularity cells.  Precomputed here so the SST kernel can take it as
    # a scalar; the NumPy SST path recomputes the identical value inline.
    _S_max = sim_vel / max(_min_face, 1e-10)

    # ── Per-cell time-step field + dt-weighted Poisson coefficients ───────────
    # _dt_field is the (time-invariant) per-cell pseudo time step used in EVERY
    # transport update (momentum / temperature / SST) and the projection.  With
    # local_dt OFF it is the constant global dt everywhere → the pressure system
    # below reduces exactly to the legacy constant-dt projection.  With local_dt
    # ON each cell gets CFL·min(local face)/(v_scale·U∞), floored at the global
    # dt's smallest stable value.
    if _local_dt:
        _min_face_cell = np.minimum(np.minimum(grid['dsE'], grid['dsW']),
                                    np.minimum(grid['dsN'], grid['dsS']))
        _dt_field = _cfl_target * _min_face_cell / (_lts_vscale * sim_vel)
        _dt_field = np.maximum(_dt_field, 1e-6)
        print(f"    [local_dt] dt_local range "
              f"[{float(_dt_field.min()):.2e}, {float(_dt_field.max()):.2e}]  "
              f"ratio {float(_dt_field.max()/_dt_field.min()):.0f}x  v_scale={_lts_vscale}")
    else:
        _dt_field = np.full((n_eta, n_xi), dt)

    # dt-weighted face coefficients for the pressure-Poisson.  The projection
    # enforces ∮u·n=0 via  Σ_f dt_face·(ds/dn)·(p_nbr−p) = ρ·∮u*·n ds, so the
    # face coefficients carry dt_face = ½(dt_local of the two adjacent cells).
    # (cE/cW/cN/cS already hold ds/dn; cS[0]=0 keeps the Neumann wall.)
    _dtfE = 0.5*(_dt_field + _dt_field[:, ip1])
    _dtfW = 0.5*(_dt_field + _dt_field[:, im1])
    _dtfN = _dt_field.copy()
    _dtfN[:-1, :] = 0.5*(_dt_field[:-1, :] + _dt_field[1:, :])
    _dtfS = _dt_field.copy()
    _dtfS[1:, :]  = 0.5*(_dt_field[1:, :] + _dt_field[:-1, :])
    cE_p = _dtfE * cE;  cW_p = _dtfW * cW
    cN_p = _dtfN * cN;  cS_p = _dtfS * cS
    coeff_denom_p = cE_p + cW_p + cN_p + cS_p + 1e-30

    Pr      = 0.71
    alpha_t = nu / Pr

    # ── k-ω SST turbulence model setup ───────────────────────────────────────
    # Activated only at Re ≥ 500; below this turbulence is negligible and the
    # model costs compute for no physical benefit.
    _kw = (Re >= 500)

    # Menter (2003) SST constants
    _a1       = 0.31
    _beta_str = 0.09          # β* — TKE destruction
    _kappa    = 0.41          # von Kármán constant
    # Zone 1: k-ω behaviour (near wall)
    _sk1 = 0.85;  _sw1 = 0.5;   _b1 = 0.075
    _g1  = _b1 / _beta_str - _sw1 * _kappa**2 / math.sqrt(_beta_str)
    # Zone 2: k-ε behaviour (far field, rewritten in ω)
    _sk2 = 1.0;   _sw2 = 0.856; _b2 = 0.0828
    _g2  = _b2 / _beta_str - _sw2 * _kappa**2 / math.sqrt(_beta_str)
    _Pr_t = 0.9   # turbulent Prandtl number

    # Wall distance: for each O-grid cell, the distance from the cell centre
    # to the MIDPOINT of the wall face below it (that face spans surface
    # vertices i and i+1).  Pairing a cell with a single vertex i is chiral —
    # every cell pairs with its CCW-trailing vertex — so use the face midpoint,
    # which is the symmetric, cell-aligned reference point.
    _xa_wmid = 0.5 * (grid['xa'] + np.roll(grid['xa'], -1))
    _ya_wmid = 0.5 * (grid['ya'] + np.roll(grid['ya'], -1))
    _d_w = np.sqrt((XC - _xa_wmid[np.newaxis, :])**2 +
                   (YC - _ya_wmid[np.newaxis, :])**2)
    _d_w = np.maximum(_d_w, 1e-10)

    # Freestream turbulence at 0.1% intensity (clean-tunnel level — 0.5%
    # seeded excess ν_t through the boundary layer and inflated Cd ~40%)
    _Tu   = 0.001
    _k_fs = 1.5 * (_Tu * sim_vel)**2
    _om_fs = math.sqrt(_k_fs) / (_beta_str**0.25 * chord)

    # Wall ω BC — Menter (1994) hydraulically-smooth wall:
    #   ω_w = 60 ν / (β₁ · d₁²)
    # d₁ = j=0 cell-centre distance to wall surface.
    # Capped at 1e5 (lowered from 1e6) to limit the ω gradient driven into
    # j=1 cells from the trailing-edge singularity where d₁ → 0.
    _om_w = np.minimum(60.0 * nu / (_b1 * _d_w[0, :]**2), 1.0e5)

    # k and ω state arrays.
    #
    # k is initialised with a linear profile (0 at wall → k_fs at far
    # field) to avoid a discontinuous ∂k/∂y at j=1.
    #
    # ω is initialised with a log-linear profile from ω_w (wall) to ω_fs
    # (far field).  This eliminates the startup gradient shock that occurs
    # with a flat ω_fs init where ω[j=0]=ω_w >> ω[j=1]=ω_fs: on the first
    # step the diffusion flux (ν/cell_area)*(ω_w - ω_fs)*dsS/dnS lands
    # entirely on j=1 cells, which can be huge for Lv4 grids where ω_w is
    # large and cell_area is small.  The log-linear profile distributes this
    # gradient smoothly over all radial cells from the start.
    _eta_frac = np.linspace(0.0, 1.0, n_eta)[:, np.newaxis]  # shape (n_eta, 1)
    tke = (_k_fs * _eta_frac) * np.ones((1, n_xi))   # linear 0 → k_fs

    _log_om_w  = np.log(np.maximum(_om_w, 1e-10))    # shape (n_xi,)
    _log_om_fs = math.log(max(_om_fs, 1e-10))
    tom = np.exp(
        _log_om_w[np.newaxis, :] * (1.0 - _eta_frac) +
        _log_om_fs * _eta_frac
    )

    tke[0,  :] = 0.0;    tom[0,  :] = _om_w          # wall BCs
    tke[-1, :] = _k_fs;  tom[-1, :] = _om_fs         # far-field BCs
    nu_t = np.zeros((n_eta, n_xi))  # eddy viscosity — zero at start

    # Real-world force scaling (matches Cartesian solver)
    _P_atm    = float(p.get("pressure", 101325.0))
    _vel_real = float(p.get("vel", 25.0))
    _rho_real = _P_atm / (287.05 * 288.15)
    _q_real   = 0.5 * _rho_real * _vel_real**2
    _nu_air   = 1.5e-5
    _chord_m_input = p.get("chord_m", None)
    if _chord_m_input is not None and float(_chord_m_input) > 0:
        _chord_m = float(_chord_m_input)          # user-specified chord [m]
    else:
        _chord_m = Re * _nu_air / max(_vel_real, 0.01)  # derived [m]

    # ── State arrays ──────────────────────────────────────────────────────────
    u     = np.full((n_eta, n_xi), u_inf)
    v     = np.full((n_eta, n_xi), v_inf)
    press = np.zeros((n_eta, n_xi))
    T     = np.full((n_eta, n_xi), T_inf)

    # Warmup = 4 chord-crossing times (same policy as Cartesian solver)
    _flow_through = int(chord / (sim_vel * dt))
    # Warmup cap: at most 5000 iterations (1 chord-crossing at dt=2e-4) or
    # max_iters//4, whichever is smaller.  The old formula (min(4*flow_through,
    # max_iters//2)) grew to 20 000+ when max_iters=50 000, blocking the
    # periodic-convergence check until iter ~20 400 even though the flow
    # plateaus at ~1 900.  That 18 000-iteration gap caused NaN crashes at
    # 512×128 (Lv4) that did NOT appear in the 3000-iter smoke tests.
    warmup = max(200, min(4 * _flow_through, 5000, max_iters // 4))

    # Physical-time-based windows (dt-independent).
    # _AVG_WIN:  number of 50-iter samples spanning 10 chord-crossing times.
    # _conv_win: iterations spanning 5 chord-crossing times for the periodic
    #            convergence check (replaces the fixed 5000-iteration window).
    _flow_cross = chord / sim_vel                                 # one chord crossing [s]
    # _AVG_WIN: a RECENT window (≈3 crossings of 50-iter samples, capped at 60)
    # so the reported Cl/Cd is the SETTLED value, not an average over the
    # developing transient.  The old "10 crossings" formula scaled as 1/dt and,
    # on fine grids (tiny dt), demanded more samples than the whole run held →
    # it averaged the rise from zero and badly under-reported lift.
    _AVG_WIN    = max(20, min(int(3.0 * _flow_cross / (dt * 50)), 60))
    # _conv_win: iterations spanning 5 chord-crossings before periodic
    # convergence may be declared.  Cap raised to max_iters//2 (was //4): with
    # the tiny dt of fine grids the per-step residual is small regardless of
    # whether the flow has physically developed, so //4 let runs "converge" at
    # ~1 crossing — under-developed, hence low lift.
    _conv_win   = max(500, min(int(5.0 * _flow_cross / dt), max_iters // 2))
    # Force-based developed-convergence (the PRIMARY criterion).  A run may NOT
    # be declared converged until the circulation has developed for at least
    # _min_cross chord-crossings of flow time.  The per-step velocity residual
    # goes small long before the LIFT settles (unsteady/Wagner development takes
    # ~10–15 crossings), so the old residual gate rubber-stamped under-developed
    # flows at ~1 crossing → Cl ~65% of steady, and worse on finer grids (tiny
    # dt → fewer crossings per iter).  After the floor, convergence is declared
    # once the windowed-mean Cl stops changing.  Both are dt-independent, hence
    # grid-independent — which removes the spurious "lift falls with refinement".
    _min_cross    = float(p.get("min_crossings", 8.0))
    _cl_conv_tol  = float(p.get("cl_conv_tol", 0.005))
    _dev_floor    = int(_min_cross * _flow_through)   # iters before convergence allowed
    # Force-plateau half-window (number of 50-iter Cl samples per comparison
    # window).  Precomputed once instead of per-step.
    _fwin         = max(6, int(2.0 * _flow_through / 50))
    # Local time-stepping breaks the "chord-crossing" clock (each cell has its
    # own pseudo-dt), so the global-dt-based development floor would block the
    # auto-stop for tens of thousands of OUTER iterations and erase the speedup.
    # Under local_dt, gate on a modest fixed iteration floor and a short plateau
    # window, and let the force-plateau (windowed Cl settling) call convergence.
    if _local_dt:
        _dev_floor = max(300, warmup)
        _fwin      = 8
    # When True, ALL auto-stop sensors (steady / periodic / developed) are
    # disabled and the run goes the full max_iters (or until the user clicks
    # Finish).  Lets the user guarantee enough chord-crossings of development on
    # fine grids, where the sensors may otherwise stop the run early.
    _no_autostop  = bool(p.get("no_autostop", False))
    cl_samples, cd_samples, cm_samples = [], [], []
    conv_state   = "Max iterations reached"
    res          = 0.0
    cl = cd = ld_ratio = cm = 0.0
    nu_val = 0.0
    st     = float('nan')
    cl_std = 0.0
    lift_n = 0.0
    drag_n = 0.0
    res_history = []

    u_old = np.empty_like(u)
    _body_mask_sent = False   # send airfoil mask to GUI once per run

    # ── Distance-weighted MUSCL setup ────────────────────────────────────────
    # _dE_circ[j,i] = Euclidean distance from cell (j,i) to cell (j,i+1).
    # Scaling the slope correction by actual spacing makes the limiter
    # independent of the varying circumferential cell size (dense near LE/TE,
    # sparser near mid-chord).  Without this, the naive Δu-based minmod
    # compares gradients from cells of very different sizes, breaking symmetry
    # for symmetric airfoils at 0 AoA and smearing the suction peak.
    _dE_circ = np.sqrt((XC[:, ip1] - XC)**2 + (YC[:, ip1] - YC)**2)

    def _mm(a, b):
        """Minmod slope limiter (returns 0 when a and b have opposite signs)."""
        return np.where(a * b > 0, np.where(np.abs(a) <= np.abs(b), a, b), 0.0)

    # ── Matrix-free Conjugate Gradient pressure solver ───────────────────────
    # Solves the same discrete Poisson system the Jacobi smoother targets,
    #   A p = -rhs,   A p = coeff_denom*p − Σ c_nbr·p_nbr,
    # reusing the precomputed face coefficients (cE/cW/cN/cS, coeff_denom) as a
    # matrix-free operator.  A is SPD (Dirichlet outer p=0 pins the level;
    # Neumann wall enters via cS[0]=0).  CG converges the low-wavenumber modes
    # in O(N) iterations where Jacobi needs O(N²), so the global pressure field
    # (stagnation, circulation) is actually resolved.
    def _press_nbr(pp):
        pE = pp[:, ip1];  pW = pp[:, im1]
        pN = np.empty_like(pp);  pS = np.empty_like(pp)
        pN[:-1, :] = pp[1:, :];  pN[-1, :] = 0.0          # Dirichlet outer
        pS[1:, :]  = pp[:-1, :]; pS[0, :] = pp[0, :]      # Neumann wall (cS[0]=0)
        return cE_p*pE + cW_p*pW + cN_p*pN + cS_p*pS      # dt-weighted coeffs

    def _solve_press_cg(p0, rhs_in):
        b = -rhs_in
        nb = float(np.sqrt(np.sum(b*b))) + 1e-30
        p = p0.copy()
        r = b - (coeff_denom_p*p - _press_nbr(p))
        if float(np.sqrt(np.sum(r*r))) / nb < _cg_rtol:
            return p
        d = r.copy();  rs = float(np.sum(r*r))
        for _ in range(_cg_maxit):
            Ad = coeff_denom_p*d - _press_nbr(d)
            al = rs / (float(np.sum(d*Ad)) + 1e-30)
            p += al*d;  r -= al*Ad
            rsn = float(np.sum(r*r))
            if (rsn**0.5) / nb < _cg_rtol:
                break
            d = r + (rsn/(rs + 1e-30))*d;  rs = rsn
        return np.clip(p, -5.0, 5.0)


    # ── Helper: apply BCs ─────────────────────────────────────────────────────
    def _apply_vel_bc(uu, vv):
        # Outer boundary: freestream + lagged point-vortex induced velocity.
        # _u_out/_v_out reduce to (u_inf, v_inf) when the vortex correction is off.
        uu[-1, :] = _u_out;  vv[-1, :] = _v_out

    # ── Helper: compute force coefficients ────────────────────────────────────
    def _compute_forces():
        """Return (cl, cd, ld, cm) using pressure + skin-friction integration."""
        p_inf = float(np.mean(press[-1, :]))
        dp    = press[0, :] - p_inf           # wall-cell pressure excess

        # South face of j=0 cells: nxS[0,:], nyS[0,:] point into body
        # Pressure force on body = Σ dp * n_into_body * ds
        Fx = float(np.sum(dp * nxS[0, :] * dsS[0, :]))
        Fy = float(np.sum(dp * nyS[0, :] * dsS[0, :]))

        # ── Skin friction (viscous shear) contribution ─────────────────────
        # Wall tangent vector (CCW around airfoil): t̂ = (-nyS, nxS)
        # This is perpendicular to the inward wall normal (nxS, nyS).
        _tx = -nyS[0, :]
        _ty =  nxS[0, :]

        # Tangential velocity at j=0 cell centre (first cell above wall).
        # u[0]=0 at the wall face (no-slip ghost), so the gradient is:
        #   ∂u_tang/∂n ≈ u_tang[j=0] / dnS[0,:]
        _utang = u[0, :] * _tx + v[0, :] * _ty

        # Wall shear stress — y⁺-aware blend of viscous sublayer and log-layer.
        #
        # Estimate u_τ from the first interior TKE (Bradshaw equilibrium):
        #   u_τ = β*^0.25 √k₁   →   y⁺ = u_τ · d_w / ν
        #
        # Viscous sublayer (y⁺ < 5):   τ_w = ν · u_tang / d_w   (linear profile)
        # Log layer       (y⁺ > 30):   τ_w = √β* · k₁            (k-based SST)
        # Transition 5 < y⁺ < 30: smooth linear blend.
        #
        # This gives accurate drag in both thin viscous-sublayer cells (fine
        # grids, low Re) and log-layer cells (coarser grids, high Re) without
        # any iterative solve for u_τ.
        # Wall shear from a VELOCITY-based law-of-the-wall (not k-based).  Solve
        #   u⁺ = u_tang/u_τ = f(y⁺),   y⁺ = u_τ·d_w/ν
        # for the friction velocity u_τ by fixed-point iteration, then
        #   τ_w = ρ·u_τ².
        # Being velocity-based, this is INSENSITIVE to the magnitude of the
        # turbulent kinetic energy, so the production limiter / eddy viscosity
        # can be raised to develop a real turbulent boundary layer without the
        # old τ_w ∝ √k closure over-reading skin friction (which inflated Cd
        # 4–6× when the limiter was raised).  f(y⁺): viscous (u⁺=y⁺) below
        # y⁺≈11.6, log law (1/κ)ln(E·y⁺), E=9.8, above (the two meet at 11.6).
        _um   = np.abs(_utang) + 1e-12
        _utau = np.sqrt(nu * _um / np.maximum(_d_w[0, :], 1e-10))    # viscous init
        for _ in range(12):
            _yp    = _utau * _d_w[0, :] / (nu + 1e-30)
            _uplus = np.where(_yp < 11.6, _yp,
                              (1.0/_kappa) * np.log(np.maximum(9.8*_yp, 1.0)))
            _utau  = _um / np.maximum(_uplus, 1e-6)
        _tau_w = rho * _utau**2 * np.sign(_utang + 1e-30)

        # Add shear force to body totals
        Fx += float(np.sum(_tau_w * _tx * dsS[0, :]))
        Fy += float(np.sum(_tau_w * _ty * dsS[0, :]))

        # Rotate into lift/drag frame (positive AoA → u_inf mostly in +x, v_inf in +y)
        cos_a = math.cos(aoa_rad);  sin_a = math.sin(aoa_rad)
        lift  =  Fy * cos_a - Fx * sin_a
        drag  =  Fx * cos_a + Fy * sin_a

        _cl  = lift / (q_ref + 1e-10)
        _cd  = drag / (q_ref + 1e-10)
        _ld  = _cl / (abs(_cd) + 1e-10)

        # Pitching moment about quarter-chord.
        # `mom` is the CCW (z-out-of-page) moment of the pressure force,
        # M_z = Σ (x−x_qc)·Fy − (y−y_ac)·Fx.  The aerodynamic pitching-moment
        # convention is positive NOSE-UP; with the LE at min-x a nose-up
        # rotation is clockwise, i.e. NEGATIVE M_z.  So Cm = −M_z.  Without the
        # negation the cambered NACA sections read +Cm (e.g. 2412 → +0.044,
        # 4412 → +0.092) — correct in magnitude but flipped from the published
        # nose-down values (−0.047, −0.092).  (Previously misattributed to grid
        # resolution; the magnitude match across airfoils showed it was a sign.)
        x_qc = float(xa_a.min()) + 0.25 * chord
        y_ac = float(ya_a.mean())
        mom  = float(np.sum(dp * (
            nyS[0, :] * (_xa_wmid - x_qc) * dsS[0, :]
            - nxS[0, :] * (_ya_wmid - y_ac) * dsS[0, :]
        )))
        _cm  = -mom / (q_ref * chord + 1e-10)
        return _cl, _cd, _ld, _cm

    # ── Main time-stepping loop ───────────────────────────────────────────────
    try:
        for i in range(max_iters):

            if gui.kill_event.is_set():
                conv_state = "Finalized"; break
            while gui.pause_event.is_set() and not gui.kill_event.is_set():
                time.sleep(0.05)
            if gui.kill_event.is_set():
                conv_state = "Finalized"; break

            # ── Apply BCs ────────────────────────────────────────────────────
            _apply_vel_bc(u, v)

            np.copyto(u_old, u)

            if _use_fortran:
                # ── Compiled momentum predictor (Fortran) ─────────────────────
                u_star, v_star = _FF.mom_predict(
                    u, v, nu_t, nxE, nyE, dsE, dnE, nxW, nyW, dsW, dnW,
                    nxN, nyN, dsN, dnN, nxS, nyS, dsS, dnS,
                    cell_area, _dE_circ, _u_out, _v_out,
                    _dt_field, nu, _jst_eps4, 1 if _adv_scheme == "central_jst" else 0)
            elif _use_mom_jit:
                # ── Compiled momentum predictor (Numba) ───────────────────────
                # Bit-faithful with the NumPy else-branch; ~one fused, multi-core
                # pass replacing ~30 temporary arrays.
                u_star, v_star = _mom_predict_jit(
                    u, v, nu_t, nxE, nyE, dsE, dnE, nxW, nyW, dsW, dnW,
                    nxN, nyN, dsN, dnN, nxS, nyS, dsS, dnS,
                    cell_area, _dE_circ, _u_out, _v_out,
                    _dt_field, nu, _jst_eps4, _adv_scheme == "central_jst")
            else:
              # ── Compute neighbour arrays ──────────────────────────────────────
              # East/west (periodic in i)
              u_E = u[:, ip1];  v_E = v[:, ip1]
              u_W = u[:, im1];  v_W = v[:, im1]

              # North (j+1); outer boundary ghost = freestream
              u_N = np.empty_like(u);  v_N = np.empty_like(v)
              u_N[:-1, :] = u[1:, :];  u_N[-1, :] = _u_out
              v_N[:-1, :] = v[1:, :];  v_N[-1, :] = _v_out

              # South (j-1); wall ghost = -cell (antisymmetric → u_face=0 at wall)
              u_S = np.empty_like(u);  v_S = np.empty_like(v)
              u_S[1:, :]  = u[:-1, :];  u_S[0, :] = -u[0, :]
              v_S[1:, :]  = v[:-1, :];  v_S[0, :] = -v[0, :]

              # ── Face-normal velocities ────────────────────────────────────────
              un_E = 0.5*(u + u_E)*nxE + 0.5*(v + v_E)*nyE
              un_W = 0.5*(u + u_W)*nxW + 0.5*(v + v_W)*nyW
              un_N = 0.5*(u + u_N)*nxN + 0.5*(v + v_N)*nyN
              un_S = 0.5*(u + u_S)*nxS + 0.5*(v + v_S)*nyS

              # ── Intermediate velocity (advection + diffusion) ─────────────────
              # ── Distance-weighted MUSCL advection (E/W) + 1st-order (N/S) ──────
              # E/W (circumferential): 2nd-order minmod-limited, scaled by actual
              # cell-to-cell Euclidean distance so the correction is independent
              # of the varying arc-length spacing around the airfoil.
              # N/S (radial): kept 1st-order — the 100:1 cell-size ratio near the
              # wall makes distance-weighted reconstruction in the radial direction
              # unstable without a full wall-function-aware gradient clamp.
              if _adv_scheme == "upwind":
                # Legacy MUSCL-limited upwind (kept for comparison; not the
                # default — its dissipation drives the symmetric-body lift bug).
                u_EE = u[:, ip2];  u_WW = u[:, im2]
                v_EE = v[:, ip2];  v_WW = v[:, im2]

                # E face  (cells i ↔ i+1)
                _sd   = (u_E - u)  / (_dE_circ          + 1e-10)   # C→E slope
                _su   = (u - u_W)  / (_dE_circ[:, im1]  + 1e-10)   # W→C slope (upwind, un>0)
                uE_pos = u   + _muscl_f * _mm(_su, _sd) * _dE_circ
                _su   = (u_EE - u_E) / (_dE_circ[:, ip1] + 1e-10)  # E→EE slope (upwind, un<0)
                uE_neg = u_E - _muscl_f * _mm(_su, _sd) * _dE_circ # _sd reused (same C→E slope)

                _sd   = (v_E - v)  / (_dE_circ          + 1e-10)
                _su   = (v - v_W)  / (_dE_circ[:, im1]  + 1e-10)
                vE_pos = v   + _muscl_f * _mm(_su, _sd) * _dE_circ
                _su   = (v_EE - v_E) / (_dE_circ[:, ip1] + 1e-10)
                vE_neg = v_E - _muscl_f * _mm(_su, _sd) * _dE_circ

                # W face  (cells i-1 ↔ i)
                _dW   = _dE_circ[:, im1]                            # d(W, C)
                _sd   = (u - u_W)  / (_dW               + 1e-10)    # W→C slope
                _su   = (u_W - u_WW) / (_dE_circ[:, im2] + 1e-10)   # WW→W slope (upwind, un>0)
                uW_pos = u_W + _muscl_f * _mm(_su, _sd) * _dW
                _su   = (u_E - u)  / (_dE_circ          + 1e-10)    # C→E slope (upwind, un<0)
                uW_neg = u   - _muscl_f * _mm(_su, _sd) * _dW

                _sd   = (v - v_W)  / (_dW               + 1e-10)
                _su   = (v_W - v_WW) / (_dE_circ[:, im2] + 1e-10)
                vW_pos = v_W + _muscl_f * _mm(_su, _sd) * _dW
                _su   = (v_E - v)  / (_dE_circ          + 1e-10)
                vW_neg = v   - _muscl_f * _mm(_su, _sd) * _dW

                uE_uw = np.where(un_E > 0, uE_pos, uE_neg)
                vE_uw = np.where(un_E > 0, vE_pos, vE_neg)
                uW_uw = np.where(un_W > 0, uW_pos, uW_neg)
                vW_uw = np.where(un_W > 0, vW_pos, vW_neg)

                # N/S: first-order upwind (radial direction)
                uN_uw = np.where(un_N > 0, u, u_N);  vN_uw = np.where(un_N > 0, v, v_N)
                uS_uw = np.where(un_S > 0, u, u_S);  vS_uw = np.where(un_S > 0, v, v_S)
              else:
                # Central face values (no directional upwind bias) — the
                # symmetry-preserving scheme.  Used by "central" and
                # "central_jst" (which adds 4th-order dissipation below).
                if _ho_faces:
                    # 4th-order central on the (periodic) E/W faces:
                    #   φ_face = (7(φL + φR) − (φLL + φRR)) / 12
                    # Symmetric stencil → preserves mirror symmetry.  N/S stay
                    # 2nd-order (radial; the wide stencil is unsafe across the
                    # ~100:1 wall-normal stretch).
                    uE_uw = (7.0*(u + u_E) - (u_W + u[:, ip2])) / 12.0
                    vE_uw = (7.0*(v + v_E) - (v_W + v[:, ip2])) / 12.0
                    uW_uw = (7.0*(u_W + u) - (u[:, im2] + u_E)) / 12.0
                    vW_uw = (7.0*(v_W + v) - (v[:, im2] + v_E)) / 12.0
                else:
                    uE_uw = 0.5*(u + u_E);  vE_uw = 0.5*(v + v_E)
                    uW_uw = 0.5*(u + u_W);  vW_uw = 0.5*(v + v_W)
                uN_uw = 0.5*(u + u_N);  vN_uw = 0.5*(v + v_N)
                uS_uw = 0.5*(u + u_S);  vS_uw = 0.5*(v + v_S)

              adv_u = (un_E*uE_uw*dsE + un_W*uW_uw*dsW
                     + un_N*uN_uw*dsN + un_S*uS_uw*dsS) / cell_area
              adv_v = (un_E*vE_uw*dsE + un_W*vW_uw*dsW
                     + un_N*vN_uw*dsN + un_S*vS_uw*dsS) / cell_area

              if _adv_scheme == "central_jst":
                # 4th-order (biharmonic) symmetric dissipation in the
                # circumferential direction — JST-style background damping of
                # the odd–even / high-wavenumber modes that pure central
                # advection admits.  Δ⁴ is negative-definite for every Fourier
                # mode (genuinely dissipative, unlike upwind's 2nd-order term),
                # and being an even operator with a smooth speed coefficient it
                # preserves the mirror symmetry exactly.
                _spd  = np.sqrt(u*u + v*v) + 1e-12
                _d4u  = u[:, ip2] - 4.0*u[:, ip1] + 6.0*u - 4.0*u[:, im1] + u[:, im2]
                _d4v  = v[:, ip2] - 4.0*v[:, ip1] + 6.0*v - 4.0*v[:, im1] + v[:, im2]
                adv_u = adv_u + _jst_eps4 * _spd * _d4u / (_dE_circ + 1e-12)
                adv_v = adv_v + _jst_eps4 * _spd * _d4v / (_dE_circ + 1e-12)

              # Point-implicit diffusion (same approach as Cartesian solver,
              # unconditionally stable for any nu)
              nbrs_u = u_E*dsE/dnE + u_W*dsW/dnW + u_N*dsN/dnN + u_S*dsS/dnS
              nbrs_v = v_E*dsE/dnE + v_W*dsW/dnW + v_N*dsN/dnN + v_S*dsS/dnS
              eff_diffcoeff = (nu + nu_t) / cell_area   # SST: ν_eff = ν + ν_t
              # Full sum including south face — dsS/dnS appears in numerator (as -u*dsS/dnS for
              # no-slip ghost) and denominator; both needed for unconditional stability.
              sum_ds_dn = dsE/dnE + dsW/dnW + dsN/dnN + dsS/dnS

              u_star = (u - _dt_field*adv_u + _dt_field*eff_diffcoeff*nbrs_u) / (1.0 + _dt_field*eff_diffcoeff*sum_ds_dn + 1e-30)
              v_star = (v - _dt_field*adv_v + _dt_field*eff_diffcoeff*nbrs_v) / (1.0 + _dt_field*eff_diffcoeff*sum_ds_dn + 1e-30)

            _apply_vel_bc(u_star, v_star)

            # ── Pressure Poisson ──────────────────────────────────────────────
            # Divergence of u* (FV: sum of face-normal fluxes)
            us_E = u_star[:, ip1];  vs_E = v_star[:, ip1]
            us_W = u_star[:, im1];  vs_W = v_star[:, im1]
            us_N = np.empty_like(u);  vs_N = np.empty_like(v)
            us_N[:-1,:] = u_star[1:,:]; us_N[-1,:] = _u_out
            vs_N[:-1,:] = v_star[1:,:]; vs_N[-1,:] = _v_out
            us_S = np.empty_like(u);  vs_S = np.empty_like(v)
            us_S[1:,:]  = u_star[:-1,:]; us_S[0,:] = -u_star[0,:]
            vs_S[1:,:]  = v_star[:-1,:]; vs_S[0,:] = -v_star[0,:]

            unE_s = 0.5*(u_star + us_E)*nxE + 0.5*(v_star + vs_E)*nyE
            unW_s = 0.5*(u_star + us_W)*nxW + 0.5*(v_star + vs_W)*nyW
            unN_s = 0.5*(u_star + us_N)*nxN + 0.5*(v_star + vs_N)*nyN
            unS_s = 0.5*(u_star + us_S)*nxS + 0.5*(v_star + vs_S)*nyS

            if _rhie_chow:
                # Rhie–Chow momentum interpolation on the E/W (circumferential)
                # faces, where the checkerboard lives.  The added face term
                #   (dt/ρ)[ avg(∇p·n)_cell − (p_nbr−p_own)/dn_compact ]
                # is the difference between the wide (cell-averaged) and compact
                # pressure gradients — zero for a smooth field, but non-zero (and
                # damping) for a saw-tooth pressure, which is what suppresses the
                # checkerboard.  Pressure is lagged (previous step's solve), so
                # this is an explicit correction; periodic in i, no ghosts.
                _pE = press[:, ip1];  _pW = press[:, im1]
                _pN = np.empty_like(press);  _pS = np.empty_like(press)
                _pN[:-1, :] = press[1:, :];  _pN[-1, :] = 0.0
                _pS[1:, :]  = press[:-1, :]; _pS[0, :] = press[0, :]
                _pfE = 0.5*(press+_pE);  _pfW = 0.5*(press+_pW)
                _pfN = 0.5*(press+_pN);  _pfS = 0.5*(press+_pS)
                _gpx = (_pfE*nxE*dsE + _pfW*nxW*dsW + _pfN*nxN*dsN + _pfS*nxS*dsS) / cell_area
                _gpy = (_pfE*nyE*dsE + _pfW*nyW*dsW + _pfN*nyN*dsN + _pfS*nyS*dsS) / cell_area
                _crc = _dt_field / rho
                unE_s = unE_s + _crc*(0.5*(_gpx+_gpx[:, ip1])*nxE
                                    + 0.5*(_gpy+_gpy[:, ip1])*nyE - (_pE-press)/dnE)
                unW_s = unW_s + _crc*(0.5*(_gpx+_gpx[:, im1])*nxW
                                    + 0.5*(_gpy+_gpy[:, im1])*nyW - (_pW-press)/dnW)

            # dt-weighted projection: RHS = ρ·∮u*·n (the /dt now lives in the
            # face coefficients cE_p…cS_p).  With a constant dt field this is
            # identical to the legacy (ρ/dt)·∮u* + constant coeffs.
            rhs = rho * (unE_s*dsE + unW_s*dsW + unN_s*dsN + unS_s*dsS)

            if _p_solver == "cg":
                # Converge the Poisson properly (low-wavenumber modes too).
                press = _solve_press_cg(press, rhs)
            elif _fortran_pref:
                # Compiled Jacobi sweep (Fortran).
                press = _FF.jacobi_press(press, cE_p, cW_p, cN_p, cS_p,
                                         coeff_denom_p, rhs, _sor_omega, _p_sweeps)
            elif _jit:
                # Compiled Jacobi sweep (Numba; same math as the NumPy branch).
                press = _jacobi_sweeps_jit(press, cE_p, cW_p, cN_p, cS_p,
                                           coeff_denom_p, rhs, _sor_omega, _p_sweeps)
            else:
                # Legacy fixed-sweep Jacobi smoother (under-converges low modes).
                for _ in range(_p_sweeps):
                    pE = press[:, ip1];  pW = press[:, im1]
                    pN = np.empty_like(press);  pS = np.empty_like(press)
                    pN[:-1, :] = press[1:, :];  pN[-1, :] = 0.0          # Dirichlet outer
                    pS[1:, :]  = press[:-1, :]; pS[0, :] = press[0, :]   # Neumann wall

                    p_new = (cE_p*pE + cW_p*pW + cN_p*pN + cS_p*pS - rhs) / coeff_denom_p
                    p_new = np.clip(p_new, -5.0, 5.0)
                    press = (1.0 - _sor_omega) * press + _sor_omega * p_new
                    press -= float(np.mean(press))

            # ── Velocity correction (projection) ──────────────────────────────
            pE = press[:, ip1];  pW = press[:, im1]
            pN = np.empty_like(press);  pS = np.empty_like(press)
            pN[:-1,:] = press[1:,:]; pN[-1,:] = 0.0
            pS[1:,:]  = press[:-1,:]; pS[0,:] = press[0,:]

            pf_E = 0.5*(press + pE);  pf_W = 0.5*(press + pW)
            pf_N = 0.5*(press + pN);  pf_S = 0.5*(press + pS)

            dpdx = (pf_E*nxE*dsE + pf_W*nxW*dsW + pf_N*nxN*dsN + pf_S*nxS*dsS) / cell_area
            dpdy = (pf_E*nyE*dsE + pf_W*nyW*dsW + pf_N*nyN*dsN + pf_S*nyS*dsS) / cell_area

            u = u_star - (_dt_field/rho) * dpdx
            v = v_star - (_dt_field/rho) * dpdy

            u = np.clip(u, -5.0, 5.0);  v = np.clip(v, -5.0, 5.0)
            _apply_vel_bc(u, v)

            # ── Temperature (passive scalar) ──────────────────────────────────
            T_E = T[:, ip1];  T_W = T[:, im1]
            T_N = np.empty_like(T);  T_S = np.empty_like(T)
            T_N[:-1,:] = T[1:,:]; T_N[-1,:] = T_inf
            T_S[1:,:]  = T[:-1,:]; T_S[0,:] = T[0,:]   # Neumann at wall (wall sets T below)

            unE_now = 0.5*(u + u[:,ip1])*nxE + 0.5*(v + v[:,ip1])*nyE
            unW_now = 0.5*(u + u[:,im1])*nxW + 0.5*(v + v[:,im1])*nyW
            u_N_now = np.empty_like(u); v_N_now = np.empty_like(v)
            u_N_now[:-1,:]=u[1:,:]; u_N_now[-1,:]=_u_out
            v_N_now[:-1,:]=v[1:,:]; v_N_now[-1,:]=_v_out
            unN_now = 0.5*(u + u_N_now)*nxN + 0.5*(v + v_N_now)*nyN
            u_S_now = np.empty_like(u); v_S_now = np.empty_like(v)
            u_S_now[1:,:]=u[:-1,:]; u_S_now[0,:]=-u[0,:]
            v_S_now[1:,:]=v[:-1,:]; v_S_now[0,:]=-v[0,:]
            unS_now = 0.5*(u + u_S_now)*nxS + 0.5*(v + v_S_now)*nyS

            TE_uw = np.where(unE_now > 0, T, T_E);  TW_uw = np.where(unW_now > 0, T, T_W)
            TN_uw = np.where(unN_now > 0, T, T_N);  TS_uw = np.where(unS_now > 0, T, T_S)

            adv_T = (unE_now*TE_uw*dsE + unW_now*TW_uw*dsW
                   + unN_now*TN_uw*dsN + unS_now*TS_uw*dsS) / cell_area

            nbrs_T = T_E*dsE/dnE + T_W*dsW/dnW + T_N*dsN/dnN + T_S*dsS/dnS
            # Effective thermal diffusivity: α_eff = ν/Pr + ν_t/Pr_t
            _a_eff_T = nu / Pr + nu_t / _Pr_t
            T = (T - _dt_field*adv_T + _dt_field*_a_eff_T/cell_area*nbrs_T) / (
                 1.0 + _dt_field*_a_eff_T/cell_area*(dsE/dnE + dsW/dnW + dsN/dnN + dsS/dnS) + 1e-30)
            T[-1, :] = T_inf    # outer boundary: freestream temperature
            T[0,  :] = T_wall   # wall: isothermal
            # Guard against T blow-up at degenerate zero-area trailing-edge cells.
            # The O-grid trailing edge has cell_area ≈ 1e-30 (from the +1e-30 floor
            # in ogrid.py).  With cell_area = 1e-30 and u ≈ 5, adv_T ≈ 10^30 →
            # T jumps to −115 000 K at that cell in the very first step, then
            # propagates outward via advection.  clip() is a no-op for all
            # physically meaningful cells; it only clamps the two degenerate
            # trailing-edge cells back into the physical temperature range.
            T = np.clip(T, min(T_inf, T_wall) - 100.0, max(T_inf, T_wall) + 100.0)

            # ── k-ω SST turbulence model ──────────────────────────────────────
            if _kw and _fortran_pref:
                # Compiled SST update (Fortran).
                tke, tom, nu_t = _FF.sst_update(
                    u, v, tke, tom, _d_w,
                    nxE, nyE, dsE, dnE, nxW, nyW, dsW, dnW,
                    nxN, nyN, dsN, dnN, nxS, nyS, dsS, dnS,
                    cell_area, _u_out, _v_out,
                    nu, _k_fs, _om_fs, _kappa, _beta_str, _b1, _b2,
                    _sk1, _sk2, _sw1, _sw2, _g1, _g2, _a1,
                    _nu_t_mult, _pk_limit, _S_max, _dt_field)
            elif _kw and _jit:
                # Compiled SST update (Numba; bit-faithful with the NumPy branch).
                tke, tom, nu_t = _sst_jit(
                    u, v, tke, tom, _d_w,
                    nxE, nyE, dsE, dnE, nxW, nyW, dsW, dnW,
                    nxN, nyN, dsN, dnN, nxS, nyS, dsS, dnS,
                    cell_area, _u_out, _v_out,
                    nu, _k_fs, _om_fs, _kappa, _beta_str, _b1, _b2,
                    _sk1, _sk2, _sw1, _sw2, _g1, _g2, _a1,
                    _nu_t_mult, _pk_limit, _S_max, _dt_field)
            elif _kw:
                # ── Automatic wall treatment (Menter 1994) ───────────────────
                # Blend the viscous-sublayer limit (accurate for y⁺ < 5) with
                # the log-law limit (accurate for y⁺ > 30) so a single ω BC
                # works across y⁺ ≈ 1 – 300.
                #
                #   ω_vis = 60 ν / (β₁ · d₁²)         [viscous sublayer]
                #   ω_log = √k₁ / (κ · d₁ · β*^0.25)  [log layer; k₁ from j=1]
                #   ω_w   = √(ω_vis² + ω_log²)         [smooth blend]
                #
                # Using k from j=1 (first interior cell not pinned to 0) as a
                # proxy for the equilibrium log-layer TKE → u_τ = β*^0.25 √k.
                if n_eta > 1:
                    _k1      = np.maximum(tke[1, :], 1e-10)
                    _om_log  = np.sqrt(_k1) / (_kappa * _d_w[0, :] * _beta_str**0.25)
                    _om_vis2 = (60.0 * nu / (_b1 * _d_w[0, :]**2))**2
                    _om_w    = np.minimum(np.sqrt(_om_vis2 + _om_log**2), 1.0e5)
                    # k wall BC: zero in viscous sublayer; Neumann (k[j=1]) in
                    # log layer.  Switch at y⁺ ≈ 11 estimated from ω_log/ω_vis.
                    _in_log  = _om_log > _om_vis2**0.5   # True where log dominates
                    tke[0, :] = np.where(_in_log, tke[1, :], 0.0)

                # Velocity gradients via Green-Gauss theorem.
                # Face-averaged velocities reuse u_N_now / u_S_now from the
                # temperature section (corrected post-projection u, v).
                _uEf = 0.5*(u + u[:, ip1]);  _vEf = 0.5*(v + v[:, ip1])
                _uWf = 0.5*(u + u[:, im1]);  _vWf = 0.5*(v + v[:, im1])
                _uNf = 0.5*(u + u_N_now);    _vNf = 0.5*(v + v_N_now)
                _uSf = 0.5*(u + u_S_now);    _vSf = 0.5*(v + v_S_now)
                # _uSf at j=0 → 0 (wall no-slip ghost): correct for ∂u/∂n

                # Raised cell-area floor for turbulence gradient computation.
                # The velocity/pressure solver uses the true cell_area; here
                # we floor at 1e-6 chord² so that degenerate trailing-edge
                # cells (cell_area → 0 at the TE singularity) don't produce
                # runaway velocity gradients that overwhelm the production
                # limiters.  For all well-resolved cells (cell_area >> 1e-6)
                # this is a no-op.
                _ca_sst = np.maximum(cell_area, 1e-6)

                _dudx = (_uEf*nxE*dsE + _uWf*nxW*dsW +
                         _uNf*nxN*dsN + _uSf*nxS*dsS) / _ca_sst
                _dudy = (_uEf*nyE*dsE + _uWf*nyW*dsW +
                         _uNf*nyN*dsN + _uSf*nyS*dsS) / _ca_sst
                _dvdx = (_vEf*nxE*dsE + _vWf*nxW*dsW +
                         _vNf*nxN*dsN + _vSf*nxS*dsS) / _ca_sst
                _dvdy = (_vEf*nyE*dsE + _vWf*nyW*dsW +
                         _vNf*nyN*dsN + _vSf*nyS*dsS) / _ca_sst

                # Strain-rate magnitude |S|² = 2 S_ij S_ij
                _S2 = np.maximum(2.0*_dudx**2 + 2.0*_dvdy**2 +
                                 (_dudy + _dvdx)**2, 0.0)
                _S  = np.sqrt(_S2)

                # Physical S ceiling: strain rate can't meaningfully exceed
                # U_∞ / min_face (≈ 1667 s⁻¹ for Lv4 with min_face=6×10⁻⁴).
                # Clips only degenerate near-singularity cells; no-op for the
                # bulk of the domain.
                _S_max = sim_vel / max(_min_face, 1e-10)
                _S2 = np.minimum(_S2, _S_max**2)
                _S  = np.minimum(_S,  _S_max)

                # Safe k and ω for division
                _k_s  = np.maximum(tke, 0.0)
                _om_s = np.maximum(tom, 1e-10)

                # k and ω neighbours (with wall/outer BCs on ghosts)
                _kE = tke[:, ip1]; _kW = tke[:, im1]
                _kN = np.empty_like(tke); _kS = np.empty_like(tke)
                _kN[:-1, :] = tke[1:, :]; _kN[-1, :] = _k_fs
                _kS[1:, :]  = tke[:-1, :]; _kS[0, :] = 0.0   # k_wall = 0

                _omE = tom[:, ip1]; _omW = tom[:, im1]
                _omN = np.empty_like(tom); _omS = np.empty_like(tom)
                _omN[:-1, :] = tom[1:, :]; _omN[-1, :] = _om_fs
                _omS[1:, :]  = tom[:-1, :]; _omS[0, :] = _om_w

                # ∇k and ∇ω (Green-Gauss) for cross-diffusion and F1
                _dkdx  = (0.5*(tke+_kE)*nxE*dsE + 0.5*(tke+_kW)*nxW*dsW +
                          0.5*(tke+_kN)*nxN*dsN + 0.5*(tke+_kS)*nxS*dsS) / _ca_sst
                _dkdy  = (0.5*(tke+_kE)*nyE*dsE + 0.5*(tke+_kW)*nyW*dsW +
                          0.5*(tke+_kN)*nyN*dsN + 0.5*(tke+_kS)*nyS*dsS) / _ca_sst
                _domdx = (0.5*(tom+_omE)*nxE*dsE + 0.5*(tom+_omW)*nxW*dsW +
                          0.5*(tom+_omN)*nxN*dsN + 0.5*(tom+_omS)*nxS*dsS) / _ca_sst
                _domdy = (0.5*(tom+_omE)*nyE*dsE + 0.5*(tom+_omW)*nyW*dsW +
                          0.5*(tom+_omN)*nyN*dsN + 0.5*(tom+_omS)*nyS*dsS) / _ca_sst

                _gk_gom = _dkdx*_domdx + _dkdy*_domdy   # ∇k · ∇ω
                # nan_to_num catches inf+(-inf)=NaN and explicit ±inf.
                # np.clip then catches large-but-finite values (e.g. 2e300)
                # that would overflow to inf in the subsequent _sw2/_om_s
                # multiplication.  1e20 is orders of magnitude above any
                # physical ∇k·∇ω yet safely below the overflow threshold.
                _gk_gom = np.clip(
                    np.nan_to_num(_gk_gom, nan=0.0,
                                  posinf=1e20, neginf=-1e20),
                    -1e20, 1e20)
                _CD_kw  = np.maximum(2.0*_sw2 / _om_s * _gk_gom, 1e-10)

                # Blending function F1 (k-ω near wall → 1, k-ε far field → 0)
                _r1a = np.sqrt(_k_s) / (_beta_str * _om_s * _d_w + 1e-30)
                _r1b = 500.0 * nu / (_om_s * _d_w**2 + 1e-30)
                _r1c = 4.0 * _sw2 * _k_s / (_CD_kw * _d_w**2 + 1e-30)
                # Clip the argument to 50 before **4: tanh saturates to 1 well
                # below that (tanh(2^4)≈1), so physics is unchanged but we
                # avoid float64 overflow that would propagate NaN downstream.
                _arg1 = np.minimum(np.maximum(_r1a, _r1b), _r1c)
                _F1   = np.tanh(np.minimum(_arg1, 50.0)**4)

                # Blending function F2 (eddy-viscosity limiter near wall)
                _r2  = np.maximum(2.0*np.sqrt(_k_s) / (_beta_str*_om_s*_d_w + 1e-30),
                                  500.0 * nu / (_om_s * _d_w**2 + 1e-30))
                _F2  = np.tanh(np.minimum(_r2, 50.0)**2)

                # Blended constants: φ = F1·φ₁ + (1-F1)·φ₂
                _sk  = _F1*_sk1 + (1.0 - _F1)*_sk2
                _sw  = _F1*_sw1 + (1.0 - _F1)*_sw2
                _bbl = _F1*_b1  + (1.0 - _F1)*_b2
                _gbl = _F1*_g1  + (1.0 - _F1)*_g2

                # Eddy viscosity = a1·k / max(a1·ω, S·F2) (Bradshaw realizability
                # limiter).  Cap raised 100ν → 1e4ν: a developed turbulent BL at
                # Re~1e6 physically reaches ν_t/ν ~ 1e3, so 100ν throttled the
                # turbulent mixing and let the BL behave laminar (separation/
                # shedding).  The earlier Cd blow-up from raising this came from
                # the τ_w∝√k wall-shear closure, now replaced by the velocity-
                # based wall function — so the higher cap no longer corrupts Cd.
                nu_t = np.minimum(
                    _a1 * _k_s / np.maximum(_a1 * _om_s, _S * _F2),
                    1.0e4 * nu)
                nu_t *= _nu_t_mult        # diagnostic global scale (1.0 = SST; 0.0 = laminar)
                nu_t[0, :] = 0.0   # no eddy viscosity at the wall

                # TKE production limiter: cap at 1×β*·k·ω (NOT the common
                # 20× Menter value).  With semi-implicit denominator having
                # only 1×β*·ω, the 20× cap allows k to grow 20× per step
                # when at the limiter — identical blow-up to Bug 1 (ω).
                # At the 1× cap: k_new = k·(1 + dt·β*·ω)/(1 + dt·β*·ω) = k
                # → exactly neutral, unconditionally stable.
                _Pk = np.minimum(nu_t * _S2,
                                 _pk_limit * _beta_str * _k_s * _om_s)

                # ── k transport ───────────────────────────────────────────────
                # Upwind advection (reuse corrected face-normal velocities)
                _kE_uw = np.where(unE_now > 0, tke, _kE)
                _kW_uw = np.where(unW_now > 0, tke, _kW)
                _kN_uw = np.where(unN_now > 0, tke, _kN)
                _kS_uw = np.where(unS_now > 0, tke, _kS)
                _adv_k = (unE_now*_kE_uw*dsE + unW_now*_kW_uw*dsW +
                          unN_now*_kN_uw*dsN + unS_now*_kS_uw*dsS) / cell_area

                _Dk = (nu + _sk * nu_t) / cell_area   # effective k diffusivity
                _nk = _kE*dsE/dnE + _kW*dsW/dnW + _kN*dsN/dnN + _kS*dsS/dnS
                _dk = dsE/dnE + dsW/dnW + dsN/dnN + dsS/dnS

                # Semi-implicit: β*ω in denominator → destruction is implicit
                # (prevents k from going negative even with large dt*β*ω)
                tke = np.maximum(
                    (tke - _dt_field*_adv_k + _dt_field*_Dk*_nk + _dt_field*_Pk) /
                    (1.0 + _dt_field*_Dk*_dk + _dt_field*_beta_str*_om_s + 1e-30),
                    0.0)

                # ── ω transport ───────────────────────────────────────────────
                # Cross-diffusion: only in zone 2 (F1→0) and where ∇k·∇ω > 0.
                # xdiff = 2·(1-F1)·σω2·(∇k·∇ω)/ω  ≡  A/ω   where A ≥ 0.
                # The 1/ω singularity makes explicit treatment explosive when ω
                # is small during startup: a large xdiff lands in the numerator
                # with nothing in the denominator to resist it.
                # Fix — semi-implicit treatment of the A/ω source:
                #   numerator   ←  +dt·xdiff        (= dt·A/ω)
                #   denominator ←  +dt·xdiff/ω_s    (= dt·A/ω²)
                # Proof of neutrality: when xdiff dominates (X = dt·A/ω² >> 1),
                #   ω_new = (ω + dt·A/ω) / (1 + dt·A/ω²)
                #         = ω·(1 + X) / (1 + X)  =  ω   ← exactly neutral.
                # For small xdiff the denominator term is negligible and we
                # recover the standard explicit treatment.
                _xdiff = np.maximum(
                    2.0 * (1.0 - _F1) * _sw2 / _om_s * _gk_gom, 0.0)

                _omE_uw = np.where(unE_now > 0, tom, _omE)
                _omW_uw = np.where(unW_now > 0, tom, _omW)
                _omN_uw = np.where(unN_now > 0, tom, _omN)
                _omS_uw = np.where(unS_now > 0, tom, _omS)
                _adv_om = (unE_now*_omE_uw*dsE + unW_now*_omW_uw*dsW +
                           unN_now*_omN_uw*dsN + unS_now*_omS_uw*dsS) / cell_area

                _Dom = (nu + _sw * nu_t) / cell_area   # effective ω diffusivity
                _nom = _omE*dsE/dnE + _omW*dsW/dnW + _omN*dsN/dnN + _omS*dsS/dnS
                _dom = dsE/dnE + dsW/dnW + dsN/dnN + dsS/dnS

                # ω production: γ·|S|² with Menter production limiter.
                # Without the limiter, P_ω is unbounded when ω_s is small
                # (startup transient) and |S| is large (near wall), causing
                # multi-million-fold overshoot in the first few iterations.
                # Cap at 1·β·ω² (NOT 20×): when the limiter fires, P_ω = β·ω²
                # exactly equals the semi-implicit destruction term D_ω = β·ω,
                # so the net production in the update step is zero and ω grows
                # only via diffusion from the wall BC.  This gives unconditional
                # stability: ω stays bounded in [ω_fs, ω_w] everywhere.
                # (With 20×, the growth factor per step → 20 when λ = dt·β·ω
                # is large, which is exactly what caused the NaN crashes.)
                _Pom = np.minimum(_gbl * _S2, _pk_limit * _bbl * _om_s**2)

                # Semi-implicit: β·ω in denominator (destruction), plus
                # dt·xdiff/ω_s for the cross-diffusion neutralisation term.
                tom = np.maximum(
                    (tom - _dt_field*_adv_om + _dt_field*_Dom*_nom + _dt_field*_Pom + _dt_field*_xdiff) /
                    (1.0 + _dt_field*_Dom*_dom + _dt_field*_bbl*_om_s
                     + _dt_field*_xdiff/_om_s + 1e-30),
                    1e-10)

                # Guard against numerical runaway at small-area cells.
                # Physical ceiling: k ≤ sim_vel² (≈ 1.0) always; ω ≤ wall
                # BC cap (1e6).  clip() is a no-op on all well-behaved cells.
                tke = np.clip(tke, 0.0, 1.0)
                tom = np.clip(tom, 1e-10, 1e6)

                # Enforce boundary conditions (overwrite transport result)
                tke[-1, :] = _k_fs;  tom[-1, :] = _om_fs   # freestream
                tke[0,  :] = 0.0;    tom[0,  :] = _om_w    # wall

            # ── Symmetry probe (diagnostic, off by default) ───────────────────
            if _mirror_every and i % _mirror_every == 0:
                _mi = n_xi - 1 - np.arange(n_xi)
                print(f"    [mirror] it={i:>6} "
                      f"du={np.max(np.abs(u - u[:, _mi])):.3e} "
                      f"dv={np.max(np.abs(v + v[:, _mi])):.3e} "
                      f"dp={np.max(np.abs(press - press[:, _mi])):.3e}",
                      flush=True)

            # ── Residual ──────────────────────────────────────────────────────
            res = float(np.mean(np.abs(u - u_old)))
            res_history.append(res)

            if (np.isnan(u).any() or np.isnan(press).any()
                    or (_kw and (np.isnan(tke).any() or np.isnan(tom).any()))):
                print(f"FATAL: NaN at BFM iteration {i}.")
                conv_state = "NaN error"; break

            # Steady convergence
            conv_threshold = float(p.get("conv_res", 1e-8))
            if not _no_autostop and i > 200 and res < conv_threshold:
                conv_state = "Converged (steady)"
                print(f"BFM converged at {i} (res={res:.2e})."); break

            # Periodic / plateau convergence
            # _conv_win is 5 physical chord-crossing times (dt-independent).
            # Absolute gate: a plateau at a large residual is a chaotic
            # transient, not a converged periodic state.  Require the
            # residual itself to be small before the relative checks apply
            # (mean |Δu| per step ≤ 1e-3 of the unit freestream).
            _cw_half = max(100, _conv_win // 10)   # half-window for oscillation check
            if (not _no_autostop and i > warmup + max(_conv_win, _dev_floor)
                    and len(res_history) >= _conv_win and res < 1e-3):
                recent = res_history[-_cw_half*2:]
                # "oscillating": residual increases in ≥20% of steps (genuine
                # oscillation, not just a single transient spike) OR second half
                # of the window trends above the first (residual not decaying).
                _n_up       = sum(1 for j in range(1, len(recent)) if recent[j] > recent[j-1]*1.005)
                sharp_osc   = _n_up >= max(3, int(0.20 * len(recent)))
                trending_up = float(np.mean(recent[_cw_half:])) > float(np.mean(recent[:_cw_half]))*1.005
                oscillating = sharp_osc or trending_up
                plateau     = min(recent) >= min(res_history[-_cw_half*4:-_cw_half*2])*0.95
                if oscillating and plateau:
                    conv_state = "Converged (periodic)"
                    print(f"BFM plateau at {i} (res={res:.2e}). Periodic."); break

            # ── Live GUI updates every 50 iters ───────────────────────────────
            elapsed = time.perf_counter() - start_time
            gui.root.after(0, gui.update_live_metrics, i, res, elapsed)

            if i % 50 == 0 and i > 0:
                cl, cd, ld_ratio, cm = _compute_forces()
                if i >= warmup:
                    cl_samples.append(cl);  cd_samples.append(cd);  cm_samples.append(cm)

                # Force-based developed-convergence: once past the _min_cross
                # flow-time floor, stop when the windowed-mean Cl has settled —
                # two consecutive ~2-crossing windows agreeing to within
                # _cl_conv_tol.  The averaging windows smooth vortex shedding, so
                # this catches both steady wakes and periodic ones, and (unlike
                # the residual gate) only fires after the lift has actually
                # developed.  This is the primary stopping criterion.  (_fwin is
                # precomputed above, and shortened under local_dt.)
                if (not _no_autostop and i >= warmup + _dev_floor
                        and len(cl_samples) >= 2 * _fwin):
                    _cl_now  = float(np.mean(cl_samples[-_fwin:]))
                    _cl_prev = float(np.mean(cl_samples[-2*_fwin:-_fwin]))
                    if abs(_cl_now - _cl_prev) <= _cl_conv_tol * (abs(_cl_now) + 1e-6):
                        conv_state = "Converged (developed)"
                        print(f"BFM developed-converged at {i} "
                              f"(~{i/max(_flow_through,1):.1f} crossings): "
                              f"Cl {_cl_prev:.4f}->{_cl_now:.4f}.")
                        break

                # Point-vortex far-field: refresh the lagged, under-relaxed bound
                # circulation Γ = −½·Cl·U·c and the induced outer-ring velocity.
                # Updated only here (every 50 steps) so the BC lags the force,
                # which together with ff_relax damps the Cl→BC→Cl feedback.
                if _farfield_vortex:
                    _g_target     = -0.5 * cl * sim_vel * chord
                    _gamma_smooth += _ff_relax * (_g_target - _gamma_smooth)
                    _vc           = _gamma_smooth / (2.0 * math.pi)
                    _u_out[:]     = u_inf - _vc * _dy_o / _r2_o
                    _v_out[:]     = v_inf + _vc * _dx_o / _r2_o

                cl_std  = float(np.std(cl_samples[-_AVG_WIN:])) if len(cl_samples) >= 2 else 0.0
                lift_n  = cl * _q_real * _chord_m
                drag_n  = cd * _q_real * _chord_m

                # Nusselt: approximate wall gradient as (T_wall - T[1,:]) / dn_wall
                # where dn_wall = dnS[0,:] + dnN[0,:] is the distance from the wall
                # to the j=1 cell centre (avoids under-estimating Nu on coarse O-grid
                # cells where j=1 sits ~0.075c from the wall vs ~0.031c in Cartesian).
                _dT_ref = abs(T_wall - T_inf)
                if _dT_ref < 0.1:           # no meaningful temperature difference
                    nu_val = 0.0
                else:
                    if n_eta > 1:
                        _dn_wall = dnS[0, :] + dnN[0, :]          # wall → j=1 cell centre
                        _grad    = np.abs(T_wall - T[1, :]) / np.maximum(_dn_wall, 1e-10)
                        nu_val   = float(np.mean(_grad)) * chord / _dT_ref
                    else:
                        nu_val = 0.0

                # Strouhal from mean-subtracted Cl zero-crossings.
                # Only meaningful when Cl genuinely oscillates: with steady
                # flow the zero-crossings are numerical noise with a ~2-sample
                # period, which produced absurd readouts (St ≈ 39).  Gate on
                # the oscillation amplitude and require the implied St to be
                # in the physically plausible shedding range.
                if len(cl_samples) >= 6 and cl_std > 0.005:
                    _cl_arr = np.array(cl_samples[-40:])
                    _cl_dev = _cl_arr - _cl_arr.mean()
                    _zc     = np.where((_cl_dev[:-1]*_cl_dev[1:]) < 0)[0]
                    if len(_zc) >= 2:
                        _half_per     = float(np.mean(np.diff(_zc)))  # half-period [50-iter samples]
                        _period_iters = 2.0 * _half_per * 50
                        _period_time  = _period_iters * dt
                        _st_cand = chord / (sim_vel * _period_time + 1e-10)
                        # Two rejections: (a) ANTI-ALIASING — a real oscillation
                        # must span several samples, so its half-period is ≥ 2
                        # samples (Nyquist); a ~1-sample "period" is just
                        # high-frequency numerical wiggle, not shedding.  (b)
                        # PHYSICAL RANGE — vortex-shedding St is always < 1
                        # (cylinders ≈0.2, airfoils ≈0.1–0.3); St near 2 was the
                        # aliased wiggle leaking through the old 2.0 cap.
                        st = (_st_cand if (_half_per >= 2.0 and 0.05 <= _st_cand <= 1.0)
                              else float('nan'))

                matrices = _interpolate_to_cartesian(u, v, press, T, T_inf, grid, chord_m=_chord_m)
                # Send airfoil body mask to GUI once (first live update)
                if not _body_mask_sent and "_mask" in matrices:
                    gui.root.after(0, gui.set_body_mask, matrices["_mask"])
                    _body_mask_sent = True
                gui.root.after(0, gui.update_live_results, matrices, cl, cd, ld_ratio,
                               cm, lift_n, drag_n, cl_std, st, nu_val)

            if i % hist_interval == 0 and i > 0:
                gui.root.after(0, gui.append_history_row, i, res, cl, cd, ld_ratio)

        # ── Final averages ─────────────────────────────────────────────────────
        if len(cl_samples) > 0:
            # Average the recent settled window, and never more than half the
            # collected samples (so a short run cannot bias the mean with its
            # early transient).
            _nw  = min(_AVG_WIN, max(20, len(cl_samples) // 2))
            _uc = cl_samples[-_nw:];  _ud = cd_samples[-_nw:]
            _um = cm_samples[-_nw:] if cm_samples else [cm]
            cl      = float(np.mean(_uc));  cd = float(np.mean(_ud))
            cm      = float(np.mean(_um));  ld_ratio = cl / (abs(cd) + 1e-10)
            cl_std  = float(np.std(_uc))
            lift_n  = cl * _q_real * _chord_m
            drag_n  = cd * _q_real * _chord_m
            print(f"BFM time-averaged (last {len(_uc)} of {len(cl_samples)} samples): "
                  f"Cl={cl:.4f} Cd={cd:.4f} L/D={ld_ratio:.4f}")
        else:
            cl, cd, ld_ratio, cm = _compute_forces()
            lift_n = cl * _q_real * _chord_m
            drag_n = cd * _q_real * _chord_m

        # Optional surface-Cp dump (diagnostic): wall-cell pressure coefficient
        # vs surface coordinate, for comparing the loading distribution to a
        # reference polar's Cp.  Off unless p["dump_cp"] is a filepath.
        if p.get("dump_cp"):
            _pinf = float(np.mean(press[-1, :]))
            _cp   = (press[0, :] - _pinf) / (q_ref + 1e-30)
            np.savetxt(p["dump_cp"], np.column_stack([grid['xa'], grid['ya'], _cp]),
                       header="x y Cp_wall", fmt="%.6f")
            print(f"  [dump_cp] wrote {p['dump_cp']}  "
                  f"Cp_min={float(_cp.min()):.3f} Cp_max={float(_cp.max()):.3f}")

        matrices = _interpolate_to_cartesian(u, v, press, T, T_inf, grid, chord_m=_chord_m)
        if not _body_mask_sent and "_mask" in matrices:
            gui.root.after(0, gui.set_body_mask, matrices["_mask"])
        print(f"BFM Finished [{conv_state}]. Cl={cl:.4f} Cd={cd:.4f} L/D={ld_ratio:.4f}")
        gui.root.after(0, gui.append_history_row, i, res, cl, cd, ld_ratio)
        gui.root.after(0, gui.display_final_results, matrices, cl, cd, ld_ratio,
                       cm, lift_n, drag_n, cl_std, st, nu_val, conv_state)

    except Exception as exc:
        import traceback
        print(f"BFM SOLVER ERROR: {exc}")
        traceback.print_exc()
    finally:
        gui.solver_running = False


# ── Cartesian interpolation for GUI display ───────────────────────────────────

def _interpolate_to_cartesian(u, v, press, T, T_inf, grid, nx_cart=320, ny_cart=320, chord_m=None):
    """
    Interpolate O-grid solution onto a regular Cartesian grid for GUI display.
    Returns a dict matching the format expected by the GUI callbacks.
    """
    XC = grid['XC'].ravel()
    YC = grid['YC'].ravel()

    chord = grid['chord']
    xa    = grid['xa']
    ya    = grid['ya']
    cy    = float(ya.mean())

    # Cartesian display domain: centred on airfoil, +/-1.5 chord
    pad   = 1.5 * chord
    xlo   = float(xa.min()) - 0.3*chord;  xhi = float(xa.max()) + pad
    ylo   = cy - pad;                     yhi = cy + pad

    # Scale-bar metadata: how many of the nx_cart cells equal 1 chord length
    _total_x      = xhi - xlo                           # display width in sim units (≈2.8*chord)
    _n_cells_pc   = nx_cart * chord / _total_x          # cells per chord on the Cartesian display
    _chord_m_disp = chord_m if chord_m else chord       # physical chord (fallback to sim units)
    _meta = {"n_cells_per_chord": _n_cells_pc, "chord_m": _chord_m_disp}

    xi_cart = np.linspace(xlo, xhi, nx_cart)
    yi_cart = np.linspace(yhi, ylo, ny_cart)   # top→bottom so row 0 = image top = physical top
    XX, YY  = np.meshgrid(xi_cart, yi_cart)    # shape (ny_cart, nx_cart)

    pts_src  = np.column_stack([XC, YC])
    pts_tgt  = np.column_stack([XX.ravel(), YY.ravel()])

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts_src)
        # k=8 inverse-distance weighting: blends 8 nearest O-grid cell centres.
        # Using 8 neighbours (vs 4) better bridges adjacent circumferential strips,
        # reducing the angular discretisation artefacts ("spokes") in the display.
        dists, idx8 = tree.query(pts_tgt, k=8)
        w = 1.0 / (dists + 1e-30)
        w /= w.sum(axis=1, keepdims=True)          # normalised weights, shape (n_tgt, 8)

        def _map(arr):
            flat = arr.ravel()
            return (flat[idx8] * w).sum(axis=1).reshape(ny_cart, nx_cart)

    except ImportError:
        # Fallback: manual nearest-neighbour (no scipy)
        idx_nn = np.argmin(
            (pts_src[:, 0:1] - pts_tgt[:, 0])**2
          + (pts_src[:, 1:2] - pts_tgt[:, 1])**2, axis=0)

        def _map(arr):
            return arr.ravel()[idx_nn].reshape(ny_cart, nx_cart)

    # ── Airfoil body mask (ray-casting polygon test) ─────────────────────────
    # xa / ya are the airfoil surface coordinates (closed polygon, n_xi points).
    # For each cell centre on the Cartesian display grid we cast a ray in the
    # +x direction and count edge crossings (even = outside, odd = inside).
    _xa_poly = np.asarray(xa, dtype=float)
    _ya_poly = np.asarray(ya, dtype=float)
    _n_poly  = len(_xa_poly)
    _px = XX.ravel()
    _py = YY.ravel()
    _inside = np.zeros(len(_px), dtype=bool)
    for _k in range(_n_poly):
        _j = (_k - 1) % _n_poly
        _xi, _yi = _xa_poly[_k], _ya_poly[_k]
        _xj, _yj = _xa_poly[_j], _ya_poly[_j]
        _cross = (_yi > _py) != (_yj > _py)
        _x_int = (_xj - _xi) * (_py - _yi) / ((_yj - _yi) + 1e-30) + _xi
        _inside ^= _cross & (_px < _x_int)
    _mask_cart = _inside.reshape(ny_cart, nx_cart)

    speed   = np.sqrt(u**2 + v**2)
    delta_T = T - float(T_inf)

    # ── Interpolate all fields to Cartesian display grid ─────────────────────
    u_c     = _map(u)
    v_c     = _map(v)
    speed_c = _map(speed)
    press_c = _map(press)
    temp_c  = _map(delta_T)

    # ── Vorticity: computed from un-smoothed u_c/v_c for accuracy ────────────
    # The BFM stores v as physical (y-upward positive); meshgrid rows increase
    # upward from ylo, so ∂u/∂y uses the standard central-difference sign.
    dx_c = (xhi - xlo) / nx_cart
    dy_c = (ylo - yhi) / ny_cart   # negative: y decreases as row index increases

    dvdx_c       = (np.roll(v_c, -1, 1) - np.roll(v_c, 1, 1)) / (2.0 * dx_c)
    dudy_c       = np.empty_like(u_c)
    dudy_c[1:-1] = (u_c[2:]  - u_c[:-2]) / (2.0 * dy_c)
    dudy_c[0]    = (u_c[1]   - u_c[0])   / dy_c
    dudy_c[-1]   = (u_c[-1]  - u_c[-2])  / dy_c
    vort_c       = dvdx_c - dudy_c

    # ── Gaussian smoothing (display only) ─────────────────────────────────────
    # Two artefacts to suppress: (1) radial "spoke" patterns from mapping the
    # ~96-cell O-grid onto the square image, and (2) high-frequency CHECKERBOARD
    # speckle in pressure and vorticity — an intrinsic trait of the collocated
    # central scheme (the raw near-wall pressure carries ~12 small ripples
    # around the airfoil).  That speckle was rendering the single leading-edge
    # suction region as a lumpy "heart" plus stray blobs.  Pressure and
    # vorticity therefore get a heavier σ (validated: σ≈2.0 merges the ripples
    # into one clean feature; less leaves ≥2 lobes); velocity/speed/temperature
    # are already smooth and keep a light σ so they stay crisp.  Display only —
    # no effect on Cl/Cd or any solver state.
    try:
        from scipy.ndimage import gaussian_filter as _gf
        _s_vel   = 1.2   # velocity / speed / temperature — already smooth
        _s_speck = 2.0   # pressure / vorticity — carry checkerboard speckle
        u_c     = _gf(u_c,     _s_vel)
        v_c     = _gf(v_c,     _s_vel)
        speed_c = _gf(speed_c, _s_vel)
        temp_c  = _gf(temp_c,  _s_vel)
        press_c = _gf(press_c, _s_speck)
        vort_c  = _gf(vort_c,  _s_speck)
    except ImportError:
        pass   # scipy unavailable — display without smoothing

    return {
        "u":           u_c,
        "speed":       speed_c,
        "pressure":    press_c,
        "vorticity":   vort_c,
        "temperature": temp_c,
        "_meta":       _meta,
        "_mask":       _mask_cart,
    }


# ── Compatibility alias ───────────────────────────────────────────────────────

# Alias so GUI / test code can call run_bfm_simulation or bfm.run interchangeably
run = run_bfm_simulation
