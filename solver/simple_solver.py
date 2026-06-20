"""
#1B — Implicit SIMPLE steady solver for the BFM O-grid.

Drops the explicit time march entirely.  Each OUTER iteration:
  1. assemble the implicit momentum coefficients from the lagged face mass
     fluxes (upwind convection + diffusion), with a deferred-correction source
     carrying the higher-order (central/JST) advection and the pressure gradient;
  2. solve the u- and v-momentum systems (under-relaxed Jacobi sweeps);
  3. build face mass fluxes with Rhie-Chow interpolation and solve the
     pressure-correction Poisson;
  4. correct pressure and velocity, under-relaxed.
Repeat until the scaled momentum + continuity residuals fall below tol.  No CFL
limit (advection is implicit), so it converges in thousands of outer iterations
instead of the explicit solver's ~500k time steps — but it is STEADY-ONLY.

Hot kernels (the 5-point linear sweep and the momentum-coefficient assembly)
are provided by a compiled Fortran module (solver/bfm_simple_f) when it has been
built with f2py; otherwise an identical Numba implementation is used.  Both paths
produce the same arithmetic, so the choice only affects speed.
"""
import numpy as np

# ── Kernel backend: prefer compiled Fortran, fall back to Numba ───────────────
_BACKEND = "none"
try:
    from . import bfm_simple_f as _ff          # f2py-compiled (gfortran)
    _BACKEND = "fortran"
except Exception:
    try:
        import bfm_simple_f as _ff             # flat import fallback
        _BACKEND = "fortran"
    except Exception:
        _ff = None

try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except Exception:                              # pragma: no cover
    _HAVE_NUMBA = False
    def njit(*a, **k):
        def deco(f): return f
        return deco
    prange = range


@njit(parallel=True, cache=True)
def _jacobi5_nb(phi, ap, ae, aw, an, as_, rhs, ghostS, omega, nsweep):
    """Under-relaxed Jacobi sweeps of  ap*phi = ae*E+aw*W+an*N+as*S + rhs  on
    rows j=0..n_eta-2 (row n_eta-1 is the fixed Dirichlet outer ring).  E/W
    periodic; wall S-ghost = ghostS (set as_[0,:]=0 for a Neumann wall).
    Numba mirror of the Fortran jacobi5 (row-major here vs column-major there,
    same arithmetic)."""
    n_eta, n_xi = phi.shape
    new = np.empty_like(phi)
    for _ in range(nsweep):
        for j in prange(n_eta - 1):
            for i in range(n_xi):
                ip = i + 1 if i + 1 < n_xi else 0
                im = i - 1 if i >= 1 else n_xi - 1
                pE = phi[j, ip]; pW = phi[j, im]
                pN = phi[j + 1, i]
                pS = phi[j - 1, i] if j >= 1 else ghostS[i]
                new[j, i] = (ae[j, i]*pE + aw[j, i]*pW + an[j, i]*pN
                             + as_[j, i]*pS + rhs[j, i]) / ap[j, i]
        for j in prange(n_eta - 1):
            for i in range(n_xi):
                phi[j, i] = (1.0 - omega)*phi[j, i] + omega*new[j, i]
    return phi


def jacobi5(phi, ap, ae, aw, an, as_, rhs, ghostS, omega, nsweep):
    """Backend-dispatching 5-point sweep (in-place on phi)."""
    if _BACKEND == "fortran":
        # Fortran expects column-major; round-trip through asfortranarray.
        pf = np.asfortranarray(phi)
        _ff.jacobi5(pf, np.asfortranarray(ap), np.asfortranarray(ae),
                    np.asfortranarray(aw), np.asfortranarray(an),
                    np.asfortranarray(as_), np.asfortranarray(rhs),
                    np.asfortranarray(ghostS), omega, nsweep)
        phi[:, :] = pf
        return phi
    return _jacobi5_nb(phi, ap, ae, aw, an, as_, rhs, ghostS, omega, nsweep)


# ── Geometry / operator helpers (vectorised NumPy; cheap relative to sweeps) ──
def _nbr_ghost(a, outer, wall_mode, wall_val=None):
    """Return E,W,N,S neighbour arrays with periodic E/W and the given outer
    (N) Dirichlet ghost and wall (S) treatment."""
    ip1 = np.r_[np.arange(1, a.shape[1]), 0]
    im1 = np.r_[a.shape[1]-1, np.arange(0, a.shape[1]-1)]
    aE = a[:, ip1]; aW = a[:, im1]
    aN = np.empty_like(a); aS = np.empty_like(a)
    aN[:-1, :] = a[1:, :]; aN[-1, :] = outer
    aS[1:, :] = a[:-1, :]
    if wall_mode == "neumann":
        aS[0, :] = a[0, :]
    elif wall_mode == "antisym":
        aS[0, :] = -a[0, :]
    else:  # dirichlet value
        aS[0, :] = wall_val if wall_val is not None else 0.0
    return aE, aW, aN, aS


def make_metrics(grid):
    """Bundle the time-invariant O-grid metrics the SIMPLE step needs."""
    g = grid
    m = dict(
        nxE=g['nxE'], nyE=g['nyE'], dsE=g['dsE'], dnE=g['dnE'],
        nxW=g['nxW'], nyW=g['nyW'], dsW=g['dsW'], dnW=g['dnW'],
        nxN=g['nxN'], nyN=g['nyN'], dsN=g['dsN'], dnN=g['dnN'],
        nxS=g['nxS'], nyS=g['nyS'], dsS=g['dsS'], dnS=g['dnS'],
        cell_area=g['cell_area'],
    )
    n_xi = g['XC'].shape[1]
    m['ip1'] = (np.arange(n_xi) + 1) % n_xi
    m['im1'] = (np.arange(n_xi) - 1) % n_xi
    # ds/dn conductance factors (geometry part of the diffusion + p' coeffs)
    m['gE'] = g['dsE']/g['dnE']; m['gW'] = g['dsW']/g['dnW']
    m['gN'] = g['dsN']/g['dnN']; m['gS'] = g['dsS']/g['dnS']
    return m


# NOTE: simple_step (one full SIMPLE outer iteration) is implemented in
# core_bfm.run_bfm_simulation's solver_mode=="simple" branch, where it can reuse
# the already-built BCs, eddy viscosity, _compute_forces and GUI callbacks.
# This module owns the reusable, backend-dispatched kernels above so they can be
# unit-tested in isolation and swapped Fortran<->Numba without touching physics.

def backend():
    return _BACKEND
