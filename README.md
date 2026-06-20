# LunarCFD

**LunarCFD v0.1.2.0** is a 2D computational fluid dynamics simulator for airfoil
analysis, with an interactive desktop GUI. It solves the incompressible
Navier–Stokes equations on a body-fitted O-grid wrapped around the airfoil, adds
the Menter k-ω SST turbulence model above Re = 500, and reports the lift, drag,
moment, and heat-transfer characteristics of the section together with a live
flow-field visualisation.

> Research / learning tool. It is accurate where stated below and explicitly not
> in others — please read **Accuracy & Margins of Error** before relying on a number.

---

## Features

- **Body-fitted O-grid (BFM)** solver — a curvilinear mesh wraps the airfoil
  surface for clean boundary-layer and pressure resolution on symmetric and
  cambered sections.
- **Incompressible Navier–Stokes**, explicit fractional-step projection method,
  central + JST advection.
- **k-ω SST turbulence model** (Menter), activated automatically at Re ≥ 500,
  with an automatic near-wall treatment (viscous-sublayer ↔ log-law blend).
- **Local time-stepping** drives runs to a steady state several times faster
  (standard in v0.1.2.0).
- **Compiled kernels** — the hot loops run as Numba and (when built) Fortran
  kernels, ~2–4× faster than plain NumPy, with an automatic
  Fortran → Numba → NumPy fallback so it always runs.
- **Live visualisation** — pressure, velocity, speed, vorticity, or temperature
  fields with a geometry outline and a physical scale bar.
- **Outputs** — Cl, Cd, L/D, Cm (about the quarter chord), physical lift/drag
  (N/m), and Nusselt number.
- **Built-in calculators** for Reynolds number, time step, and required
  iterations.

NACA 0012 / 2412 / 4412 / 0006 and a 1×8 rectangle are built in; custom `.dat`
airfoil coordinate files are also supported.

---

## Requirements

- Python 3.11
- `numpy`, `scipy`, `numba`  (Tkinter ships with standard CPython)
- Optional: `psutil` (RAM/priority readout). A compiled Fortran kernel module is
  used if present, but is not required — the solver falls back to Numba/NumPy.

```bash
pip install numpy scipy numba psutil
```

## Running

```bash
python main.py
```

This opens the GUI. Set the airfoil, Reynolds number, angle of attack and grid,
then **Run Solver**. The run auto-stops when the lift has converged.

---

## Inputs (left panel)

| Field | Meaning | Default |
|---|---|---|
| Re | Reynolds number (target flow regime) | 500 000 |
| Velocity (m/s) | Freestream speed; scales physical forces, not the dimensionless physics | 100 |
| AoA (deg) | Angle of attack (inflow is rotated; the airfoil stays axis-aligned) | 0 |
| Air Pressure (Pa) | Sets air density for physical force output | 101325 |
| Max Iterations | Iteration cap (the solver may stop earlier on convergence) | 150000 |
| dt (time step) | Dimensionless step; auto-clamped to a CFL-stable value | 2e-4 |
| Omega | Pressure-solver relaxation (clamped to 0.05–1.0) | 0.6 |
| Convergence Residual | Steady-state residual threshold | 1e-8 |
| Wall Temp / Air Temp (K) | Thermal boundary conditions (drive Nusselt number) | 320 / 300 |
| Chord (m, blank = auto) | Physical chord for force scaling; blank derives it from Re/ν/V | 1 |
| Airfoil | NACA 0012 / 2412 / 4412 / 0006, rectangle, or custom `.dat` | NACA 0012 |
| Grid Size | BFM O-grid resolution (see below) | BFM 96×48 |

Higher-order faces, Rhie–Chow momentum interpolation, and local time-stepping are
**standard** in v0.1.2.0 (always on).

### Grid sizes

The grid is the BFM O-grid as *circumferential × radial* cells. The flow-field
always renders at 320×320 pixels regardless of solve resolution.

| Grid | Use |
|---|---|
| BFM 64×32 | Coarsest, fastest — quick exploration |
| BFM 96×48 | Recommended starting point |
| BFM 128×64 | Good standard accuracy |
| BFM 256×128 | Fine near-wall resolution |
| BFM 320×160 | Highest available; best for high-Re boundary layers |

All grids use a 15-chord far-field radius and sinh near-wall stretching.

---

## Calculators (toolbar)

Three helper windows size your inputs correctly:

- **Re Calculator** — enter real velocity (m/s), chord (m), and air temperature
  (°C). It computes the kinematic viscosity of air via Sutherland's law and then
  `Re = V · L / ν`, shows the flow regime with a real-world example, and applies
  the result to the Re field.
- **dt Calculator** — choose a grid and a CFL target (0.05 is the stable cap for
  the default scheme). It finds the smallest cell on that grid and returns
  `dt = CFL × min_face`, plus the number of steps per chord-crossing. The solver
  re-clamps any dt to its own stable limit, so the suggestion is always safe.
- **Max Iterations Calculator** — choose a grid, a target number of
  chord-crossings (≈15 gives ~95 %+ developed lift — the Wagner effect), and a dt;
  it returns `iterations ≈ crossings ÷ dt`. Use it so a run develops far enough
  for Cl to be accurate instead of stopping early and under-reading lift.

---

## Outputs

- **Cl, Cd, L/D** — lift, drag, and lift-to-drag ratio.
- **Cm** — pitching moment about the quarter chord (nose-up positive).
- **Lift / Drag (N/m)** — physical forces per unit span from the real density.
- **Nusselt** — convective heat transfer (needs Wall Temp ≠ Air Temp).

> Because local time-stepping is standard, the solver targets the **steady** state
> and is not time-accurate. Vortex shedding, **Strouhal number**, and the Cl
> oscillation amplitude are therefore not physically meaningful and the Strouhal
> readout normally shows "—".

---

## Accuracy & Margins of Error

Validated against published NACA polars (Abbott & von Doenhoff / NASA TMR), fully
developed (~15 chord-crossings) with the standard settings. **Tested up to
Re ≈ 500 000 — relatively low for aerospace — with spot checks near Re = 1 000 000.**

- **Lift (Cl): within ~±5 % of published** for attached flow (AoA ≲ 8°).
  Measured examples:

  | Case | LunarCFD Cl | Reference | Agreement |
  |---|---|---|---|
  | NACA 0012, Re 1e6, α = 0° | ≈ 0 | 0 | ✓ |
  | NACA 0012, Re 1e6, α = 4° | 0.416 | 0.43 | 97 % |
  | NACA 0012, Re 1e6, α = 8° | 0.800 | 0.84 | 95 % |
  | NACA 2412, Re 5e5, α = 0° | 0.21 | ~0.22 | 95 % |

- **Drag (Cd): over-predicted, roughly 1.3–2× the reference.** This is inherent
  to the method (fully-turbulent assumption with no transition, 2D, steady RANS).
  Use Cd comparatively or as an order-of-magnitude estimate, not for a precise
  drag budget.
- **Moment (Cm): matches published** quarter-chord values — ≈ 0 for symmetric
  sections, NACA 2412 ≈ −0.05, NACA 4412 ≈ −0.10 (measured 2412 α = 0° → −0.051).
- **Lift develops slowly** (the Wagner effect): a run stopped too early reads low.
  Use the Max Iterations Calculator, or let the auto-stop settle it.

### Limitations

- **2D, steady RANS** — no 3D structures (tip vortices, spanwise variation); no
  time-accurate unsteadiness (shedding/Strouhal are not meaningful).
- **Fully turbulent** — no laminar run / transition modelling, which inflates Cd.
- **Post-stall** (typically AoA > 12–15° for NACA 0012) is not reliable at any Re.
- Above the tested band (Re > ~1e6) results are extrapolation.

---

## Method

- Cell-centred finite-volume incompressible Navier–Stokes on a body-fitted O-grid.
- Explicit fractional-step (projection) time advance; central + JST 4th-difference
  advection; point-implicit diffusion.
- Pressure-Poisson via a Jacobi smoother (matrix-free CG available).
- Menter k-ω SST turbulence with automatic wall treatment (Re ≥ 500).
- Local time-stepping (per-cell pseudo time step) for fast steady convergence.
- See `REFERENCES.md` for the method citations (Chorin, Menter SST, JST,
  Rhie–Chow, …).

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE). © 2026 LunarCFD
Development Team.
