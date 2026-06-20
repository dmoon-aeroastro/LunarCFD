# LunarCFD — Methods & References

LunarCFD is an **original, clean-room implementation** of standard, published
computational-fluid-dynamics methods.  No source code was copied from any other
solver or publication; every routine was written from the mathematical
descriptions in the references below.  This file documents the provenance of
each method so the methods can be properly cited and so it is clear which parts
are established science (the algorithms) versus this project's own code.

Using these published methods *with* citation is standard scientific practice.
The references are listed per solver component.

## Governing equations & pressure–velocity coupling
- **Incompressible Navier–Stokes, fractional-step / projection method**
  Chorin, A. J. (1968). *Numerical solution of the Navier–Stokes equations.*
  Mathematics of Computation 22(104), 745–762.
- **Pressure–velocity coupling on collocated grids (Rhie–Chow momentum
  interpolation)** — used to suppress checkerboard pressure modes.
  Rhie, C. M. & Chow, W. L. (1983). *Numerical study of the turbulent flow past
  an airfoil with trailing edge separation.* AIAA Journal 21(11), 1525–1532.
- **Finite-volume method, SIMPLE family, pressure correction**
  Patankar, S. V. (1980). *Numerical Heat Transfer and Fluid Flow.* Hemisphere.
  Ferziger, J. H. & Perić, M. (2002). *Computational Methods for Fluid
  Dynamics*, 3rd ed. Springer.

## Spatial discretisation
- **Artificial dissipation (scalar 2nd/4th-order, "JST")**
  Jameson, A., Schmidt, W. & Turkel, E. (1981). *Numerical solutions of the
  Euler equations by finite volume methods using Runge–Kutta time-stepping
  schemes.* AIAA Paper 81-1259.
- **MUSCL reconstruction / minmod & flux limiters** (legacy upwind option)
  van Leer, B. (1979). *Towards the ultimate conservative difference scheme. V.*
  Journal of Computational Physics 32(1), 101–136.

## Turbulence model
- **Menter k-ω SST** (constants, F1/F2 blending, Bradshaw eddy-viscosity limiter)
  Menter, F. R. (1994). *Two-equation eddy-viscosity turbulence models for
  engineering applications.* AIAA Journal 32(8), 1598–1605.
  Menter, F. R., Kuntz, M. & Langtry, R. (2003). *Ten years of industrial
  experience with the SST turbulence model.* Turbulence, Heat and Mass
  Transfer 4, 625–632.
- **Bradshaw equilibrium assumption** (shear-stress limiter)
  Bradshaw, P., Ferriss, D. H. & Atwell, N. P. (1967). *Calculation of boundary
  layer development using the turbulent energy equation.* JFM 28(3), 593–616.

## Geometry & grid
- **NACA 4-digit airfoil definition**
  Jacobs, E. N., Ward, K. E. & Pinkerton, R. M. (1933). *The characteristics of
  78 related airfoil sections from tests in the variable-density wind tunnel.*
  NACA Report 460.
  Abbott, I. H. & von Doenhoff, A. E. (1959). *Theory of Wing Sections.* Dover.
- **Body-fitted O-grid via algebraic transfinite interpolation (TFI) with
  sinh near-wall stretching**
  Thompson, J. F., Warsi, Z. U. A. & Mastin, C. W. (1985). *Numerical Grid
  Generation: Foundations and Applications.* North-Holland.

## Far-field boundary treatment (planned upgrade)
- **Point-vortex (and source) far-field correction for finite domains**
  Lagally, M. (1922); Filon, L. N. G. (1926); and for the modern airfoil
  application see e.g. *Far-field boundary conditions for airfoil simulation in
  steady, incompressible, two-dimensional flow*, arXiv:2411.13077 (2024).

## Auxiliary
- **Sutherland's viscosity law** (Reynolds-number calculator)
  Sutherland, W. (1893). *The viscosity of gases and molecular force.*
  Philosophical Magazine 36, 507–531.

## Notes on originality and licensing
- All solver code is written from the equations in the above works; no code was
  copied from GPL or other licensed projects.
- Third-party libraries used (NumPy, SciPy) are BSD-licensed (permissive).
- The project license is declared in the source headers; choose it deliberately
  before public release.
- This is an implementation/engineering project, not a claim of novel numerical
  research — the methods are attributed to their originators above.
