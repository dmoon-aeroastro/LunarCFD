# LunarCFD v0.1.1.0

A 2D computational fluid dynamics simulator for airfoil analysis, built in Python with a Tkinter GUI.

## What's New in v0.1.1.0

- **2–3× faster solver** — pressure Poisson SOR is now JIT-compiled via Numba
- **Multi-core support** — CPU Cores selector (1–6) in the left panel
- **Convergence reason** — status button now shows why the simulation ended (Converged: Periodic, Max Iterations Reached, etc.)
- **Per-iteration metrics** — iteration count, elapsed time, and residual update every iteration
- **Matrix toggle** — Show/Update velocity matrix checkbox to pause the flowfield display and reduce CPU load

## Overview

LunarCFD solves the 2D incompressible Navier-Stokes equations using an explicit projection method on a staggered Cartesian grid. The pressure Poisson equation is solved with Successive Over-Relaxation (SOR). It simulates flow around a NACA 0012 airfoil and computes lift coefficient (Cl), drag coefficient (Cd), and lift-to-drag ratio (L/D).

Key features:
- Real-time velocity field visualization
- Adjustable Reynolds number, angle of attack, grid size, and solver parameters
- Color-coded simulation status indicator with convergence reason
- Session save/load support
- Integrated help documentation
- Integration test suite

## Capabilities

- Simulates 2D incompressible flow around a NACA 0012 airfoil
- Supports Reynolds numbers from 1 to 10,000
- Angle of attack adjustable across a wide range (positive and negative)
- Three grid resolutions: 80x80, 160x160, and 320x320
- Real-time velocity field visualization during simulation
- Automatic convergence detection for periodic (vortex shedding) flow regimes
- Computes lift coefficient (Cl), drag coefficient (Cd), and lift-to-drag ratio (L/D)
- Session save and load for resuming or comparing runs
- Built-in integration test suite to verify solver correctness

## Limitations

- **2D only** — no 3D effects, spanwise flow, tip vortices, or finite wing behavior
- **NACA 0012 only** — no support for other airfoil profiles in this release
- **Laminar flow only** — no turbulence modeling; results at high Reynolds numbers may not match experimental data
- **Incompressible flow only** — not valid for high-speed or transonic/supersonic regimes
- **Cartesian grid** — the airfoil is approximated on a fixed grid; no body-fitted mesh
- **Slow at high resolution** — 160x160 runs take a few minutes; 320x320 can take significantly longer

## Installation

**Windows PowerShell:**

> Please ensure you have installed [Python 3.11](https://www.python.org/downloads/release/python-3110/) and [Git](https://git-scm.com/download/win) before running these commands.

```powershell
git clone https://github.com/dmoon-aeroastro/LunarCFD.git

cd LunarCFD

py -3.11 -m venv env

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

env\Scripts\activate

pip install -r requirements.txt

py -3.11 main.py
```

## Updating

If you have already installed LunarCFD and want to get the latest version:

```powershell
cd LunarCFD

env\Scripts\activate

git pull

pip install -r requirements.txt

py -3.11 main.py
```

## Running Tests

```powershell
py -3.11 run_tests.py
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| Re | Reynolds number (1–10000) |
| Velocity | Inflow velocity magnitude |
| AoA | Angle of attack in degrees |
| Air Pressure | Ambient pressure for force calculation |
| Max Iterations | Hard stop for the time-stepping loop |
| dt | Time step size |
| Omega | SOR relaxation factor (0 < omega < 2) |
| Convergence Residual | Residual threshold for periodic-regime detection |
| THETA | Angle-of-attack rotation parameter (mirrors AoA) |
| Grid Size | Simulation grid resolution (80x80 / 160x160 / 320x320) |
| CPU Cores | Number of cores used by the parallel pressure solver (1–6) |

## Outputs

| Output | Description |
|--------|-------------|
| Cl | Lift coefficient |
| Cd | Drag coefficient |
| L/D | Lift-to-drag ratio |

## Future Updates

- Support for additional airfoil profiles (NACA 4-digit series, custom geometry import)
- Pressure field and streamline visualization
- Results export to CSV
- Higher Reynolds number stability improvements
- Turbulence modeling
- Body-fitted mesh generation for improved airfoil resolution

## Project Structure

```
LunarCFD/
├── main.py           # Application entry point
├── run_tests.py      # Integration test suite
├── gui/
│   └── main_window.py
├── solver/
│   ├── core.py       # Navier-Stokes time-stepping and SOR pressure solver
│   ├── geometry.py   # NACA 0012 airfoil mask generation
│   └── guards.py     # Input validation
├── file_io/
│   └── session.py    # Session save/load
└── mesh/
    └── generator.py
```

## License

MIT License
