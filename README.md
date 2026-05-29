# LunarCFD v0.1.0.0

A 2D computational fluid dynamics simulator for airfoil analysis, built in Python with a Tkinter GUI.

## Overview

LunarCFD solves the 2D incompressible Navier-Stokes equations using an explicit projection method on a staggered Cartesian grid. The pressure Poisson equation is solved with Successive Over-Relaxation (SOR). It simulates flow around a NACA 0012 airfoil and computes lift coefficient (Cl), drag coefficient (Cd), and lift-to-drag ratio (L/D).

Key features:
- Real-time velocity field visualization
- Adjustable Reynolds number, angle of attack, grid size, and solver parameters
- Color-coded simulation status indicator
- Session save/load support
- Integrated help documentation
- Integration test suite

## Requirements

- Python 3.9+
- numpy
- psutil
- scipy

Install dependencies:

```
pip install -r requirements.txt
```

## Running

```
python main.py
```

## Running Tests

```
python run_tests.py
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

## Outputs

| Output | Description |
|--------|-------------|
| Cl | Lift coefficient |
| Cd | Drag coefficient |
| L/D | Lift-to-drag ratio |

## Project Structure

```
LunarCFD_v0.1.0.0/
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
