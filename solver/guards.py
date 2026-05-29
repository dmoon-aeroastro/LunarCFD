"""Safety and Solution Quality Monitoring Subsystems.

Validates current flow conditions against stability laws and mathematical 
divergence ceilings during active matrix processing loops.

Dependencies:
    - numpy, math
"""

# GNU General Public License v3.0 Header
# Copyright (C) 2026 LunarCFD Development Team

import numpy as np
import math

class ConvergenceGuard:
    """Evaluates stability criteria and limits matrix overflow vectors."""
    
    def __init__(self, residual_ceiling=1e6):
        self.residual_ceiling = residual_ceiling

    def verify_cfl_limits(self, u_matrix, dt, dx):
        """Validates current fields against Courant-Friedrichs-Lewy limits.

        ===========================================================================
        MATHEMATICAL CONCEPT: Courant-Friedrichs-Lewy (CFL) Condition
        ===========================================================================
        To guarantee stability in explicit transient hyperbolic PDE schemes, 
        numerical information cannot propagate faster than physical wave speeds:
            C = (u * Δt) / Δx  <=  C_max (Typically 1.0)
        ===========================================================================

        Args:
            u_matrix (np.ndarray): Local horizontal velocity grid array.
            dt (float): Iterative step duration value.
            dx (float): Discretization length step across spatial cells.

        Returns:
            bool: True if condition constraints are violated, otherwise False.
        """
        max_velocity = np.max(np.abs(u_matrix))
        if dx == 0:
            return True
        cfl_number = (max_velocity * dt) / dx
        return cfl_number > 1.0

    def check_divergence_breach(self, computed_residual):
        """Monitors simulation convergence to prevent uncontrolled matrix failure.
        
        Args:
            computed_residual (float): L2 structural error metric value.
            
        Returns:
            bool: True if divergence is imminent or NaN/Inf is encountered.
        """
        if math.isnan(computed_residual) or math.isinf(computed_residual):
            return True
        if computed_residual > self.residual_ceiling:
            return True
        return False