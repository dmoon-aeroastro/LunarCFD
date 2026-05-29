"""Discrete Array Field Grid Domain Generation Tools.

Constructs 2D numerical discretization node fields and handles memory 
footprint mapping estimation prior to large allocation cycles.

Dependencies:
    - numpy
"""

# GNU General Public License v3.0 Header
# Copyright (C) 2026 LunarCFD Development Team

import numpy as np

class DiscretizationGridGenerator:
    """Manages structural geometry calculations for continuous spatial grid fields."""
    
    @staticmethod
    def estimate_memory_allocation_mb(nx, ny):
        """Calculates expected floating-point allocation footprints before grid creation.
        
        Args:
            nx (int): Spatial mesh horizontal cell element count.
            ny (int): Spatial mesh vertical cell element count.
            
        Returns:
            float: Size target computation mapping scaled in Megabytes.
        """
        # Estimating allocation of core solution vector matrices (u, v, p, sources)
        total_floats = nx * ny * 8 
        bytes_per_float = 8 # 64-bit precision metrics float standard
        return (total_floats * bytes_per_float) / (1024 * 1024)

    @staticmethod
    def build_structured_mesh(nx, ny):
        """Constructs uniform discrete boundaries across standard dimensional spans.
        
        Args:
            nx (int): Resolution coordinate scaling vector count along X axis.
            ny (int): Resolution coordinate scaling vector count along Y axis.
            
        Returns:
            tuple: (X, Y) matrices containing structural space layout coordinates.
        """
        # Patankar 1980, Chapter 3: Grid selection and control volume layout frameworks.
        # Control volumes are structured orthogonally for staggered grid velocity point layouts.
        x = np.linspace(0.0, 2.0, nx)
        y = np.linspace(0.0, 1.0, ny)
        return np.meshgrid(x, y)