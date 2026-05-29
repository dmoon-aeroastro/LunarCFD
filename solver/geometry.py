import numpy as np

def get_naca_mask(ny, nx, chord_length=40, center_x=30, center_y=40):
    mask = np.zeros((ny, nx), dtype=bool)
    x = np.linspace(0, 1, chord_length)
    
    # NACA 0012 thickness formula
    t = 0.12 / 0.2 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    
    for i in range(len(x)):
        # Calculate upper and lower bounds relative to grid centers
        half_thickness = t[i] * chord_length
        
        # Use floor and ceil to make sure we don't collapse to a 0-pixel height anywhere
        y_upper = int(np.ceil(center_y + half_thickness))
        y_lower = int(np.floor(center_y - half_thickness))
        
        # Slicing is exclusive of the stop index, so adding +1 makes it inclusive 
        # of the y_upper pixel, ensuring an airtight solid boundary layer
        mask[y_lower:y_upper + 1, center_x + i] = True
        
    return mask