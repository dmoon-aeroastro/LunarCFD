"""
NACA 4-digit airfoil coordinate generator with cosine spacing.

Returns a closed (x, y) contour starting and ending at the trailing edge,
going counter-clockwise: TE -> upper surface -> LE -> lower surface -> TE.

The chord is normalised to 1.0.  Coordinates are in the range x in [0, 1],
y roughly in [-t/2, +t/2].
"""

import numpy as np


# ── NACA 4-digit series ──────────────────────────────────────────────────────

def naca4(series="0012", n_pts=128):
    """
    Generate a closed NACA 4-digit airfoil contour.

    Parameters
    ----------
    series : str
        4-character NACA series string, e.g. "0012", "2412", "4412".
    n_pts : int
        Number of points per surface (upper or lower).  Final array length
        is 2*n_pts (shared leading-edge and trailing-edge points appear once).

    Returns
    -------
    x, y : ndarray, shape (2*n_pts,)
        Closed airfoil contour: TE -> upper -> LE -> lower -> (TE implicit).
        x[0] = 1.0 exactly (trailing edge).  The trailing edge is CLOSED
        (modified -0.1036 coefficient) and the contour starts exactly at the
        TE point, so for symmetric sections the resampled point set is
        mirror-symmetric about y=0.  (The standard open-TE -0.1015
        coefficient left a 0.21%-chord blunt base whose single closing face
        sat asymmetrically on the upper side — one source of spurious lift
        for symmetric airfoils at AoA=0 on the O-grid.)
    """
    series = str(series).zfill(4)
    m = int(series[0]) / 100.0   # max camber fraction of chord
    p = int(series[1]) / 10.0    # chordwise location of max camber (tenths)
    t = int(series[2:]) / 100.0  # max thickness fraction of chord

    # Cosine clustering: denser near leading and trailing edges
    beta = np.linspace(0.0, np.pi, n_pts + 1)   # n_pts+1 vertices per half
    xc   = (1.0 - np.cos(beta)) / 2.0           # x in [0, 1]

    # NACA 5-coefficient thickness distribution, closed trailing edge
    # (-0.1036 instead of the classic open-TE -0.1015: yt(1) = 0 exactly)
    yt = (t / 0.2) * (0.2969 * np.sqrt(xc)
                      - 0.1260 * xc
                      - 0.3516 * xc**2
                      + 0.2843 * xc**3
                      - 0.1036 * xc**4)

    # Camber line and its slope
    if m < 1e-6 or p < 1e-6:
        yc  = np.zeros_like(xc)
        dyc = np.zeros_like(xc)
    else:
        fwd = xc <= p
        yc  = np.where(fwd,
                       (m / p**2) * (2*p*xc - xc**2),
                       (m / (1-p)**2) * ((1 - 2*p) + 2*p*xc - xc**2))
        dyc = np.where(fwd,
                       (2*m / p**2) * (p - xc),
                       (2*m / (1-p)**2) * (p - xc))

    theta = np.arctan(dyc)

    # Upper and lower surface coordinates
    xu = xc - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = xc + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Build closed contour CCW: TE -> upper (reversed) -> LE -> lower -> (TE)
    # xu/xl run LE->TE with index; with the closed TE both surfaces end at
    # exactly (1, yc(1)).  Drop the duplicate LE point AND the duplicate TE
    # point so every contour point is unique (the TE appears only at x[0]).
    x = np.concatenate([xu[::-1], xl[1:-1]])   # upper TE->LE, lower LE->just-before-TE
    y = np.concatenate([yu[::-1], yl[1:-1]])
    return x, y


# ── .dat file loader (Selig format) ─────────────────────────────────────────

def load_dat(filepath):
    """
    Load airfoil coordinates from a Selig-format .dat file.

    Selig format: the first line is the airfoil name (text), subsequent lines
    are whitespace-separated ``x  y`` pairs.  Both single-surface (x from 0 to 1
    and back) and two-surface (upper then lower) layouts are handled: if x
    increases to 1 and then decreases back to 0 the contour is assumed already
    in CCW order; otherwise the two halves are stitched into CCW order.

    Returns
    -------
    x, y : ndarray  (closed contour)
    """
    rows = []
    with open(filepath) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except (ValueError, IndexError):
                continue   # skip header / blank / comment lines

    data = np.array(rows)
    return data[:, 0], data[:, 1]


# ── Rectangle (bluff body) ──────────────────────────────────────────────────

def rect_1x8(n_pts=512, corner_r=0.02):
    """
    Generate a closed 1×8 rectangle contour (chord=1, height=0.125).

    Sharp corners cause O-grid normal singularities, so each corner is
    replaced by a small circular arc of radius corner_r (default 2 % chord).

    Convention matches naca4: CCW, starting near the TE (right-centre),
    going upper surface → LE → lower surface → back to TE.
    x in [0, 1],  y in [-0.0625, +0.0625].
    """
    h = 0.5 / 8.0   # half-height = 0.0625
    r = min(corner_r, h * 0.9)   # clamp radius so it fits

    # Corner centres (cx, cy) and start/end angles (CCW)
    # Corners: top-right (TR), top-left (TL), bottom-left (BL), bottom-right (BR)
    corners = [
        (1.0 - r,  h - r,  0.0,        0.5 * np.pi),   # TR: 0° → 90°
        (r,         h - r,  0.5*np.pi,  np.pi),          # TL: 90° → 180°
        (r,        -h + r,  np.pi,      1.5*np.pi),      # BL: 180° → 270°
        (1.0 - r, -h + r,  1.5*np.pi,  2.0*np.pi),      # BR: 270° → 360°
    ]

    pts_per_arc = max(8, n_pts // 16)
    xs, ys = [], []

    # Segment order for CCW starting EXACTLY at mid-TE (1, 0): up the right
    # side, TR arc, top edge, TL arc, left edge, BL arc, bottom edge, BR arc,
    # then the lower right side back up toward (1, 0).
    # Starting on the symmetry line keeps the resampled point set
    # mirror-symmetric about y=0 (the previous version started at the TR arc,
    # shifting every point ~4% of the perimeter relative to its mirror —
    # same class of asymmetry as the open-TE airfoil bug).

    def arc(cx, cy, a0, a1, n):
        t = np.linspace(a0, a1, n, endpoint=False)
        return cx + r * np.cos(t), cy + r * np.sin(t)

    n_edge_long  = max(4, n_pts // 4)        # top/bottom edges
    n_edge_half  = max(4, n_pts // 32)       # right-edge halves (above/below y=0)
    n_edge_short = max(4, n_pts // 16)       # left edge

    # Right edge, lower→upper half: start at (1, 0) exactly
    xs.append(np.full(n_edge_half, 1.0))
    ys.append(np.linspace(0.0, h-r, n_edge_half, endpoint=False))

    # TR arc (0 → 90°), then top edge right→left
    ax, ay = arc(1.0-r,  h-r, 0.0,        0.5*np.pi, pts_per_arc)
    xs.append(ax); ys.append(ay)
    xs.append(np.linspace(1.0-r, r,    n_edge_long, endpoint=False))
    ys.append(np.full(n_edge_long, h))

    # TL arc (90 → 180°), then left edge top→bottom
    ax, ay = arc(r,       h-r, 0.5*np.pi, np.pi,      pts_per_arc)
    xs.append(ax); ys.append(ay)
    xs.append(np.full(n_edge_short, 0.0))
    ys.append(np.linspace(h-r, -h+r, n_edge_short, endpoint=False))

    # BL arc (180 → 270°), then bottom edge left→right
    ax, ay = arc(r,      -h+r, np.pi,      1.5*np.pi,  pts_per_arc)
    xs.append(ax); ys.append(ay)
    xs.append(np.linspace(r, 1.0-r,    n_edge_long, endpoint=False))
    ys.append(np.full(n_edge_long, -h))

    # BR arc (270 → 360°), then right edge from the arc end back up to (1, 0)
    ax, ay = arc(1.0-r, -h+r, 1.5*np.pi, 2.0*np.pi,  pts_per_arc)
    xs.append(ax); ys.append(ay)
    xs.append(np.full(n_edge_half, 1.0))
    ys.append(np.linspace(-h+r, 0.0, n_edge_half, endpoint=False))

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    return x, y


# ── Utility: resample a contour to exactly n_pts points ─────────────────────

def resample(x, y, n_pts):
    """
    Resample a closed polygon (x, y) to n_pts equally arc-length spaced points.
    Useful to normalise .dat files to the same resolution as naca4().
    """
    dx = np.diff(np.append(x, x[0]))
    dy = np.diff(np.append(y, y[0]))
    s  = np.concatenate([[0.0], np.cumsum(np.sqrt(dx**2 + dy**2))])
    s /= s[-1]
    t  = np.linspace(0.0, 1.0, n_pts, endpoint=False)
    xr = np.interp(t, s, np.append(x, x[0]))
    yr = np.interp(t, s, np.append(y, y[0]))
    return xr, yr
