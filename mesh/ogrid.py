"""
O-grid generator for body-fitted CFD around a closed airfoil contour.

Topology
--------
Structured O-grid: i-index goes around the airfoil (periodic, CCW), j-index
goes radially outward from the body surface.

    j = 0      : first fluid cell (wall-adjacent layer)
    j = n_eta-1: outermost cell (adjacent to far-field boundary)

Vertex array XV[j, i], YV[j, i] has shape (n_eta+1, n_xi).
Cell-centre array XC[j, i], YC[j, i] has shape (n_eta, n_xi).

Grid generation
---------------
Algebraic Transfinite Interpolation (TFI) with sinh stretching in eta to
cluster cells near the wall.  The outer boundary is a circle of radius R
centred on the airfoil centroid, with angular distribution matching the
airfoil parametrisation so grid lines are approximately wall-normal.

Metric arrays (precomputed, shape (n_eta, n_xi) for cell-centre quantities)
---------------------------------------------------------------------------
Face-normal vectors and face lengths for each of the four faces per cell:

    East  face : between (j, i) and (j, i+1%n_xi)  -- xi-faces
    West  face : between (j, i) and (j, i-1%n_xi)
    North face : between (j, i) and (j+1, i)        -- eta-faces
    South face : between (j, i) and (j-1, i)

Each face stores an *outward* unit normal (nx, ny) from the cell and the
face length ds, plus the centre-to-centre distance dn used in diffusion /
pressure-gradient calculations.

Sign convention
---------------
For every face the outward normal is computed by rotating the face-edge
vector 90° clockwise, then confirming it points from cell-centre toward the
face-centre (if not, flipped).  This removes any dependency on grid chirality.
"""

import numpy as np


# ── Stretching helper ────────────────────────────────────────────────────────

def _sinh_stretch(n, alpha=3.0):
    """
    One-sided sinh stretching: returns n+1 values in [0, 1] denser near 0.

    eta[j] = sinh(alpha * s[j]) / sinh(alpha),  s = j/n.

    alpha = 3 gives first-cell fraction ~0.006, last ~0.25 (for n=48).
    """
    s   = np.linspace(0.0, 1.0, n + 1)
    eta = np.sinh(alpha * s) / np.sinh(alpha)
    return eta


# ── Main grid builder ────────────────────────────────────────────────────────

def build_ogrid(x_a, y_a, n_xi=None, n_eta=48, R=8.0, alpha_stretch=3.0):
    """
    Build an O-grid around a closed airfoil contour.

    Parameters
    ----------
    x_a, y_a : array-like, length N_airfoil
        Airfoil surface coordinates (closed CCW contour).  The function
        resamples to n_xi equi-arc-length points internally.
    n_xi : int or None
        Number of cells in the circumferential direction.  If None, the
        length of x_a is used directly (no resampling).
    n_eta : int
        Number of cells in the radial direction.
    R : float
        Far-field radius as a multiple of chord length.
    alpha_stretch : float
        Stretching factor for the sinh wall clustering.  Larger = tighter
        near-wall spacing.

    Returns
    -------
    grid : dict with keys:
        XV, YV   : vertex coordinates, shape (n_eta+1, n_xi)
        XC, YC   : cell-centre coordinates, shape (n_eta, n_xi)
        --- East face (between (j,i) and (j, i+1%n_xi)) ---
        nxE, nyE : outward normal from cell (j,i) through east face
        dsE      : east face length
        dnE      : centre-to-centre distance projected onto east face normal
        --- West face ---
        nxW, nyW, dsW, dnW
        --- North face (between (j,i) and (j+1,i)) ---
        nxN, nyN, dsN, dnN
        --- South face (between (j,i) and (j-1,i)) ---
        nxS, nyS, dsS, dnS
        --- Wall face (south face of j=0 row) ---
        nxW_wall, nyW_wall, dsW_wall : airfoil surface face outward normals
        --- Area ---
        cell_area : shape (n_eta, n_xi)
        --- Metadata ---
        n_xi, n_eta, R, chord
    """
    x_a = np.asarray(x_a, dtype=float)
    y_a = np.asarray(y_a, dtype=float)

    # ── Chord and centroid ────────────────────────────────────────────────────
    chord = float(x_a.max() - x_a.min())
    cx    = float(x_a.mean())
    cy    = float(y_a.mean())

    # ── Resample airfoil to n_xi equi-arc-length points ─────────────────────
    if n_xi is None:
        n_xi = len(x_a)

    dx  = np.diff(np.append(x_a, x_a[0]))
    dy  = np.diff(np.append(y_a, y_a[0]))
    arc = np.concatenate([[0.0], np.cumsum(np.sqrt(dx**2 + dy**2))])
    arc /= arc[-1]
    t   = np.linspace(0.0, 1.0, n_xi, endpoint=False)
    xa  = np.interp(t, arc, np.append(x_a, x_a[0]))
    ya  = np.interp(t, arc, np.append(y_a, y_a[0]))

    # ── Outer boundary: circle of radius R*chord centred at centroid ─────────
    # Angular position of each airfoil point from the centroid, used to
    # distribute outer boundary points — keeps grid lines approximately radial.
    theta = np.arctan2(ya - cy, xa - cx)
    xo    = cx + R * chord * np.cos(theta)
    yo    = cy + R * chord * np.sin(theta)

    # ── Radial stretching parameter (sinh) ────────────────────────────────────
    eta   = _sinh_stretch(n_eta, alpha_stretch)   # shape (n_eta+1,)

    # ── Vertex array : TFI linear interpolation between inner and outer ───────
    XV = np.empty((n_eta + 1, n_xi), dtype=float)
    YV = np.empty((n_eta + 1, n_xi), dtype=float)
    for j in range(n_eta + 1):
        XV[j, :] = (1.0 - eta[j]) * xa + eta[j] * xo
        YV[j, :] = (1.0 - eta[j]) * ya + eta[j] * yo

    # ── Cell centres (average of four corners, with periodic i wrap) ──────────
    ip1 = (np.arange(n_xi) + 1) % n_xi
    XC  = (XV[:-1, :] + XV[:-1, ip1] + XV[1:, :] + XV[1:, ip1]) / 4.0
    YC  = (YV[:-1, :] + YV[:-1, ip1] + YV[1:, :] + YV[1:, ip1]) / 4.0

    # ── Face metric helper ────────────────────────────────────────────────────
    def _face_metrics(v1x, v1y, v2x, v2y, cx_cell, cy_cell):
        """
        Given two vertices (v1, v2) defining a face edge and the cell centre
        (cx_cell, cy_cell), return (nx, ny, ds, dn_dummy) where:
          - (nx, ny) is the unit outward normal from the cell
          - ds is the face length
          - dn_dummy is a placeholder (filled later with c-c distances)
        """
        edx = v2x - v1x
        edy = v2y - v1y
        ds  = np.sqrt(edx**2 + edy**2) + 1e-30

        # Raw normal: rotate edge CW
        nx_raw = edy / ds
        ny_raw = -edx / ds

        # Face midpoint
        fmx = (v1x + v2x) / 2.0
        fmy = (v1y + v2y) / 2.0

        # Vector from cell centre to face midpoint
        dcx = fmx - cx_cell
        dcy = fmy - cy_cell

        # Flip if the raw normal points toward cell interior
        dot = dcx * nx_raw + dcy * ny_raw
        sign = np.where(dot >= 0.0, 1.0, -1.0)

        return sign * nx_raw, sign * ny_raw, ds

    # ── East face: between cell (j,i) and (j, i+1%n_xi) ─────────────────────
    # East face vertices: (j, i+1) and (j+1, i+1)
    v1x_E = XV[:-1, ip1];  v1y_E = YV[:-1, ip1]
    v2x_E = XV[1:,  ip1];  v2y_E = YV[1:,  ip1]
    nxE, nyE, dsE = _face_metrics(v1x_E, v1y_E, v2x_E, v2y_E, XC, YC)

    # Centre-to-centre distance projected onto face normal
    XC_E = XC[:, ip1]   # east neighbour cell centres
    YC_E = YC[:, ip1]
    dxE  = XC_E - XC
    dyE  = YC_E - YC
    dnE  = np.abs(dxE * nxE + dyE * nyE) + 1e-30

    # ── West face: between cell (j,i) and (j, i-1%n_xi) ─────────────────────
    im1 = (np.arange(n_xi) - 1) % n_xi
    # West face vertices: (j, i) and (j+1, i) — outward = pointing west
    v1x_W = XV[:-1, :];  v1y_W = YV[:-1, :]
    v2x_W = XV[1:,  :];  v2y_W = YV[1:,  :]
    # Raw edge from vertex 1 to 2 (going north); outward from cell = west = reversed sign
    edx_W = v2x_W - v1x_W;  edy_W = v2y_W - v1y_W
    ds_W  = np.sqrt(edx_W**2 + edy_W**2) + 1e-30
    nx_W_raw = edy_W / ds_W;  ny_W_raw = -edx_W / ds_W
    fmx_W = (v1x_W + v2x_W) / 2.0;  fmy_W = (v1y_W + v2y_W) / 2.0
    dcx_W = fmx_W - XC;  dcy_W = fmy_W - YC
    dot_W = dcx_W * nx_W_raw + dcy_W * ny_W_raw
    sign_W = np.where(dot_W >= 0.0, 1.0, -1.0)
    nxW = sign_W * nx_W_raw;  nyW = sign_W * ny_W_raw;  dsW = ds_W

    XC_W = XC[:, im1];  YC_W = YC[:, im1]
    dxW  = XC_W - XC;   dyW  = YC_W - YC
    dnW  = np.abs(dxW * nxW + dyW * nyW) + 1e-30

    # ── North face: between cell (j,i) and (j+1,i) ───────────────────────────
    # North face vertices: (j+1, i) and (j+1, i+1)
    v1x_N = XV[1:, :];    v1y_N = YV[1:, :]
    v2x_N = XV[1:, ip1];  v2y_N = YV[1:, ip1]
    nxN, nyN, dsN = _face_metrics(v1x_N, v1y_N, v2x_N, v2y_N, XC, YC)

    # North neighbour cell centres (only valid for j < n_eta-1)
    XC_N = np.empty_like(XC);  YC_N = np.empty_like(YC)
    XC_N[:-1, :] = XC[1:, :];  XC_N[-1, :] = XC[-1, :] + (XC[-1, :] - XC[-2, :])
    YC_N[:-1, :] = YC[1:, :];  YC_N[-1, :] = YC[-1, :] + (YC[-1, :] - YC[-2, :])
    dxN  = XC_N - XC;  dyN = YC_N - YC
    dnN  = np.abs(dxN * nxN + dyN * nyN) + 1e-30

    # ── South face: between cell (j,i) and (j-1,i) ───────────────────────────
    # South face vertices: (j, i) and (j, i+1)
    v1x_S = XV[:-1, :];    v1y_S = YV[:-1, :]
    v2x_S = XV[:-1, ip1];  v2y_S = YV[:-1, ip1]
    nxS, nyS, dsS = _face_metrics(v1x_S, v1y_S, v2x_S, v2y_S, XC, YC)

    # South neighbour cell centres (only valid for j > 0)
    XC_S = np.empty_like(XC);  YC_S = np.empty_like(YC)
    XC_S[1:, :]  = XC[:-1, :]; XC_S[0, :] = XC[0, :] - (XC[1, :] - XC[0, :])
    YC_S[1:, :]  = YC[:-1, :]; YC_S[0, :] = YC[0, :] - (YC[1, :] - YC[0, :])
    dxS  = XC_S - XC;  dyS = YC_S - YC
    dnS  = np.abs(dxS * nxS + dyS * nyS) + 1e-30

    # ── Cell area (shoelace on the four vertices) ─────────────────────────────
    # Vertices go: SW, SE, NE, NW (CCW for outward-j convention — sign may vary
    # but we take the absolute value)
    sw_x = XV[:-1, :];    sw_y = YV[:-1, :]
    se_x = XV[:-1, ip1];  se_y = YV[:-1, ip1]
    ne_x = XV[1:,  ip1];  ne_y = YV[1:,  ip1]
    nw_x = XV[1:,  :];    nw_y = YV[1:,  :]
    cell_area = 0.5 * np.abs(
        (se_x - sw_x) * (nw_y - sw_y) - (nw_x - sw_x) * (se_y - sw_y) +
        (ne_x - se_x) * (sw_y - se_y) - (sw_x - se_x) * (ne_y - se_y)
    ) + 1e-8

    # ── Wall-face outward normal (south face of j=0 cells) ────────────────────
    # These are the faces coinciding with the airfoil surface.
    # Used for force integration.
    nxW_wall = nxS[0, :]
    nyW_wall = nyS[0, :]
    dsW_wall = dsS[0, :]

    return dict(
        XV=XV, YV=YV,
        XC=XC, YC=YC,
        nxE=nxE, nyE=nyE, dsE=dsE, dnE=dnE,
        nxW=nxW, nyW=nyW, dsW=dsW, dnW=dnW,
        nxN=nxN, nyN=nyN, dsN=dsN, dnN=dnN,
        nxS=nxS, nyS=nyS, dsS=dsS, dnS=dnS,
        nxW_wall=nxW_wall, nyW_wall=nyW_wall, dsW_wall=dsW_wall,
        cell_area=cell_area,
        n_xi=n_xi, n_eta=n_eta, R=R, chord=chord,
        xa=xa, ya=ya, xo=xo, yo=yo,
    )
