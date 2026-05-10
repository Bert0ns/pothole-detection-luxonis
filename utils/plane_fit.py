from __future__ import annotations

import numpy as np

_MIN_ROAD_PIXELS = 100   # below this the fit is unreliable
_MAX_SAMPLES = 5000      # cap to keep lstsq fast on large frames


def fit_road_plane(
    depth_frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    depth_min: int = 200,
    depth_max: int = 2000,
    max_samples: int = _MAX_SAMPLES,
) -> np.ndarray | None:
    """Fit plane z = a*u + b*v + c to road pixels outside pothole bbox.

    Pixels inside bbox are excluded so only road surface contributes to the fit.
    Threshold filtering is applied internally — no pre-processing needed.

    Args:
        depth_frame: full H×W depth frame in mm.
        bbox:        (xmin, ymin, xmax, ymax) pothole region to exclude.
        depth_min:   minimum valid depth in mm.
        depth_max:   maximum valid depth in mm (tune to camera height + margin).
        max_samples: random subsample cap — keeps lstsq fast on dense frames.

    Returns:
        Coefficients [a, b, c] where road_depth(u, v) = a*u + b*v + c,
        or None if there are fewer than _MIN_ROAD_PIXELS valid road pixels.
    """
    xmin, ymin, xmax, ymax = bbox

    valid = (depth_frame >= depth_min) & (depth_frame <= depth_max)
    valid[ymin:ymax, xmin:xmax] = False  # exclude pothole region

    n = int(valid.sum())
    if n < _MIN_ROAD_PIXELS:
        return None

    v_coords, u_coords = np.where(valid)
    z = depth_frame[valid].astype(np.float32)

    if n > max_samples:
        idx = np.random.choice(n, max_samples, replace=False)
        u_coords, v_coords, z = u_coords[idx], v_coords[idx], z[idx]

    A = np.column_stack([u_coords, v_coords, np.ones(len(u_coords), dtype=np.float32)])
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    return coeffs


def road_depth_at(coeffs: np.ndarray, u: float, v: float) -> float:
    """Expected road surface depth at pixel (u, v) using plane coefficients."""
    return float(coeffs[0] * u + coeffs[1] * v + coeffs[2])
