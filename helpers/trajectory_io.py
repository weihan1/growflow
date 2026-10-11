"""Point selection for legacy and visibility-aware ground-truth trajectories."""

import numpy as np


def trajectory_validity(points):
    """Missing points have three NaNs; partial NaNs and infinities are errors."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Trajectory frame must have shape (N, 3)")
    valid = np.isfinite(points).all(axis=1)
    missing = np.isnan(points).all(axis=1)
    if not (valid | missing).all():
        raise ValueError("Trajectory contains infinity or a partially missing point")
    return valid


def select_trajectory_points(points, indices=None):
    """Return finite points and their ORIGINAL track IDs, without renumbering."""
    points = np.asarray(points)
    valid = trajectory_validity(points)
    if indices is None:
        indices = np.arange(len(points), dtype=np.int64)
    else:
        indices = np.asarray(indices)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError("Track indices must be a one-dimensional integer array")
        if np.any(indices < 0) or np.any(indices >= len(points)):
            raise ValueError("Track index is out of bounds; regenerate closest_indices.pt")
    indices = indices[valid[indices]]
    return points[indices], indices
