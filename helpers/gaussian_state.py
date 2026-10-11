"""Shared layout for persistent Gaussian ODE states.

Geometry uses [means(3), quaternions(4), log-scales(3)]. Optional SH-degree-0
color flow appends the three DC SH coefficients, not clipped RGB values.
Opacity and Gaussian identity are not part of the evolving state.
"""

import torch


def pack_gaussian_state(splats, deformed_params_list):
    """Build an initial state without changing the Gaussian row/ID order."""
    parts = []
    for name in deformed_params_list:
        if name in ("means", "quats", "scales"):
            parts.append(splats[name])
        elif name == "shs":
            if splats["sh0"].shape[1:] != (1, 3) or splats["shN"].numel() != 0:
                raise ValueError("Color flow currently requires --sh-degree 0")
            parts.append(splats["sh0"].flatten(start_dim=1))
        else:
            raise ValueError(f"Unsupported evolving Gaussian parameter: {name}")
    return torch.cat(parts, dim=-1)


def validate_gaussian_state(state, color_dim=0):
    """Catch incompatible geometry-only/color checkpoints and trajectories."""
    if color_dim not in (0, 3):
        raise ValueError("Color flow supports three SH-degree-0 coefficients")
    expected = 10 + color_dim
    if state.ndim < 2 or state.shape[-1] != expected:
        raise ValueError(
            f"Expected a {expected}-dimensional Gaussian state, got {tuple(state.shape)}. "
            "Use matching --learn-shs / --sh-degree settings for boundary training, "
            "global training, and rendering; regenerate incompatible boundary trajectories."
        )


def unpack_gaussian_state(state, color_dim=0):
    """Return differentiable rasterizer inputs for [..., N, D] states."""
    validate_gaussian_state(state, color_dim)
    params = {
        "means": state[..., :3],
        "quats": state[..., 3:7],
        "scales": state[..., 7:10],
    }
    if color_dim:
        params["colors"] = state[..., 10:13].unsqueeze(-2)  # [..., N, 1, 3]
    return params


def validate_boundary_trajectory(trajectory, initial_state, num_timesteps):
    """A cached trajectory must preserve time, Gaussian IDs, and state width."""
    expected = (num_timesteps, *initial_state.shape)
    if tuple(trajectory.shape) != expected:
        raise ValueError(
            f"Boundary trajectory shape {tuple(trajectory.shape)} does not match {expected}. "
            "Regenerate it from the same static checkpoint with matching --learn-shs "
            "and --sh-degree settings."
        )
