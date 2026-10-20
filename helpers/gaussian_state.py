"""Shared layout for persistent Gaussian ODE states.

Geometry uses [means(3), quaternions(4), log-scales(3)]. Optional color flow
appends all SH coefficients (DC first, then higher orders), not clipped RGB.
Opacity and Gaussian identity are not part of the evolving state.
"""

import torch


def validate_color_dim(color_dim):
    """Allow complete RGB SH bases through the rasterizer's maximum degree 4."""
    if color_dim not in (0, 3, 12, 27, 48, 75):
        raise ValueError(
            "Color flow requires 3 * (sh_degree + 1)^2 coefficients "
            "for SH degrees 0 through 4, or color_dim=0 for fixed color"
        )


def pack_gaussian_state(splats, deformed_params_list):
    """Build an initial state without changing the Gaussian row/ID order."""
    parts = []
    for name in deformed_params_list:
        if name in ("means", "quats", "scales"):
            parts.append(splats[name])
        elif name == "shs":
            sh0, shN = splats["sh0"], splats["shN"]
            if (sh0.ndim != 3 or sh0.shape[1:] != (1, 3)
                    or shN.ndim != 3 or shN.shape[0] != sh0.shape[0]
                    or shN.shape[-1] != 3):
                raise ValueError("Expected SH tensors shaped [N, 1, 3] and [N, K-1, 3]")
            colors = torch.cat((sh0, shN), dim=1).flatten(start_dim=1)
            validate_color_dim(colors.shape[-1])
            parts.append(colors)
        else:
            raise ValueError(f"Unsupported evolving Gaussian parameter: {name}")
    return torch.cat(parts, dim=-1)


def validate_gaussian_state(state, color_dim=0):
    """Catch incompatible geometry-only/color checkpoints and trajectories."""
    validate_color_dim(color_dim)
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
        params["colors"] = state[..., 10:].unflatten(-1, (color_dim // 3, 3))
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
