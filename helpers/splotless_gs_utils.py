import torch
from typing import Dict
#TODO: fill the missing imports if using this code


"""
Code below is taken from spotlesssplats https://github.com/lilygoli/SpotLessSplats
"""
@torch.no_grad()
def update_running_stats(runner, info: Dict):
    """Update running stats."""
    cfg = runner.cfg

    # normalize grads to [-1, 1] screen space
    if cfg.absgrad:
        grads = info["means2d"].absgrad.clone()
    else:
        grads = info["means2d"].grad.clone()
    if cfg.ubp:
        sqrgrads = info["means2d"].sqrgrad.clone()
    grads[..., 0] *= info["width"] / 2.0 * cfg.batch_size
    grads[..., 1] *= info["height"] / 2.0 * cfg.batch_size

    runner.running_stats["hist_err"] = (
        0.95 * runner.running_stats["hist_err"] + info["err"]
    )
    mid_err = torch.sum(runner.running_stats["hist_err"]) * cfg.robust_percentile
    runner.running_stats["avg_err"] = torch.linspace(0, 1, cfg.bin_size + 1)[
        torch.where(torch.cumsum(runner.running_stats["hist_err"], 0) >= mid_err)[0][
            0
        ]
    ]

    lower_err = torch.sum(runner.running_stats["hist_err"]) * cfg.lower_bound
    upper_err = torch.sum(runner.running_stats["hist_err"]) * cfg.upper_bound

    runner.running_stats["lower_err"] = torch.linspace(0, 1, cfg.bin_size + 1)[
        torch.where(torch.cumsum(runner.running_stats["hist_err"], 0) >= lower_err)[
            0
        ][0]
    ]
    runner.running_stats["upper_err"] = torch.linspace(0, 1, cfg.bin_size + 1)[
        torch.where(torch.cumsum(runner.running_stats["hist_err"], 0) >= upper_err)[
            0
        ][0]
    ]

    if cfg.packed:
        # grads is [nnz, 2]
        gs_ids = info["gaussian_ids"]  # [nnz] or None
        runner.running_stats["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        runner.running_stats["count"].index_add_(
            0, gs_ids, torch.ones_like(gs_ids).int()
        )
        if cfg.ubp:
            runner.running_stats["sqrgrad"].index_add_(
                0, gs_ids, torch.sum(sqrgrads, dim=-1)
            )
    else:
        # grads is [C, N, 2]
        sel = info["radii"] > 0.0  # [C, N]
        gs_ids = torch.where(sel)[1]  # [nnz]
        runner.running_stats["grad2d"].index_add_(0, gs_ids, grads[sel].norm(dim=-1))
        runner.running_stats["count"].index_add_(
            0, gs_ids, torch.ones_like(gs_ids).int()
        )
        if cfg.ubp:
            runner.running_stats["sqrgrad"].index_add_(
                0, gs_ids, torch.sum(sqrgrads[sel], dim=-1)
            )

@torch.no_grad()
def reset_opa(runner, value: float = 0.01):
    """Utility function to reset opacities."""
    opacities = torch.clamp(
        runner.splats["opacities"], max=torch.logit(torch.tensor(value)).item()
    )
    for optimizer in runner.optimizers:
        for i, param_group in enumerate(optimizer.param_groups):
            if param_group["name"] != "opacities":
                continue
            p = param_group["params"][0]
            p_state = optimizer.state[p]
            del optimizer.state[p]
            for key in p_state.keys():
                if key != "step":
                    p_state[key] = torch.zeros_like(p_state[key])
            p_new = torch.nn.Parameter(opacities)
            optimizer.param_groups[i]["params"] = [p_new]
            optimizer.state[p_new] = p_state
            runner.splats[param_group["name"]] = p_new
    torch.cuda.empty_cache()

@torch.no_grad()
def reset_sh(runner, value: float = 0.001):
    """Utility function to reset SH specular coefficients."""
    colors = torch.clamp(
        runner.splats["shN"], max=torch.abs(torch.tensor(value)).item()
    )
    for optimizer in runner.optimizers:
        for i, param_group in enumerate(optimizer.param_groups):
            if param_group["name"] != "shN":
                continue
            p = param_group["params"][0]
            p_state = optimizer.state[p]
            del optimizer.state[p]
            for key in p_state.keys():
                if key != "step":
                    p_state[key] = torch.zeros_like(p_state[key])
            p_new = torch.nn.Parameter(colors)
            optimizer.param_groups[i]["params"] = [p_new]
            optimizer.state[p_new] = p_state
            runner.splats[param_group["name"]] = p_new
    torch.cuda.empty_cache()

@torch.no_grad()
def refine_split(runner, mask: Tensor):
    """Utility function to grow GSs."""
    device = runner.device

    sel = torch.where(mask)[0]
    rest = torch.where(~mask)[0]

    scales = torch.exp(runner.splats["scales"][sel])  # [N, 3]
    quats = F.normalize(runner.splats["quats"][sel], dim=-1)  # [N, 4]
    rotmats = _quat_to_rotmat(quats)  # [N, 3, 3]
    samples = torch.einsum(
        "nij,nj,bnj->bni",
        rotmats,
        scales,
        torch.randn(2, len(scales), 3, device=device),
    )  # [2, N, 3]

    for optimizer in runner.optimizers:
        for i, param_group in enumerate(optimizer.param_groups):
            p = param_group["params"][0]
            name = param_group["name"]
            # create new params
            if name == "means3d":
                p_split = (p[sel] + samples).reshape(-1, 3)  # [2N, 3]
            elif name == "scales":
                p_split = torch.log(scales / 1.6).repeat(2, 1)  # [2N, 3]
            else:
                repeats = [2] + [1] * (p.dim() - 1)
                p_split = p[sel].repeat(repeats)
            p_new = torch.cat([p[rest], p_split])
            p_new = torch.nn.Parameter(p_new)
            # update optimizer
            p_state = optimizer.state[p]
            del optimizer.state[p]
            for key in p_state.keys():
                if key == "step":
                    continue
                v = p_state[key]
                # new params are assigned with zero optimizer states
                # (worth investigating it)
                v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=device)
                p_state[key] = torch.cat([v[rest], v_split])
            optimizer.param_groups[i]["params"] = [p_new]
            optimizer.state[p_new] = p_state
            runner.splats[name] = p_new
    for k, v in runner.running_stats.items():
        if v is None or k.find("err") != -1:
            continue
        repeats = [2] + [1] * (v.dim() - 1)
        v_new = v[sel].repeat(repeats)
        if k == "sqrgrad":
            v_new = torch.ones_like(
                v_new
            )  # the new ones are assumed to have high utilization in the start
        runner.running_stats[k] = torch.cat((v[rest], v_new))
    torch.cuda.empty_cache()

@torch.no_grad()
def refine_duplicate(runner, mask: Tensor):
    """Unility function to duplicate GSs."""
    sel = torch.where(mask)[0]
    for optimizer in runner.optimizers:
        for i, param_group in enumerate(optimizer.param_groups):
            p = param_group["params"][0]
            name = param_group["name"]
            p_state = optimizer.state[p]
            del optimizer.state[p]
            for key in p_state.keys():
                if key != "step":
                    # new params are assigned with zero optimizer states
                    # (worth investigating it as it will lead to a lot more GS.)
                    v = p_state[key]
                    v_new = torch.zeros(
                        (len(sel), *v.shape[1:]), device=runner.device
                    )
                    # v_new = v[sel]
                    p_state[key] = torch.cat([v, v_new])
            p_new = torch.nn.Parameter(torch.cat([p, p[sel]]))
            optimizer.param_groups[i]["params"] = [p_new]
            optimizer.state[p_new] = p_state
            runner.splats[name] = p_new
    for k, v in runner.running_stats.items():
        if k.find("err") != -1:
            continue
        if k == "sqrgrad":
            runner.running_stats[k] = torch.cat(
                (v, torch.ones_like(v[sel]))
            )  # new ones are assumed to have high utilization
        else:
            runner.running_stats[k] = torch.cat((v, v[sel]))
    torch.cuda.empty_cache()

@torch.no_grad()
def refine_keep(runner, mask: Tensor):
    """Unility function to prune GSs."""
    sel = torch.where(mask)[0]
    for optimizer in runner.optimizers:
        for i, param_group in enumerate(optimizer.param_groups):
            p = param_group["params"][0]
            name = param_group["name"]
            p_state = optimizer.state[p]
            del optimizer.state[p]
            for key in p_state.keys():
                if key != "step":
                    p_state[key] = p_state[key][sel]
            p_new = torch.nn.Parameter(p[sel])
            optimizer.param_groups[i]["params"] = [p_new]
            optimizer.state[p_new] = p_state
            runner.splats[name] = p_new
    for k, v in runner.running_stats.items():
        if k.find("err") != -1:
            continue
        runner.running_stats[k] = v[sel]
    torch.cuda.empty_cache()