import matplotlib.pyplot as plt
import torch
from gsplat import rasterization
import math
import struct
from typing import Optional, Tuple
from typing_extensions import Literal, assert_never
import torch.nn.functional as F
from torch import Tensor
import open3d as o3d
import numpy as np
import cv2
from tqdm import tqdm
import os

### These helper functions are taken from gsplat.
def convert_3d_to_2d(means3d, w2c, K):
    """
    """
    C, N, _ = means.shape

    tx, ty, tz = torch.unbind(means, dim=-1)  # [C, N]
    tz2 = tz**2  # [C, N]

    fx = Ks[..., 0, 0, None]  # [C, 1]
    fy = Ks[..., 1, 1, None]  # [C, 1]
    cx = Ks[..., 0, 2, None]  # [C, 1]
    cy = Ks[..., 1, 2, None]  # [C, 1]
    tan_fovx = 0.5 * width / fx  # [C, 1]
    tan_fovy = 0.5 * height / fy  # [C, 1]

    lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
    lim_x_neg = cx / fx + 0.3 * tan_fovx
    lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
    lim_y_neg = cy / fy + 0.3 * tan_fovy
    tx = tz * torch.clamp(tx / tz, min=-lim_x_neg, max=lim_x_pos)
    ty = tz * torch.clamp(ty / tz, min=-lim_y_neg, max=lim_y_pos)

    O = torch.zeros((C, N), device=means.device, dtype=means.dtype)
    J = torch.stack(
        [fx / tz, O, -fx * tx / tz2, O, fy / tz, -fy * ty / tz2], dim=-1
    ).reshape(C, N, 2, 3)

    cov2d = torch.einsum("...ij,...jk,...kl->...il", J, covars, J.transpose(-1, -2))
    means2d = torch.einsum("cij,cnj->cni", Ks[:, :2, :3], means)  # [C, N, 2]
    means2d = means2d / tz[..., None]  # [C, N, 2]
    return means2d, cov2d  # [C, N, 2], [C, N, 2, 2]


def _quat_scale_to_covar_preci(
    quats: Tensor,  # [N, 4],
    scales: Tensor,  # [N, 3],
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """PyTorch implementation of `gsplat.cuda._wrapper.quat_scale_to_covar_preci()`."""
    R = _quat_to_rotmat(quats)  # (..., 3, 3)

    if compute_covar:
        M = R * scales[..., None, :]  # (..., 3, 3)
        covars = torch.bmm(M, M.transpose(-1, -2))  # (..., 3, 3)
        if triu:
            covars = covars.reshape(covars.shape[:-2] + (9,))  # (..., 9)
            covars = (
                covars[..., [0, 1, 2, 4, 5, 8]] + covars[..., [0, 3, 6, 4, 7, 8]]
            ) / 2.0  # (..., 6)
    if compute_preci:
        P = R * (1 / scales[..., None, :])  # (..., 3, 3)
        precis = torch.bmm(P, P.transpose(-1, -2))  # (..., 3, 3)
        if triu:
            precis = precis.reshape(precis.shape[:-2] + (9,))
            precis = (
                precis[..., [0, 1, 2, 4, 5, 8]] + precis[..., [0, 3, 6, 4, 7, 8]]
            ) / 2.0

    return covars if compute_covar else None, precis if compute_preci else None


def _fully_fused_projection(
    means: Tensor,  # [N, 3]
    covars: Tensor,  # [N, 3, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """PyTorch implementation of `gsplat.cuda._wrapper.fully_fused_projection()`

    .. note::

        This is a minimal implementation of fully fused version, which has more
        arguments. Not all arguments are supported.
    """
    means_c, covars_c = _world_to_cam(means, covars, viewmats)
    means2d, covars2d = _persp_proj(means_c, covars_c, Ks, width, height)
    covars2d = covars2d + torch.eye(2, device=means.device, dtype=means.dtype) * eps2d
    det = (
        covars2d[..., 0, 0] * covars2d[..., 1, 1]
        - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    )
    det = det.clamp(min=1e-10)

    conics = torch.stack(
        [
            covars2d[..., 1, 1] / det,
            -(covars2d[..., 0, 1] + covars2d[..., 1, 0]) / 2.0 / det,
            covars2d[..., 0, 0] / det,
        ],
        dim=-1,
    )  # [C, N, 3]
    return means2d,covars2d,conics

def _quat_to_rotmat(quats: Tensor) -> Tensor:
    """Convert quaternion to rotation matrix."""
    quats = F.normalize(quats, p=2, dim=-1)
    w, x, y, z = torch.unbind(quats, dim=-1)
    R = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return R.reshape(quats.shape[:-1] + (3, 3))


def _quat_scale_to_matrix(
    quats: Tensor,  # [N, 4],
    scales: Tensor,  # [N, 3],
) -> Tensor:
    """Convert quaternion and scale to a 3x3 matrix (R * S)."""
    R = _quat_to_rotmat(quats)  # (..., 3, 3)
    M = R * scales[..., None, :]  # (..., 3, 3)
    return M


def _quat_scale_to_covar_preci(
    quats: Tensor,  # [N, 4],
    scales: Tensor,  # [N, 3],
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """PyTorch implementation of `gsplat.cuda._wrapper.quat_scale_to_covar_preci()`."""
    R = _quat_to_rotmat(quats)  # (..., 3, 3)

    if compute_covar:
        M = R * scales[..., None, :]  # (..., 3, 3)
        covars = torch.bmm(M, M.transpose(-1, -2))  # (..., 3, 3)
        if triu:
            covars = covars.reshape(covars.shape[:-2] + (9,))  # (..., 9)
            covars = (
                covars[..., [0, 1, 2, 4, 5, 8]] + covars[..., [0, 3, 6, 4, 7, 8]]
            ) / 2.0  # (..., 6)
    if compute_preci:
        P = R * (1 / scales[..., None, :])  # (..., 3, 3)
        precis = torch.bmm(P, P.transpose(-1, -2))  # (..., 3, 3)
        if triu:
            precis = precis.reshape(precis.shape[:-2] + (9,))
            precis = (
                precis[..., [0, 1, 2, 4, 5, 8]] + precis[..., [0, 3, 6, 4, 7, 8]]
            ) / 2.0

    return covars if compute_covar else None, precis if compute_preci else None


def _persp_proj(
    means: Tensor,  # [C, N, 3]
    covars: Tensor,  # [C, N, 3, 3]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation of perspective projection for 3D Gaussians.

    Args:
        means: Gaussian means in camera coordinate system. [C, N, 3].
        covars: Gaussian covariances in camera coordinate system. [C, N, 3, 3].
        Ks: Camera intrinsics. [C, 3, 3].
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **means2d**: Projected means. [C, N, 2].
        - **cov2d**: Projected covariances. [C, N, 2, 2].
    """
    C, N, _ = means.shape

    tx, ty, tz = torch.unbind(means, dim=-1)  # [C, N] , tz usually is just camera position
    tz2 = tz**2  # [C, N]

    fx = Ks[..., 0, 0, None]  # [C, 1]
    fy = Ks[..., 1, 1, None]  # [C, 1]
    cx = Ks[..., 0, 2, None]  # [C, 1]
    cy = Ks[..., 1, 2, None]  # [C, 1]
    tan_fovx = 0.5 * width / fx  # [C, 1]
    tan_fovy = 0.5 * height / fy  # [C, 1]

    lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
    lim_x_neg = cx / fx + 0.3 * tan_fovx
    lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
    lim_y_neg = cy / fy + 0.3 * tan_fovy
    tx = tz * torch.clamp(tx / tz, min=-lim_x_neg, max=lim_x_pos)
    ty = tz * torch.clamp(ty / tz, min=-lim_y_neg, max=lim_y_pos)

    O = torch.zeros((C, N), device=means.device, dtype=means.dtype)
    J = torch.stack(
        [fx / tz, O, -fx * tx / tz2, O, fy / tz, -fy * ty / tz2], dim=-1
    ).reshape(C, N, 2, 3)

    cov2d = torch.einsum("...ij,...jk,...kl->...il", J, covars, J.transpose(-1, -2))
    means2d = torch.einsum("cij,cnj->cni", Ks[:, :2, :3], means)  # [C, N, 2]
    means2d = means2d / tz[..., None]  # [C, N, 2]
    return means2d, cov2d  # [C, N, 2], [C, N, 2, 2]


def _world_to_cam(
    means: Tensor,  # [N, 3]
    covars: Tensor,  # [N, 3, 3]
    viewmats: Tensor,  # [C, 4, 4]
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation of world to camera transformation on Gaussians.

    Args:
        means: Gaussian means in world coordinate system. [C, N, 3].
        covars: Gaussian covariances in world coordinate system. [C, N, 3, 3].
        viewmats: world to camera transformation matrices. [C, 4, 4].

    Returns:
        A tuple:

        - **means_c**: Gaussian means in camera coordinate system. [C, N, 3].
        - **covars_c**: Gaussian covariances in camera coordinate system. [C, N, 3, 3].
    """
    R = viewmats[:, :3, :3]  # [C, 3, 3]
    t = viewmats[:, :3, 3]  # [C, 3]
    means_c = torch.einsum("cij,nj->cni", R, means) + t[:, None, :]  # (C, N, 3)
    covars_c = torch.einsum("cij,njk,clk->cnil", R, covars, R)  # [C, N, 3, 3]
    return means_c, covars_c


def spawn_gaussians(device, lst_of_dct, num_gaussians=2):
    """
    Setting up camera params and returning num_gaussians with similar initial conditions.
    Sets up the fixed parameters for the neural ODE (i.e. parameters that don't 
    appear in the keys of dict_of_fns)
    """
    viewmats = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 8.0],
            [0.0, 0.0, 0.0, 1.0],
        ], # the translation vector controls the depth of view
        device=device,
    )
    fov_x = math.pi / 2.0
    height, width = 800,800 
    focal = 0.5 * float(width) / math.tan(0.5 * fov_x)

    Ks = torch.tensor(
    [
        [focal, 0, width / 2],
        [0, focal, height / 2],
        [0, 0, 1],
    ],
    device=device,
    )
    #NOTE: make sure these keys are the same arguments as the rasterization function.
    default_params = {
        "means": torch.tensor([[5.0, 5.0, 0.0], [0.0,0.0,0.0]], device=device),
        "quats": torch.tensor([[1, 0, 0, 0]], device=device, dtype=torch.float32).repeat(num_gaussians, 1), #(N, 4)
        "scales": torch.tensor([[1., 1., 1.]], device=device).repeat(num_gaussians, 1), #(N, 3)
        "opacities": torch.tensor(1.0, device=device).repeat(num_gaussians), #(N,)
        "colors": torch.tensor([[1.0, 0.0, 0.0]], device=device).repeat(num_gaussians, 1), #(N,3)
        "Ks": Ks[None], #(C, 3, 3)
        "viewmats": viewmats[None], #(C, 4, 4)
        "height": height,
        "width": width
    }

    # Filter out keys that appear in dict_of_fns 
    if len(lst_of_dct) != 0:
        gaussian_params = {
            key: value 
            for key, value in default_params.items() 
            if key not in lst_of_dct[0] #just check the first gaussian -- good enough 
        }
    else:
        gaussian_params = {
            key: value 
            for key, value in default_params.items() 
        }

    return gaussian_params

def quat_mult(q1, q2):
    """
    Performs quaternion multiplication
    """
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T

def o3d_knn(pts, num_knn):
    """
    Find num_knn neighbors of 3D points.
    """
    indices = []
    sq_dists = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for p in pcd.points:
        [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
        indices.append(i[1:])
        sq_dists.append(d[1:])
    return np.array(sq_dists), np.array(indices)


def calculate_neighbor_weights(initial_params, num_knn, lamb=2000):
    """
    Given initial params at timestep 0, and num_knn desired number of neighbors, calculate the neighbor weights.
    Args:
    -initial_params: initial gaussian parameters learned in the coarse stage. (N, feature_dim)
    -num_knn: int

    return:
    -neighbor_weight, the unnormalized weight 
    """
    init_fg_pts = initial_params[:,:3]
    neighbor_sq_dist, neighbor_indices = o3d_knn(init_fg_pts.detach().cpu().numpy(), num_knn) #excludes itself as neighbor
    neighbor_weights = np.exp(-lamb * neighbor_sq_dist) #(N, num_knn)
    neighbor_dists = np.sqrt(neighbor_sq_dist) #(N, num_knn)
    return neighbor_weights, neighbor_indices, neighbor_dists


def create_optimization_progress_video(timestep_renders, output_path, fps=30):
    """
    Creates separate videos for each timestep, showing ground truth on left and render progression on right.
    
    Args:
        timestep_renders: Dictionary with timestep as key and list of render data as values
        output_dir: Directory to save the output videos
        fps: Frames per second for the output videos
    """
    if not timestep_renders:
        print("No timestep renders data provided.")
        return
    
    # Create output directory if it doesn't exist
    
    # Process each timestep separately
    for timestep in sorted(timestep_renders.keys()):
        # Skip if no renders for this timestep
        if not timestep_renders[timestep]:
            print(f"No renders found for timestep {timestep}")
            continue
        
        # Sort renders by optimization step
        sorted_renders = sorted(timestep_renders[timestep], key=lambda x: x["current_step"])
        
        # Get dimensions from first render
        first_render = sorted_renders[0]
        gt_img = first_render["gt_img"]
        h, w, c = gt_img.shape
        
        # Create video file name
        video_path = os.path.join(output_path, f"timestep_{timestep:.4f}.mp4")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, fps, (w*2, h))
        
        # Find min and max optimization steps for this timestep
        min_step = min(r["current_step"] for r in sorted_renders)
        max_step = max(r["current_step"] for r in sorted_renders)
        
        print(f"Creating video for timestep {timestep} with {len(sorted_renders)} frames...")
        
        # Process each render for this timestep
        for render_data in tqdm(sorted_renders):
            rendered_img = render_data["rendered_img"]
            gt_img = render_data["gt_img"]
            current_step = render_data["current_step"]
            psnr = render_data["psnr"].item()
            
            # Create side-by-side comparison
            combined_img = np.hstack([gt_img, rendered_img])

            if c == 3:  # Only convert if it's a 3-channel image
                combined_img = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR) 

            # Add text overlays
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Left side - Ground Truth
            cv2.putText(combined_img, "Ground Truth", (10, 30), font, 0.8, (255, 255, 255), 2)
            
            # Right side - Current Render
            cv2.putText(combined_img, "Current Render", (w + 10, 30), font, 0.8, (255, 255, 255), 2)
            
            # Optimization info
            cv2.putText(combined_img, f"Timestep: {timestep:.4f}", (10, 70), font, 0.8, (255, 255, 255), 1)
            cv2.putText(combined_img, f"Step: {current_step}/{max_step}", (10, 100), font, 0.8, (255, 255, 255), 1)
            
            cv2.putText(combined_img, f"PSNR: {psnr:.2f} dB", (10, 130), font, 0.8, (255, 255, 255), 1)
            # Progress bar at the bottom
            # progress = (current_step - min_step) / (max_step - min_step) if max_step > min_step else 1.0
            # bar_length = int(progress * w * 2)
            # cv2.rectangle(combined_img, (0, h-10), (bar_length, h-5), (0, 255, 0), -1)
            
            # Write frame to video
            video.write(combined_img)
        
        # Release video writer
        video.release()
        print(f"Video saved to {video_path}")
    
    print(f"Created {len(timestep_renders)} videos in {output_path}")




if __name__ == "__main__":
    gaussian_params = spawn_gaussians("cuda", [])
    #1. first build 3D covar using _quat_scale_to_covar_preci
    covars3d = _quat_scale_to_covar_preci(gaussian_params["quats"],gaussian_params["scales"])[0]
    #2. convert means and covar to camera coords
    means2d, cov2d, conics = _fully_fused_projection(gaussian_params["means"], covars3d, gaussian_params["viewmats"], gaussian_params["Ks"],
                                             gaussian_params["width"], gaussian_params["height"])
    renders,_,meta = rasterization(**gaussian_params)
    # assert (meta["means2d"] == means2d).all()
    # assert (meta["conics"] == conics).all()
    img = renders[0]
    plt.imshow(img.cpu().numpy())
    plt.axis("off")
    plt.savefig("image")
    plt.close()
    print(f"the gaussians 2d positions and covariances are {means2d}\n {cov2d}")