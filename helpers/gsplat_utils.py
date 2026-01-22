import os
import imageio
import math
import numpy as np
import torch
from torch import Tensor
from typing import Dict
import torch.nn.functional as F
from typing import Optional, Tuple
from gsplat.optimizers import SelectiveAdam
from sklearn.neighbors import NearestNeighbors
from gsplat.strategy import DefaultStrategy, MCMCStrategy
import open3d as o3d
import trimesh
from helpers.pc_viz_utils import visualize_point_cloud
import random


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def knn(x: Tensor, K: int = 4) -> Tensor:
    """
    Return K closest neighbor distances from x.
    """
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)

def o3d_knn(pts, num_knn):
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

def create_splats_with_optimizers(
    parser, 
    first_mesh_path,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    deformed_params_list: list[str] = ["means", "scales", "quats"], #start with these basic ones.
    device: str = "cuda",
    learn_mask: bool=False
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    """
    Initialize the gaussians with specific initialization scheme.
    Furthermore, initialize the optimizers.
    Return:
    gaussians: ParameterDict
    optimizers: Dict[name: optimizer]
    param_feature_dim: int
    """

    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
        # visualize_point_cloud(points, color=rgbs)
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        print(f"points are initialized in [{points.min().item()}, {points.max().item()}]")
        rgbs = torch.rand((init_num_pts, 3))
    elif init_type == "blender_pts": #sample from mesh instead of using mesh vertices
        mesh = trimesh.load_mesh(first_mesh_path, process=True) #NOTE: must set process=True, otherwise duplicate gaussians
        vpos, _,_ = trimesh.sample.sample_surface(mesh, init_num_pts, sample_color=True)
        points = torch.tensor(vpos, dtype=torch.float32, device="cuda")
        # points = torch.from_numpy(mesh.vertices).to(torch.float32)
        # colors = colors/255.0
        # visualize_point_cloud(points)
        # rgbs = torch.tensor(colors, dtype=torch.float32, device="cuda")
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random or blender_pts")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    assert torch.isfinite(points).all(), "means is nan/inf"
    assert torch.isfinite(scales).all(), "scales is nan/inf"
    assert torch.isfinite(quats).all(), "quats is nan/inf"
    assert torch.isfinite(rgbs).all(), "rgbs is nan/inf"
    assert torch.isfinite(opacities).all(), "opacities is nan/inf"
    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3] 
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    if learn_mask:
        masks = torch.zeros((points.shape[0], 1), dtype=torch.float32, device="cuda")
        params.append(("masks", torch.nn.Parameter(masks), 2.5e-3))
        
    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    deformed_params_dict = {}
    param_feature_dim = 0 #use for neural ode initialization

    #If deforming colors, just store the whole spherical harmonic colors, don't do it separately
    for param in splats.keys():
        if param in deformed_params_list: #this adds everything except colors
            if param == "opacities": # the default opacities has dim -1 being 1
                param_feature_dim += 1
                deformed_params_dict[param] = 1
            else:
                param_feature_dim += splats[param].shape[-1]
                deformed_params_dict[param] = splats[param].shape[-1]

    if "shs" in deformed_params_list: #add the color shsape directly
        N = splats["means"].shape[0]
        reshaped_sh0 = splats["sh0"].reshape(N, -1)
        reshaped_shN = splats["shN"].reshape(N, -1)
        colors_shape = reshaped_sh0.shape[-1] + reshaped_shN.shape[-1]
        deformed_params_dict["shs"] = colors_shape
        param_feature_dim += colors_shape

    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(batch_size), "name": name}],
            eps=1e-15 / math.sqrt(batch_size),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers, param_feature_dim, deformed_params_dict

def prepare_times(gaussians, config, train_dataset, fixed_initial_params, deformed_params_dict, deformed_params_list, percent_train=1, out_dir="./"):
    """
    Prepare continuous times and train time indices for the fine training and the raster parameters
    """
    #1. Figure out how many timesteps there are in the dataset.
    # num_timesteps = train_dataset.num_timesteps()
    # int_times = torch.arange(0, num_timesteps) #integer times from 0 to number of images, for indexing
    # cont_times = int_times/(num_timesteps-1) #continuous times from 0 to 1 for integration
    # available_time_targets = int_times[1:]
    # train_time_index = available_time_targets

    #respect json dataloader...
    cont_times = torch.tensor(train_dataset.times)  # The times you stored in dataloader
    train_time_index = torch.arange(1, len(cont_times))  # [1, 2, 3, ..., len-1]
    config.train_time_index = torch.cat((torch.tensor([0]), train_time_index)).tolist()
    if config.data_type == "blender":
        raster_params = get_raster_params_blender(config, gaussians,train_dataset, deformed_params_dict)
    elif config.data_type == "colmap":
        raster_params = get_raster_params_captured(config, gaussians, train_dataset, deformed_params_dict)
    check_fixed_initial_params(gaussians, fixed_initial_params, deformed_params_dict, deformed_params_list) #check whether the order 
    print(f"the training indices are {train_time_index.tolist()}")
    # if percent_train <1: #if we use less than all samples, save a video showing in increasing order the samples we are using
    #     save_video(gt_images, train_time_index, out_dir)
    return train_time_index, cont_times, raster_params

def check_fixed_initial_params(gaussians, fixed_initial_params, deformed_params_dict, deformed_params_list):
    """
    Check if the order in the fixed_initial_params matches that of the deformed_params_dict
    NOTE: we need to iterate over deformed_params_list since the order of keys are not preserved in deformed_params_dict
    """
    print("checking if the order in fixed initial params matches that of the deformed_params_dict") 
    index = 0 
    for k in deformed_params_list:
        v = deformed_params_dict[k]
        if k != "shs":
            assert (torch.squeeze(fixed_initial_params[:, index: index+v]) == gaussians[k].data).all()
            index += v
        else:
            N = gaussians["means"].shape[0]
            concatenated_shs = torch.cat((gaussians["sh0"].reshape(N, -1), gaussians["shN"].reshape(N, -1)), dim=-1)
            assert (fixed_initial_params[:, index: index+v] == concatenated_shs).all()
            index += v

    print("check complete")

def get_raster_params_blender(cfg, gaussians, dataset, deformed_params_dict):
    """
    Given deformed_params_dict, select the un-deformed gaussian parameters and 
    return a set of fixed raster parameters. 
    """
    raster_params = {}
    first_timestep = dataset[0]
    num_cameras = len(dataset.images.keys()) if dataset.cam_batch_size == -1 else dataset.cam_batch_size
    raster_params["Ks"] = first_timestep["K"][None].expand(num_cameras, -1, -1).to("cuda") #(C,3,3)
    raster_params["width"] =dataset.image_width
    raster_params["height"] = dataset.image_height
    raster_params["near_plane"] = cfg.near_plane
    raster_params["far_plane"] = cfg.far_plane
    raster_params["render_mode"] = "RGB"
    raster_params["sh_degree"] = cfg.sh_degree #this is fine since get_raster_params is called during dyn train, which all deg have been act
    raster_params["camera_model"] = cfg.camera_model
    raster_params["rasterize_mode"] = "antialiased" if cfg.antialiased else "classic"
    raster_params["absgrad"] = cfg.strategy.absgrad if isinstance(cfg.strategy, DefaultStrategy) else False
    raster_params["backgrounds"] = torch.tensor(cfg.bkgd_color, device="cuda")[None].expand(num_cameras, -1).to(torch.float32)
    #NOTE: for c2w, we append them during training
    for k,v in gaussians.items():
        if k == "masks":
            continue #dont want to store that in the deformed params dict
        if k not in ["sh0", "shN"]:
            if k not in deformed_params_dict:  #here we first test by freezing opacities and colors
                v.requires_grad_(False)
                raster_params[k] = v
        else:
            if "shs" not in deformed_params_dict: #spherical harmonics is stored as shs in deformed_params_dict
                v.requires_grad_(False)
                raster_params[k] = v

    if "sh0" in raster_params and "shN" in raster_params: #replacing sh0, shN with colors
        raster_params["colors"] =  torch.cat([raster_params["sh0"], raster_params["shN"]], 1)  # [N, K, 3]
        del raster_params["sh0"]
        del raster_params["shN"]

    return raster_params




def get_raster_params_captured(cfg, gaussians, dataset, deformed_params_dict):
    """
    Given deformed_params_dict, select the un-deformed gaussian parameters and 
    return a set of fixed raster parameters. 
    
    For the raster_params, we are most likely not going to sample all cameras in one batch so set it equal
    to the camera batch size.
    """
    raster_params = {}
    first_timestep = 0
    first_image_idx = sorted(dataset.timestep_images[0].keys())[0] #selects first camera index
    #NOTE: this code would only return one intrinsics when cam_batch_size = -1
    if dataset.split == "train": #need to use camera_filter here to select the first training camera
        num_cameras_in_batch = len(dataset.camera_filter[0]) if dataset.cam_batch_size == -1 else dataset.cam_batch_size
    else:
        num_cameras_in_batch = 1 #keep it 1 so we can expand properly 

    first_image_shape = dataset.timestep_images[first_timestep][first_image_idx] #()
    height, width = first_image_shape.shape[0], first_image_shape.shape[1]
    raster_params["Ks"] = torch.from_numpy(dataset.timestep_intrinsics[first_timestep][1][None]).expand(num_cameras_in_batch, -1, -1) #(num_cam, 3, 3)
    raster_params["width"] = width
    raster_params["height"] = height
    raster_params["near_plane"] = cfg.near_plane
    raster_params["far_plane"] = cfg.far_plane
    raster_params["render_mode"] = "RGB"
    raster_params["sh_degree"] = cfg.sh_degree #this is fine since get_raster_params is called during dyn train, which all deg have been act
    raster_params["camera_model"] = cfg.camera_model
    raster_params["rasterize_mode"] = "antialiased" if cfg.antialiased else "classic"
    raster_params["absgrad"] = cfg.strategy.absgrad if isinstance(cfg.strategy, DefaultStrategy) else False
    raster_params["backgrounds"] = (
        torch.tensor(cfg.bkgd_color, device="cuda")[None]
        .expand(num_cameras_in_batch, -1)
        .to(torch.float32)
    )
    #NOTE: for c2w, we append them during training
    #TODO: code below is redundant, we already freeze the gaussians.
    for k,v in gaussians.items():
        if k != "masks":
            if k not in ["sh0", "shN"]:
                if k not in deformed_params_dict:  #here we first test by freezing opacities and colors
                    v.requires_grad_(False)
                    raster_params[k] = v
            else:
                if "shs" not in deformed_params_dict: #spherical harmonics is stored as shs in deformed_params_dict
                    v.requires_grad_(False)
                    raster_params[k] = v

    if "sh0" in raster_params and "shN" in raster_params: #replacing sh0, shN with colors
        raster_params["colors"] =  torch.cat([raster_params["sh0"], raster_params["shN"]], 1)  # [N, K, 3]
        del raster_params["sh0"]
        del raster_params["shN"]

    return raster_params


def save_video(img_lst, train_time_index, path):
    """
    Convert img_lst as a torch tensor, index it by train_time_index and time 0 and save to path
    """
    fps=21
    img_tensor = torch.stack(img_lst, dim=0) #(t, h, w, 3)
    img_frames = img_tensor[torch.cat((torch.tensor([0]), train_time_index))]
    frames = [(frame.cpu().numpy()*255).astype(np.uint8) for frame in img_frames]
    img_file = os.path.join(path, "gt_samples.mp4")
    imageio.mimwrite(img_file, frames, fps=fps)


def create_batch(temp_batch_size, train_time_index, train_dataset, cont_times, limit=None, piecewise_ode=False):
    """
    Create a training batch of data to be fed to the neural ode based on gt_images and cont_times
    Return time input to the neural ode, the image batch
    Args:
    -temp_batch_size: int
    -train_time_index: an array of training time indices
    -train_dataset: Dataset object containing images across cameras/times
    -cont_times: normalized times (t,)

    Return:
    -inp_t: [b+1,] 
    -image_batch: [b, h, w, 3] or [h,w,3] if single image
    """
    #1. Need to get all cameras for all timesteps
    all_times = range(0, train_dataset.num_timesteps())
    c2ws, gt_images, _ = train_dataset.__getitems__(all_times) #c2ws: [N, 4, 4], gt_images: [N, T, H, W, 3]

    #sample a batch of indices 
    initial_index = 0 
    num_elements = len(train_time_index)
    probs = torch.ones(num_elements) / num_elements #generate probabilities 
    batch_indices = torch.multinomial(probs, temp_batch_size, replacement=False) #sampling in {0...num_elements-1}

    # Select the exact training time
    train_indices, _ = train_time_index[batch_indices].sort(descending=False) 

    if limit is not None:
        if piecewise_ode:
            if limit == 1:
                #If limit is 1, then we do 0 -> 1, else we do 1->2, 2->3 etc.
                train_indices = train_indices[0]
                inp_t = torch.cat((cont_times[initial_index][None], cont_times[train_indices][None]), dim=0).to("cuda")
            elif limit > 1:
                initial_index = limit - 1
                train_indices = train_indices[initial_index]
                inp_t = torch.cat((cont_times[initial_index][None], cont_times[train_indices][None]), dim=0).to("cuda")
            else:
                raise ValueError("limit is below 1")
        else:
            train_indices = train_indices[:limit]
            inp_t = torch.cat((cont_times[initial_index][None], cont_times[train_indices]), dim=0).to("cuda")
    else: 
        #in non-progressive training scenarios, we still append 0.
        inp_t = torch.cat((cont_times[initial_index][None], cont_times[train_indices]), dim=0).to("cuda")

    # Get an image batch
    image_batch = gt_images[:, train_indices] #(N, len(train_indices), H, W, 3)

    # Sample some time, must be 1D and start with 0.0
    assert inp_t.dim() == 1 #the time input to the neural ode has to be 1 dimensional
    image_batch = image_batch.to("cuda") 
    image_batch = torch.squeeze(image_batch) 
    return inp_t, image_batch, c2ws

def map_cont_to_int(cont_t, num_images, factor=1):
    """
    Logic: int_t = (cont_t/factor) * (num_images-1)
    """
    int_t = (cont_t /factor)* (num_images - 1)
    int_t = torch.round(int_t).to(torch.int32) 
    return int_t

def save_gt_video(cfg, video, cam_index, full_out_path, is_reverse):
    """
    Given an array of frames, and cam index, save that video to full_out_path
    """
    #no need to background blend
    selected_video = video[cam_index]
    fps = selected_video.shape[0]/cfg.video_duration
    if is_reverse:
        selected_video = selected_video.flip(0)
    imageio.mimwrite(full_out_path, (selected_video.detach().cpu().numpy()*255).astype(np.uint8), fps=fps)


def _persp_proj(
    means,  # [C, N, 3]
    covars,  # [C, N, 3, 3]
    Ks,  # [C, 3, 3]
    width,
    height,
):
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




def create_reverse_map_dict(total_num_timesteps, offset=0):
    """
    Create a reverse map dict to map frames with an offset from the end.
    For instance if given total_num_timesteps = 35 and offset = 0, then map:
    00000.png <-> 00034.png, 00001.png <-> 00033.png, etc.
    
    If offset is 5, then map:
    00000.png <-> 00029.png, 00001.png <-> 00030.png, 00002.png <-> 00031.png, etc.
    """
    result = {}
    last_idx = total_num_timesteps - 1  # Last valid index (e.g., 34 for 35 frames)
    
    for i in range(total_num_timesteps):
        file_name = f"{i:05d}.png"
        
        # For offset 0: map 0->34, 1->33 (symmetric from ends)
        # For offset 5: map 0->29, 1->30, 2->31 (start 5 from end, then increment)
        if offset == 0:
            # Original symmetric mapping
            mapped_number = last_idx - i
        else:
            # Start from (last_idx - offset) and increment
            mapped_number = (last_idx - offset) + i
            
        # Ensure we don't exceed valid indices
        if 0 <= mapped_number < total_num_timesteps:
            mapped_file_name = f"{mapped_number:05d}.png"
            result[file_name] = mapped_file_name
    
    return result




def find_closest_gauss(gt, gauss, return_unique=False, batch_size=1024):
    gt = torch.tensor(gt, dtype=torch.float32)
    gauss = torch.tensor(gauss, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt = gt.to(device)
    gauss = gauss.to(device)

    indices = []

    for i in range(0, len(gt), batch_size):
        gt_batch = gt[i:i+batch_size]  # (B, 3)
        dists = torch.cdist(gt_batch, gauss)  # (B, N_gauss)
        min_idx = torch.argmin(dists, dim=1)  # (B,)
        indices.append(min_idx)

    indices = torch.cat(indices, dim=0).cpu()

    if return_unique:
        return torch.unique(indices)
    else:
        return indices

    
    
def reset_adam_states(optimizers):
    """
    Reset adam states for each optimizer in optimizers
    """
    for optimizer_name, optimizer in optimizers.items():
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param in optimizer.state:
                    state = optimizer.state[param]
                    if 'exp_avg' in state:
                        state['exp_avg'] = torch.zeros_like(param)
                    if 'exp_avg_sq' in state:
                        state['exp_avg_sq'] = torch.zeros_like(param)

                        
                        
def create_dataloader_during_prog(trainset, lst_of_prog_indices, shuffle_ind, temp_batch_size, mixed_init_training=False, num_init_conditions=4):

    from torch.utils.data import DataLoader
    from datasets.sampler import InfiniteNeuralODEDataSampler, NeuralODEDataSampler_MixedInit
    if mixed_init_training:
        print(f"initializing the data sampler with {num_init_conditions} number of initials") 
        prog_sampler = NeuralODEDataSampler_MixedInit(
            trainset, lst_of_prog_indices, seq_length=temp_batch_size, 
            num_initial_conditions = num_init_conditions
        )  # we use the sampler only during training
        full_batch_size = temp_batch_size * num_init_conditions
        trainloader = DataLoader(
                trainset, 
                batch_size=full_batch_size,
                shuffle=False,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                sampler=prog_sampler,
                collate_fn=trainset.custom_collate_fn
            )
    else:
        prog_sampler = InfiniteNeuralODEDataSampler(
            trainset,
            lst_of_prog_indices,
            shuffle = shuffle_ind
        )
        # temp_batch_size = temp_batch_size if cfg.progressive_batch_size else len(lst_of_prog_indices) 
        temp_batch_size = min(len(lst_of_prog_indices), temp_batch_size)
        trainloader = DataLoader(
                trainset, 
                batch_size=temp_batch_size,
                shuffle=False,
                num_workers=0,
                persistent_workers=False,
                pin_memory=True,
                sampler=prog_sampler,
                collate_fn=trainset.custom_collate_fn
            )
    trainloader_iter = iter(trainloader)
    return trainloader_iter


def world_to_cam_means(
    means,
    viewmats
):
    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]
    assert means.shape == batch_dims + (N, 3), means.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape

    R = viewmats[..., :3, :3]  # [..., C, 3, 3]
    t = viewmats[..., :3, 3]  # [..., C, 3]
    means_c = (
        torch.einsum("...cij,...nj->...cni", R, means) + t[..., None, :]
    )  # [..., C, N, 3]
    return means_c

    
    
def pers_proj_means(
    means: Tensor,  # [..., C, N, 3]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
) -> Tensor:
    """PyTorch implementation of perspective projection for 3D Gaussians."""
    batch_dims = means.shape[:-3]
    C, N = means.shape[-3:-1]
    assert means.shape == batch_dims + (C, N, 3), means.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape
    
    tx, ty, tz = torch.unbind(means, dim=-1)  # [..., C, N]
    
    # Extract intrinsic parameters
    fx = Ks[..., 0, 0, None]  # [..., C, 1]
    fy = Ks[..., 1, 1, None]  # [..., C, 1]
    cx = Ks[..., 0, 2, None]  # [..., C, 1]
    cy = Ks[..., 1, 2, None]  # [..., C, 1]
    
    # Calculate field of view limits
    tan_fovx = 0.5 * width / fx  # [..., C, 1]
    tan_fovy = 0.5 * height / fy  # [..., C, 1]
    lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
    lim_x_neg = cx / fx + 0.3 * tan_fovx
    lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
    lim_y_neg = cy / fy + 0.3 * tan_fovy
    
    # Clamp to avoid extreme projections
    tx = tz * torch.clamp(tx / tz, min=-lim_x_neg, max=lim_x_pos)
    ty = tz * torch.clamp(ty / tz, min=-lim_y_neg, max=lim_y_pos)
    
    # Project to 2D
    means2d = torch.einsum(
        "...ij,...nj->...ni", Ks[..., :2, :3], torch.stack([tx, ty, tz], dim=-1)
    )  # [..., C, N, 2]
    means2d = means2d / tz[..., None]  # [..., C, N, 2]
    
    return means2d

    
    
def build_rotation(q):
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3), device='cuda')
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T
    
def save_point_cloud_to_ply(points: torch.Tensor, filename: str):
    """
    Save a point cloud to a .ply file.
    
    Args:
        points: torch.Tensor of shape (N, 3) or (N, 6) — xyz or xyz+rgb
        filename: output path (e.g., 'output.ply')
    """
    assert points.ndim == 2 and points.shape[1] in [3, 6], "Expected shape (N, 3) or (N, 6)"
    points = points.detach().cpu().numpy()

    has_color = points.shape[1] == 6

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points.shape[0]}",
        "property float x",
        "property float y",
        "property float z"
    ]

    if has_color:
        header += [
            "property uchar red",
            "property uchar green",
            "property uchar blue"
        ]

    header.append("end_header")

    with open(filename, 'w') as f:
        f.write('\n'.join(header) + '\n')
        for p in points:
            xyz = p[:3]
            if has_color:
                rgb = (p[3:] * 255).clip(0, 255).astype(int)
                f.write(f"{xyz[0]} {xyz[1]} {xyz[2]} {rgb[0]} {rgb[1]} {rgb[2]}\n")
            else:
                f.write(f"{xyz[0]} {xyz[1]} {xyz[2]}\n")

