from dataclasses import dataclass, field
from typing import Optional, Literal, List, Union, Tuple 
from gsplat.strategy import DefaultStrategy, MCMCStrategy
import yaml
from typing_extensions import assert_never

@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = True 
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    static_ckpt: Optional[List[str]] = None
    #per timestep static
    per_timestep_static_ckpt_dir: Optional[List[str]] = None
    #per_segment_folder
    per_segment_ckpt_dir: str = ""
    # # Whether or not 
    # eval_static: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    dynamic_ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"
    # Video duration in secs, consistent across all saved videos
    video_duration: int = 3
    time_normalize_factor: float = 1 
    #whether to include image at t=0 
    include_zero: bool=False
    include_end: bool=False
    upper_bound_exp: bool = False #when using upper_bound_exp, this gaussians from current timestep will depend on previous timestep.
    per_time_upper_bound: Optional[int] = None #when using per_time_upper_bound, you're just training one gaussians.
    #ours is concatenate all params, ours2 is split architecture, ours3 is breaking time into segments, 4dgs is 4dgs, ours_hybrid is using hybrid representation
    version: Literal["ours", "4dgs"] = "ours"
    previous_init_params_path: str = ""
    align_timesteps: bool=False #whether to run ICP to align the timesteps.
    dates: Optional[List[str]] = None
    subsample_factor: int = 1 #subsampling the timesteps
    start_from: int=0 #where to start from, if 0 start from beginning, use positive integers
    end_until:int=0


    ## Neural ODE splatting config
    # Path to the dataset
    data_dir: str = "./data/dynamic/captured/pi_rose"

    #whether to combine train and test 
    combine_train_test:bool=False 
    #Whether or not the dataset is reversed or not, default to False since dataset is already flipped.
    is_reverse: bool = False 
    # training_time indices
    train_time_index: list = field(default_factory=list) #NOTE: setting this as cmd line args doesn't work yet
    # temporal batch_size for dynamic training
    temp_batch_size: int = 1 
    # camera batch size for dynamic training, if -1, use all cameras
    cam_batch_size: int = -1
    #Temporal downsampling
    downsample_factor: float = 1 #choose below 1, the code samples 1/downsample_factor
    #shuffle training timesteps
    shuffle_ind: bool=True
    #Use progressive training
    use_progressive_training: bool=False
    # This FREEZES earlier timesteps and trains specifically for later timesteps.
    use_progressive_training_two:bool =False
    # Progressive training intervals (only used if progressive_training is enabled) 
    progressive_training_intervals: List[int] = field(default_factory=list)
    # Whether to use a batch size of 1 during progressive training. NOTE: setting it to False leads to OOM
    progressive_batch_size: bool=True 
    # progressive_training options
    progressive_option: Literal["uniform", "linear", "non-linear", "d3dgs"] = "d3dgs"
    # Progressive training weighting
    base: float = 1.2
    # number of iterations where we unlock all timesteps.
    num_train_all: int = 500 
    min_iterations_req:int = 500
    scale_activation: str = "exp"
    #number of neighbors for rigid loss, 20 default val
    num_knn: int = 20
    not_skip_encoder: bool = True
    not_skip_model: bool = True
    
    #Neural ODE & friends
    augment_dim: int = 0
    hidden_dim: int = 256
    hidden_depth: int = 8
    min_step_size: float = 1e-4 #min_step_size to prevent underflow in adaptive methods
    encoding: str = "hexplane"
    bbox_expansion: bool=False
    use_timenet: bool=True #whether or not to use timenet for time encoding
    ours_multires: list[int] = field(default_factory=lambda:[1, 2])  # multi resolution of voxel grid
    spatial_temp_resolution :list[int] = field(default_factory=lambda:[64, 64, 64, 150])
    non_linearity_name: str = "relu" #relu performs slightly better but is more prone to underflow when training for longer. 
    feature_out_output_dim: int = 64 #the size of the feature mixer and the size of the mlp heads
    unscaled_neural_ode_lr_init:float= 1.6e-4
    unscaled_encoder_lr_init: float =1.6e-3 
    gamma: float=0.1
    adjust_lr_w_scene: bool = False #whether to adjust lr w.r.t scene
    concat_remaining:bool= True #whether or not to concatenate the remaining parameters
    method: str="dopri5"
    rtol: float = 1.e-4
    atol: float = 1.e-5
    adjoint: bool =True
    learn_pos: bool =True #learning position trajectory
    learn_quat: bool=True #learning quaternion trajectory
    learn_scales: bool=True #learning scale trajectory
    learn_shs: bool=False #learning colors trajectory
    learn_opacities: bool=False #learning opacity trajectory
    use_tanh_act:bool=False #whether to use tanh activation
    resnet_init:bool=False
    encode_other:bool=True #whether to encode other parameters
    x_multires: int=6
    time_conditioned: bool=False #whether to use frequency encoding for time.
    reset_adam: bool =True #whether to reset adam states when ur adding a new timestep
    reverse_scheduler: bool =False #whether to reverse the scheduler for the neural ODE
    scheduler_train_all:bool = False #apply scheduler train all
    load_optimizers:bool = True #whether to load optimizers when resume training
    use_skip:bool = False #whether to use skip connections in my neural ODE
    mixed_init_training:bool =True #whether to train the representation using mixed initial conditionsm, TODO: remember to turn this back on
    num_init_conditions:int = 1 #the number of initial conditions to use when doing mixed initial training
    full_trajectory_path: str="" #the path of the full trajectory
    image_supervision: bool = True #default, using image supervision
    geometry_supervision: bool = False #supervising on the geometry (pseudo gt trajectory)
    reset_ode: bool=True #whether to reset the neural ODE
    skip_static_eval:bool=True
    global_integration_interval: int=50 #how often to do global integration
    global_integration_start:int = 1000
    compute_tv_loss_ours:bool = False
    plane_tv_weight_ours: float = 0.0001  # TV loss of spatial grid
    time_smoothness_weight_ours: float = 0.01  # TV loss of temporal grid
    l1_time_planes_weight_ours: float = 0.0001  # TV loss of temporal grid
    use_bounding_box: bool=True
    use_mask_proj:bool =False
    use_mask_intersection:bool=False #False = union, true = intersection. NOTE: union will include more than the masked stuff
    dilation_iters: int = 10 
    apply_mask: bool=True
    use_own_impl: bool=True
    learn_masks: bool=True
    learn_masks_from: int =3000
    masks_reg: float=0.1
    cache_trajectory:bool =True 
    cache_trajectory_split:bool =False
    mask_threshold: float=0.5 #the threshold of the mask that we use to select the foreground gaussians.



    #ingp configs default values
    ingp_otype: str = "HashGrid"
    ingp_n_levels: int = 16
    ingp_n_feat_per_lvl: int=2
    ingp_log2_hashmap_size: int=15
    ingp_base_resolution:int=16
    ingp_per_level_scale:int=1.5
 
    ######################## 4dgs ###################################
    #TODO: only using this when using 4dgs model
    net_width: int = 64  # width of deformation MLP, larger will increase the rendering quality and decrease the training/rendering speed.
    timebase_pe: int = 4  # useless
    defor_depth: int = 1  # depth of deformation MLP, larger will increase the rendering quality and decrease the training/rendering speed.
    posebase_pe: int = 10  # useless
    scale_rotation_pe: int = 2  # useless
    opacity_pe: int = 2  # useless
    timenet_width: int = 64  # useless
    timenet_output: int = 32  # useless
    bounds: float = 1.6  # bounds parameter is a float
    compute_tv_loss: bool = False
    plane_tv_weight: float = 0.0001  # TV loss of spatial grid
    time_smoothness_weight: float = 0.01  # TV loss of temporal grid
    l1_time_planes_weight: float = 0.0001  # TV loss of temporal grid
    multires: list[int] = field(default_factory=lambda:[1, 2, 4, 8])  # multi resolution of voxel grid, note more multires than ours
    no_dx: bool = False  # cancel the deformation of Gaussians' position
    no_grid: bool = False  # cancel the spatial-temporal hexplane.
    no_ds: bool = False  # cancel the deformation of Gaussians' scaling
    no_dr: bool = False  # cancel the deformation of Gaussians' rotations
    no_do: bool = True  # cancel the deformation of Gaussians' opacity
    no_dshs: bool = True  # cancel dshs
    empty_voxel: bool = False  # useless
    grid_pe: int = 0  # useless, I was trying to add positional encoding to hexplane's features
    static_mlp: bool = False  # useless
    apply_rotation: bool = False  # useless
    ######################## 4dgs ###################################

    # Type of the dataset (e.g. COLMAP or Blender)
    data_type: Literal["colmap", "blender"] = "colmap"
    # Downsample factor for the dataset
    data_factor: int =1
    debug_data_loading:bool = False #if set to true, can use pdb to check each timestep's data
    crop_imgs: bool=False #NOTE: dont use this, messes up optimization, crop first
    use_crops: bool=False #the difference with above is you just use cropped images a priori
    use_bg_masks: bool=True#if using bg masks, u mask out like the plant + pot
    # Directory to save results 
    result_dir: str = ""
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"
    # Whether to downsample time for eval (allows us to visualize interpolation)
    downsample_eval: bool = True
    # image shape:
    target_shape: Tuple[int, int] = (400,400)
    use_dense: bool = False

    # Port for the viewer server
    port: int = 65432 
    # whether to be in debug mode 1, debug mode 1 just visualizes psnr over time compared to upper bound
    debug: bool = True 
    # whether to be in debug mode 2, debug mode 2 allows u to train a particular timestep to understand per-time deformation.
    debug2: bool = False
    # whether or not to debug for nan gradients
    debug_nan:bool =False
    debug_every:int = 100
    #whether we want to plot a reference plot (like when are we only training a subset and we want to see the full trajectory)
    viz_reference: bool = False
    # # we can do debug mode 1, but without showing upper
    # debug_no_upper: bool = False
    #whether to debug eval
    debug_eval:bool=False
    
    # Batch size for static training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0
    # Whether to load from config 
    load_from_cfg: str = ""
    # Test folder, only used in metrics
    test_folder: str =  ""

    ## Static training 
    # Number of training steps
    static_max_steps: int = 30_000 
    # Steps to evaluate the model
    static_eval_steps: List[int] = field(default_factory=lambda: [1, 3_500, 7_000, 15_000, 23_000, 30_000, 50_000, 70_000, 90_000, 100_000])
    # Steps to save the model
    static_save_steps: List[int] = field(default_factory=lambda: [1, 7_000, 30_000, 100_000])
    # Whether to save ply file (storage size can be large)
    static_save_ply: bool = False
    # Steps to save the model as ply
    static_ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    ## Dynamic training 
    # Number of training steps
    dynamic_max_steps: int = 30_000
    # Steps to evaluate the model
    dynamic_eval_steps: List[int] = field(default_factory=lambda: [1, 1000, 2000, 4000,  5000, 6000, 8000, 12000, 15_000, 30_000, 60_000, 80_000, 100_000])
    # # Steps to save the model
    dynamic_save_steps: List[int] = field(default_factory=lambda: [1, 1000, 3000, 5000, 7000, 9000, 10_000, 15_000, 30_000, 50_000, 40_000, 60_000, 80_000, 100_000, 120_000, 140_000, 160_000, 180_000, 200_000, 250_000, 270_000])
    run_eval: bool = True
    # Resume dynamic training
    resume_dyn_training: bool =False

    # Initialization strategy
    init_type: Literal["sfm", "random", "blender_pts"] = "sfm" #use random for blender, sfm for captured
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000 
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 0.5
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2
    # Number of vertices to sample on gt mesh, independent on number of vertices. The number of sampled vertices
    # need not be equal to the number of gaussian means.
    num_vertices_sampled: int = 2500000
    #monotonic scale loss
    #masked loss
    return_mask:bool = True
    use_masked_loss: bool = False 
    use_masked_loss_v2: bool= False
    viz_mask: bool = False #use this to visualize the mask
    
    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False
    # Use absgrad for pruning/splitting
    use_absgrad: bool = False

    # Use random background for training to encourage alpha consistency w/ source (helps with white floaters on image) 
    random_bkgd: bool = False 
    # Fixed background color to use w/ transparent source images for evaluation (and training if random_bkgd is False)
    bkgd_color: List[int] = field(default_factory=lambda: [0, 0, 0])

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0
    # Isometry regularization 
    isometry_reg: float = 0.0 #default to 0.3
    local_isometry_reg: float=0.0 
    #Rigid regularization
    rigid_reg: float = 0.0
    #boundary supervision loss
    param_loss_reg:float = 0.0
    #Monotonic scale regularization
    monotonic_lambda: float = 0.0 #default to 0.1
    #Gravity regularization
    gravity_reg: float=0.0
    #Boundary conditions regularization
    boundary_condition_reg:float =0.0
    velocity_reg:float=0.0
    acceleration_reg:float=0.0
    scale_acceleration_reg:float=0.0
    scale_velocity_reg:float=0.0
    chamfer_reg:float = 0.0 
    static_chamfer_reg:float = 0.0
    chamfer_num_points:int = 10_000

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    #Logging
    #Use wandb
    use_wandb: bool = True
    # Dump information to wandb every this steps
    wandb_every: int = 100
    # Save training images to wandb
    wandb_save_image: bool = False

    
    #Full eval settings, use for comparison against baselines
    track_path: str =""
    render_tracks: bool =False #TODO: enable this later...
    tracking_window: int=35
    render_white:bool =False
    skip_train: bool = True
    skip_test: bool = False 
    animate_pc:bool=True
    save_pc_imgs:bool=True
    render_spacetime_viz:bool=False
    render_demo_viz:bool=False
    skip_pc: bool=False
    skip_4dgs: bool =False
    skip_4dgaussians: bool=False
    skip_dynamic3dgs: bool=False
    skip_upper_bound: bool=False
    skip_rendering: bool=False
    interpolation_factor:int =1 #set this > 1 if u want to interpolate frames
    render_interpolation_frames:bool=False
    task_name: str = "dense_supervision" 
    existing_result_path: str = ""
    flip_x:bool=True
    flip_y:bool=False
    flip_z:bool=True
    use_mask_psnr:bool = True

    def adjust_steps(self, factor: float):
        """
        Adjust all steps (eval, saving, gaussians pruning) based on a multiplicative factor
        """
        self.static_eval_steps = [int(i * factor) for i in self.static_eval_steps]
        self.static_save_steps = [int(i * factor) for i in self.static_save_steps]
        self.static_ply_steps = [int(i * factor) for i in self.static_ply_steps]
        self.static_max_steps = int(self.static_max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)