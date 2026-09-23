"""Configuration shared by synthetic and captured GrowFlow pipelines."""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Union

from gsplat.strategy import DefaultStrategy, MCMCStrategy
from typing_extensions import assert_never


@dataclass
class CommonConfig:
    """Shared CLI contract for training, rendering, and evaluation.

    Dataset-specific config classes inherit from this class and override only the
    defaults that genuinely differ. Keeping these fields flat preserves the Tyro
    flags used by the submitted commands.
    """

    # Runtime and checkpoints.
    disable_viewer: bool = True
    static_ckpt: Optional[List[str]] = None
    per_timestep_static_ckpt_dir: Optional[List[str]] = None
    per_segment_ckpt_dir: str = ""
    dynamic_ckpt: Optional[List[str]] = None
    compression: Optional[Literal["png"]] = None
    render_traj_path: str = "360"
    video_duration: int = 3
    time_normalize_factor: float = 1
    include_zero: bool = False
    upper_bound_exp: bool = False
    per_time_upper_bound: Optional[int] = None
    version: Literal["ours", "4dgs"] = "ours"
    previous_init_params_path: str = ""

    # Dataset and temporal sampling.
    data_dir: str = (
        "./data/dynamic/blender/360/multi-view/31_views/"
        "rose_transparent_final_small_vase_70_timesteps"
    )
    combine_train_test: bool = False
    is_reverse: bool = False
    train_time_index: list = field(default_factory=list)
    temp_batch_size: int = 1
    cam_batch_size: int = -1
    downsample_factor: float = 1
    shuffle_ind: bool = True

    # Progressive trajectory training.
    use_progressive_training: bool = False
    use_progressive_training_two: bool = False
    progressive_training_intervals: List[int] = field(default_factory=list)
    progressive_batch_size: bool = True
    progressive_option: Literal["uniform", "linear", "non-linear", "d3dgs"] = (
        "d3dgs"
    )
    base: float = 1.2
    num_train_all: int = 500
    min_iterations_req: int = 300
    scale_activation: str = "exp"
    num_knn: int = 20
    not_skip_encoder: bool = True
    not_skip_model: bool = True

    # Neural ODE.
    augment_dim: int = 0
    hidden_dim: int = 64
    hidden_depth: int = 3
    min_step_size: float = 1e-4
    encoding: str = "hexplane"
    bbox_expansion: bool = False
    use_timenet: bool = True
    ours_multires: list[int] = field(default_factory=lambda: [1, 2])
    spatial_temp_resolution: list[int] = field(
        default_factory=lambda: [64, 64, 64, 25]
    )
    non_linearity_name: str = "relu"
    unscaled_neural_ode_lr_init: float = 1.6e-4
    unscaled_encoder_lr_init: float = 1.6e-3
    gamma: float = 0.1
    adjust_lr_w_scene: bool = False
    concat_remaining: bool = True
    method: str = "dopri5"
    rtol: float = 1e-4
    atol: float = 1e-5
    adjoint: bool = True
    learn_pos: bool = True
    learn_quat: bool = True
    learn_scales: bool = True
    learn_shs: bool = False
    learn_opacities: bool = False
    use_tanh_act: bool = False
    resnet_init: bool = False
    encode_other: bool = True
    x_multires: int = 6
    time_conditioned: bool = False
    reset_adam: bool = True
    reverse_scheduler: bool = False
    scheduler_train_all: bool = False
    load_optimizers: bool = True
    use_skip: bool = False
    mixed_init_training: bool = True
    num_init_conditions: int = 1
    full_trajectory_path: str = ""
    image_supervision: bool = True
    geometry_supervision: bool = False
    reset_ode: bool = True
    skip_static_eval: bool = True
    global_integration_interval: int = 50
    global_integration_start: int = 5000
    compute_tv_loss_ours: bool = False
    plane_tv_weight_ours: float = 0.0001
    time_smoothness_weight_ours: float = 0.01
    l1_time_planes_weight_ours: float = 0.0001
    learn_masks: bool = False
    learn_masks_from: int = 3000
    masks_reg: float = 0.1
    mask_threshold: float = 0.5
    cache_trajectory: bool = True
    cache_trajectory_split: bool = False

    # Instant-NGP encoder options.
    ingp_otype: str = "HashGrid"
    ingp_n_levels: int = 16
    ingp_n_feat_per_lvl: int = 2
    ingp_log2_hashmap_size: int = 15
    ingp_base_resolution: int = 16
    ingp_per_level_scale: int = 1.5

    # 4DGS baseline options.
    net_width: int = 64
    timebase_pe: int = 4
    defor_depth: int = 1
    posebase_pe: int = 10
    scale_rotation_pe: int = 2
    opacity_pe: int = 2
    timenet_width: int = 64
    timenet_output: int = 32
    bounds: float = 1.6
    compute_tv_loss: bool = False
    plane_tv_weight: float = 0.0001
    time_smoothness_weight: float = 0.01
    l1_time_planes_weight: float = 0.0001
    multires: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    no_dx: bool = False
    no_grid: bool = False
    no_ds: bool = False
    no_dr: bool = False
    no_do: bool = True
    no_dshs: bool = True
    empty_voxel: bool = False
    grid_pe: int = 0
    static_mlp: bool = False
    apply_rotation: bool = False

    # Data loading and output.
    data_type: Literal["colmap", "blender"] = "blender"
    data_factor: int = 4
    result_dir: str = ""
    test_every: int = 8
    patch_size: Optional[int] = None
    global_scale: float = 1.0
    normalize_world_space: bool = True
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"
    downsample_eval: bool = True
    target_shape: Tuple[int, int] = (400, 400)

    # Debugging and scheduling.
    port: int = 65432
    debug: bool = True
    debug2: bool = False
    debug_nan: bool = False
    debug_every: int = 500
    viz_reference: bool = False
    debug_eval: bool = False
    batch_size: int = 1
    steps_scaler: float = 1.0
    load_from_cfg: str = ""
    test_folder: str = ""

    # Static reconstruction.
    static_max_steps: int = 30_000
    static_eval_steps: List[int] = field(
        default_factory=lambda: [
            1,
            3_500,
            7_000,
            15_000,
            23_000,
            30_000,
            50_000,
            70_000,
            90_000,
            100_000,
        ]
    )
    static_save_steps: List[int] = field(
        default_factory=lambda: [1, 7_000, 30_000, 100_000]
    )
    static_save_ply: bool = False
    static_ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # Dynamic optimization.
    dynamic_max_steps: int = 30_000
    dynamic_eval_steps: List[int] = field(
        default_factory=lambda: [
            15_000,
            20_000,
            25_000,
            27_000,
            30_000,
            60_000,
            80_000,
            100_000,
        ]
    )
    dynamic_save_steps: List[int] = field(
        default_factory=lambda: [
            1,
            15_000,
            20_000,
            25_000,
            30_000,
            50_000,
            40_000,
            60_000,
            80_000,
            100_000,
            120_000,
            140_000,
            160_000,
            180_000,
            200_000,
            250_000,
            270_000,
        ]
    )
    run_eval: bool = True
    resume_dyn_training: bool = False

    # Gaussian initialization and static optimization.
    init_type: Literal["sfm", "random", "blender_pts"] = "blender_pts"
    init_num_pts: int = 100_000
    init_extent: float = 0.5
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    init_opa: float = 0.1
    init_scale: float = 1.0
    ssim_lambda: float = 0.2
    num_vertices_sampled: int = 2_500_000
    return_mask: bool = False
    use_masked_loss: bool = False
    use_masked_loss_v2: bool = False
    near_plane: float = 0.01
    far_plane: float = 1e10
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    packed: bool = False
    sparse_grad: bool = False
    visible_adam: bool = False
    antialiased: bool = False
    use_absgrad: bool = False
    random_bkgd: bool = False
    bkgd_color: List[int] = field(default_factory=lambda: [0, 0, 0])

    # Regularization.
    opacity_reg: float = 0.0
    scale_reg: float = 0.0
    isometry_reg: float = 0.0
    local_isometry_reg: float = 0.0
    rigid_reg: float = 0.0
    monotonic_lambda: float = 0.0
    gravity_reg: float = 0.0
    boundary_condition_reg: float = 0.0
    velocity_reg: float = 0.0
    acceleration_reg: float = 0.0
    scale_acceleration_reg: float = 0.0
    scale_velocity_reg: float = 0.0
    chamfer_reg: float = 0.0
    static_chamfer_reg: float = 0.0
    chamfer_num_points: int = 10_000

    # Camera, appearance, bilateral-grid, and depth optimization.
    pose_opt: bool = False
    pose_opt_lr: float = 1e-5
    pose_opt_reg: float = 1e-6
    pose_noise: float = 0.0
    app_opt: bool = False
    app_embed_dim: int = 16
    app_opt_lr: float = 1e-3
    app_opt_reg: float = 1e-6
    use_bilateral_grid: bool = False
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)
    depth_loss: bool = False
    depth_lambda: float = 1e-2

    # Logging and evaluation.
    use_wandb: bool = False
    wandb_every: int = 100
    wandb_save_image: bool = False
    track_path: str = ""
    render_white: bool = False
    use_bounding_box: bool = True
    render_tracks: bool = True
    tracking_window: int = 5
    skip_train: bool = True
    skip_test: bool = False
    animate_pc: bool = True
    skip_4dgs: bool = False
    skip_4dgaussians: bool = False
    skip_dynamic3dgs: bool = False
    skip_upper_bound: bool = False
    skip_rendering: bool = False
    task_name: str = "dense_supervision"
    existing_result_path: str = ""

    def adjust_steps(self, factor: float) -> None:
        """Scale static-training and densification schedules."""
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
