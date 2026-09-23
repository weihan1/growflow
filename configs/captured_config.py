"""Configuration overrides for captured scenes."""

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from configs.common_config import CommonConfig


@dataclass
class Config(CommonConfig):
    """Captured-scene settings layered on the shared training config."""

    # Temporal selection and alignment.
    render_traj_path: str = "interp"
    include_end: bool = False
    align_timesteps: bool = False
    dates: Optional[List[str]] = None
    subsample_factor: int = 1
    start_from: int = 0
    end_until: int = 0

    # Captured-data defaults.
    data_dir: str = "./data/dynamic/captured/pi_rose"
    min_iterations_req: int = 500
    hidden_dim: int = 256
    hidden_depth: int = 8
    spatial_temp_resolution: list[int] = field(
        default_factory=lambda: [64, 64, 64, 150]
    )
    feature_out_output_dim: int = 64
    global_integration_start: int = 1000

    # Foreground-mask handling.
    use_mask_proj: bool = False
    use_mask_intersection: bool = False
    dilation_iters: int = 10
    apply_mask: bool = True
    use_own_impl: bool = True
    learn_masks: bool = True

    # Loader settings.
    data_type: Literal["colmap", "blender"] = "colmap"
    data_factor: int = 1
    debug_data_loading: bool = False
    crop_imgs: bool = False
    use_crops: bool = False
    use_bg_masks: bool = True
    use_dense: bool = False
    debug_every: int = 100

    # Captured-scene schedules.
    dynamic_eval_steps: List[int] = field(
        default_factory=lambda: [
            1,
            1_000,
            2_000,
            4_000,
            5_000,
            6_000,
            8_000,
            12_000,
            15_000,
            30_000,
            60_000,
            80_000,
            100_000,
        ]
    )
    dynamic_save_steps: List[int] = field(
        default_factory=lambda: [
            1,
            1_000,
            3_000,
            5_000,
            7_000,
            9_000,
            10_000,
            15_000,
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
    run_eval: bool = False
    init_type: Literal["sfm", "random", "blender_pts"] = "sfm"
    return_mask: bool = True
    viz_mask: bool = False
    param_loss_reg: float = 0.0
    use_wandb: bool = True

    # Rendering and evaluation.
    render_tracks: bool = False
    tracking_window: int = 35
    save_pc_imgs: bool = True
    render_spacetime_viz: bool = False
    render_demo_viz: bool = False
    skip_pc: bool = False
    interpolation_factor: int = 1
    render_interpolation_frames: bool = False
    flip_x: bool = True
    flip_y: bool = False
    flip_z: bool = True
    use_mask_psnr: bool = True
