"""Configuration overrides for synthetic Blender scenes."""

from dataclasses import dataclass

from configs.common_config import CommonConfig


@dataclass
class Config(CommonConfig):
    """Synthetic-scene settings layered on the shared training config."""

    half_normalize: bool = False

    # This was historically an unannotated class variable. Keep it out of the
    # dataclass and CLI field sets for backwards compatibility.
    feature_out_output_dim = 64

    rtol_train_all: float = 1e-4
    atol_train_all: float = 1e-5
    adjoint_train_all: bool = True

    chamfer_reg_box: float = 0.0
    use_mesh_vertices: bool = True

    render_foreground: bool = True
    render_only_foreground: bool = False
    use_intersection: bool = True
    use_mask_projection: bool = False
    train_interp: bool = True
    compute_masked_psnr: bool = False
    metric_scene: str = ""
