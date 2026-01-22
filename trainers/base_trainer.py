import torch
import torch.nn.functional as F
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.compression import PngCompression
from helpers.utils import AppearanceOptModule, CameraOptModule
from typing_extensions import assert_never
import nerfview
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from helpers.criterions import psnr
import viser
import math

class BaseEngine:
    """
    Base class for Trainer and Evaluator
    Stores shared attributes but no shared methods
    """
    def __init__(self, cfg, gaussians, dynamical_model, model_list, model_index, trainset,testset, paths, device):
        self.cfg = cfg
        self.gaussians = gaussians
        self.dynamical_model = dynamical_model
        self.model_list = model_list
        self.model_index = model_index
        self.device = device
        self.trainset = trainset
        self.testset =testset 
        self.paths = paths 

        #Sets up the paths
        self.ckpt_dir = self.paths["ckpt_dir"]
        self.stats_dir_static = self.paths["stats_dir_static"]
        self.stats_dir_dynamic = self.paths["stats_dir_dynamic"]
        self.render_dir_static = self.paths["render_dir_static"]
        self.render_dir_dynamic = self.paths["render_dir_dynamic"]
        self.gt_videos_dir  = self.paths["gt_videos_dir"]
        print(f"saving checkpoint at {self.paths['ckpt_dir']}")

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.gaussians.splats, self.gaussians.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.gaussians.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        self.gaussians.strategy = self.cfg.strategy #set the strategy of the gaussians
        self.gaussians.strategy.absgrad = self.cfg.use_absgrad
        if self.cfg.use_absgrad:
            print("Using absgrad for pruning/splitting")

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)

        self.app_optimizers = []
        if cfg.app_opt:
            feature_dim = 32 if cfg.app_opt else None
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
        self.fixed_bkgd = (
            torch.tensor(cfg.bkgd_color, device=self.device)[None, :] 
        )
        assert (self.fixed_bkgd <= 1).all() and (self.fixed_bkgd >= 0).all(), "bkgd color needs to be normalized bw [0,1]"

        #Metrics, for psnr, we use our own implementation as it supports mask psnr
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = psnr 
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(self.device) #. If set to True will instead expect input to be in the [0,1] range.

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )