import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from typing import Tuple
from datetime import datetime
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import secrets
from datasets.colmap import DynamicParser
from helpers.utils import safe_state, setup_directories, verify_optimizer_parameter_references
from trainers.trainer import Trainer
from trainers.evaluate import Evaluator
from models.gaussian_model import Gaussians


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, cfg
    ) -> None:
        safe_state(42) #get seed

        self.cfg = cfg
        self.device = "cuda:0"
        unique_str = secrets.token_hex(4) #in case we run two exps at same time.
        now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        self.now = now
        checkpoint_path = (cfg.dynamic_ckpt or cfg.static_ckpt or [None])[0]
        if cfg.result_dir == "": #if result_dir exists, just use it
            if checkpoint_path:
                cfg.result_dir = checkpoint_path.split('/ckpts/')[0]  #Assume the checkpoints are always contained in the ckpts folder.

            elif cfg.per_time_upper_bound is not None:  #NOTE: if per timestep static ckpt dir 
                scene = cfg.data_dir.split("/")[-1] #scene is always after the last /
                # cfg.result_dir = f"{cfg.result_dir}_{now}"
                if cfg.result_dir == "":
                    cfg.result_dir = f"./results/{scene}/per_timestep_static_{cfg.per_time_upper_bound}"
                os.makedirs(f"{cfg.result_dir}", exist_ok=True)
            
            else:
                # Where to dump results.
                scene = cfg.data_dir.split("/")[-1] #scene is always after the last /
                # cfg.result_dir = f"{cfg.result_dir}_{now}"
                cfg.result_dir = f"./results/{scene}/{unique_str}_{now}"
                os.makedirs(f"{cfg.result_dir}", exist_ok=True)
                # Setup output directories.

        elif cfg.per_segment_ckpt_dir != "":
            cfg.result_dir = os.path.join(cfg.per_segment_ckpt_dir, "combined")
            os.makedirs(cfg.result_dir, exist_ok=True)

        self.paths = setup_directories(cfg.result_dir)

        # Wandb 
        #Only use wandb to log the dynamic training
        if not cfg.use_wandb:
            print("disabling wandb")
            os.environ['WANDB_MODE'] = 'disabled'

        #if disabled, we still initialize
        self.wandb_run = wandb.init(
            project="my-awesome-project",
            name=f"{cfg.result_dir}_{now}",
            config=vars(cfg) #some configs set after are not important to the run.
        )

        if cfg.data_type == "colmap":
            from datasets.colmap import Dynamic_Dataset, Dynamic_Datasetshared

            white_bkgd = cfg.bkgd_color == [1,1,1]
            # Load data: Training data should contain initial points and colors.
            self.parser = DynamicParser(
                data_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=cfg.normalize_world_space,
                test_every=cfg.test_every,
                align_timesteps = cfg.align_timesteps,
                dates = cfg.dates,
                use_dense = cfg.use_dense,
                subsample_factor = cfg.subsample_factor,
                start_from=cfg.start_from,
                crop_imgs=cfg.crop_imgs,
                use_crops=cfg.use_crops,
                use_bg_masks = cfg.use_bg_masks,
                end_until=cfg.end_until,
                include_end=cfg.include_end,
                white_bkgd=white_bkgd
            )

            #This loads all the data and stores it in a dict
            self.shared_dataset = Dynamic_Datasetshared(
                parser=self.parser,
                apply_mask=cfg.apply_mask,
                debug_data_loading=cfg.debug_data_loading
            )
            shared_data = self.shared_dataset.get_shared_data()
            self.trainset = Dynamic_Dataset(
                parser=self.parser,
                split="train",
                shared_data=shared_data,
                is_reverse=cfg.is_reverse,
                downsample_factor=cfg.downsample_factor,
                include_zero=cfg.include_zero,
                cam_batch_size = cfg.cam_batch_size,
                time_normalize_factor=cfg.time_normalize_factor,
                return_mask = cfg.return_mask,
            )
            
            self.testset = Dynamic_Dataset(
                parser=self.parser,
                split="test",
                shared_data=shared_data,
                is_reverse=cfg.is_reverse,
                downsample_factor=cfg.downsample_factor,
                prepend_zero=True,
                downsample_eval=cfg.downsample_eval,
                cam_batch_size = -1,
                time_normalize_factor=cfg.time_normalize_factor,
                return_mask = cfg.return_mask,
            )
            self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
            print("Scene scale:", self.scene_scale)

        elif cfg.data_type == "blender":
            from datasets.blender import Dynamic_Dataset
            self.parser = None
            self.trainset = Dynamic_Dataset(
                cfg.data_dir,
                split="train",
                is_reverse=cfg.is_reverse,
                include_zero=cfg.include_zero,
                downsample_factor=cfg.downsample_factor,
                cam_batch_size=cfg.cam_batch_size,
                half_normalize=cfg.half_normalize,
                target_shape=cfg.target_shape,
                bkgd_color= cfg.bkgd_color,
                return_mask= cfg.return_mask
            )
            #don't prepend zero in the val set so that we can pass it t=0
            #also get all cameras during eval
            #Also here test set also gets downsampled
            self.testset = Dynamic_Dataset(
                cfg.data_dir,
                split="test",
                is_reverse=cfg.is_reverse,
                downsample_factor=1, #never downsample during testing
                prepend_zero=False, 
                downsample_eval=cfg.downsample_eval,
                cam_batch_size = -1,
                half_normalize = cfg.half_normalize,
                target_shape=cfg.target_shape,
                bkgd_color= cfg.bkgd_color,
                return_mask=cfg.return_mask
            ) #using val for test

            self.scene_scale = self.trainset.scene_scale * 1.1 * cfg.global_scale #creates a little buffer
            print("dataset loaded successfully")

        # Creating the static gaussians.
        # Convention: if using SHs, concatenate them directly to begin with
        feature_dim = 32 if cfg.app_opt else None
        deformed_params_list = []
        if cfg.learn_pos:
            print("Learning gaussian means trajectory")
            deformed_params_list.append("means")
        if cfg.learn_quat:
            print("Learning gaussians quaternion trajectory")
            deformed_params_list.append("quats")
        if cfg.learn_scales:
            print("Learning gaussians scales trajectory")
            deformed_params_list.append("scales")
        if cfg.learn_shs:
            print("Learning gaussians colors trajectory")
            deformed_params_list.append("shs")
        if cfg.learn_opacities:
            print("Learning gaussians opacities trajectory")
            deformed_params_list.append("opacities")

        print("Initializing gaussians")
        self.gaussians = Gaussians(
            parser=self.parser,
            first_mesh_path= self.trainset.first_mesh_path,
            init_type=cfg.init_type,
            app_opt = cfg.app_opt,
            packed  = cfg.packed,
            antialiased = cfg.antialiased,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            deformed_params_list = deformed_params_list,
            scale_activation=self.cfg.scale_activation,
            device=self.device,
            learn_mask=cfg.learn_masks
        )

        print("Model initialized. Number of GS:", len(self.gaussians.splats["means"]))

        self.model_list = []
        self.model_index = 0
        #Creating the neural ODE.
        if cfg.version == "4dgs":
            from models.deformation import deform_network
            print("using 4dgs version")
            self.dynamical_model = deform_network(cfg).to(self.device)

        elif cfg.version == "ours":
            from models.neural_ode import DynamicalModel
            print("using ours non-split")
            self.dynamical_model = DynamicalModel(
                self.gaussians.param_feature_dim,
                augment_dim=cfg.augment_dim,
                hidden_dim=cfg.hidden_dim,
                hidden_depth=cfg.hidden_depth,
                encoding=cfg.encoding,
                non_linearity_name=cfg.non_linearity_name,
                rtol=cfg.rtol,
                atol=cfg.atol,
                adjoint=cfg.adjoint,
                neural_ode_lr = cfg.unscaled_neural_ode_lr_init, 
                encoder_lr = cfg.unscaled_encoder_lr_init,
                gamma= cfg.gamma,
                scene_scale = self.scene_scale if cfg.data_type =="blender" else self.scene_scale,
                event_fn=None,
                concat_remaining=cfg.concat_remaining,
                max_steps = cfg.dynamic_max_steps,
                adjust_lr_w_scene = cfg.adjust_lr_w_scene,
                resnet_init = cfg.resnet_init,
                use_timenet = cfg.use_timenet,
                use_skip = cfg.use_skip,
                data_type = cfg.data_type,
                multires = cfg.ours_multires,
                spatial_temp_resolution = cfg.spatial_temp_resolution,
                method = cfg.method,
                feature_out_output_dim = cfg.feature_out_output_dim
            ).to(self.device)

        print(self.dynamical_model)

        #Use this for rendering images and computing some PSNR during training
        self.evaluator = Evaluator(
            cfg=cfg,
            parser=self.parser,
            gaussians=self.gaussians,
            dynamical_model=self.dynamical_model,
            model_list=self.model_list,
            model_index=self.model_index,
            trainset=self.trainset,
            testset=self.testset,
            paths=self.paths,
            device=self.device,
        )

        self.trainer = Trainer(
            cfg=cfg,
            parser=self.parser,
            gaussians=self.gaussians,
            dynamical_model=self.dynamical_model,
            model_list = self.model_list,
            model_index = self.model_index,
            trainset=self.trainset,
            testset=self.testset,
            paths=self.paths,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            device=self.device,
        )

    def reset_gaussians(self):
        """Reset gaussians by re-initializing them"""
        feature_dim = 32 if self.cfg.app_opt else None
        deformed_params_list = []
        if self.cfg.learn_pos:
            print("Learning gaussian means trajectory")
            deformed_params_list.append("means")
        if self.cfg.learn_quat:
            print("Learning gaussians quaternion trajectory")
            deformed_params_list.append("quats")
        if self.cfg.learn_scales:
            print("Learning gaussians scales trajectory")
            deformed_params_list.append("scales")
        if self.cfg.learn_shs:
            print("Learning gaussians colors trajectory")
            deformed_params_list.append("shs")
        if self.cfg.learn_opacities:
            print("Learning gaussians opacities trajectory")
            deformed_params_list.append("opacities")
        self.gaussians = Gaussians(
            parser=self.parser,
            init_type=self.cfg.init_type,
            app_opt = self.cfg.app_opt,
            packed  = self.cfg.packed,
            antialiased = self.cfg.antialiased,
            init_num_pts=self.cfg.init_num_pts,
            init_extent=self.cfg.init_extent,
            init_opacity=self.cfg.init_opa,
            init_scale=self.cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=self.cfg.sh_degree,
            sparse_grad=self.cfg.sparse_grad,
            visible_adam=self.cfg.visible_adam,
            batch_size=self.cfg.batch_size,
            feature_dim=feature_dim,
            deformed_params_list = deformed_params_list,
            scale_activation=self.cfg.scale_activation,
            device=self.device,
        )

    def run(self):
        """
        Main entrypoint for training/eval
        """
        cfg = self.cfg
        if cfg.dynamic_ckpt is None:
            #Static training
            skip_static_eval = cfg.skip_static_eval
            if cfg.static_ckpt is None:
                self.trainer.train_static(0)

            else:
                #Load normal gaussian 
                ckpts = [
                    torch.load(file, map_location=self.device, weights_only=False)
                    for file in cfg.static_ckpt
                ]
                ckpt = ckpts[0] #always only using one checkpoint
                for k in self.gaussians.splats.keys():
                    self.gaussians.splats[k].data = ckpt["splats"][k]
                if "optimizer_state_dict" in ckpt: 
                    for k, state_dict in ckpt["optimizer_state_dict"].items():
                        if k in self.gaussians.optimizers:
                            self.gaussians.optimizers[k].load_state_dict(state_dict)
                verify_optimizer_parameter_references(self.gaussians) #verify stuff is loaded correctly.
                num_gauss = self.gaussians.splats["means"].shape[0]
                print(f"loaded gaussian ckpts, we have {num_gauss} gaussians")
                step = ckpts[0]["step"]

                #2. Load initial_parameters
                if not skip_static_eval:    
                    self.evaluator.static_eval(step=step, time_index=0)
                    self.evaluator.static_render_traj(step=step, time_index=0)
                    if cfg.compression is not None:
                        self.run_compression(step=step)
            print("starting dynamic training!")
            full_trajectory = np.load(cfg.full_trajectory_path, allow_pickle=True) if cfg.full_trajectory_path != "" else None
            if cfg.data_type == "blender":
                if cfg.train_interp:
                    self.trainer.train_dynamic_blender_masked(full_trajectory=full_trajectory)
                else:
                    self.trainer.train_dynamic_blender(full_trajectory=full_trajectory)
            elif cfg.data_type == "colmap":
                self.trainer.train_dynamic_captured(full_trajectory=full_trajectory)

            step = cfg.dynamic_max_steps - 1
            self.wandb_run.alert(title="Job done", text=f"{cfg.result_dir}_{self.now} is done") #only cover training.
            init_params = self.trainer.init_params

        else:
            ckpts = [
                torch.load(file, map_location=self.device, weights_only=False)
                for file in cfg.dynamic_ckpt
            ]

            self.dynamical_model.load_state_dict(ckpts[0]["neural_ode"])
            init_params = ckpts[0]["init_params"]
            for k in self.gaussians.splats.keys():
                self.gaussians.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
            step = ckpts[0]["step"]
            if cfg.resume_dyn_training:
                print(f"resume training at step {step}")
                if cfg.load_optimizers:
                    print("loading optimizers schedulers when resume training!")
                    if "encoder_optimizer_state_dict" in ckpts[0].keys(): 
                        if cfg.encoding in ["hexplane", "ingp"]:
                            self.trainer.dynamical_model.optimizers["encoder_optimizer"].load_state_dict(ckpts[0]["encoder_optimizer_state_dict"])
                            self.trainer.dynamical_model.schedulers["encoder_scheduler"].load_state_dict(ckpts[0]["encoder_scheduler_state_dict"])
                            
                    self.trainer.dynamical_model.optimizers["neural_ode_optimizer"].load_state_dict(ckpts[0]["neural_ode_optimizer_state_dict"])
                    self.trainer.dynamical_model.schedulers["neural_ode_scheduler"].load_state_dict(ckpts[0]["neural_ode_scheduler_state_dict"])
                timestep_counter = ckpts[0]["timestep_counter"]
                lst_of_prog_indices = ckpts[0]["lst_of_prog_indices"]
                active_timesteps = ckpts[0]["active_timesteps"]
                next_t = ckpts[0]["next_t"]
                if cfg.version == "ours3":
                    if cfg.previous_init_params_path != "":
                        previous_init_params = torch.load(cfg.previous_init_params_path)
                    else:
                        previous_init_params = None
                    self.trainer.train_split(previous_init_params=previous_init_params, init_step=step, timestep_counter=timestep_counter)
                else:
                    full_trajectory = np.load(cfg.full_trajectory_path) if cfg.full_trajectory_path != "" else None
                    if cfg.data_type == "blender":
                        if cfg.train_interp:
                            self.trainer.train_dynamic_blender_masked(
                                init_step=step,
                                timestep_counter=timestep_counter,
                                lst_of_prog_indices=lst_of_prog_indices,
                                next_t=next_t,
                                active_timesteps=active_timesteps,
                                full_trajectory=full_trajectory,
                            )
                        else:
                            self.trainer.train_dynamic_blender(
                                init_step=step,
                                timestep_counter=timestep_counter,
                                lst_of_prog_indices=lst_of_prog_indices,
                                next_t=next_t,
                                active_timesteps=active_timesteps,
                                full_trajectory=full_trajectory
                            )
                    elif cfg.data_type == "colmap":
                        self.trainer.train_dynamic_captured(
                            init_step=step,
                            timestep_counter=timestep_counter,
                            lst_of_prog_indices=lst_of_prog_indices,
                            next_t=next_t,
                            active_timesteps=active_timesteps,
                            full_trajectory=full_trajectory
                        )
                        

        if cfg.run_eval:
            if cfg.data_type == "blender":
                self.evaluator.dynamic_eval(init_params, step=step)
            elif cfg.data_type == "colmap":
                raise NotImplementedError #need to add the bounding box and stuff.
                self.evaluator.dynamic_eval_captured(init_params, step=step)


    def full_eval(self):
        """
        Main entrypoint for full evaluation (i.e. rendering all train/test images in an uniform format)
        Only doing this on the dynamic rendering 
        """
        cfg = self.cfg
        ckpts = [
            torch.load(file, map_location=self.device, weights_only=False)
            for file in cfg.dynamic_ckpt
        ]
        self.dynamical_model.load_state_dict(ckpts[0]["neural_ode"])
        init_params = ckpts[0]["init_params"]
        for k in self.gaussians.splats.keys():
            self.gaussians.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        if cfg.data_type == "blender":
            self.evaluator.full_dynamic_eval(init_params, step=step)
        elif cfg.data_type == "colmap":
            self.evaluator.full_dynamic_eval_captured(init_params, step=step)

    def generate_trajectory(self):
        """
        Generate trajectory before training
        """
        cfg = self.cfg
        ckpts = [
            torch.load(file, map_location=self.device, weights_only=False)
            for file in cfg.static_ckpt
        ]
        ckpt = ckpts[0] #always only using one checkpoint
        for k in self.gaussians.splats.keys():
            self.gaussians.splats[k].data = ckpt["splats"][k]
        if "optimizer_state_dict" in ckpt: 
            for k, state_dict in ckpt["optimizer_state_dict"].items():
                if k in self.gaussians.optimizers:
                    self.gaussians.optimizers[k].load_state_dict(state_dict)
        verify_optimizer_parameter_references(self.gaussians) #verify stuff is loaded correctly.
        num_gauss = self.gaussians.splats["means"].shape[0]
        print(f"loaded gaussian ckpts, we have {num_gauss} gaussians")

        self.trainer.generate_trajectory()
        

    def generate_gt(self):
        """
        Use this to generate ground truth images/point clouds
        """
        cfg = self.cfg
        self.evaluator.full_render_gt()
            
    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        compress_dir = f"{self.cfg.result_dir}/compression/"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()
